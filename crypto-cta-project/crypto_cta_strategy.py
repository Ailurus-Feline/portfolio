from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ccxt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


pd.set_option("display.max_columns", 20)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"
RESULTS = PROJECT_ROOT / "results"
CLASS1_CSV = RESULTS / "class1_trend" / "csv"
CLASS1_FIGURES = RESULTS / "class1_trend" / "figures"
CLASS2_CSV = RESULTS / "class2_factor" / "csv"
CLASS2_FIGURES = RESULTS / "class2_factor" / "figures"
CLASS3_CSV = RESULTS / "class3_combo" / "csv"
CLASS3_BACKTESTS = CLASS3_CSV / "backtests"
CLASS3_FIGURES = RESULTS / "class3_combo" / "figures"
CLASS4_CSV = RESULTS / "class4_risk" / "csv"
CLASS4_FIGURES = RESULTS / "class4_risk" / "figures"
CSV_OUT = CLASS1_CSV
FIGURES = CLASS1_FIGURES
FACTOR_DATA_OUT = CLASS2_CSV
FACTOR_RESULTS_OUT = CLASS2_CSV
FACTOR_FIGURES_OUT = CLASS2_FIGURES
COMBO_RESULTS_OUT = CLASS3_CSV
COMBO_BACKTESTS_OUT = CLASS3_BACKTESTS
COMBO_FIGURES_OUT = CLASS3_FIGURES
for path_obj in [
    DATA_RAW,
    DATA_CLEAN,
    CLASS1_CSV,
    CLASS1_FIGURES,
    CLASS2_CSV,
    CLASS2_FIGURES,
    CLASS3_CSV,
    CLASS3_BACKTESTS,
    CLASS3_FIGURES,
    CLASS4_CSV,
    CLASS4_FIGURES,
]:
    path_obj.mkdir(parents=True, exist_ok=True)

# Baseline configuration
EXCHANGE_ID = "binance"
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
SINCE = "2020-01-01T00:00:00Z"
LIMIT_PER_REQUEST = 1000
MAX_FETCH_BATCHES = 200
EXTENDED_SYMBOL = "SOL/USDT"
TOP_FACTOR_COUNTS = [3, 5]
MAX_TOP_FACTOR_ANALYSIS = max(TOP_FACTOR_COUNTS)
HOURS_PER_DAY = 24
HOURS_PER_WEEK = 24 * 7

COMBO_HORIZONS: dict[str, dict[str, int]] = {
    "1h": {"bars": 1, "periods_per_year": 24 * 365},
    "1d": {"bars": HOURS_PER_DAY, "periods_per_year": 365},
    "1w": {"bars": HOURS_PER_WEEK, "periods_per_year": 52},
}
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
COMBO_TOP_FACTOR_COUNT = 5
COMBO_RIDGE_ALPHA = 20.0
COMBO_DEFAULT_QUANTILE = 0.90
COMBO_SENSITIVITY_QUANTILES = [0.60, 0.65, 0.70, 0.75, 0.80]
COMBO_SENSITIVITY_FEES_BPS = [2.0, 5.0, 10.0]
COMBO_LOOKBACK_WINDOWS = [12 * HOURS_PER_DAY, 24 * HOURS_PER_DAY, 48 * HOURS_PER_DAY]

# Class 4: exit rules + multi-asset risk allocation
CLASS4_SYMBOLS = ["BTC/USDT", "ETH/USDT"]
CLASS4_METHOD = "ridge"
CLASS4_HORIZON = "1h"
CLASS4_QUANTILE = COMBO_DEFAULT_QUANTILE
CLASS4_FEE_BPS = 2.0
CLASS4_FIXED_TP_PCT = 0.02
CLASS4_FIXED_SL_PCT = 0.01
CLASS4_ATR_WINDOW = 24
CLASS4_ATR_TP_MULT = 2.0
CLASS4_ATR_SL_MULT = 1.0
CLASS4_TIME_STOP_BARS = 24
CLASS4_TRAIL_PCT = 0.10
CLASS4_MVO_RISK_AVERSION = 5.0
CLASS4_PERIODS_PER_YEAR = 24 * 365


def make_exchange(exchange_id: str = EXCHANGE_ID):
    """Create a CCXT exchange client with rate limiting enabled."""
    exchange_cls = getattr(ccxt, exchange_id)
    return exchange_cls({"enableRateLimit": True})


def fetch_ohlcv_loop(
    exchange,
    symbol: str,
    timeframe: str,
    since_iso: str,
    limit: int = 1000,
    max_batches: int = 50,
) -> pd.DataFrame:
    """Fetch OHLCV candles in batches."""
    since_ms = exchange.parse8601(since_iso)
    all_rows: list[list[float]] = []

    for batch in range(max_batches):
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not rows:
            break

        all_rows.extend(rows)
        last_ts = rows[-1][0]
        next_since = last_ts + 1

        print(f"{symbol} batch {batch + 1:02d}: {len(rows)} rows, last = {exchange.iso8601(last_ts)}")

        if next_since <= since_ms:
            break
        since_ms = next_since

        time.sleep(exchange.rateLimit / 1000 if getattr(exchange, "rateLimit", None) else 0.2)

        if since_ms >= int(datetime.now(timezone.utc).timestamp() * 1000):
            break

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def generate_demo_ohlcv(
    symbol: str,
    start: str = SINCE,
    periods: int = 24 * 180,
    freq: str = "1h",
    seed: int = 7,
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV for offline fallback."""
    rng = np.random.default_rng(seed + sum(ord(char) for char in symbol))
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    drift = 0.00002
    vol = 0.015 if symbol.startswith("BTC") else 0.020
    rets = drift + vol * rng.standard_normal(periods)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(0.004 * close * rng.standard_normal(periods))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(loc=1000, scale=300, size=periods))
    return pd.DataFrame(
        {
            "timestamp": idx.astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def parse_timestamp_auto(ts: pd.Series) -> pd.Series:
    """Parse timestamps that may be in s/ms/us/ns epoch or ISO string format."""
    ts_num = pd.to_numeric(ts, errors="coerce")
    if ts_num.notna().mean() > 0.8 and ts_num.notna().any():
        abs_med = float(ts_num.abs().median())
        if abs_med > 1e17:
            unit = "ns"
        elif abs_med > 1e14:
            unit = "us"
        elif abs_med > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(ts_num, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(ts, utc=True, errors="coerce")


def download_or_demo(symbols: list[str] = SYMBOLS) -> dict[str, pd.DataFrame]:
    """Download OHLCV for all symbols; fallback to demo data on failure."""
    data: dict[str, pd.DataFrame] = {}
    try:
        exchange = make_exchange(EXCHANGE_ID)
        exchange.load_markets()
        for symbol in symbols:
            df = fetch_ohlcv_loop(
                exchange,
                symbol,
                TIMEFRAME,
                SINCE,
                LIMIT_PER_REQUEST,
                max_batches=MAX_FETCH_BATCHES,
            )
            if len(df) == 0:
                raise RuntimeError(f"No data returned for {symbol}")
            safe_name = symbol.replace("/", "_")
            raw_path = DATA_RAW / f"{safe_name}_{TIMEFRAME}_raw.csv"
            df.to_csv(raw_path, index=False)
            print(f"Saved raw download: {raw_path}")
            data[symbol] = df
    except Exception as error:
        print("Data download failed; using demo data instead.")
        print("Reason:", repr(error))
        for symbol in symbols:
            demo_df = generate_demo_ohlcv(symbol)
            safe_name = symbol.replace("/", "_")
            raw_path = DATA_RAW / f"{safe_name}_{TIMEFRAME}_raw_demo.csv"
            demo_df.to_csv(raw_path, index=False)
            print(f"Saved raw demo data: {raw_path}")
            data[symbol] = demo_df
    return data


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV into analysis-ready format."""
    out = df.copy()
    out["datetime"] = parse_timestamp_auto(out["timestamp"])
    out = out.dropna(subset=["datetime"])
    out = out.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    out = out[["datetime", "open", "high", "low", "close", "volume"]]

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def momentum(close: pd.Series, lookback: int) -> pd.Series:
    return close / close.shift(lookback) - 1


def volume_price_trend(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    ret = close.pct_change()
    vol_z = zscore(volume, window)
    return ret.rolling(window).sum() * vol_z


def range_position(close: pd.Series, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    rolling_high = high.rolling(window).max()
    rolling_low = low.rolling(window).min()
    return (close - rolling_low) / (rolling_high - rolling_low) - 0.5


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    ret = close.pct_change()
    gain = ret.clip(lower=0)
    loss = (-ret).clip(lower=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rolling_volatility(ret: pd.Series, window: int) -> pd.Series:
    return ret.rolling(window).std()


def rolling_autocorr(ret: pd.Series, window: int) -> pd.Series:
    return ret.rolling(window).corr(ret.shift(1))


def bollinger_position(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pos = (close - lower) / (upper - lower)
    return pos


def macd_signal_normalized(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return (macd - macd_signal) / close


def distance_from_ma(close: pd.Series, window: int) -> pd.Series:
    ma = close.rolling(window).mean()
    return close / ma - 1


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    pos_flow = np.where(tp > tp.shift(1), rmf, 0.0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0.0)
    pos_sum = pd.Series(pos_flow, index=tp.index).rolling(window).sum()
    neg_sum = pd.Series(neg_flow, index=tp.index).rolling(window).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    return mfi


def _stochastic(series: pd.Series, window: int) -> pd.Series:
    low = series.rolling(window).min()
    high = series.rolling(window).max()
    return (series - low) / (high - low) * 100


def schaff_trend_cycle(close: pd.Series, cycle: int = 10) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    pf1 = _stochastic(macd, cycle).ewm(span=3, adjust=False).mean()
    stc = _stochastic(pf1, cycle).ewm(span=3, adjust=False).mean()
    return stc


def dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    alpha = 1 / max(window, 1)
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    plus_dm_s = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    minus_dm_s = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    plus_di = 100 * plus_dm_s / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})


def build_alpha_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1h"] = out["close"].pct_change()
    out["future_ret_1h"] = out["close"].shift(-1) / out["close"] - 1

    # Momentum.
    out["factor_mom_4w"] = momentum(out["close"], 4 * HOURS_PER_WEEK)
    out["factor_mom_12w"] = momentum(out["close"], 12 * HOURS_PER_WEEK)
    out["factor_mom_26w"] = momentum(out["close"], 26 * HOURS_PER_WEEK)
    out["factor_mom_52w"] = momentum(out["close"], 52 * HOURS_PER_WEEK)

    # Reversal.
    out["factor_rev_1w"] = -out["close"].pct_change(HOURS_PER_WEEK)
    out["factor_rev_2w"] = -out["close"].pct_change(2 * HOURS_PER_WEEK)

    # Volatility and volatility change.
    out["factor_vol_4w"] = rolling_volatility(out["ret_1h"], 4 * HOURS_PER_WEEK)
    out["factor_vol_12w"] = rolling_volatility(out["ret_1h"], 12 * HOURS_PER_WEEK)
    out["factor_vol_26w"] = rolling_volatility(out["ret_1h"], 26 * HOURS_PER_WEEK)
    out["factor_vol_change"] = out["factor_vol_4w"] / out["factor_vol_12w"].replace(0, np.nan)

    # Return autocorrelation.
    out["factor_ret_autocorr_12w"] = rolling_autocorr(out["ret_1h"], 12 * HOURS_PER_WEEK)

    # Existing baseline factors.
    out["factor_vol_price"] = volume_price_trend(out["close"], out["volume"], 24)
    out["factor_range_pos"] = range_position(out["close"], out["high"], out["low"], 48)
    out["factor_rsi_14"] = calculate_rsi(out["close"], window=14)

    # Bollinger / MACD / distance to MA.
    out["factor_bb_position_20"] = bollinger_position(out["close"], window=20)
    out["factor_macd_signal"] = macd_signal_normalized(out["close"], fast=12, slow=26, signal=9)
    out["factor_dist_ma_10"] = distance_from_ma(out["close"], window=10)
    out["factor_dist_ma_50"] = distance_from_ma(out["close"], window=50)

    # Advanced indicators.
    out["factor_mfi_14"] = money_flow_index(out["high"], out["low"], out["close"], out["volume"], window=14)
    out["factor_stc"] = schaff_trend_cycle(out["close"], cycle=10)
    dmi = dmi_adx(out["high"], out["low"], out["close"], window=14)
    out["factor_plus_di"] = dmi["plus_di"]
    out["factor_minus_di"] = dmi["minus_di"]
    out["factor_adx"] = dmi["adx"]
    out["factor_dmi_spread"] = dmi["plus_di"] - dmi["minus_di"]

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def calc_ic_table(data: pd.DataFrame, factors: list[str], target: str = "future_ret_1h") -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for factor in factors:
        tmp = data[[factor, target]].dropna()
        pearson = tmp[factor].corr(tmp[target], method="pearson")
        # Clip extreme factor values before correlating — dampens outlier-driven IC
        clip_pearson = (
            tmp[factor]
            .clip(tmp[factor].quantile(0.05), tmp[factor].quantile(0.95))
            .corr(tmp[target], method="pearson")
        )
        spearman = tmp[factor].corr(tmp[target], method="spearman")
        rows.append(
            {
                "factor": factor,
                "pearson_ic": float(pearson) if pd.notna(pearson) else np.nan,
                "clipped_ic": float(clip_pearson) if pd.notna(clip_pearson) else np.nan,
                "spearman_ic": float(spearman) if pd.notna(spearman) else np.nan,
                "n_obs": len(tmp),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_ic", key=lambda s: s.abs(), ascending=False)


def rolling_ic(factor: pd.Series, target: pd.Series, window: int = 500, method: str = "pearson") -> pd.Series:
    if method == "pearson":
        return factor.rolling(window).corr(target)
    return factor.rank().rolling(window).corr(target.rank())


def factor_slug(name: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in name.lower())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "factor"


def quantile_monetization(
    data: pd.DataFrame,
    factor: str,
    target: str = "future_ret_1h",
    q_high: float = 0.80,
    q_low: float = 0.20,
    window: int = 24 * 60,
    fee_bps: float = 2.0,
    direction: int = 1,
) -> pd.DataFrame:
    out = data[["datetime", factor, target]].copy()
    out[factor] = out[factor] * direction  # flip if Class-2 IC was negative

    # Thresholds use only past factor values (shift + rolling quantile)
    hist_factor = out[factor].shift(1)
    out["q_high"] = hist_factor.rolling(window, min_periods=window // 6).quantile(q_high)
    out["q_low"] = hist_factor.rolling(window, min_periods=window // 6).quantile(q_low)

    out["signal"] = 0
    out.loc[out[factor] > out["q_high"], "signal"] = 1
    out.loc[out[factor] < out["q_low"], "signal"] = -1

    out["position"] = out["signal"]  # position at t earns target return over t -> t+1
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["fee"] = out["turnover"] * fee_bps / 10_000
    out["pnl"] = out["position"] * out[target] - out["fee"]
    out["equity"] = (1 + out["pnl"].fillna(0)).cumprod()
    return out


def backtest_metrics(bt: pd.DataFrame, periods_per_year: int = 24 * 365) -> pd.Series:
    pnl = bt["pnl"].dropna()
    equity = bt["equity"].dropna()
    ann_return = equity.iloc[-1] ** (periods_per_year / len(equity)) - 1 if len(equity) > 0 else np.nan
    sharpe = pnl.mean() / pnl.std() * np.sqrt(periods_per_year) if pnl.std(ddof=1) != 0 else np.nan
    mdd = max_drawdown(equity)
    avg_turnover = bt["turnover"].mean()
    exposure = bt["position"].abs().mean()
    win_rate = (pnl > 0).mean()
    return pd.Series(
        {
            "annual_return": ann_return,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "avg_hourly_turnover": avg_turnover,
            "avg_exposure": exposure,
            "win_rate": win_rate,
        }
    )


def run_factor_research_workflow(clean_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | pd.Series]:
    btc_df = clean_data["BTC/USDT"].copy()
    alpha_df = build_alpha_dataset(btc_df)
    factor_cols = [col for col in alpha_df.columns if col.startswith("factor_")]

    ic_table = calc_ic_table(alpha_df, factor_cols)

    # Rank by |clipped IC| with a minimum sample-size guard
    min_obs = max(1000, int(len(alpha_df) * 0.2))
    ranked = ic_table[(ic_table["n_obs"] >= min_obs)].copy()
    ranked["score"] = ranked["clipped_ic"].abs()
    ranked["score"] = ranked["score"].fillna(ranked["pearson_ic"].abs())
    ranked = ranked.dropna(subset=["score"]).sort_values("score", ascending=False).reset_index(drop=True)

    if ranked.empty:
        ranked = ic_table.copy()
        ranked["score"] = ranked["clipped_ic"].abs()
        ranked["score"] = ranked["score"].fillna(ranked["pearson_ic"].abs())
        ranked = ranked.dropna(subset=["score"]).sort_values("score", ascending=False).reset_index(drop=True)

    if ranked.empty:
        raise RuntimeError("No valid factors available for ranking after IC calculation.")

    top_count = min(MAX_TOP_FACTOR_ANALYSIS, len(ranked))
    top_factors = ranked.head(top_count).copy()
    top_factors.insert(0, "rank", range(1, len(top_factors) + 1))
    top_factors["direction"] = np.where(top_factors["pearson_ic"] < 0, -1, 1)  # feed into Class 3 combo matrix

    selected_factor = str(top_factors.iloc[0]["factor"])
    selected_ic = float(top_factors.iloc[0]["pearson_ic"])
    selected_direction = int(top_factors.iloc[0]["direction"])

    top_rolling_rows: list[pd.DataFrame] = []
    top_metrics_rows: list[dict[str, float | int | str]] = []
    top_sensitivity_rows: list[dict[str, float | int | str]] = []
    top_equity_curves = pd.DataFrame({"datetime": alpha_df["datetime"]})

    for row in top_factors.itertuples(index=False):
        factor_name = str(row.factor)
        direction = int(row.direction)

        valid_n_i = int(alpha_df[[factor_name, "future_ret_1h"]].dropna().shape[0])
        rolling_window_i = min(2400, max(200, valid_n_i // 5))
        rolling_i = rolling_ic(alpha_df[factor_name], alpha_df["future_ret_1h"], window=rolling_window_i)
        top_rolling_rows.append(
            pd.DataFrame(
                {
                    "datetime": alpha_df["datetime"],
                    "factor": factor_name,
                    "rolling_ic": rolling_i,
                    "rolling_window": rolling_window_i,
                }
            )
        )

        bt_i = quantile_monetization(
            alpha_df,
            factor_name,
            q_high=0.80,
            q_low=0.20,
            window=24 * 300,
            fee_bps=2.0,
            direction=direction,
        )
        top_equity_curves[factor_name] = bt_i["equity"]

        metrics_i = backtest_metrics(bt_i).to_dict()
        top_metrics_rows.append(
            {
                "factor": factor_name,
                "direction": direction,
                "n_obs": valid_n_i,
                "rolling_window": rolling_window_i,
                "pearson_ic": float(row.pearson_ic),
                "clipped_ic": float(row.clipped_ic) if pd.notna(row.clipped_ic) else np.nan,
                "spearman_ic": float(row.spearman_ic) if pd.notna(row.spearman_ic) else np.nan,
                **{k: float(v) if pd.notna(v) else np.nan for k, v in metrics_i.items()},
            }
        )

        factor_sensitivity_rows: list[pd.Series] = []
        for q in [0.60, 0.70, 0.75, 0.80, 0.85]:
            tmp_bt = quantile_monetization(
                alpha_df,
                factor_name,
                q_high=q,
                q_low=1 - q,
                window=24 * 60,
                fee_bps=2.0,
                direction=direction,
            )
            tmp_metrics = backtest_metrics(tmp_bt)
            tmp_metrics["q_high"] = q
            tmp_metrics["q_low"] = 1 - q
            factor_sensitivity_rows.append(tmp_metrics)
            top_sensitivity_rows.append(
                {
                    "factor": factor_name,
                    "q_high": q,
                    "q_low": 1 - q,
                    **{k: float(v) if pd.notna(v) else np.nan for k, v in tmp_metrics.to_dict().items()},
                }
            )

        sensitivity_i = pd.DataFrame(factor_sensitivity_rows)
        slug = factor_slug(factor_name)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(alpha_df["datetime"], rolling_i, label="Rolling IC")
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"Rolling IC: {factor_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("IC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FACTOR_FIGURES_OUT / f"factor_top_{slug}_rolling_ic.png", dpi=150)
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(bt_i["datetime"], bt_i["equity"], label="Equity")
        ax.set_title(f"Quantile Monetization Equity: {factor_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FACTOR_FIGURES_OUT / f"factor_top_{slug}_equity_curve.png", dpi=150)
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sensitivity_i["q_high"], sensitivity_i["sharpe"], marker="o")
        ax.set_title(f"Sensitivity (Sharpe): {factor_name}")
        ax.set_xlabel("Upper Quantile")
        ax.set_ylabel("Sharpe")
        fig.tight_layout()
        fig.savefig(FACTOR_FIGURES_OUT / f"factor_top_{slug}_sensitivity_sharpe.png", dpi=150)
        plt.show()
        plt.close(fig)

    valid_n = int(alpha_df[[selected_factor, "future_ret_1h"]].dropna().shape[0])
    rolling_window = min(2400, max(200, valid_n // 5))
    alpha_df["rolling_ic"] = rolling_ic(
        alpha_df[selected_factor],
        alpha_df["future_ret_1h"],
        window=rolling_window,
    )

    bt = quantile_monetization(
        alpha_df,
        selected_factor,
        q_high=0.80,
        q_low=0.20,
        window=24 * 300,
        fee_bps=2.0,
        direction=selected_direction,
    )
    metrics = backtest_metrics(bt)
    metrics["selected_direction"] = selected_direction
    metrics["selected_factor"] = selected_factor

    sensitivity_rows: list[pd.Series] = []
    for q in [0.60, 0.70, 0.75, 0.80, 0.85]:
        tmp_bt = quantile_monetization(
            alpha_df,
            selected_factor,
            q_high=q,
            q_low=1 - q,
            window=24 * 60,
            fee_bps=2.0,
            direction=selected_direction,
        )
        tmp_metrics = backtest_metrics(tmp_bt)
        tmp_metrics["q_high"] = q
        tmp_metrics["q_low"] = 1 - q
        sensitivity_rows.append(tmp_metrics)
    sensitivity = pd.DataFrame(sensitivity_rows)

    top_rolling_ic = pd.concat(top_rolling_rows, ignore_index=True)
    top_metrics = pd.DataFrame(top_metrics_rows)
    top_sensitivity = pd.DataFrame(top_sensitivity_rows)

    alpha_df.to_csv(FACTOR_DATA_OUT / "factor_dataset.csv", index=False)
    ic_table.to_csv(FACTOR_DATA_OUT / "factor_ic_table.csv", index=False)
    alpha_df[["datetime", selected_factor, "future_ret_1h", "rolling_ic"]].to_csv(
        FACTOR_DATA_OUT / "factor_rolling_ic.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "selected_factor": selected_factor,
                "selected_pearson_ic": selected_ic,
                "selected_direction": selected_direction,
                "selection_min_obs": min_obs,
                "selected_n_obs": valid_n,
                "rolling_window": rolling_window,
            }
        ]
    ).to_csv(FACTOR_DATA_OUT / "factor_selection.csv", index=False)
    top_factors[
        ["rank", "factor", "pearson_ic", "clipped_ic", "spearman_ic", "n_obs", "direction", "score"]
    ].to_csv(FACTOR_DATA_OUT / "factor_top_factors.csv", index=False)
    top_rolling_ic.to_csv(FACTOR_DATA_OUT / "factor_top_rolling_ic.csv", index=False)

    for top_n in TOP_FACTOR_COUNTS:
        top_factors.head(min(top_n, len(top_factors)))[
            ["rank", "factor", "pearson_ic", "clipped_ic", "spearman_ic", "n_obs", "direction", "score"]
        ].to_csv(FACTOR_DATA_OUT / f"factor_top{top_n}_factors.csv", index=False)

    metrics.to_frame(name="value").to_csv(FACTOR_RESULTS_OUT / "factor_backtest_metrics.csv")
    sensitivity.to_csv(FACTOR_RESULTS_OUT / "factor_sensitivity.csv", index=False)
    bt.to_csv(FACTOR_RESULTS_OUT / "factor_quantile_bt_full.csv", index=False)
    bt[["datetime", "signal", "pnl", "equity"]].to_csv(FACTOR_RESULTS_OUT / "factor_quantile_bt_core.csv", index=False)
    top_metrics.to_csv(FACTOR_RESULTS_OUT / "factor_top_metrics.csv", index=False)
    top_sensitivity.to_csv(FACTOR_RESULTS_OUT / "factor_top_sensitivity.csv", index=False)
    top_equity_curves.to_csv(FACTOR_RESULTS_OUT / "factor_top_equity_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ic_table["factor"], ic_table["pearson_ic"], alpha=0.8, label="Pearson IC")
    ax.plot(ic_table["factor"], ic_table["clipped_ic"], marker="o", label="Clipped Pearson")
    ax.plot(ic_table["factor"], ic_table["spearman_ic"], marker="s", label="Spearman")
    ax.set_title("Factor IC Comparison")
    ax.set_ylabel("IC")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FACTOR_FIGURES_OUT / "factor_ic_comparison.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(alpha_df["datetime"], alpha_df["rolling_ic"], label="Rolling IC")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title(f"Rolling IC: {selected_factor}")
    ax.set_xlabel("Time")
    ax.set_ylabel("IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FACTOR_FIGURES_OUT / "factor_rolling_ic.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt["datetime"], bt["equity"], label="Equity")
    ax.set_title(f"Quantile Monetization Equity: {selected_factor}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FACTOR_FIGURES_OUT / "factor_equity_curve.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sensitivity["q_high"], sensitivity["sharpe"], marker="o")
    ax.set_title("Sensitivity: Quantile Threshold vs Sharpe")
    ax.set_xlabel("Upper Quantile")
    ax.set_ylabel("Sharpe")
    fig.tight_layout()
    fig.savefig(FACTOR_FIGURES_OUT / "factor_sensitivity_sharpe.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top_factors["factor"], top_factors["score"], alpha=0.85)
    ax.set_title(f"Top {len(top_factors)} Factors by |Clipped IC| (fallback: |Pearson IC|)")
    ax.set_ylabel("Ranking score")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FACTOR_FIGURES_OUT / "factor_top_ranked_ic.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for factor_name in top_factors["factor"]:
        factor_curve = top_rolling_ic[top_rolling_ic["factor"] == factor_name]
        ax.plot(factor_curve["datetime"], factor_curve["rolling_ic"], label=factor_name)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title(f"Rolling IC Comparison: Top {len(top_factors)} Factors")
    ax.set_xlabel("Time")
    ax.set_ylabel("IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FACTOR_FIGURES_OUT / "factor_top_rolling_ic_compare.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for factor_name in top_factors["factor"]:
        ax.plot(top_equity_curves["datetime"], top_equity_curves[factor_name], label=factor_name)
    ax.set_title(f"Top {len(top_factors)} Factor Equity Comparison")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FACTOR_FIGURES_OUT / "factor_top_equity_compare.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for factor_name in top_factors["factor"]:
        subset = top_sensitivity[top_sensitivity["factor"] == factor_name]
        ax.plot(subset["q_high"], subset["sharpe"], marker="o", label=factor_name)
    ax.set_title(f"Sensitivity (Sharpe) Comparison: Top {len(top_factors)} Factors")
    ax.set_xlabel("Upper Quantile")
    ax.set_ylabel("Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FACTOR_FIGURES_OUT / "factor_top_sensitivity_compare.png", dpi=150)
    plt.show()
    plt.close(fig)

    print("Factor data saved to:", FACTOR_DATA_OUT.resolve())
    print("Factor reports saved to:", FACTOR_RESULTS_OUT.resolve())
    print("Factor figures saved to:", FACTOR_FIGURES_OUT.resolve())
    print("Selected factor:", selected_factor, "direction:", selected_direction)
    print("Selected n_obs:", valid_n, "rolling_window:", rolling_window)
    print("Rolling IC mean:", float(alpha_df["rolling_ic"].mean()))
    print("Rolling IC std :", float(alpha_df["rolling_ic"].std()))
    print("Top factors for batch analysis:", top_factors["factor"].tolist())

    return {
        "alpha_data": alpha_df,
        "ic_table": ic_table,
        "metrics": metrics,
        "sensitivity": sensitivity,
        "bt": bt,
        "top_factors": top_factors,
        "top_metrics": top_metrics,
        "top_sensitivity": top_sensitivity,
        "top_rolling_ic": top_rolling_ic,
    }


def data_quality_report(df: pd.DataFrame, timeframe: str = TIMEFRAME) -> dict[str, object]:
    """Compute data-quality summary for one OHLCV table."""
    expected = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq=timeframe, tz="UTC")
    actual = pd.DatetimeIndex(df["datetime"])
    missing = expected.difference(actual)
    return {
        "start": df["datetime"].min(),
        "end": df["datetime"].max(),
        "rows": len(df),
        "duplicates": int(df["datetime"].duplicated().sum()),
        "missing_bars": len(missing),
    }


def add_ma_signal(df: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.DataFrame:
    # Class 1 trend baseline: signal from MA cross, execute with one-bar lag
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    out[f"ma_{fast}"] = out["close"].rolling(fast).mean()
    out[f"ma_{slow}"] = out["close"].rolling(slow).mean()
    out["signal"] = np.where(out[f"ma_{fast}"] > out[f"ma_{slow}"], 1, -1)
    out.loc[out[f"ma_{slow}"].isna(), "signal"] = 0
    out["position"] = out["signal"].shift(1).fillna(0)  # anti-look-ahead
    return out


def backtest_signal(df: pd.DataFrame, fee_bps: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    fee_rate = fee_bps / 10_000
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["cost"] = out["turnover"] * fee_rate
    out["strategy_ret"] = out["position"] * out["ret"] - out["cost"]
    out["strategy_ret"] = out["strategy_ret"].fillna(0)
    out["equity"] = 1 + (out["strategy_ret"]).cumsum()
    out["buy_hold"] = 1 + (out["ret"].fillna(0)).cumsum()
    return out


def max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""
    running_max = equity.cummax()
    dd = (equity - running_max) / 1
    return float(dd.min())


def performance_summary(bt: pd.DataFrame, periods_per_year: int = 24 * 365) -> pd.Series:
    """Summarize key performance metrics for a backtest."""
    r = bt["strategy_ret"].dropna()
    ann_ret = (bt["equity"].iloc[-1]) ** (periods_per_year / max(len(bt), 1)) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return pd.Series(
        {
            "Total Return": bt["equity"].iloc[-1] - 1,
            "Annualized Return": ann_ret,
            "Annualized Vol": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_drawdown(bt["equity"]),
            "Average Turnover": bt["turnover"].mean(),
        }
    )


def add_ma_signal_long_only(df: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.DataFrame:
    """Build long-only MA signal variant."""
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    out[f"ma_{fast}"] = out["close"].rolling(fast).mean()
    out[f"ma_{slow}"] = out["close"].rolling(slow).mean()
    out["signal"] = np.where(out[f"ma_{fast}"] > out[f"ma_{slow}"], 1, 0)
    out.loc[out[f"ma_{slow}"].isna(), "signal"] = 0
    out["position"] = out["signal"].shift(1).fillna(0)
    return out


def plot_price_and_mas(bt: pd.DataFrame, fast: int, slow: int, title: str, save_path: Path | None = None) -> None:
    """Plot close price and moving averages."""
    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt["datetime"], bt["close"], label="Close")
    ax.plot(bt["datetime"], bt[f"ma_{fast}"], label=f"MA_{fast}")
    ax.plot(bt["datetime"], bt[f"ma_{slow}"], label=f"MA_{slow}")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_equity(bt: pd.DataFrame, title: str, save_path: Path | None = None) -> None:
    """Plot strategy equity against buy-and-hold."""
    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt["datetime"], bt["equity"], label="MA strategy")
    ax.plot(bt["datetime"], bt["buy_hold"], label="Buy and hold")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def evaluate_ma_strategy(clean_df: pd.DataFrame, fast: int = 40, slow: int = 100, fee_bps: float = 2.0) -> tuple[pd.DataFrame, pd.Series]:
    """Return MA backtest table and summary for one symbol."""
    signal_df = add_ma_signal(clean_df, fast=fast, slow=slow)
    bt_df = backtest_signal(signal_df, fee_bps=fee_bps)
    return bt_df, performance_summary(bt_df)


def run_baseline_workflow(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, dict[str, pd.DataFrame | pd.Series]]]:
    """Run baseline trend workflow and persist outputs."""
    raw_data = download_or_demo(symbols)

    clean_data: dict[str, pd.DataFrame] = {}
    for symbol, raw_df in raw_data.items():
        clean_df = clean_ohlcv(raw_df)
        clean_data[symbol] = clean_df
        safe_name = symbol.replace("/", "_")
        clean_df.to_parquet(DATA_CLEAN / f"{safe_name}_{TIMEFRAME}.parquet", index=False)
        clean_df.to_csv(CSV_OUT / f"{safe_name}_{TIMEFRAME}.csv", index=False)
        print(symbol, data_quality_report(clean_df))

    fast = 40
    slow = 100
    results: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
    for symbol, clean_df in clean_data.items():
        bt_df, summary = evaluate_ma_strategy(clean_df, fast=fast, slow=slow, fee_bps=2.0)
        results[symbol] = {"backtest": bt_df, "summary": summary}

    summary_table = pd.DataFrame({symbol: obj["summary"] for symbol, obj in results.items()}).T
    summary_table.to_csv(CSV_OUT / "ma_strategy_summary.csv")

    for symbol, obj in results.items():
        safe_name = symbol.replace("/", "_")
        bt_df = obj["backtest"]
        if isinstance(bt_df, pd.DataFrame):
            bt_df.to_csv(CSV_OUT / f"{safe_name}_{TIMEFRAME}_ma_backtest.csv", index=False)

    return clean_data, summary_table, results


def run_strategy_scenarios(clean_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Run scenario analyses on top of the baseline workflow."""
    outputs: dict[str, pd.DataFrame] = {}

    window_pairs = [(10, 30), (20, 60), (40, 100), (60, 180)]
    rows: list[dict[str, float | int | str]] = []
    base_symbol = "BTC/USDT"
    for fast, slow in window_pairs:
        signal_df = add_ma_signal(clean_data[base_symbol], fast=fast, slow=slow)
        bt_df = backtest_signal(signal_df, fee_bps=2.0)
        perf = performance_summary(bt_df)
        rows.append(
            {
                "symbol": base_symbol,
                "fast": fast,
                "slow": slow,
                "total_return": float(perf["Total Return"]),
                "annualized_return": float(perf["Annualized Return"]),
                "annualized_vol": float(perf["Annualized Vol"]),
                "sharpe": float(perf["Sharpe"]),
                "max_drawdown": float(perf["Max Drawdown"]),
            }
        )
    window_sweep = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    window_sweep.to_csv(CSV_OUT / "window_sweep.csv", index=False)
    outputs["window_sweep"] = window_sweep

    fast, slow = 40, 100
    ls_signal = add_ma_signal(clean_data[base_symbol], fast=fast, slow=slow)
    lo_signal = add_ma_signal_long_only(clean_data[base_symbol], fast=fast, slow=slow)
    ls_bt = backtest_signal(ls_signal, fee_bps=2.0)
    lo_bt = backtest_signal(lo_signal, fee_bps=2.0)
    long_only_vs_long_short = pd.DataFrame(
        {
            "long_short": performance_summary(ls_bt),
            "long_only": performance_summary(lo_bt),
        }
    ).T
    long_only_vs_long_short.to_csv(CSV_OUT / "long_only_vs_long_short.csv")
    outputs["long_only_vs_long_short"] = long_only_vs_long_short

    fee_bps_grid = [0.5, 1.0, 2.0, 5.0, 10.0]
    fee_rows: list[dict[str, float]] = []
    for fee_bps in fee_bps_grid:
        bt_df = backtest_signal(ls_signal, fee_bps=fee_bps)
        perf = performance_summary(bt_df)
        fee_rows.append(
            {
                "fee_bps": fee_bps,
                "total_return": float(perf["Total Return"]),
                "annualized_return": float(perf["Annualized Return"]),
                "sharpe": float(perf["Sharpe"]),
                "avg_turnover": float(perf["Average Turnover"]),
            }
        )
    fee_sensitivity = pd.DataFrame(fee_rows).sort_values("fee_bps").reset_index(drop=True)
    fee_sensitivity.to_csv(CSV_OUT / "fee_sensitivity.csv", index=False)
    outputs["fee_sensitivity"] = fee_sensitivity

    extended_clean = dict(clean_data)
    if EXTENDED_SYMBOL not in extended_clean:
        ext_raw = download_or_demo([EXTENDED_SYMBOL])[EXTENDED_SYMBOL]
        ext_df = clean_ohlcv(ext_raw)
        extended_clean[EXTENDED_SYMBOL] = ext_df
        safe_name = EXTENDED_SYMBOL.replace("/", "_")
        ext_df.to_parquet(DATA_CLEAN / f"{safe_name}_{TIMEFRAME}.parquet", index=False)
        ext_df.to_csv(CSV_OUT / f"{safe_name}_{TIMEFRAME}.csv", index=False)
        print(EXTENDED_SYMBOL, data_quality_report(ext_df))

    summary_plus = pd.DataFrame(
        {
            symbol: evaluate_ma_strategy(df, fast=40, slow=100, fee_bps=2.0)[1]
            for symbol, df in extended_clean.items()
        }
    ).T
    summary_plus.to_csv(CSV_OUT / "extended_symbol_summary.csv")
    outputs["extended_symbol_summary"] = summary_plus

    print("Saved scenario outputs to:", RESULTS.resolve())
    return outputs


def add_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "future_ret_1h" not in out.columns:
        out["future_ret_1h"] = out["close"].shift(-1) / out["close"] - 1
    out["future_ret_1d"] = out["close"].shift(-HOURS_PER_DAY) / out["close"] - 1
    out["future_ret_1w"] = out["close"].shift(-HOURS_PER_WEEK) / out["close"] - 1
    return out


def load_selected_factors(
    top_n: int = COMBO_TOP_FACTOR_COUNT,
    factor_results: dict[str, pd.DataFrame | pd.Series] | None = None,
) -> pd.DataFrame:
    if factor_results and "top_factors" in factor_results:
        top_factors = factor_results["top_factors"]
        if isinstance(top_factors, pd.DataFrame):
            return top_factors.head(top_n).copy()

    for path in [CLASS2_CSV / f"factor_top{top_n}_factors.csv", CLASS2_CSV / "factor_top_factors.csv"]:
        if path.exists():
            return pd.read_csv(path).head(top_n).copy()

    raise RuntimeError("No Class 2 factor ranking found. Run factor research workflow first.")


def prepare_directed_alpha_matrix(
    alpha_df: pd.DataFrame,
    selected_factors: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    out = alpha_df.copy()
    combo_cols: list[str] = []
    for row in selected_factors.itertuples(index=False):
        factor_name = str(row.factor)
        direction = int(getattr(row, "direction", 1))
        combo_name = f"combo_{factor_name}"
        out[combo_name] = out[factor_name] * direction
        combo_cols.append(combo_name)
    return out, combo_cols


def time_split_index(
    index: pd.Index,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    n = len(index)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return index[:train_end], index[train_end:valid_end], index[valid_end:]


def prepare_model_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "future_ret_1h",
    datetime_col: str = "datetime",
) -> tuple[pd.DataFrame, pd.Index, pd.Index, pd.Index]:
    cols = feature_cols + [target_col, "ret_1h"]
    if datetime_col in df.columns:
        data = df[[datetime_col, *cols]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        data = data.set_index(datetime_col)
    else:
        data = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, valid_idx, test_idx = time_split_index(data.index)
    return data, train_idx, valid_idx, test_idx


def safe_standardize_by_train(data: pd.DataFrame, cols: list[str], train_idx: pd.Index) -> pd.DataFrame:
    train = data.loc[train_idx, cols]
    mu = train.mean()
    sd = train.std().replace(0, np.nan)
    return (data[cols] - mu) / sd


def equal_weight_signal(data: pd.DataFrame, feature_cols: list[str], train_idx: pd.Index) -> pd.Series:
    standardized = safe_standardize_by_train(data, feature_cols, train_idx)
    return standardized.mean(axis=1).rename("sig_equal_weight")


def ic_weight_signal(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_idx: pd.Index,
    long_only_weight: bool = False,
) -> tuple[pd.Series, pd.Series]:
    standardized = safe_standardize_by_train(data, feature_cols, train_idx)

    ics: dict[str, float] = {}
    for col in feature_cols:
        ics[col] = data.loc[train_idx, col].corr(data.loc[train_idx, target_col])
    weights = pd.Series(ics).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if long_only_weight:
        weights = weights.clip(lower=0.0)
    if weights.abs().sum() == 0:
        weights[:] = 1.0
    weights = weights / weights.abs().sum()
    signal = standardized.mul(weights, axis=1).sum(axis=1)
    return signal.rename("sig_ic_weight"), weights.sort_values(ascending=False)


def model_signal(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_idx: pd.Index,
    model_type: str = "ridge",
    alpha: float = COMBO_RIDGE_ALPHA,
) -> tuple[pd.Series, object]:
    standardized = safe_standardize_by_train(data, feature_cols, train_idx)
    y = data[target_col]
    x_train = standardized.loc[train_idx]
    y_train = y.loc[train_idx]

    if model_type == "linear":
        model: LinearRegression | Ridge | DecisionTreeRegressor = LinearRegression(fit_intercept=False)
    elif model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "tree":
        model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=100, random_state=42)
    else:
        raise ValueError("model_type should be linear, ridge, or tree")

    model.fit(x_train, y_train)
    pred = pd.Series(model.predict(standardized), index=standardized.index, name=f"sig_{model_type}")
    return pred, model


def signal_to_position(
    signal: pd.Series,
    train_idx: pd.Index,
    method: str = "quantile",
    q: float = COMBO_DEFAULT_QUANTILE,
    lookback_window: int | None = None,
) -> pd.Series:
    clean_signal = signal.replace([np.inf, -np.inf], np.nan)
    position = pd.Series(0.0, index=signal.index)

    if method == "quantile":
        if lookback_window is None:
            train_signal = clean_signal.loc[train_idx.intersection(clean_signal.index)]
            hi = train_signal.quantile(q)
            lo = train_signal.quantile(1 - q)
            position[clean_signal > hi] = 1.0
            position[clean_signal < lo] = -1.0
        else:
            hist_signal = clean_signal.shift(1)
            min_periods = max(5, lookback_window // 4)
            hi = hist_signal.rolling(lookback_window, min_periods=min_periods).quantile(q)
            lo = hist_signal.rolling(lookback_window, min_periods=min_periods).quantile(1 - q)
            position[clean_signal > hi] = 1.0
            position[clean_signal < lo] = -1.0
    else:
        raise ValueError("method should be quantile")
    return position.fillna(0.0)


def calc_signal_metrics(
    pnl: pd.Series,
    position: pd.Series | None = None,
    signal: pd.Series | None = None,
    returns: pd.Series | None = None,
    periods_per_year: int = 24 * 365,
) -> pd.Series:
    pnl = pnl.dropna()
    ann_return = pnl.mean() * periods_per_year
    ann_vol = pnl.std(ddof=1) * np.sqrt(periods_per_year) if len(pnl) > 1 else np.nan
    sharpe = ann_return / ann_vol if ann_vol not in (0, np.nan) and pd.notna(ann_vol) else np.nan
    equity = 1 + pnl.cumsum()
    metrics: dict[str, float] = {
        "ann_return": float(ann_return) if pd.notna(ann_return) else np.nan,
        "ann_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": max_drawdown(equity),
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else np.nan,
        "hit_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
    }
    if position is not None:
        metrics["avg_abs_position"] = float(position.abs().mean())
        metrics["turnover_per_bar"] = float(position.diff().abs().fillna(position.abs()).mean())
    if signal is not None and returns is not None:
        aligned = pd.concat([signal.rename("signal"), returns.rename("ret")], axis=1).dropna()
        if len(aligned) > 1:
            metrics["ic"] = float(aligned["signal"].corr(aligned["ret"]))
        else:
            metrics["ic"] = np.nan
    return pd.Series(metrics)


def signal_backtest(
    data: pd.DataFrame,
    signal: pd.Series,
    train_idx: pd.Index,
    method: str = "quantile",
    q: float = COMBO_DEFAULT_QUANTILE,
    fee_bps: float = 2.0,
    return_col: str = "ret_1h",
    lookback_window: int | None = None,
) -> pd.DataFrame:
    aligned = pd.concat([data[return_col].rename("ret"), signal.rename("signal")], axis=1).dropna()
    position = signal_to_position(
        aligned["signal"],
        train_idx,
        method=method,
        q=q,
        lookback_window=lookback_window,
    ).reindex(aligned.index).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    fee = turnover * fee_bps / 10_000
    pnl = position.shift(1).fillna(0.0) * aligned["ret"] - fee
    return pd.DataFrame(
        {
            "ret": aligned["ret"],
            "signal": aligned["signal"],
            "position": position,
            "turnover": turnover,
            "fee": fee,
            "pnl": pnl,
        }
    )


def metrics_by_period(
    result: pd.DataFrame,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    test_idx: pd.Index,
    periods_per_year: int = 24 * 365,
) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    for name, idx in {"train": train_idx, "valid": valid_idx, "test": test_idx}.items():
        subset = result.loc[result.index.intersection(idx)]
        rows[name] = calc_signal_metrics(
            subset["pnl"],
            subset["position"],
            subset["signal"],
            subset["ret"],
            periods_per_year=periods_per_year,
        )
    return pd.DataFrame(rows).T


def _calendar_period_returns(pnl: pd.Series, freq: str) -> pd.Series:
    clean = pnl.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index, utc=True)
    grouped = clean.groupby(pd.Grouper(freq=freq))
    return grouped.apply(lambda s: (1 + s).prod() - 1).dropna()


def calendar_returns_table(
    result: pd.DataFrame,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    test_idx: pd.Index,
    method: str,
    horizon: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    freq_map = {"month": "ME", "quarter": "QE", "year": "YE"}

    for split_name, idx in {"train": train_idx, "valid": valid_idx, "test": test_idx}.items():
        subset = result.loc[result.index.intersection(idx)]
        if subset.empty:
            continue
        for calendar_name, freq in freq_map.items():
            period_returns = _calendar_period_returns(subset["pnl"], freq)
            for period_end, period_ret in period_returns.items():
                detail_rows.append(
                    {
                        "method": method,
                        "horizon": horizon,
                        "split": split_name,
                        "calendar": calendar_name,
                        "period_end": period_end,
                        "period_return": float(period_ret),
                    }
                )
            if len(period_returns) > 0:
                summary_rows.append(
                    {
                        "method": method,
                        "horizon": horizon,
                        "split": split_name,
                        "calendar": calendar_name,
                        "n_periods": len(period_returns),
                        "mean_period_return": float(period_returns.mean()),
                        "std_period_return": float(period_returns.std(ddof=1)) if len(period_returns) > 1 else np.nan,
                        "win_rate": float((period_returns > 0).mean()),
                        "total_return": float((1 + period_returns).prod() - 1),
                    }
                )

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def build_combination_signals(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_idx: pd.Index,
) -> tuple[dict[str, pd.Series], pd.Series]:
    signals: dict[str, pd.Series] = {
        "equal_weight": equal_weight_signal(data, feature_cols, train_idx),
    }
    ic_signal, ic_weights = ic_weight_signal(data, feature_cols, target_col, train_idx)
    signals["ic_weight"] = ic_signal

    for model_type in ["linear", "ridge", "tree"]:
        pred, _ = model_signal(
            data,
            feature_cols,
            target_col,
            train_idx,
            model_type=model_type,
            alpha=COMBO_RIDGE_ALPHA,
        )
        signals[model_type] = pred

    return signals, ic_weights


def combination_sensitivity_test(
    data: pd.DataFrame,
    signal: pd.Series,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    test_idx: pd.Index,
    method: str = "",
    qs: list[float] | tuple[float, ...] = tuple(COMBO_SENSITIVITY_QUANTILES),
    fees_bps: list[float] | tuple[float, ...] = tuple(COMBO_SENSITIVITY_FEES_BPS),
    lookbacks: list[int | None] | tuple[int | None, ...] = (None, *COMBO_LOOKBACK_WINDOWS),
    periods_per_year: int = 24 * 365,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for fee_bps in fees_bps:
        for q in qs:
            for lookback in lookbacks:
                bt = signal_backtest(
                    data,
                    signal,
                    train_idx,
                    method="quantile",
                    q=q,
                    fee_bps=fee_bps,
                    lookback_window=lookback,
                )
                for period_name, idx in {"train": train_idx, "valid": valid_idx, "test": test_idx}.items():
                    subset = bt.loc[bt.index.intersection(idx)]
                    metrics = calc_signal_metrics(
                        subset["pnl"],
                        subset["position"],
                        subset["signal"],
                        subset["ret"],
                        periods_per_year=periods_per_year,
                    )
                    rows.append(
                        {
                            "method": method,
                            "lookback": -1 if lookback is None else lookback,
                            "q": q,
                            "fee_bps": fee_bps,
                            "period": period_name,
                            **{k: float(v) if pd.notna(v) else np.nan for k, v in metrics.to_dict().items()},
                        }
                    )
    return pd.DataFrame(rows)


def _plot_combo_equity_curves(
    bt_results: dict[str, pd.DataFrame],
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, result in bt_results.items():
        equity = (1 + result["pnl"].fillna(0)).cumprod()
        ax.plot(result.index, equity.values, label=name)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_combo_sensitivity_heatmap(
    sensitivity: pd.DataFrame,
    period: str,
    save_path: Path,
    lookback: int = -1,
) -> None:
    subset = sensitivity[(sensitivity["period"] == period) & (sensitivity["lookback"] == lookback)]
    if subset.empty:
        subset = sensitivity[sensitivity["period"] == period]
        if subset.empty:
            return
        subset = subset[subset["lookback"] == subset["lookback"].iloc[0]]
    pivot = subset.pivot(index="q", columns="fee_bps", values="sharpe")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("fee_bps")
    ax.set_ylabel("threshold quantile")
    ax.set_title(f"{period.title()} Sharpe sensitivity")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_model_combination_workflow(
    clean_data: dict[str, pd.DataFrame],
    factor_results: dict[str, pd.DataFrame | pd.Series] | None = None,
    symbol: str = "BTC/USDT",
) -> dict[str, object]:
    alpha_df = build_alpha_dataset(clean_data[symbol].copy())
    alpha_df = add_horizon_targets(alpha_df)
    selected_factors = load_selected_factors(COMBO_TOP_FACTOR_COUNT, factor_results)
    alpha_df, combo_cols = prepare_directed_alpha_matrix(alpha_df, selected_factors)

    all_period_metrics: list[pd.DataFrame] = []
    all_sensitivity: list[pd.DataFrame] = []
    all_calendar_detail: list[pd.DataFrame] = []
    all_calendar_summary: list[pd.DataFrame] = []
    all_ic_weights: list[dict[str, float | int | str]] = []
    outputs_by_horizon: dict[str, dict[str, object]] = {}

    for horizon_name, horizon_cfg in COMBO_HORIZONS.items():
        target_col = f"future_ret_{horizon_name}"
        periods_per_year = int(horizon_cfg["periods_per_year"])
        model_data, train_idx, valid_idx, test_idx = prepare_model_data(alpha_df, combo_cols, target_col=target_col)

        signals, ic_weights = build_combination_signals(model_data, combo_cols, target_col, train_idx)
        bt_results = {
            name: signal_backtest(
                model_data,
                sig,
                train_idx,
                method="quantile",
                q=COMBO_DEFAULT_QUANTILE,
                fee_bps=2.0,
            )
            for name, sig in signals.items()
        }

        period_rows: list[pd.DataFrame] = []
        horizon_sensitivity: list[pd.DataFrame] = []
        for name, result in bt_results.items():
            period_df = metrics_by_period(
                result,
                train_idx,
                valid_idx,
                test_idx,
                periods_per_year=periods_per_year,
            )
            period_df = period_df.reset_index(names=["period"])
            period_df["method"] = name
            period_df["horizon"] = horizon_name
            period_rows.append(period_df)

            calendar_detail, calendar_summary = calendar_returns_table(
                result,
                train_idx,
                valid_idx,
                test_idx,
                method=name,
                horizon=horizon_name,
            )
            if not calendar_detail.empty:
                all_calendar_detail.append(calendar_detail)
            if not calendar_summary.empty:
                all_calendar_summary.append(calendar_summary)

            sensitivity = combination_sensitivity_test(
                model_data,
                signals[name],
                train_idx,
                valid_idx,
                test_idx,
                method=name,
                periods_per_year=periods_per_year,
            )
            sensitivity["horizon"] = horizon_name
            horizon_sensitivity.append(sensitivity)

        period_metrics = pd.concat(period_rows, ignore_index=True)
        all_period_metrics.append(period_metrics)
        sensitivity = pd.concat(horizon_sensitivity, ignore_index=True)
        all_sensitivity.append(sensitivity)

        for factor_name, weight in ic_weights.items():
            all_ic_weights.append({"horizon": horizon_name, "factor": factor_name, "weight": float(weight)})

        horizon_slug = horizon_name.replace("/", "_")
        _plot_combo_equity_curves(
            bt_results,
            title=f"{symbol} signal combination equity ({horizon_name})",
            save_path=COMBO_FIGURES_OUT / f"combo_equity_{horizon_slug}.png",
        )
        for name in signals:
            method_sensitivity = sensitivity[sensitivity["method"] == name]
            if method_sensitivity.empty:
                continue
            _plot_combo_sensitivity_heatmap(
                method_sensitivity,
                period="test",
                save_path=COMBO_FIGURES_OUT / f"combo_sensitivity_{horizon_slug}_{name}.png",
            )

        for name, result in bt_results.items():
            result_out = result.copy()
            result_out.insert(0, "datetime", result_out.index)
            result_out.to_csv(COMBO_BACKTESTS_OUT / f"combo_{horizon_slug}_{name}_backtest.csv", index=False)

        outputs_by_horizon[horizon_name] = {
            "model_data": model_data,
            "signals": signals,
            "bt_results": bt_results,
            "period_metrics": period_metrics,
            "sensitivity": sensitivity,
            "ic_weights": ic_weights,
            "selected_factors": selected_factors,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
        }

    period_metrics_df = pd.concat(all_period_metrics, ignore_index=True)
    sensitivity_df = pd.concat(all_sensitivity, ignore_index=True)
    calendar_detail_df = pd.concat(all_calendar_detail, ignore_index=True) if all_calendar_detail else pd.DataFrame()
    calendar_summary_df = pd.concat(all_calendar_summary, ignore_index=True) if all_calendar_summary else pd.DataFrame()
    ic_weights_df = pd.DataFrame(all_ic_weights)

    selected_factors.to_csv(COMBO_RESULTS_OUT / "combo_selected_factors.csv", index=False)
    period_metrics_df.to_csv(COMBO_RESULTS_OUT / "combo_period_metrics.csv", index=False)
    sensitivity_df.to_csv(COMBO_RESULTS_OUT / "combo_sensitivity.csv", index=False)
    if not calendar_detail_df.empty:
        calendar_detail_df.to_csv(COMBO_RESULTS_OUT / "combo_calendar_returns.csv", index=False)
    if not calendar_summary_df.empty:
        calendar_summary_df.to_csv(COMBO_RESULTS_OUT / "combo_calendar_summary.csv", index=False)
    ic_weights_df.to_csv(COMBO_RESULTS_OUT / "combo_ic_weights.csv", index=False)
    period_metrics_df.to_csv(COMBO_RESULTS_OUT / "combo_summary.csv", index=False)

    print("Class 3 reports saved to:", CLASS3_CSV.resolve())
    print("Class 3 backtests saved to:", CLASS3_BACKTESTS.resolve())
    print("Class 3 figures saved to:", CLASS3_FIGURES.resolve())
    print("Selected factors:", selected_factors["factor"].tolist())
    print("Horizons tested:", list(COMBO_HORIZONS.keys()))
    print("Combination methods:", list(outputs_by_horizon["1h"]["bt_results"].keys()))

    return {
        "selected_factors": selected_factors,
        "period_metrics": period_metrics_df,
        "sensitivity": sensitivity_df,
        "calendar_returns": calendar_detail_df,
        "calendar_summary": calendar_summary_df,
        "ic_weights": ic_weights_df,
        "by_horizon": outputs_by_horizon,
    }


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 24) -> pd.Series:
    return true_range(high, low, close).rolling(window, min_periods=max(5, window // 4)).mean()


def build_class4_signal(
    clean_df: pd.DataFrame,
    selected_factors: pd.DataFrame,
    method: str = CLASS4_METHOD,
    horizon: str = CLASS4_HORIZON,
) -> tuple[pd.DataFrame, pd.Series, pd.Index, pd.Index, pd.Index]:
    alpha_df = build_alpha_dataset(clean_df.copy())
    alpha_df = add_horizon_targets(alpha_df)
    alpha_df, combo_cols = prepare_directed_alpha_matrix(alpha_df, selected_factors)
    target_col = f"future_ret_{horizon}"
    model_data, train_idx, valid_idx, test_idx = prepare_model_data(alpha_df, combo_cols, target_col=target_col)
    signals, _ = build_combination_signals(model_data, combo_cols, target_col, train_idx)
    if method not in signals:
        raise KeyError(f"Unknown Class 4 method '{method}'. Available: {list(signals)}")
    return model_data, signals[method], train_idx, valid_idx, test_idx


def apply_exit_rules(
    ohlcv: pd.DataFrame,
    target_position: pd.Series,
    mode: str = "none",
    fee_bps: float = CLASS4_FEE_BPS,
    tp_pct: float = CLASS4_FIXED_TP_PCT,
    sl_pct: float = CLASS4_FIXED_SL_PCT,
    atr: pd.Series | None = None,
    atr_tp_mult: float = CLASS4_ATR_TP_MULT,
    atr_sl_mult: float = CLASS4_ATR_SL_MULT,
    time_stop_bars: int = CLASS4_TIME_STOP_BARS,
    trail_pct: float = CLASS4_TRAIL_PCT,
) -> pd.DataFrame:
    # Conservative OHLC assumption: if both TP and SL hit in one bar, SL fires first.
    bars = ohlcv[["open", "high", "low", "close"]].copy()
    bars["ret"] = bars["close"].pct_change()
    aligned = bars.join(target_position.rename("target"), how="inner").dropna(subset=["ret"])
    if atr is not None:
        aligned = aligned.join(atr.rename("atr"), how="left")

    n = len(aligned)
    position = np.zeros(n, dtype=float)
    exit_reason = np.array([""] * n, dtype=object)
    entry_price = np.full(n, np.nan)
    hold_bars = np.zeros(n, dtype=int)

    cur_pos = 0.0
    cur_entry = np.nan
    cur_hold = 0
    extreme = np.nan  # running high for long / running low for short

    high = aligned["high"].to_numpy()
    low = aligned["low"].to_numpy()
    close = aligned["close"].to_numpy()
    target = aligned["target"].to_numpy()
    atr_vals = aligned["atr"].to_numpy() if "atr" in aligned.columns else np.full(n, np.nan)

    for i in range(n):
        desired = float(target[i]) if pd.notna(target[i]) else 0.0

        # Enter / flip when flat or desired sign changes and exit rule is not forcing flat.
        if cur_pos == 0.0 and desired != 0.0:
            cur_pos = desired
            cur_entry = close[i]
            cur_hold = 0
            extreme = close[i]
            exit_reason[i] = "enter"
        elif cur_pos != 0.0 and desired != 0.0 and np.sign(desired) != np.sign(cur_pos):
            cur_pos = desired
            cur_entry = close[i]
            cur_hold = 0
            extreme = close[i]
            exit_reason[i] = "flip"
        elif cur_pos != 0.0:
            cur_hold += 1
            if cur_pos > 0:
                extreme = close[i] if not np.isfinite(extreme) else max(extreme, high[i])
            else:
                extreme = close[i] if not np.isfinite(extreme) else min(extreme, low[i])

            hit_sl = False
            hit_tp = False
            reason = ""

            if mode == "fixed":
                if cur_pos > 0:
                    hit_sl = low[i] <= cur_entry * (1.0 - sl_pct)
                    hit_tp = high[i] >= cur_entry * (1.0 + tp_pct)
                else:
                    hit_sl = high[i] >= cur_entry * (1.0 + sl_pct)
                    hit_tp = low[i] <= cur_entry * (1.0 - tp_pct)
            elif mode == "atr":
                atr_i = atr_vals[i]
                if np.isfinite(atr_i) and atr_i > 0 and np.isfinite(cur_entry):
                    if cur_pos > 0:
                        hit_sl = low[i] <= cur_entry - atr_sl_mult * atr_i
                        hit_tp = high[i] >= cur_entry + atr_tp_mult * atr_i
                    else:
                        hit_sl = high[i] >= cur_entry + atr_sl_mult * atr_i
                        hit_tp = low[i] <= cur_entry - atr_tp_mult * atr_i
            elif mode == "trailing":
                if np.isfinite(extreme):
                    if cur_pos > 0:
                        stop = extreme * (1.0 - trail_pct)
                        hit_sl = low[i] <= stop
                    else:
                        stop = extreme * (1.0 + trail_pct)
                        hit_sl = high[i] >= stop
            elif mode == "time":
                if cur_hold >= time_stop_bars:
                    hit_sl = True
                    reason = "time_stop"

            if mode in {"fixed", "atr"} and (hit_sl or hit_tp):
                # Same-bar ambiguity: prefer stop-loss.
                if hit_sl:
                    reason = "stop_loss"
                else:
                    reason = "take_profit"
                cur_pos = 0.0
                cur_entry = np.nan
                cur_hold = 0
                extreme = np.nan
                exit_reason[i] = reason
            elif mode == "trailing" and hit_sl:
                cur_pos = 0.0
                cur_entry = np.nan
                cur_hold = 0
                extreme = np.nan
                exit_reason[i] = "trailing_stop"
            elif mode == "time" and hit_sl:
                cur_pos = 0.0
                cur_entry = np.nan
                cur_hold = 0
                extreme = np.nan
                exit_reason[i] = reason
            elif desired == 0.0:
                cur_pos = 0.0
                cur_entry = np.nan
                cur_hold = 0
                extreme = np.nan
                exit_reason[i] = "signal_flat"

        position[i] = cur_pos
        entry_price[i] = cur_entry
        hold_bars[i] = cur_hold

    out = aligned.copy()
    out["position"] = position
    out["entry_price"] = entry_price
    out["hold_bars"] = hold_bars
    out["exit_reason"] = exit_reason
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["fee"] = out["turnover"] * fee_bps / 10_000
    out["pnl"] = out["position"].shift(1).fillna(0.0) * out["ret"] - out["fee"]
    out["equity"] = (1 + out["pnl"].fillna(0.0)).cumprod()
    return out


def compare_exit_rules(
    ohlcv: pd.DataFrame,
    target_position: pd.Series,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    test_idx: pd.Index,
    symbol: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    atr = average_true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"], window=CLASS4_ATR_WINDOW)
    modes = {
        "none": {},
        "fixed": {"tp_pct": CLASS4_FIXED_TP_PCT, "sl_pct": CLASS4_FIXED_SL_PCT},
        "atr": {"atr": atr, "atr_tp_mult": CLASS4_ATR_TP_MULT, "atr_sl_mult": CLASS4_ATR_SL_MULT},
        "time": {"time_stop_bars": CLASS4_TIME_STOP_BARS},
        "trailing": {"trail_pct": CLASS4_TRAIL_PCT},
    }

    bt_by_mode: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, float | int | str]] = []
    for mode, kwargs in modes.items():
        bt = apply_exit_rules(ohlcv, target_position, mode=mode, fee_bps=CLASS4_FEE_BPS, **kwargs)
        bt = bt.copy()
        bt["signal"] = target_position.reindex(bt.index)
        bt_by_mode[mode] = bt
        period_df = metrics_by_period(
            bt,
            train_idx,
            valid_idx,
            test_idx,
            periods_per_year=CLASS4_PERIODS_PER_YEAR,
        ).reset_index(names=["period"])
        period_df["exit_mode"] = mode
        period_df["symbol"] = symbol
        rows.extend(period_df.to_dict("records"))

    return bt_by_mode, pd.DataFrame(rows)


def allocate_portfolio_weights(
    train_returns: pd.DataFrame,
    method: str = "equal",
    risk_aversion: float = CLASS4_MVO_RISK_AVERSION,
) -> pd.Series:
    assets = list(train_returns.columns)
    n = len(assets)
    if n == 0:
        return pd.Series(dtype=float)

    if method == "equal":
        weights = np.ones(n) / n
    elif method == "sharpe":
        mu = train_returns.mean()
        vol = train_returns.std(ddof=1).replace(0, np.nan)
        sharpe = (mu / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        if sharpe.sum() == 0:
            weights = np.ones(n) / n
        else:
            weights = (sharpe / sharpe.sum()).to_numpy()
    elif method == "risk_target":
        vol = train_returns.std(ddof=1).replace(0, np.nan)
        inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if inv_vol.sum() == 0:
            weights = np.ones(n) / n
        else:
            weights = (inv_vol / inv_vol.sum()).to_numpy()
    elif method == "mvo":
        mu_hat = train_returns.mean().to_numpy()
        cov_hat = np.cov(train_returns.T)
        if cov_hat.ndim == 0:
            cov_hat = np.array([[float(cov_hat)]])

        def objective(w: np.ndarray) -> float:
            return -(w @ mu_hat - risk_aversion * (w @ cov_hat @ w))

        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        weights = result.x if result.success else np.ones(n) / n
    else:
        raise ValueError("method should be equal, sharpe, risk_target, or mvo")

    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() == 0:
        weights = np.ones(n) / n
    else:
        weights = weights / weights.sum()
    return pd.Series(weights, index=assets, name=method)


def build_portfolio_returns(
    asset_pnl: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    aligned = asset_pnl[weights.index].fillna(0.0)
    return aligned.mul(weights, axis=1).sum(axis=1).rename("portfolio_pnl")


def run_risk_layer_workflow(
    clean_data: dict[str, pd.DataFrame],
    factor_results: dict[str, pd.DataFrame | pd.Series] | None = None,
    symbols: list[str] | None = None,
) -> dict[str, object]:
    symbols = symbols or CLASS4_SYMBOLS
    selected_factors = load_selected_factors(COMBO_TOP_FACTOR_COUNT, factor_results)

    asset_signals: dict[str, pd.Series] = {}
    asset_targets: dict[str, pd.Series] = {}
    asset_splits: dict[str, tuple[pd.Index, pd.Index, pd.Index]] = {}
    exit_metrics_rows: list[pd.DataFrame] = []
    asset_baseline_pnl: dict[str, pd.Series] = {}
    asset_fixed_pnl: dict[str, pd.Series] = {}

    for symbol in symbols:
        if symbol not in clean_data:
            raw = download_or_demo([symbol])[symbol]
            clean_data[symbol] = clean_ohlcv(raw)

        clean_df = clean_data[symbol].copy()
        model_data, signal, train_idx, valid_idx, test_idx = build_class4_signal(
            clean_df,
            selected_factors,
            method=CLASS4_METHOD,
            horizon=CLASS4_HORIZON,
        )
        baseline_bt = signal_backtest(
            model_data,
            signal,
            train_idx,
            method="quantile",
            q=CLASS4_QUANTILE,
            fee_bps=CLASS4_FEE_BPS,
        )
        target_position = baseline_bt["position"]
        asset_targets[symbol] = target_position
        asset_signals[symbol] = signal
        asset_splits[symbol] = (train_idx, valid_idx, test_idx)

        ohlcv = clean_df.set_index("datetime")[["open", "high", "low", "close"]]
        bt_by_mode, metrics_df = compare_exit_rules(
            ohlcv,
            target_position,
            train_idx,
            valid_idx,
            test_idx,
            symbol=symbol,
        )
        exit_metrics_rows.append(metrics_df)

        safe = symbol.replace("/", "_")
        for mode, bt in bt_by_mode.items():
            out = bt.copy()
            out.insert(0, "datetime", out.index)
            out.to_csv(CLASS4_CSV / f"risk_{safe}_{mode}_backtest.csv", index=False)

        asset_baseline_pnl[symbol] = bt_by_mode["none"]["pnl"].rename(symbol)
        asset_fixed_pnl[symbol] = bt_by_mode["fixed"]["pnl"].rename(symbol)

        fig, ax = plt.subplots(figsize=(12, 5))
        for mode, bt in bt_by_mode.items():
            ax.plot(bt.index, bt["equity"], label=mode)
        ax.set_title(f"{symbol} exit-rule equity comparison ({CLASS4_METHOD}/{CLASS4_HORIZON})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(CLASS4_FIGURES / f"risk_{safe}_exit_equity.png", dpi=150)
        plt.close(fig)

    exit_metrics = pd.concat(exit_metrics_rows, ignore_index=True)
    exit_metrics.to_csv(CLASS4_CSV / "risk_exit_metrics.csv", index=False)

    # Portfolio allocation uses BTC+ETH strategy pnl; weights fit on the common train window.
    baseline_pnl_df = pd.concat(asset_baseline_pnl.values(), axis=1).dropna(how="any")
    fixed_pnl_df = pd.concat(asset_fixed_pnl.values(), axis=1).reindex(baseline_pnl_df.index).fillna(0.0)

    # Use BTC split as the chronological reference for portfolio train/test.
    ref_train, ref_valid, ref_test = asset_splits[symbols[0]]
    common_index = baseline_pnl_df.index
    train_mask = common_index.isin(ref_train)
    test_mask = common_index.isin(ref_test)
    train_returns = baseline_pnl_df.loc[train_mask]

    allocation_methods = ["equal", "sharpe", "risk_target", "mvo"]
    weight_rows: list[dict[str, float | str]] = []
    portfolio_metric_rows: list[dict[str, float | str]] = []
    portfolio_curves = pd.DataFrame({"datetime": common_index})

    for sleeve_name, sleeve_pnl in {"baseline": baseline_pnl_df, "fixed_tpsl": fixed_pnl_df}.items():
        train_sleeve = sleeve_pnl.loc[train_mask]
        for method in allocation_methods:
            weights = allocate_portfolio_weights(train_sleeve, method=method)
            for asset, weight in weights.items():
                weight_rows.append(
                    {
                        "sleeve": sleeve_name,
                        "method": method,
                        "asset": asset,
                        "weight": float(weight),
                    }
                )
            port_pnl = build_portfolio_returns(sleeve_pnl, weights)
            portfolio_curves[f"{sleeve_name}_{method}"] = (1 + port_pnl.fillna(0.0)).cumprod().to_numpy()

            for period_name, mask in {"train": train_mask, "valid": common_index.isin(ref_valid), "test": test_mask}.items():
                sub = port_pnl.loc[mask]
                metrics = calc_signal_metrics(sub, periods_per_year=CLASS4_PERIODS_PER_YEAR)
                portfolio_metric_rows.append(
                    {
                        "sleeve": sleeve_name,
                        "method": method,
                        "period": period_name,
                        **{k: float(v) if pd.notna(v) else np.nan for k, v in metrics.to_dict().items()},
                    }
                )

    weights_df = pd.DataFrame(weight_rows)
    portfolio_metrics_df = pd.DataFrame(portfolio_metric_rows)
    weights_df.to_csv(CLASS4_CSV / "risk_portfolio_weights.csv", index=False)
    portfolio_metrics_df.to_csv(CLASS4_CSV / "risk_portfolio_metrics.csv", index=False)
    portfolio_curves.to_csv(CLASS4_CSV / "risk_portfolio_equity_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in portfolio_curves.columns:
        if col == "datetime":
            continue
        if col.startswith("baseline_"):
            ax.plot(portfolio_curves["datetime"], portfolio_curves[col], label=col)
    ax.set_title("Multi-asset portfolio equity (baseline sleeve, no TP/SL)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CLASS4_FIGURES / "risk_portfolio_equity_baseline.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in portfolio_curves.columns:
        if col == "datetime":
            continue
        if col.startswith("fixed_tpsl_"):
            ax.plot(portfolio_curves["datetime"], portfolio_curves[col], label=col)
    ax.set_title("Multi-asset portfolio equity (fixed TP/SL sleeve)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CLASS4_FIGURES / "risk_portfolio_equity_fixed_tpsl.png", dpi=150)
    plt.close(fig)

    # Side-by-side summary: baseline vs fixed TP/SL on test for BTC and portfolio equal-weight.
    summary_rows: list[dict[str, float | str]] = []
    for _, row in exit_metrics[(exit_metrics["period"] == "test")].iterrows():
        summary_rows.append(
            {
                "scope": "single_asset",
                "symbol": row["symbol"],
                "exit_mode": row["exit_mode"],
                "sharpe": row.get("sharpe", np.nan),
                "ann_return": row.get("ann_return", np.nan),
                "max_drawdown": row.get("max_drawdown", np.nan),
                "hit_rate": row.get("hit_rate", np.nan),
            }
        )
    for _, row in portfolio_metrics_df[portfolio_metrics_df["period"] == "test"].iterrows():
        summary_rows.append(
            {
                "scope": "portfolio",
                "symbol": "BTC+ETH",
                "exit_mode": f"{row['sleeve']}:{row['method']}",
                "sharpe": row.get("sharpe", np.nan),
                "ann_return": row.get("ann_return", np.nan),
                "max_drawdown": row.get("max_drawdown", np.nan),
                "hit_rate": row.get("hit_rate", np.nan),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(CLASS4_CSV / "risk_summary.csv", index=False)

    print("Class 4 reports saved to:", CLASS4_CSV.resolve())
    print("Class 4 figures saved to:", CLASS4_FIGURES.resolve())
    print("Exit modes compared: none / fixed / atr / time / trailing")
    print("Portfolio methods:", allocation_methods)
    print("Assets:", symbols)

    return {
        "selected_factors": selected_factors,
        "exit_metrics": exit_metrics,
        "portfolio_weights": weights_df,
        "portfolio_metrics": portfolio_metrics_df,
        "portfolio_curves": portfolio_curves,
        "summary": summary_df,
        "asset_signals": asset_signals,
        "asset_targets": asset_targets,
    }


def main() -> None:
    # Six-stage pipeline: Class1 trend -> scenarios -> Class2 factors -> Class3 combo -> Class4 risk
    print("[Stage 1/6] Running baseline workflow...")
    clean_data, summary_table, results = run_baseline_workflow(SYMBOLS)

    print("[Stage 2/6] Baseline summary:")
    print(summary_table)

    btc = results["BTC/USDT"]["backtest"]
    if isinstance(btc, pd.DataFrame):
        print("[Stage 2/6] Generating baseline plots...")
        plot_price_and_mas(
            btc,
            fast=40,
            slow=100,
            title="BTC/USDT close price and moving averages",
            save_path=FIGURES / "baseline_btc_price_ma.png",
        )
        plot_equity(
            btc,
            title="BTC/USDT equity curve",
            save_path=FIGURES / "baseline_btc_equity.png",
        )

    print("[Stage 3/6] Running scenario analyses...")
    run_strategy_scenarios(clean_data)

    print("[Stage 4/6] Running factor research workflow...")
    factor_results = run_factor_research_workflow(clean_data)

    print("[Stage 5/6] Running Class 3 model combination workflow...")
    run_model_combination_workflow(clean_data, factor_results=factor_results)

    print("[Stage 6/6] Running Class 4 risk layer workflow...")
    run_risk_layer_workflow(clean_data, factor_results=factor_results)


if __name__ == "__main__":
    main()
