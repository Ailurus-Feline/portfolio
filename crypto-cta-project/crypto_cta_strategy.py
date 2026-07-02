from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ccxt


pd.set_option("display.max_columns", 20)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
CSV_OUT = RESULTS / "csv"
FACTOR_DATA_OUT = CSV_OUT
FACTOR_RESULTS_OUT = CSV_OUT
FACTOR_FIGURES_OUT = FIGURES
for path_obj in [DATA_RAW, DATA_CLEAN, RESULTS, FIGURES, CSV_OUT]:
    path_obj.mkdir(parents=True, exist_ok=True)

# Baseline configuration from notebook
EXCHANGE_ID = "binance"  # If Binance is not accessible, try "okx" or "bybit".
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
    """Build factor features and future-return target on cleaned OHLCV."""
    out = df.copy()
    out["ret_1h"] = out["close"].pct_change()
    out["future_ret_1h"] = out["close"].shift(-1) / out["close"] - 1

    # Momentum (weekly windows mapped to hourly bars).
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
    out[factor] = out[factor] * direction

    hist_factor = out[factor].shift(1)
    out["q_high"] = hist_factor.rolling(window, min_periods=window // 6).quantile(q_high)
    out["q_low"] = hist_factor.rolling(window, min_periods=window // 6).quantile(q_low)

    out["signal"] = 0
    out.loc[out[factor] > out["q_high"], "signal"] = 1
    out.loc[out[factor] < out["q_low"], "signal"] = -1

    out["position"] = out["signal"]
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
    """Run factor workflow and persist outputs."""
    btc_df = clean_data["BTC/USDT"].copy()
    alpha_df = build_alpha_dataset(btc_df)
    factor_cols = [col for col in alpha_df.columns if col.startswith("factor_")]

    ic_table = calc_ic_table(alpha_df, factor_cols)

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
    top_factors["direction"] = np.where(top_factors["pearson_ic"] < 0, -1, 1)

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
    """Build long-short MA signal with one-bar execution lag."""
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    out[f"ma_{fast}"] = out["close"].rolling(fast).mean()
    out[f"ma_{slow}"] = out["close"].rolling(slow).mean()
    out["signal"] = np.where(out[f"ma_{fast}"] > out[f"ma_{slow}"], 1, -1)
    out.loc[out[f"ma_{slow}"].isna(), "signal"] = 0
    out["position"] = out["signal"].shift(1).fillna(0)
    return out


def backtest_signal(df: pd.DataFrame, fee_bps: float = 2.0) -> pd.DataFrame:
    """Run vectorized backtest with turnover-based costs."""
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


def main() -> None:
    """Execute baseline workflow, scenarios, and factor workflow."""
    print("[Stage 1/4] Running baseline workflow...")
    clean_data, summary_table, results = run_baseline_workflow(SYMBOLS)

    print("[Stage 2/4] Baseline summary:")
    print(summary_table)

    btc = results["BTC/USDT"]["backtest"]
    if isinstance(btc, pd.DataFrame):
        print("[Stage 2/3] Generating baseline plots...")
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

    print("[Stage 3/4] Running scenario analyses...")
    run_strategy_scenarios(clean_data)

    print("[Stage 4/4] Running factor research workflow...")
    run_factor_research_workflow(clean_data)


if __name__ == "__main__":
    main()
