from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ccxt

"""Crypto CTA moving-average strategy research script.

This file is intentionally written as a readable learning baseline.
It demonstrates one complete quant-research loop:

1. Acquire OHLCV data from exchange APIs (or generate demo data on failure).
2. Clean and validate time-series data.
3. Build MA-based trading signals with anti-look-ahead handling.
4. Run a simple transaction-cost-aware backtest.
5. Compare scenarios and export artifacts for review.

The implementation prefers clarity over micro-optimization so that each step
can be studied and modified independently.
"""


pd.set_option("display.max_columns", 20)

# End-to-end flow:
# 1) data access -> 2) cleaning -> 3) signal -> 4) backtest -> 5) summary + plots
# The script keeps each step in a separate pure-ish function for easier learning/testing.

# Project paths
PROJECT_ROOT = Path.cwd()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"
RESULTS = PROJECT_ROOT / "results"
OUTPUTS_CLASS2 = PROJECT_ROOT / "outputs_class2"
for path_obj in [DATA_RAW, DATA_CLEAN, RESULTS, OUTPUTS_CLASS2]:
    path_obj.mkdir(parents=True, exist_ok=True)

# Baseline configuration from notebook
EXCHANGE_ID = "binance"  # If Binance is not accessible, try "okx" or "bybit".
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
SINCE = "2020-01-01T00:00:00Z"
LIMIT_PER_REQUEST = 1000


def make_exchange(exchange_id: str = EXCHANGE_ID):
    """Create a configured CCXT exchange client.

    Parameters
    ----------
    exchange_id:
        Exchange name supported by ccxt (for example: "binance", "okx", "bybit").

    Returns
    -------
    Exchange instance
        A ccxt exchange client with built-in rate limiting enabled.
    """
    # `getattr` allows switching exchanges by string config, e.g., "binance" -> ccxt.binance.
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
    """Fetch OHLCV candles in batches until data is exhausted or limits are hit.

    Why loop in batches:
    - Most exchange endpoints cap each response by a fixed limit.
    - A single request usually cannot retrieve a long history.
    - We advance the `since` cursor using the last returned timestamp.

    Parameters
    ----------
    exchange:
        ccxt exchange instance.
    symbol:
        Instrument symbol in ccxt format, e.g., BTC/USDT.
    timeframe:
        Candle interval, e.g., 1m, 1h, 1d.
    since_iso:
        ISO timestamp string for the start point.
    limit:
        Max rows per request.
    max_batches:
        Safety cap for total requests in one call.

    Returns
    -------
    DataFrame
        Columns: timestamp, open, high, low, close, volume.
    """
    # CCXT APIs accept/start from millisecond timestamps.
    since_ms = exchange.parse8601(since_iso)
    all_rows: list[list[float]] = []

    for batch in range(max_batches):
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not rows:
            break

        all_rows.extend(rows)
        last_ts = rows[-1][0]
        # +1 ms avoids refetching the last candle in the next request.
        next_since = last_ts + 1

        print(f"{symbol} batch {batch + 1:02d}: {len(rows)} rows, last = {exchange.iso8601(last_ts)}")

        if next_since <= since_ms:
            break
        since_ms = next_since

        # Be polite to the exchange API.
        time.sleep(exchange.rateLimit / 1000 if getattr(exchange, "rateLimit", None) else 0.2)

        # Stop when we are close to current time.
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
    """Generate synthetic OHLCV data for offline development.

    This fallback lets the full research pipeline run even when:
    - network access is blocked,
    - the selected exchange endpoint is unstable,
    - or API credentials/regions are restricted.

    The synthetic process is simple on purpose:
    - random-walk-like returns with symbol-specific volatility,
    - deterministic seeding for reproducibility,
    - plausible high/low spread around open/close.
    """
    # Use symbol-dependent seed so each asset is reproducible but still different.
    rng = np.random.default_rng(seed + sum(ord(char) for char in symbol))
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    drift = 0.00002
    vol = 0.015 if symbol.startswith("BTC") else 0.020
    rets = drift + vol * rng.standard_normal(periods)
    # Build a positive price path from returns in log space.
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(0.004 * close * rng.standard_normal(periods))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(loc=1000, scale=300, size=periods))
    return pd.DataFrame(
        {
            # Use ISO strings to avoid unit ambiguity across pandas versions.
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
    """Download OHLCV for all symbols; fallback to demo data on failure.

    Design choice:
    This function intentionally catches broad exceptions so the pipeline remains
    runnable in educational/research settings. In production, you would usually
    narrow exception classes and add retries/alerting.
    """
    data: dict[str, pd.DataFrame] = {}
    try:
        # One exchange instance is reused for all symbols.
        exchange = make_exchange(EXCHANGE_ID)
        exchange.load_markets()
        for symbol in symbols:
            df = fetch_ohlcv_loop(
                exchange,
                symbol,
                TIMEFRAME,
                SINCE,
                LIMIT_PER_REQUEST,
                max_batches=10,
            )
            if len(df) == 0:
                raise RuntimeError(f"No data returned for {symbol}")
            data[symbol] = df
    except Exception as error:
        # For teaching/research workflows, fallback keeps the pipeline runnable.
        print("Data download failed; using demo data instead.")
        print("Reason:", repr(error))
        for symbol in symbols:
            data[symbol] = generate_demo_ohlcv(symbol)
    return data


def clean_ohlcv(df: pd.DataFrame, timeframe: str = TIMEFRAME) -> pd.DataFrame:
    """Clean raw OHLCV into a consistent analysis-ready table.

    Cleaning steps:
    1. Convert integer timestamps to timezone-aware UTC datetime.
    2. Sort chronologically and remove duplicate timestamps.
    3. Keep canonical OHLCV columns.
    4. Coerce non-numeric values to NaN and drop invalid rows.

    Returns a clean DataFrame used by signal and backtest functions.
    """
    _ = timeframe  # Kept for interface compatibility with notebook.
    out = df.copy()
    # Always normalize to UTC for consistent cross-exchange handling.
    out["datetime"] = parse_timestamp_auto(out["timestamp"])
    out = out.dropna(subset=["datetime"])
    out = out.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    out = out[["datetime", "open", "high", "low", "close", "volume"]]

    # Coerce bad values to NaN, then drop; this is safer than  failing mid-pipeline.
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


def build_alpha_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build class2 alpha factors and future-return target on cleaned OHLCV."""
    out = df.copy()
    out["ret_1h"] = out["close"].pct_change()
    out["future_ret_1h"] = out["close"].shift(-1) / out["close"] - 1

    out["factor_mom_4h"] = momentum(out["close"], 4)
    out["factor_mom_24h"] = momentum(out["close"], 24)
    out["factor_mom_72h"] = momentum(out["close"], 72)
    out["factor_vol_price"] = volume_price_trend(out["close"], out["volume"], 24)
    out["factor_range_pos"] = range_position(out["close"], out["high"], out["low"], 48)
    out["factor_rsi_14"] = calculate_rsi(out["close"], window=14)
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


def run_class2_alpha_workflow(clean_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | pd.Series]:
    """Reproduce class2 notebook requirements from this script only.

    Outputs are exported to outputs_class2/ for direct report usage.
    """
    btc_df = clean_data["BTC/USDT"].copy()
    alpha_df = build_alpha_dataset(btc_df)
    factor_cols = [col for col in alpha_df.columns if col.startswith("factor_")]

    ic_table = calc_ic_table(alpha_df, factor_cols)

    selected_factor = "factor_mom_24h"
    alpha_df["rolling_ic"] = rolling_ic(alpha_df[selected_factor], alpha_df["future_ret_1h"], window=2400)

    bt = quantile_monetization(
        alpha_df,
        selected_factor,
        q_high=0.80,
        q_low=0.20,
        window=24 * 300,
        fee_bps=2.0,
        direction=-1,
    )
    metrics = backtest_metrics(bt)

    sensitivity_rows: list[pd.Series] = []
    for q in [0.60, 0.70, 0.75, 0.80, 0.85]:
        tmp_bt = quantile_monetization(
            alpha_df,
            selected_factor,
            q_high=q,
            q_low=1 - q,
            window=24 * 60,
            fee_bps=2.0,
            direction=-1,
        )
        tmp_metrics = backtest_metrics(tmp_bt)
        tmp_metrics["q_high"] = q
        tmp_metrics["q_low"] = 1 - q
        sensitivity_rows.append(tmp_metrics)
    sensitivity = pd.DataFrame(sensitivity_rows)

    ic_table.to_csv(OUTPUTS_CLASS2 / "ic_table.csv", index=False)
    metrics.to_frame(name="value").to_csv(OUTPUTS_CLASS2 / "backtest_metrics.csv")
    sensitivity.to_csv(OUTPUTS_CLASS2 / "sensitivity.csv", index=False)
    bt[["datetime", "signal", "pnl", "equity"]].to_csv(OUTPUTS_CLASS2 / "quantile_bt_core.csv", index=False)

    print("Class2 outputs saved to:", OUTPUTS_CLASS2.resolve())
    print("Rolling IC mean:", float(alpha_df["rolling_ic"].mean()))
    print("Rolling IC std :", float(alpha_df["rolling_ic"].std()))

    return {
        "alpha_data": alpha_df,
        "ic_table": ic_table,
        "metrics": metrics,
        "sensitivity": sensitivity,
        "bt": bt,
    }


def data_quality_report(df: pd.DataFrame, timeframe: str = TIMEFRAME) -> dict[str, object]:
    """Compute a compact data-quality summary for quick sanity checks.

    The report helps answer:
    - Are there duplicate timestamps?
    - Are expected bars missing in the time grid?
    - What is the actual start/end coverage?
    """
    # Build the expected evenly spaced timeline and compare with actual timestamps.
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
    """Build a long-short moving-average signal with anti-look-ahead handling.

    Signal logic:
    - fast MA > slow MA  -> +1 (long)
    - fast MA <= slow MA -> -1 (short)

    Execution logic:
    - position is signal shifted by 1 bar, so decisions at t affect trading at t+1.
    - this is critical to avoid look-ahead bias in backtests.
    """
    out = df.copy()
    # Return is calculated on close-to-close bars.
    out["ret"] = out["close"].pct_change()
    out[f"ma_{fast}"] = out["close"].rolling(fast).mean()
    out[f"ma_{slow}"] = out["close"].rolling(slow).mean()
    out["signal"] = np.where(out[f"ma_{fast}"] > out[f"ma_{slow}"], 1, -1)
    # Before slow MA is available, there is no valid signal.
    out.loc[out[f"ma_{slow}"].isna(), "signal"] = 0
    # One-bar shift is essential: trade at t+1 using signal formed at t.
    out["position"] = out["signal"].shift(1).fillna(0)
    return out


def backtest_signal(df: pd.DataFrame, fee_bps: float = 2.0) -> pd.DataFrame:
    """Run a simple vectorized backtest with turnover-based transaction costs.

    Return model per bar:
    strategy_ret = position * ret - cost

    Cost model:
    - turnover = absolute change in position.
    - fee is charged in basis points on turnover.

    Notes:
    - This is intentionally minimal and does not model slippage/latency.
    - Equity here is cumulative arithmetic return around a 1.0 baseline.
    """
    out = df.copy()
    # Convert basis points into decimal fee rate.
    fee_rate = fee_bps / 10_000
    # Turnover is absolute position change; 0->1 and 1->-1 are both charged.
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["cost"] = out["turnover"] * fee_rate
    # Strategy PnL: exposure * return - transaction costs.
    out["strategy_ret"] = out["position"] * out["ret"] - out["cost"]
    out["strategy_ret"] = out["strategy_ret"].fillna(0)
    # This notebook-style equity uses cumulative returns around 1.0.
    out["equity"] = 1 + (out["strategy_ret"]).cumsum()
    out["buy_hold"] = 1 + (out["ret"].fillna(0)).cumsum()
    return out


def max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve.

    Drawdown is measured as the distance from running peak equity.
    The minimum drawdown value over time is reported as max drawdown.
    """
    running_max = equity.cummax()
    # Drawdown is distance from running peak; min value is max drawdown.
    dd = (equity - running_max) / 1
    return float(dd.min())


def performance_summary(bt: pd.DataFrame, periods_per_year: int = 24 * 365) -> pd.Series:
    """Summarize key performance metrics for a backtest result table.

    Included metrics:
    - Total Return
    - Annualized Return
    - Annualized Volatility
    - Sharpe (return/vol)
    - Max Drawdown
    - Average Turnover

    Annualization assumes equally spaced bars and uses periods_per_year
    (default configured for 1-hour bars).
    """
    r = bt["strategy_ret"].dropna()
    # Annualization assumes equally spaced bars.
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
    """Build a long-only MA signal variant.

    Difference from the baseline long-short signal:
    - Bull regime: +1
    - Non-bull regime: 0 (flat), not -1

    This isolates whether short exposure is helping or hurting performance.
    """
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    out[f"ma_{fast}"] = out["close"].rolling(fast).mean()
    out[f"ma_{slow}"] = out["close"].rolling(slow).mean()
    out["signal"] = np.where(out[f"ma_{fast}"] > out[f"ma_{slow}"], 1, 0)
    out.loc[out[f"ma_{slow}"].isna(), "signal"] = 0
    out["position"] = out["signal"].shift(1).fillna(0)
    return out


def plot_price_and_mas(bt: pd.DataFrame, fast: int, slow: int, title: str) -> None:
    """Plot close price and moving averages for visual sanity checks.

    This chart is useful to verify:
    - MA crossover timing,
    - trend regime behavior,
    - and whether signal windows are too fast/slow for the asset.
    """
    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt["datetime"], bt["close"], label="Close")
    ax.plot(bt["datetime"], bt[f"ma_{fast}"], label=f"MA_{fast}")
    ax.plot(bt["datetime"], bt[f"ma_{slow}"], label=f"MA_{slow}")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_equity(bt: pd.DataFrame, title: str) -> None:
    """Plot strategy equity versus buy-and-hold benchmark.

    This chart helps compare active signal behavior against passive exposure.
    """
    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt["datetime"], bt["equity"], label="MA strategy")
    ax.plot(bt["datetime"], bt["buy_hold"], label="Buy and hold")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    plt.show()


def run_baseline_workflow(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, dict[str, pd.DataFrame | pd.Series]]]:
    """Run the baseline research workflow end-to-end.

    Pipeline stages:
    1. Download/fallback data.
    2. Clean + validate + persist data.
    3. Build signals and run backtests per symbol.
    4. Persist summary and detailed outputs.

    Returns
    -------
    clean_data:
        Per-symbol cleaned datasets.
    summary_table:
        Per-symbol performance summary table.
    results:
        Per-symbol dict containing backtest table and summary series.
    """
    # Step 1: ingest raw market data (real or demo fallback).
    raw_data = download_or_demo(symbols)

    # Step 2: clean and persist a canonical dataset per symbol.
    clean_data: dict[str, pd.DataFrame] = {}
    for symbol, raw_df in raw_data.items():
        clean_df = clean_ohlcv(raw_df)
        clean_data[symbol] = clean_df
        safe_name = symbol.replace("/", "_")
        clean_df.to_parquet(DATA_CLEAN / f"{safe_name}_{TIMEFRAME}.parquet", index=False)
        print(symbol, data_quality_report(clean_df))

    fast = 40
    slow = 100
    # Step 3: generate baseline signal/backtest for each symbol.
    results: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
    for symbol, clean_df in clean_data.items():
        signal_df = add_ma_signal(clean_df, fast=fast, slow=slow)
        bt_df = backtest_signal(signal_df, fee_bps=2.0)
        results[symbol] = {"backtest": bt_df, "summary": performance_summary(bt_df)}

    summary_table = pd.DataFrame({symbol: obj["summary"] for symbol, obj in results.items()}).T
    summary_table.to_csv(RESULTS / "ma_strategy_summary.csv")

    # Persist per-symbol backtest details for later inspection.
    for symbol, obj in results.items():
        safe_name = symbol.replace("/", "_")
        bt_df = obj["backtest"]
        if isinstance(bt_df, pd.DataFrame):
            bt_df.to_csv(RESULTS / f"{safe_name}_{TIMEFRAME}_ma_backtest.csv", index=False)

    return clean_data, summary_table, results


def run_strategy_scenarios(clean_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Run scenario analyses on top of the same baseline framework.

    Scenarios:
    1. MA window sweep.
    2. Long-only vs long-short.
    3. Transaction-cost sensitivity.
    4. Symbol-universe extension.

    Keeping all scenarios under the same function set ensures comparability.
    """
    outputs: dict[str, pd.DataFrame] = {}

    # Scenario 1: Fast/slow window sweep.
    window_pairs = [(10, 30), (20, 60), (40, 100), (60, 180)]
    rows: list[dict[str, float | int | str]] = []
    base_symbol = "BTC/USDT"
    for fast, slow in window_pairs:
        # Re-run the full signal/backtest stack under each parameter pair.
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
    window_sweep.to_csv(RESULTS / "window_sweep.csv", index=False)
    outputs["window_sweep"] = window_sweep

    # Scenario 2: Long-only versus long-short.
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
    long_only_vs_long_short.to_csv(RESULTS / "long_only_vs_long_short.csv")
    outputs["long_only_vs_long_short"] = long_only_vs_long_short

    # Scenario 3: Transaction-cost sensitivity.
    fee_bps_grid = [0.5, 1.0, 2.0, 5.0, 10.0]
    fee_rows: list[dict[str, float]] = []
    for fee_bps in fee_bps_grid:
        # Isolate fee impact by reusing the same signal and only changing fee.
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
    fee_sensitivity.to_csv(RESULTS / "fee_sensitivity.csv", index=False)
    outputs["fee_sensitivity"] = fee_sensitivity

    # Scenario 4: Add one more symbol and rerun baseline.
    extended_symbols = sorted(set(list(clean_data.keys()) + ["SOL/USDT"]))
    # Reusing baseline workflow keeps evaluation rules identical across scenarios.
    clean_plus, summary_plus, _ = run_baseline_workflow(extended_symbols)
    _ = clean_plus
    summary_plus.to_csv(RESULTS / "extended_symbol_summary.csv")
    outputs["extended_symbol_summary"] = summary_plus

    print("Saved scenario outputs to:", RESULTS.resolve())
    return outputs


def main() -> None:
    """Execute baseline workflow and then scenario analyses.

    This main function is intentionally linear so new learners can trace
    the full research path in one pass.
    """
    print("[Stage 1/3] Running baseline workflow...")
    clean_data, summary_table, results = run_baseline_workflow(SYMBOLS)

    print("[Stage 2/3] Baseline summary:")
    print(summary_table)

    btc = results["BTC/USDT"]["backtest"]
    if isinstance(btc, pd.DataFrame):
        print("[Stage 2/3] Generating baseline plots...")
        plot_price_and_mas(btc, fast=40, slow=100, title="BTC/USDT close price and moving averages")
        plot_equity(btc, title="BTC/USDT equity curve")

    print("[Stage 3/3] Running scenario analyses...")
    run_strategy_scenarios(clean_data)

    print("[Stage 4/4] Running class2 alpha workflow...")
    run_class2_alpha_workflow(clean_data)


if __name__ == "__main__":
    main()
