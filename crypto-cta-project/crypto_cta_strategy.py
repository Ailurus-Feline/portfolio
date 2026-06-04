from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ccxt


pd.set_option("display.max_columns", 20)

# Project paths
PROJECT_ROOT = Path.cwd()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"
RESULTS = PROJECT_ROOT / "results"
for path_obj in [DATA_RAW, DATA_CLEAN, RESULTS]:
    path_obj.mkdir(parents=True, exist_ok=True)

# Baseline configuration from notebook
EXCHANGE_ID = "binance"  # If Binance is not accessible, try "okx" or "bybit".
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
SINCE = "2024-01-01T00:00:00Z"
LIMIT_PER_REQUEST = 1000


def make_exchange(exchange_id: str = EXCHANGE_ID):
    """Create a CCXT exchange object with rate limit enabled."""
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
    """Fetch OHLCV bars by repeatedly calling fetch_ohlcv and moving `since` forward."""
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
    """Generate demo OHLCV data for offline use when exchange data is unavailable."""
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
            "timestamp": (idx.view("int64") // 1_000_000).astype("int64"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def download_or_demo(symbols: list[str] = SYMBOLS) -> dict[str, pd.DataFrame]:
    """Download OHLCV from exchange; fallback to demo data on any failure."""
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
                max_batches=10,
            )
            if len(df) == 0:
                raise RuntimeError(f"No data returned for {symbol}")
            data[symbol] = df
    except Exception as error:
        print("Data download failed; using demo data instead.")
        print("Reason:", repr(error))
        for symbol in symbols:
            data[symbol] = generate_demo_ohlcv(symbol)
    return data


def clean_ohlcv(df: pd.DataFrame, timeframe: str = TIMEFRAME) -> pd.DataFrame:
    """Apply basic cleaning: timestamp conversion, sorting, deduping, and numeric enforcement."""
    _ = timeframe  # Kept for interface compatibility with notebook.
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out = out.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    out = out[["datetime", "open", "high", "low", "close", "volume"]]

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def data_quality_report(df: pd.DataFrame, timeframe: str = TIMEFRAME) -> dict[str, object]:
    """Return a quick data quality summary including missing bars and duplicates."""
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
    """Build long-short MA signal and one-bar delayed position to avoid look-ahead bias."""
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    out[f"ma_{fast}"] = out["close"].rolling(fast).mean()
    out[f"ma_{slow}"] = out["close"].rolling(slow).mean()
    out["signal"] = np.where(out[f"ma_{fast}"] > out[f"ma_{slow}"], 1, -1)
    out.loc[out[f"ma_{slow}"].isna(), "signal"] = 0
    out["position"] = out["signal"].shift(1).fillna(0)
    return out


def backtest_signal(df: pd.DataFrame, fee_bps: float = 2.0) -> pd.DataFrame:
    """Run a simple bar-by-bar backtest with turnover-based transaction costs."""
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
    """Compute maximum drawdown from an equity curve represented in cumulative-return space."""
    running_max = equity.cummax()
    dd = (equity - running_max) / 1
    return float(dd.min())


def performance_summary(bt: pd.DataFrame, periods_per_year: int = 24 * 365) -> pd.Series:
    """Summarize strategy performance with annualized return/volatility and turnover metrics."""
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
    """Build a long-only MA signal where non-bull regimes stay flat instead of short."""
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    out[f"ma_{fast}"] = out["close"].rolling(fast).mean()
    out[f"ma_{slow}"] = out["close"].rolling(slow).mean()
    out["signal"] = np.where(out[f"ma_{fast}"] > out[f"ma_{slow}"], 1, 0)
    out.loc[out[f"ma_{slow}"].isna(), "signal"] = 0
    out["position"] = out["signal"].shift(1).fillna(0)
    return out


def plot_price_and_mas(bt: pd.DataFrame, fast: int, slow: int, title: str) -> None:
    """Plot close price and moving averages for visual sanity checks."""
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
    """Plot strategy equity against buy-and-hold benchmark."""
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
    """Run the notebook baseline pipeline and return cleaned data, summary table, and full results."""
    raw_data = download_or_demo(symbols)

    clean_data: dict[str, pd.DataFrame] = {}
    for symbol, raw_df in raw_data.items():
        clean_df = clean_ohlcv(raw_df)
        clean_data[symbol] = clean_df
        safe_name = symbol.replace("/", "_")
        clean_df.to_parquet(DATA_CLEAN / f"{safe_name}_{TIMEFRAME}.parquet", index=False)
        print(symbol, data_quality_report(clean_df))

    fast = 40
    slow = 100
    results: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
    for symbol, clean_df in clean_data.items():
        signal_df = add_ma_signal(clean_df, fast=fast, slow=slow)
        bt_df = backtest_signal(signal_df, fee_bps=2.0)
        results[symbol] = {"backtest": bt_df, "summary": performance_summary(bt_df)}

    summary_table = pd.DataFrame({symbol: obj["summary"] for symbol, obj in results.items()}).T
    summary_table.to_csv(RESULTS / "ma_strategy_summary.csv")

    for symbol, obj in results.items():
        safe_name = symbol.replace("/", "_")
        bt_df = obj["backtest"]
        if isinstance(bt_df, pd.DataFrame):
            bt_df.to_csv(RESULTS / f"{safe_name}_{TIMEFRAME}_ma_backtest.csv", index=False)

    return clean_data, summary_table, results


def run_strategy_scenarios(clean_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Run scenario analyses built on top of the baseline MA strategy methods."""
    outputs: dict[str, pd.DataFrame] = {}

    # Scenario 1: Fast/slow window sweep.
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
    clean_plus, summary_plus, _ = run_baseline_workflow(extended_symbols)
    _ = clean_plus
    summary_plus.to_csv(RESULTS / "extended_symbol_summary.csv")
    outputs["extended_symbol_summary"] = summary_plus

    print("Saved scenario outputs to:", RESULTS.resolve())
    return outputs


def main() -> None:
    """Execute baseline workflow and then run scenario analyses."""
    clean_data, summary_table, results = run_baseline_workflow(SYMBOLS)

    print("Baseline summary:")
    print(summary_table)

    btc = results["BTC/USDT"]["backtest"]
    if isinstance(btc, pd.DataFrame):
        plot_price_and_mas(btc, fast=40, slow=100, title="BTC/USDT close price and moving averages")
        plot_equity(btc, title="BTC/USDT equity curve")

    run_strategy_scenarios(clean_data)


if __name__ == "__main__":
    main()
