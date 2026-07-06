"""Long-only backtest engine for v3 alpha strategies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy_rules import PositionRule

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestConfig:
    """Capital and transaction-cost assumptions."""

    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0002
    stamp_tax_rate: float = 0.001
    leverage_max: float = 1.0
    slippage_bps: float = 0.0


def load_backtest_config(path: Path) -> BacktestConfig:
    """Load key-value settings from backtest_config.yaml without PyYAML."""
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        values[key.strip()] = float(raw_value.strip())

    return BacktestConfig(
        initial_capital=values.get("initial_capital", BacktestConfig.initial_capital),
        commission_rate=values.get("commission_rate", BacktestConfig.commission_rate),
        stamp_tax_rate=values.get("stamp_tax_rate", BacktestConfig.stamp_tax_rate),
        leverage_max=values.get("leverage_max", BacktestConfig.leverage_max),
        slippage_bps=values.get("slippage_bps", BacktestConfig.slippage_bps),
    )


def _trading_cost(
    prev_position: float,
    new_position: float,
    portfolio_value: float,
    config: BacktestConfig,
) -> float:
    """Commission on traded notional plus stamp tax on sells."""
    turnover = abs(new_position - prev_position)
    if turnover == 0.0 or portfolio_value <= 0.0:
        return 0.0

    buy_notional = max(0.0, new_position - prev_position) * portfolio_value
    sell_notional = max(0.0, prev_position - new_position) * portfolio_value
    traded_notional = buy_notional + sell_notional

    commission = traded_notional * config.commission_rate
    stamp_tax = sell_notional * config.stamp_tax_rate
    slippage = traded_notional * (config.slippage_bps / 10_000.0)
    return commission + stamp_tax + slippage


def run_backtest(
    signals: pd.DataFrame,
    config: BacktestConfig,
    target_fn: PositionRule,
    initial_position: float = 0.0,
) -> pd.DataFrame:
    """
    Backtest a long-only strategy on the sector index.

    Signal at close on day t (signal_z) sets position held on day t+1.
    """
    frame = signals.sort_values("date").reset_index(drop=True).copy()
    required = {"date", "signal_z", "index_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Signal input missing columns: {sorted(missing)}")

    frame["index_return"] = frame["index_close"].pct_change()
    n_rows = len(frame)

    positions = np.zeros(n_rows, dtype=float)
    target_positions = np.full(n_rows, np.nan, dtype=float)
    gross_returns = np.zeros(n_rows, dtype=float)
    trading_costs = np.zeros(n_rows, dtype=float)
    net_returns = np.zeros(n_rows, dtype=float)
    strategy_nav = np.full(n_rows, np.nan, dtype=float)
    benchmark_nav = np.full(n_rows, np.nan, dtype=float)

    strategy_nav[0] = config.initial_capital
    benchmark_nav[0] = config.initial_capital
    positions[0] = float(np.clip(initial_position, 0.0, config.leverage_max))

    for day in range(1, n_rows):
        prev_position = positions[day - 1]
        prev_z = frame.at[day - 1, "signal_z"]

        if pd.isna(prev_z):
            target_positions[day] = prev_position
        else:
            target_positions[day] = target_fn(float(prev_z), prev_position)

        positions[day] = float(np.clip(target_positions[day], 0.0, config.leverage_max))

        index_return = frame.at[day, "index_return"]
        if pd.isna(index_return):
            strategy_nav[day] = strategy_nav[day - 1]
            benchmark_nav[day] = benchmark_nav[day - 1]
            continue

        gross_returns[day] = positions[day] * index_return
        trading_costs[day] = _trading_cost(
            prev_position=positions[day - 1],
            new_position=positions[day],
            portfolio_value=strategy_nav[day - 1],
            config=config,
        )
        net_returns[day] = gross_returns[day] - trading_costs[day] / strategy_nav[day - 1]
        strategy_nav[day] = strategy_nav[day - 1] * (1.0 + net_returns[day])
        benchmark_nav[day] = benchmark_nav[day - 1] * (1.0 + index_return)

    frame["position"] = positions
    frame["target_position"] = target_positions
    frame["signal_z_lagged"] = frame["signal_z"].shift(1)
    if "regime" in frame.columns:
        frame["signal_regime"] = frame["regime"].shift(1)
    frame["gross_return"] = gross_returns
    frame["trading_cost"] = trading_costs
    frame["net_return"] = net_returns
    frame["strategy_nav"] = strategy_nav
    frame["benchmark_nav"] = benchmark_nav
    frame["strategy_cum_return"] = frame["strategy_nav"] / config.initial_capital - 1.0
    frame["benchmark_cum_return"] = frame["benchmark_nav"] / config.initial_capital - 1.0
    frame["excess_cum_return"] = frame["strategy_cum_return"] - frame["benchmark_cum_return"]
    frame["position_change"] = frame["position"].diff().fillna(frame["position"]).abs()
    return frame


DualPositionRule = Callable[[float, float, float], float]


def run_dual_backtest(
    signals: pd.DataFrame,
    config: BacktestConfig,
    target_fn: DualPositionRule,
    initial_position: float = 0.0,
) -> pd.DataFrame:
    """Backtest using two lagged z-scores (signal_z, signal_z_2)."""
    frame = signals.sort_values("date").reset_index(drop=True).copy()
    required = {"date", "signal_z", "signal_z_2", "index_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dual signal input missing columns: {sorted(missing)}")

    frame["index_return"] = frame["index_close"].pct_change()
    n_rows = len(frame)

    positions = np.zeros(n_rows, dtype=float)
    target_positions = np.full(n_rows, np.nan, dtype=float)
    gross_returns = np.zeros(n_rows, dtype=float)
    trading_costs = np.zeros(n_rows, dtype=float)
    net_returns = np.zeros(n_rows, dtype=float)
    strategy_nav = np.full(n_rows, np.nan, dtype=float)
    benchmark_nav = np.full(n_rows, np.nan, dtype=float)

    strategy_nav[0] = config.initial_capital
    benchmark_nav[0] = config.initial_capital
    positions[0] = float(np.clip(initial_position, 0.0, config.leverage_max))

    for day in range(1, n_rows):
        prev_position = positions[day - 1]
        prev_z1 = frame.at[day - 1, "signal_z"]
        prev_z2 = frame.at[day - 1, "signal_z_2"]

        if pd.isna(prev_z1) or pd.isna(prev_z2):
            target_positions[day] = prev_position
        else:
            target_positions[day] = target_fn(float(prev_z1), float(prev_z2), prev_position)

        positions[day] = float(np.clip(target_positions[day], 0.0, config.leverage_max))

        index_return = frame.at[day, "index_return"]
        if pd.isna(index_return):
            strategy_nav[day] = strategy_nav[day - 1]
            benchmark_nav[day] = benchmark_nav[day - 1]
            continue

        gross_returns[day] = positions[day] * index_return
        trading_costs[day] = _trading_cost(
            prev_position=positions[day - 1],
            new_position=positions[day],
            portfolio_value=strategy_nav[day - 1],
            config=config,
        )
        net_returns[day] = gross_returns[day] - trading_costs[day] / strategy_nav[day - 1]
        strategy_nav[day] = strategy_nav[day - 1] * (1.0 + net_returns[day])
        benchmark_nav[day] = benchmark_nav[day - 1] * (1.0 + index_return)

    frame["position"] = positions
    frame["target_position"] = target_positions
    frame["signal_z_lagged"] = frame["signal_z"].shift(1)
    frame["signal_z_2_lagged"] = frame["signal_z_2"].shift(1)
    frame["gross_return"] = gross_returns
    frame["trading_cost"] = trading_costs
    frame["net_return"] = net_returns
    frame["strategy_nav"] = strategy_nav
    frame["benchmark_nav"] = benchmark_nav
    frame["strategy_cum_return"] = frame["strategy_nav"] / config.initial_capital - 1.0
    frame["benchmark_cum_return"] = frame["benchmark_nav"] / config.initial_capital - 1.0
    frame["excess_cum_return"] = frame["strategy_cum_return"] - frame["benchmark_cum_return"]
    frame["position_change"] = frame["position"].diff().fillna(frame["position"]).abs()
    return frame


# Backward-compatible alias used during the first v3 iteration.
def run_long_only_backtest(signals: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    from src.strategy_rules import baseline_target

    if "signal_z" not in signals.columns and "sentiment_z" in signals.columns:
        signals = signals.rename(columns={"sentiment_z": "signal_z"})
    return run_backtest(signals, config, baseline_target)


@dataclass(frozen=True)
class PerformanceSummary:
    """Segment-level performance metrics."""

    split: str
    n_days: int
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe: float
    n_trades: int
    total_cost: float
    avg_position: float
    excess_return_vs_benchmark: float


def _max_drawdown(nav: pd.Series) -> float:
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return float(drawdown.min())


def _annualized_return(daily_returns: pd.Series, n_days: int) -> float:
    clean = daily_returns.dropna()
    if clean.empty or n_days <= 0:
        return float("nan")
    cumulative = float((1.0 + clean).prod())
    if cumulative <= 0:
        return float("nan")
    years = n_days / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    return cumulative ** (1.0 / years) - 1.0


def _sharpe_ratio(daily_returns: pd.Series) -> float:
    clean = daily_returns.dropna()
    if len(clean) < 2:
        return float("nan")
    std = float(clean.std(ddof=0))
    if std == 0:
        return float("nan")
    return float(clean.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def summarize_performance(
    daily: pd.DataFrame,
    split: str | None = None,
) -> PerformanceSummary:
    """Compute performance statistics for one segment."""
    segment = daily if split is None else daily.loc[daily["split"] == split].copy()
    if segment.empty:
        raise ValueError(f"No rows available for split={split!r}.")

    start_nav = float(segment["strategy_nav"].iloc[0])
    end_nav = float(segment["strategy_nav"].iloc[-1])
    start_bench = float(segment["benchmark_nav"].iloc[0])
    end_bench = float(segment["benchmark_nav"].iloc[-1])

    total_return = end_nav / start_nav - 1.0
    benchmark_return = end_bench / start_bench - 1.0
    n_days = len(segment)

    return PerformanceSummary(
        split=split or "all",
        n_days=n_days,
        total_return=total_return,
        annualized_return=_annualized_return(segment["net_return"], n_days),
        max_drawdown=_max_drawdown(segment["strategy_nav"]),
        sharpe=_sharpe_ratio(segment["net_return"]),
        n_trades=int((segment["position"].diff().abs() > 1e-9).sum()),
        total_cost=float(segment["trading_cost"].sum()),
        avg_position=float(segment["position"].mean()),
        excess_return_vs_benchmark=total_return - benchmark_return,
    )


def performance_summary_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Return metrics for all, train, valid, and final splits."""
    splits = ["all", "train", "valid", "final"]
    rows: list[dict[str, float | int | str]] = []
    for split in splits:
        segment = daily if split == "all" else daily.loc[daily["split"] == split]
        if segment.empty:
            continue
        summary = summarize_performance(daily, None if split == "all" else split)
        rows.append(
            {
                "split": summary.split,
                "n_days": summary.n_days,
                "total_return": summary.total_return,
                "annualized_return": summary.annualized_return,
                "max_drawdown": summary.max_drawdown,
                "sharpe": summary.sharpe,
                "n_trades": summary.n_trades,
                "total_cost": summary.total_cost,
                "avg_position": summary.avg_position,
                "excess_vs_benchmark": summary.excess_return_vs_benchmark,
            }
        )
    return pd.DataFrame(rows)
