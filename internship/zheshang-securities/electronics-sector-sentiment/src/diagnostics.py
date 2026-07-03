"""Signal diagnostics for v2 parameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    FINAL_RATIO,
    GRID_TOP_N,
    IC_HORIZONS,
    IC_PRIMARY_HORIZON,
    IC_SCORE_WEIGHTS,
    TRAIN_RATIO,
    VALID_RATIO,
)

SplitName = str


@dataclass(frozen=True)
class ICSummary:
    """Rank IC metrics for one segment and evaluation horizon."""

    horizon: int
    rank_ic: float
    ic_mean: float
    ic_std: float
    ic_ir: float
    n_obs: int


@dataclass(frozen=True)
class SegmentDiagnostics:
    """Diagnostics for one chronological segment."""

    split: SplitName
    ic_by_horizon: dict[int, ICSummary]
    score: float
    event_study: pd.DataFrame
    quintile_returns: pd.DataFrame


def assign_time_splits(
    dates: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
) -> pd.Series:
    """Assign train / valid / final labels using chronological 6-2-2 splits."""
    if not np.isclose(train_ratio + valid_ratio + FINAL_RATIO, 1.0):
        raise ValueError("Split ratios must sum to 1.")

    ordered = dates.sort_values().reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_ratio)
    valid_end = int(n_rows * (train_ratio + valid_ratio))

    split = pd.Series("final", index=ordered.index, dtype="object")
    split.iloc[:train_end] = "train"
    split.iloc[train_end:valid_end] = "valid"
    return split


def attach_splits(data: pd.DataFrame) -> pd.DataFrame:
    """Add a split column based on chronological ordering."""
    result = data.sort_values("date").reset_index(drop=True).copy()
    result["split"] = assign_time_splits(result["date"])
    return result


def add_forward_returns(data: pd.DataFrame, horizons: tuple[int, ...] = IC_HORIZONS) -> pd.DataFrame:
    """Attach index forward simple returns for each horizon."""
    result = data.sort_values("date").reset_index(drop=True).copy()
    for horizon in horizons:
        result[f"fwd_ret_{horizon}d"] = result["index_close"].shift(-horizon) / result["index_close"] - 1.0
    return result


def _valid_pairs(signal: pd.Series, forward_return: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"signal": signal, "forward_return": forward_return}).dropna()
    return frame


def rank_ic(signal: pd.Series, forward_return: pd.Series) -> float:
    """Pooled Spearman correlation between signal and forward return."""
    frame = _valid_pairs(signal, forward_return)
    if len(frame) < 3:
        return float("nan")
    return float(frame["signal"].corr(frame["forward_return"], method="spearman"))


def monthly_rank_ic(signal: pd.Series, forward_return: pd.Series, dates: pd.Series) -> pd.Series:
    """Monthly Spearman IC series used to compute IC mean / IC IR."""
    frame = _valid_pairs(signal, forward_return)
    frame["date"] = dates.loc[frame.index].to_numpy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M").astype(str)
    if frame.empty:
        return pd.Series(dtype=float)

    def _month_ic(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return float("nan")
        return float(group["signal"].corr(group["forward_return"], method="spearman"))

    return frame.groupby("month", sort=True).apply(_month_ic, include_groups=False)


def summarize_ic(
    data: pd.DataFrame,
    horizon: int,
    split: SplitName | None = None,
) -> ICSummary:
    """Compute pooled and monthly Rank IC metrics for one segment."""
    segment = data if split is None else data.loc[data["split"] == split]
    signal = segment["sentiment_z"]
    forward_return = segment[f"fwd_ret_{horizon}d"]
    dates = segment["date"]

    pooled = rank_ic(signal, forward_return)
    monthly = monthly_rank_ic(signal, forward_return, dates).dropna()
    ic_mean = float(monthly.mean()) if not monthly.empty else float("nan")
    ic_std = float(monthly.std(ddof=0)) if len(monthly) > 1 else float("nan")
    ic_ir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) and ic_std > 0 else float("nan")

    frame = _valid_pairs(signal, forward_return)
    return ICSummary(
        horizon=horizon,
        rank_ic=pooled,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        n_obs=len(frame),
    )


def ic_score(ic_by_horizon: dict[int, ICSummary]) -> float:
    """Weighted IC score; more negative is better for contrarian sentiment."""
    total_weight = 0.0
    weighted_sum = 0.0
    for horizon, weight in IC_SCORE_WEIGHTS.items():
        summary = ic_by_horizon.get(horizon)
        if summary is None or np.isnan(summary.rank_ic):
            continue
        weighted_sum += weight * summary.rank_ic
        total_weight += weight
    if total_weight == 0:
        return float("nan")
    return weighted_sum / total_weight


def event_study_table(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Average forward return by regime for one horizon."""
    frame = data.loc[:, ["regime", f"fwd_ret_{horizon}d"]].dropna()
    if frame.empty:
        return pd.DataFrame(columns=["regime", "mean_forward_return", "count"])

    grouped = (
        frame.groupby("regime", observed=False)[f"fwd_ret_{horizon}d"]
        .agg(mean_forward_return="mean", count="count")
        .reset_index()
    )
    order = pd.Categorical(grouped["regime"], categories=["overcooled", "neutral", "overheated"], ordered=True)
    return grouped.assign(regime=order).sort_values("regime").reset_index(drop=True)


def quintile_forward_returns(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Mean forward return by sentiment z-score quintile."""
    frame = data.loc[:, ["sentiment_z", f"fwd_ret_{horizon}d"]].dropna().copy()
    if frame.empty:
        return pd.DataFrame(columns=["quintile", "mean_forward_return", "count"])

    try:
        frame["quintile"] = pd.qcut(frame["sentiment_z"], 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    except ValueError:
        return pd.DataFrame(columns=["quintile", "mean_forward_return", "count"])

    grouped = (
        frame.groupby("quintile", observed=False)[f"fwd_ret_{horizon}d"]
        .agg(mean_forward_return="mean", count="count")
        .reset_index()
    )
    return grouped


def evaluate_segment(data: pd.DataFrame, split: SplitName) -> SegmentDiagnostics:
    """Compute IC, event study, and quintile diagnostics for one split."""
    segment = data.loc[data["split"] == split].copy()
    ic_by_horizon = {horizon: summarize_ic(segment, horizon) for horizon in IC_HORIZONS}
    return SegmentDiagnostics(
        split=split,
        ic_by_horizon=ic_by_horizon,
        score=ic_score(ic_by_horizon),
        event_study=event_study_table(segment, IC_HORIZONS[0]),
        quintile_returns=quintile_forward_returns(segment, IC_HORIZONS[0]),
    )


def split_date_range(data: pd.DataFrame, split: SplitName) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive start/end dates for a split."""
    segment = data.loc[data["split"] == split, "date"]
    if segment.empty:
        raise ValueError(f"No rows found for split={split!r}.")
    return pd.Timestamp(segment.min()), pd.Timestamp(segment.max())


def plot_quintile_returns(quintiles: pd.DataFrame, output_path: Path, horizon: int, split: SplitName) -> None:
    """Bar chart: mean forward return by sentiment quintile (Q1=coldest, Q5=hottest)."""
    if quintiles.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        quintiles["quintile"].astype(str),
        quintiles["mean_forward_return"] * 100.0,
        color="#4c72b0",
    )
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(
        f"Sentiment Quintile vs Forward Return ({horizon}d, {split})\n"
        "Ideal pattern: Q1 (low sentiment) > Q5 (high sentiment)"
    )
    ax.set_xlabel("Sentiment quintile (Q1 = low, Q5 = high)")
    ax.set_ylabel("Mean forward return (%)")
    ax.grid(True, axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_event_study(event_study: pd.DataFrame, output_path: Path, horizon: int, split: SplitName) -> None:
    """Bar chart: mean forward return when sentiment is overcooled / neutral / overheated."""
    if event_study.empty:
        return

    colors = {
        "overcooled": "#d62728",
        "neutral": "#1f77b4",
        "overheated": "#2ca02c",
    }
    bar_colors = [colors.get(regime, "#777777") for regime in event_study["regime"].astype(str)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        event_study["regime"].astype(str),
        event_study["mean_forward_return"] * 100.0,
        color=bar_colors,
    )
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(
        f"Regime Event Study ({horizon}d, {split})\n"
        "Ideal pattern: overcooled > neutral > overheated"
    )
    ax.set_xlabel("Sentiment regime")
    ax.set_ylabel("Mean forward return (%)")
    ax.grid(True, axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ic_by_split(winner: pd.Series, output_path: Path) -> None:
    """Bar chart: Rank IC on train / valid / final for the selected config."""
    splits = ["train", "valid", "final"]
    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, horizon in enumerate(IC_HORIZONS):
        values = [winner[f"{split}_rank_ic_{horizon}d"] for split in splits]
        offset = (index - 0.5) * width
        ax.bar(x + offset, values, width, label=f"{horizon}d forward return")

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([split.capitalize() for split in splits])
    ax.set_ylabel("Rank IC")
    ax.set_title(
        f"Selected Config IC by Split ({winner['config_id']})\n"
        "Negative IC = higher sentiment predicts lower future return (contrarian)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_grid_top_results(
    results: pd.DataFrame,
    output_path: Path,
    winner: pd.Series,
    top_n: int = GRID_TOP_N,
) -> None:
    """Horizontal bar chart of the best grid configs by valid IC score."""
    top = results.sort_values("valid_score", ascending=True).head(top_n).copy()
    top = top.sort_values("valid_score", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(top) + 1.5)))
    colors = ["#2ca02c" if config_id == winner["config_id"] else "#4c72b0" for config_id in top["config_id"]]
    ax.barh(top["config_id"], top["valid_score"], color=colors)
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Valid IC score (more negative is better)")
    ax.set_ylabel("Config")
    ax.set_title(
        f"Top {len(top)} Grid Configs by Valid IC Score\n"
        f"Green = selected ({winner['config_id']})"
    )
    ax.grid(True, axis="x", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_selection_candidates(
    results: pd.DataFrame,
    output_path: Path,
    winner: pd.Series,
    top_k_train: int = 5,
) -> None:
    """Show the train top-K pool and why the winner was picked on valid IC."""
    pool = results.sort_values("train_score", ascending=True).head(top_k_train).copy()
    pool = pool.sort_values("valid_score", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.45 * len(pool) + 1.5)))
    colors = ["#2ca02c" if config_id == winner["config_id"] else "#4c72b0" for config_id in pool["config_id"]]
    ax.barh(pool["config_id"], pool["valid_score"], color=colors)
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Valid IC score (more negative is better)")
    ax.set_ylabel("Config")
    ax.set_title(
        f"Selection Pool: Train Top-{top_k_train}, Ranked by Valid IC\n"
        f"Green = selected ({winner['config_id']})"
    )
    ax.grid(True, axis="x", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _draw_quintile_panel(ax, quintiles: pd.DataFrame, horizon: int) -> None:
    if quintiles.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.bar(
        quintiles["quintile"].astype(str),
        quintiles["mean_forward_return"] * 100.0,
        color="#4c72b0",
    )
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(f"Quintile Returns ({horizon}d, final)")
    ax.set_xlabel("Quintile")
    ax.set_ylabel("Mean fwd return (%)")


def _draw_event_panel(ax, event_study: pd.DataFrame, horizon: int) -> None:
    if event_study.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    colors = {"overcooled": "#d62728", "neutral": "#1f77b4", "overheated": "#2ca02c"}
    bar_colors = [colors.get(str(regime), "#777777") for regime in event_study["regime"]]
    ax.bar(event_study["regime"].astype(str), event_study["mean_forward_return"] * 100.0, color=bar_colors)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(f"Regime Event Study ({horizon}d, final)")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Mean fwd return (%)")


def _draw_ic_panel(ax, winner: pd.Series) -> None:
    splits = ["train", "valid", "final"]
    x = np.arange(len(splits))
    width = 0.35
    for index, horizon in enumerate(IC_HORIZONS):
        values = [winner[f"{split}_rank_ic_{horizon}d"] for split in splits]
        offset = (index - 0.5) * width
        ax.bar(x + offset, values, width, label=f"{horizon}d")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([split.capitalize() for split in splits])
    ax.set_title("Rank IC by Split (selected config)")
    ax.set_ylabel("Rank IC")
    ax.legend(fontsize=8)


def _draw_grid_panel(ax, results: pd.DataFrame, winner: pd.Series, top_k_train: int = 5) -> None:
    pool = results.sort_values("train_score", ascending=True).head(top_k_train).copy()
    pool = pool.sort_values("valid_score", ascending=True)
    colors = ["#2ca02c" if row.config_id == winner["config_id"] else "#4c72b0" for row in pool.itertuples()]
    ax.barh(pool["config_id"], pool["valid_score"], color=colors)
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"Train Top-{top_k_train} by Valid IC (green = selected)")
    ax.set_xlabel("Valid IC score")
    ax.tick_params(axis="y", labelsize=8)


def plot_v2_summary(
    winner: pd.Series,
    results: pd.DataFrame,
    final_diag: SegmentDiagnostics,
    output_path: Path,
) -> None:
    """Four-panel dashboard summarizing the key v2 findings."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"v2 Summary — {winner['config_id']}  "
        f"(weights={winner['w_new_high_low']:.2f}/{winner['w_above_ma']:.2f}/{winner['w_positive_return']:.2f}, "
        f"EMA={int(winner['ema_span'])})",
        fontsize=13,
        y=0.98,
    )

    _draw_ic_panel(axes[0, 0], winner)
    _draw_event_panel(axes[0, 1], final_diag.event_study, IC_PRIMARY_HORIZON)
    _draw_quintile_panel(axes[1, 0], final_diag.quintile_returns, IC_PRIMARY_HORIZON)
    _draw_grid_panel(axes[1, 1], results, winner)

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
