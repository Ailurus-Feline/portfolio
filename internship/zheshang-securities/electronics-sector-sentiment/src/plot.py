"""Visualization for the v1 sentiment pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

REGIME_COLORS = {
    "neutral": "#1f77b4",
    "overheated": "#2ca02c",
    "overcooled": "#d62728",
}

FIGURE_SIZE = (14.0, 7.0)
PLOT_BBOX = [0.08, 0.10, 0.84, 0.70]
HEADER_BBOX = [0.08, 0.82, 0.84, 0.16]

LEGEND_ENTRIES = (
    ("neutral", "Neutral"),
    ("overheated", "Overheated"),
    ("overcooled", "Overcooled"),
    ("zscore", "Z-score"),
)


def _legend_handles() -> list[Line2D]:
    color_map = {
        "neutral": REGIME_COLORS["neutral"],
        "overheated": REGIME_COLORS["overheated"],
        "overcooled": REGIME_COLORS["overcooled"],
        "zscore": "black",
    }
    width_map = {"zscore": 1.2, "neutral": 2.0, "overheated": 2.0, "overcooled": 2.0}
    return [
        Line2D([0], [0], color=color_map[key], linewidth=width_map[key], label=label)
        for key, label in LEGEND_ENTRIES
    ]


def _plot_colored_index_line(ax: Axes, dates, log_index: pd.Series, regimes: pd.Series) -> None:
    x = pd.to_datetime(dates)
    y = log_index.to_numpy()
    regime_values = regimes.to_numpy()

    for i in range(len(x) - 1):
        color = REGIME_COLORS.get(regime_values[i], REGIME_COLORS["neutral"])
        ax.plot(x[i : i + 2], y[i : i + 2], color=color, linewidth=1.6)


def _draw_plot_panel(plot_ax: Axes, ax_right: Axes, data: pd.DataFrame) -> None:
    _plot_colored_index_line(plot_ax, data["date"], data["index_log"], data["regime"])

    plot_ax.set_ylabel("ln(Shenwan Electronics Index)")
    plot_ax.set_xlabel("Date")
    plot_ax.grid(True, alpha=0.25)

    ax_right.plot(data["date"], data["sentiment_z"], color="black", linewidth=1.0)
    ax_right.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_right.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax_right.axhline(-1.0, color="gray", linewidth=0.8, linestyle=":")
    ax_right.set_ylabel("Sentiment z-score (std)")


def _draw_header(header_ax: Axes, title: str) -> None:
    header_ax.axis("off")
    header_ax.text(
        0.5,
        1.0,
        title,
        ha="center",
        va="top",
        fontsize=11,
        transform=header_ax.transAxes,
    )
    header_ax.legend(
        handles=_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(0.0, 0.82),
        ncol=1,
        fontsize=9,
        framealpha=0.95,
    )


def _build_figure(data: pd.DataFrame, title: str) -> Figure:
    fig = plt.figure(figsize=FIGURE_SIZE)

    plot_ax = fig.add_axes(PLOT_BBOX)
    ax_right = plot_ax.twinx()
    _draw_plot_panel(plot_ax, ax_right, data)

    header_ax = fig.add_axes(HEADER_BBOX)
    header_ax.set_zorder(10)
    _draw_header(header_ax, title)

    return fig


def plot_sentiment_vs_index(
    data: pd.DataFrame,
    output_path: Path,
    title: str = "Electronics Sector Sentiment vs Shenwan Electronics Index",
) -> None:
    """Create the dual-axis chart used as the main sentiment deliverable."""
    plot_df = data.dropna(subset=["index_log", "sentiment_z"]).copy()
    if plot_df.empty:
        raise ValueError("No rows available for plotting after dropping NaNs.")

    fig = _build_figure(plot_df, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_v1_v2_zscore_comparison(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    output_path: Path,
) -> None:
    """Overlay v1 and v2 sentiment z-scores to show parameter impact."""
    merged = v1.loc[:, ["date", "sentiment_z"]].merge(
        v2.loc[:, ["date", "sentiment_z"]],
        on="date",
        suffixes=("_v1", "_v2"),
    )
    diff = merged["sentiment_z_v2"] - merged["sentiment_z_v1"]
    corr = merged["sentiment_z_v1"].corr(merged["sentiment_z_v2"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(merged["date"], merged["sentiment_z_v1"], color="#1f77b4", linewidth=1.0, label="v1 (equal 1/3, EMA 90)")
    axes[0].plot(merged["date"], merged["sentiment_z_v2"], color="#d62728", linewidth=1.0, alpha=0.85, label="v2 best (0.25/0.25/0.50, EMA 60)")
    axes[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
    axes[0].axhline(-1.0, color="gray", linewidth=0.6, linestyle=":")
    axes[0].set_ylabel("Sentiment z-score")
    axes[0].set_title(f"v1 vs v2 Sentiment Z-score  (corr = {corr:.3f})")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(merged["date"], diff, color="#9467bd", linewidth=0.9)
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("v2 − v1")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Difference panel (non-zero = parameters matter)")
    axes[1].grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
