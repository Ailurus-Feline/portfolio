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


def _draw_header(header_ax: Axes) -> None:
    header_ax.axis("off")
    header_ax.text(
        0.5,
        1.0,
        "Electronics Sector Sentiment vs Shenwan Electronics Index",
        ha="center",
        va="top",
        fontsize=12,
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


def _build_figure(data: pd.DataFrame) -> Figure:
    fig = plt.figure(figsize=FIGURE_SIZE)

    plot_ax = fig.add_axes(PLOT_BBOX)
    ax_right = plot_ax.twinx()
    _draw_plot_panel(plot_ax, ax_right, data)

    header_ax = fig.add_axes(HEADER_BBOX)
    header_ax.set_zorder(10)
    _draw_header(header_ax)

    return fig


def plot_sentiment_vs_index(data: pd.DataFrame, output_path: Path) -> None:
    """Create the dual-axis chart used as the main v1 deliverable."""
    plot_df = data.dropna(subset=["index_log", "sentiment_z"]).copy()
    if plot_df.empty:
        raise ValueError("No rows available for plotting after dropping NaNs.")

    fig = _build_figure(plot_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
