"""Visualization for the v1 sentiment pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REGIME_COLORS = {
    "neutral": "#1f77b4",
    "overheated": "#2ca02c",
    "overcooled": "#d62728",
}


def _plot_colored_index_line(ax, dates, log_index: pd.Series, regimes: pd.Series) -> None:
    """Draw the log-index line in regime-colored segments."""
    x = pd.to_datetime(dates)
    y = log_index.to_numpy()
    regime_values = regimes.to_numpy()

    for i in range(len(x) - 1):
        color = REGIME_COLORS.get(regime_values[i], REGIME_COLORS["neutral"])
        ax.plot(x[i : i + 2], y[i : i + 2], color=color, linewidth=1.6)


def plot_sentiment_vs_index(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the dual-axis chart used as the main v1 deliverable."""
    plot_df = data.dropna(subset=["index_log", "sentiment_z"]).copy()
    if plot_df.empty:
        raise ValueError("No rows available for plotting after dropping NaNs.")

    fig, ax_left = plt.subplots(figsize=(14, 6))

    _plot_colored_index_line(
        ax_left,
        plot_df["date"],
        plot_df["index_log"],
        plot_df["regime"],
    )
    ax_left.set_ylabel("ln(Shenwan Electronics Index)")
    ax_left.grid(True, alpha=0.25)

    ax_right = ax_left.twinx()
    ax_right.plot(
        plot_df["date"],
        plot_df["sentiment_z"],
        color="black",
        linewidth=1.0,
        label="Sentiment z-score",
    )
    ax_right.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_right.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax_right.axhline(-1.0, color="gray", linewidth=0.8, linestyle=":")
    ax_right.set_ylabel("Sentiment z-score (std)")

    ax_left.set_title("Electronics Sector Sentiment vs Shenwan Electronics Index")
    ax_left.set_xlabel("Date")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
