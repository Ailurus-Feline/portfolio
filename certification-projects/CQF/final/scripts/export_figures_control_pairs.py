"""
Export summary figures for control pairs XLE–XOP and GLD–GDX.

    PYTHONPATH=src python scripts/export_figures_control_pairs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from ts_pairs.config import PROJECT_ROOT
from ts_pairs.data import align_pair
from ts_pairs.walkforward import run_train_test_backtest


def _savefig(fig: plt.Figure, name: str) -> Path:
    out = PROJECT_ROOT / "figures" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _pair_figure(y: str, x: str) -> Path:
    panel = align_pair(y, x)
    tt = run_train_test_backtest(panel, train_frac=0.7)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    axes[0].plot(panel.index, panel["y"], label=y)
    axes[0].plot(panel.index, panel["x"], label=x)
    axes[0].set_title(f"{y} vs {x} adjusted closes")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tt.train.equity.index, tt.train.equity, label="Train")
    axes[1].plot(tt.test.equity.index, tt.test.equity, label="Test")
    axes[1].axvline(tt.split_date, color="black", ls=":", lw=0.9)
    axes[1].set_title(
        f"{y}/{x} train/test equity (OOS ret={tt.test.total_return:.1%}, "
        f"Sharpe={tt.test.sharpe:.2f})"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    return _savefig(fig, f"{y.lower()}_{x.lower()}_overview.png")


def main() -> None:
    paths = [
        _pair_figure("XLE", "XOP"),
        _pair_figure("GLD", "GDX"),
    ]
    print("Wrote:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
