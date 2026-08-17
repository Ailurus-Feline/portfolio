"""
Stress-test tables + one comparison figure for the main pair.

    PYTHONPATH=src python scripts/export_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from ts_pairs.compare import adf_gate_backtest, cost_sensitivity, hedge_alternatives, z_grid_is_oos
from ts_pairs.config import PAIRS, PROJECT_ROOT
from ts_pairs.data import align_pair


def main() -> None:
    out = PROJECT_ROOT / "results"
    figdir = PROJECT_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    y, x, _, _ = PAIRS[0]
    panel = align_pair(y, x)

    ztab = z_grid_is_oos(panel)
    ctab = cost_sensitivity(panel)
    htab = hedge_alternatives(panel)
    gate = adf_gate_backtest(panel)

    ztab.to_csv(out / "sensitivity_z_is_oos.csv", index=False)
    ctab.to_csv(out / "sensitivity_cost.csv", index=False)
    htab.to_csv(out / "sensitivity_hedge.csv", index=False)

    gate_row = pd.DataFrame(
        [
            {
                "book": "ungated",
                "total_return": gate["ungated"].total_return,
                "sharpe": gate["ungated"].sharpe,
                "max_drawdown": gate["ungated"].max_drawdown,
                "n_trades": gate["ungated"].n_trades,
                "frac_allowed": 1.0,
            },
            {
                "book": "adf_10pct_gate",
                "total_return": gate["gated"].total_return,
                "sharpe": gate["gated"].sharpe,
                "max_drawdown": gate["gated"].max_drawdown,
                "n_trades": gate["gated"].n_trades,
                "frac_allowed": gate["frac_allowed"],
            },
        ]
    )
    gate_row.to_csv(out / "sensitivity_adf_gate.csv", index=False)

    # Z vs Sharpe, train vs test
    fig, ax = plt.subplots(figsize=(8, 4))
    for seg, color in (("train", "steelblue"), ("test", "firebrick")):
        sl = ztab[ztab["segment"] == seg]
        ax.plot(sl["Z"], sl["sharpe"], marker="o", color=color, label=seg)
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_xlabel("Z entry threshold")
    ax.set_ylabel("Annualised Sharpe")
    ax.set_title("EWA–EWC: Z grid Sharpe, train vs frozen test")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "ewa_ewc_z_sharpe_is_oos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("=== Z grid IS/OOS ===")
    print(ztab.to_string(index=False))
    print("\n=== Cost sensitivity ===")
    print(ctab.to_string(index=False))
    print("\n=== Hedge alternatives ===")
    print(htab.to_string(index=False))
    print("\n=== ADF gate ===")
    print(gate_row.to_string(index=False))
    print("frac_allowed", gate["frac_allowed"])


if __name__ == "__main__":
    main()
