"""
End-to-end smoke for all locked pairs: EG → OU → Z* → backtest.

    PYTHONPATH=src python scripts/smoke_all_pairs.py

Writes ``results/smoke_all_pairs_summary.csv`` for the report tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from ts_pairs.backtest import run_pairs_backtest
from ts_pairs.cointegration import engle_granger_from_panel
from ts_pairs.config import PAIRS, PROJECT_ROOT
from ts_pairs.data import align_pair
from ts_pairs.johansen import johansen_from_panel
from ts_pairs.ou_process import fit_ou_ar1
from ts_pairs.rolling import run_rolling_beta_experiment
from ts_pairs.signals import evaluate_z, scan_z_grid
from ts_pairs.walkforward import run_train_test_backtest


def main() -> None:
    rows = []
    for y_ticker, x_ticker, role, story in PAIRS:
        panel = align_pair(y_ticker, x_ticker)
        eg = engle_granger_from_panel(panel)
        joh = johansen_from_panel(panel)
        ou = fit_ou_ar1(eg.residual)
        scan = scan_z_grid(eg.residual, ou)
        sig = evaluate_z(eg.residual, ou, scan.recommended_z)
        bt = run_pairs_backtest(
            panel, sig.positions, beta=eg.beta, z_entry=scan.recommended_z
        )
        tt = run_train_test_backtest(panel, train_frac=0.7)
        roll = run_rolling_beta_experiment(panel, z_star=scan.recommended_z)

        print("=" * 60)
        print(f"{y_ticker}/{x_ticker} [{role}] — {story}")
        print(eg.summary())
        print(joh.summary())
        print(ou.summary())
        print(bt.summary())
        print(tt.summary())
        print(roll.summary())

        rows.append(
            {
                "pair": f"{y_ticker}_{x_ticker}",
                "role": role,
                "beta_eg": eg.beta,
                "adf_tau": eg.adf.tau,
                "eg_coint_5pct": eg.is_cointegrated_5pct,
                "johansen_rank_5pct": joh.rank_trace_5pct,
                "half_life": ou.half_life,
                "z_star": scan.recommended_z,
                "is_return": bt.total_return,
                "is_sharpe": bt.sharpe,
                "oos_return": tt.test.total_return,
                "oos_sharpe": tt.test.sharpe,
                "roll_return": roll.rolling.total_return,
                "roll_sharpe": roll.rolling.sharpe,
                "fixed_return": roll.fixed.total_return,
            }
        )

    out = PROJECT_ROOT / "results" / "smoke_all_pairs_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print("\nWrote", out)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
