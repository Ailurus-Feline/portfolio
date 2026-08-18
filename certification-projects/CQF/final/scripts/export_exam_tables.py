"""
Export the extra tables the CQF brief asks to *discuss* (I(1) screen,
sub-period EG, VAR lag/stability, historical VaR).

    PYTHONPATH=src python scripts/export_exam_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from ts_pairs.backtest import run_pairs_backtest
from ts_pairs.cointegration import (
    engle_granger_from_panel,
    integration_order_check,
    split_by_periods,
)
from ts_pairs.config import PAIRS, PROJECT_ROOT
from ts_pairs.data import align_pair
from ts_pairs.johansen import var_lag_stability
from ts_pairs.metrics import historical_var
from ts_pairs.ou_process import fit_ou_ar1
from ts_pairs.signals import evaluate_z, scan_z_grid
from ts_pairs.walkforward import run_train_test_backtest


def main() -> None:
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    i1_rows = []
    sub_rows = []
    var_rows = []
    var_risk_rows = []

    for y_ticker, x_ticker, role, _story in PAIRS:
        panel = align_pair(y_ticker, x_ticker)
        for col, name in (("y", y_ticker), ("x", x_ticker)):
            chk = integration_order_check(panel[col], name=name)
            chk["pair"] = f"{y_ticker}_{x_ticker}"
            chk["role"] = role
            i1_rows.append(chk)

        for label, chunk in split_by_periods(panel, ["2020-03-01", "2022-01-01"]):
            if len(chunk) < 120:
                continue
            eg = engle_granger_from_panel(chunk, y_ticker=y_ticker, x_ticker=x_ticker)
            sub_rows.append(
                {
                    "pair": f"{y_ticker}_{x_ticker}",
                    "period": label,
                    "nobs": len(chunk),
                    "alpha": eg.alpha,
                    "beta": eg.beta,
                    "adf_tau": eg.adf.tau,
                    "coint_5pct": eg.is_cointegrated_5pct,
                    "coint_10pct": eg.adf.decide("10%"),
                    "ecm_lambda": eg.ecm.get("lambda"),
                    "ecm_t": eg.ecm.get("lambda_tstat"),
                }
            )

        vd = var_lag_stability(panel)
        vd["pair"] = f"{y_ticker}_{x_ticker}"
        var_rows.append(vd)

        eg = engle_granger_from_panel(panel)
        ou = fit_ou_ar1(eg.residual)
        scan = scan_z_grid(eg.residual, ou)
        sig = evaluate_z(eg.residual, ou, scan.recommended_z)
        bt = run_pairs_backtest(panel, sig.positions, beta=eg.beta, z_entry=scan.recommended_z)
        tt = run_train_test_backtest(panel, train_frac=0.7)
        var_risk_rows.append(
            {
                "pair": f"{y_ticker}_{x_ticker}",
                "is_var95": historical_var(bt.pnl.fillna(0.0), 0.05),
                "is_vol_ann": float(bt.pnl.fillna(0.0).std() * (252**0.5)),
                "oos_var95": historical_var(tt.test.pnl.fillna(0.0), 0.05),
                "oos_vol_ann": float(tt.test.pnl.fillna(0.0).std() * (252**0.5)),
            }
        )

    i1 = pd.DataFrame(i1_rows)
    sub = pd.DataFrame(sub_rows)
    var_lag = pd.DataFrame(var_rows)
    risk = pd.DataFrame(var_risk_rows)

    i1.to_csv(out_dir / "integration_order.csv", index=False)
    sub.to_csv(out_dir / "subperiod_eg.csv", index=False)
    var_lag.to_csv(out_dir / "var_lag_stability.csv", index=False)
    risk.to_csv(out_dir / "var_risk.csv", index=False)

    print("=== I(1) screen ===")
    print(i1.to_string(index=False))
    print("\n=== Sub-period EG ===")
    print(sub.to_string(index=False))
    print("\n=== VAR on log-returns ===")
    print(var_lag.to_string(index=False))
    print("\n=== Historical VaR / vol ===")
    print(risk.to_string(index=False))


if __name__ == "__main__":
    main()
