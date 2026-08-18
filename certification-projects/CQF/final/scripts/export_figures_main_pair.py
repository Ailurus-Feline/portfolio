"""
Export core figures for the main pair (EWA–EWC) into ``figures/``.

    PYTHONPATH=src python scripts/export_figures_main_pair.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from ts_pairs.cointegration import engle_granger_from_panel
from ts_pairs.config import PROJECT_ROOT
from ts_pairs.data import align_pair
from ts_pairs.metrics import drawdown_series, rolling_sharpe
from ts_pairs.ou_process import fit_ou_ar1, zscore_series
from ts_pairs.rolling import run_rolling_beta_experiment
from ts_pairs.signals import scan_z_grid
from ts_pairs.walkforward import run_train_test_backtest


def _savefig(fig: plt.Figure, name: str) -> Path:
    out = PROJECT_ROOT / "figures" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    panel = align_pair("EWA", "EWC")
    eg = engle_granger_from_panel(panel)
    ou = fit_ou_ar1(eg.residual)
    z = zscore_series(eg.residual, ou)
    scan = scan_z_grid(eg.residual, ou)
    tt = run_train_test_backtest(panel, train_frac=0.7)
    roll = run_rolling_beta_experiment(panel, z_star=scan.recommended_z)

    paths: list[Path] = []

    # 1) Prices
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(panel.index, panel["y"], label="EWA")
    ax.plot(panel.index, panel["x"], label="EWC")
    ax.set_title("EWA vs EWC adjusted closes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    paths.append(_savefig(fig, "ewa_ewc_prices.png"))

    # 2) Residual + bands
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eg.residual.index, eg.residual, color="steelblue", lw=0.8, label="e_t")
    upper = ou.mu + scan.recommended_z * ou.sigma_eq
    lower = ou.mu - scan.recommended_z * ou.sigma_eq
    ax.axhline(ou.mu, color="black", ls="--", lw=0.8, label="μ")
    ax.axhline(upper, color="firebrick", ls="--", lw=0.8, label=f"±Z*σ (Z*={scan.recommended_z:g})")
    ax.axhline(lower, color="firebrick", ls="--", lw=0.8)
    ax.set_title("EG residual with OU entry bands (full sample)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    paths.append(_savefig(fig, "ewa_ewc_residual_bands.png"))

    # 3) Z* trade-count table as bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    tab = scan.to_frame()
    ax.bar(tab.index.astype(str), tab["n_trades"], color="slategray")
    ax.set_xlabel("Z")
    ax.set_ylabel("Number of round-trip entries")
    ax.set_title("Z* grid: trade count vs threshold")
    ax.grid(True, axis="y", alpha=0.3)
    paths.append(_savefig(fig, "ewa_ewc_zstar_trades.png"))

    # 4) Train vs test equity
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tt.train.equity.index, tt.train.equity, label="Train equity")
    ax.plot(tt.test.equity.index, tt.test.equity, label="Test equity")
    ax.axvline(tt.split_date, color="black", ls=":", lw=0.9, label="Split")
    ax.set_title("Train/test equity (frozen α, β, Z*)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    paths.append(_savefig(fig, "ewa_ewc_train_test_equity.png"))

    # 5) Fixed vs rolling equity + drawdown
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(roll.fixed.equity.index, roll.fixed.equity, label="Fixed β")
    axes[0].plot(roll.rolling.equity.index, roll.rolling.equity, label="Rolling β")
    axes[0].set_title("Fixed vs rolling β equity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    dd_f = drawdown_series(roll.fixed.equity)
    dd_r = drawdown_series(roll.rolling.equity)
    axes[1].plot(dd_f.index, dd_f, label="Fixed β DD")
    axes[1].plot(dd_r.index, dd_r, label="Rolling β DD")
    axes[1].set_title("Drawdowns")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    paths.append(_savefig(fig, "ewa_ewc_fixed_vs_rolling.png"))

    # 6) Rolling Sharpe (test pnl if long enough else full IS pnl)
    fig, ax = plt.subplots(figsize=(9, 4))
    rs = rolling_sharpe(tt.test.pnl.fillna(0.0), window=63)
    ax.plot(rs.index, rs, color="darkgreen")
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_title("Rolling 63d Sharpe on TEST P&L")
    ax.grid(True, alpha=0.3)
    paths.append(_savefig(fig, "ewa_ewc_test_rolling_sharpe.png"))

    # 7) Beta path
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(roll.beta_path.index, roll.beta_path, color="purple", lw=0.9)
    ax.set_title("Rolling EG β path (8m window)")
    ax.grid(True, alpha=0.3)
    paths.append(_savefig(fig, "ewa_ewc_beta_path.png"))

    # Save z-scan table for the report
    tab_path = PROJECT_ROOT / "results" / "ewa_ewc_zstar_table.csv"
    tab_path.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(tab_path)

    print("Wrote figures:")
    for p in paths:
        print(" ", p)
    print("Wrote", tab_path)


if __name__ == "__main__":
    main()
