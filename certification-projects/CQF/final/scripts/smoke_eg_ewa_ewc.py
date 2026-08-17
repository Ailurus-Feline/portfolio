"""
Smoke check — Part I slice: data download + Engle–Granger on the main pair.

Run from the project root::

    PYTHONPATH=src python scripts/smoke_eg_ewa_ewc.py

This is *not* the graded report; it only prints full-sample and sub-period EG.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``import ts_pairs`` without an editable install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_pairs.cointegration import engle_granger_from_panel, split_by_periods
from ts_pairs.data import align_pair


def main() -> None:
    panel = align_pair("EWA", "EWC", start="2019-01-01", force=False)
    print(
        f"Panel EWA/EWC: {panel.shape[0]} rows | "
        f"{panel.index.min().date()} → {panel.index.max().date()}"
    )

    full = engle_granger_from_panel(panel)
    print(full.summary())

    # Multi-period EG — matches the brief's "split the dataset" suggestion.
    print("\n--- Sub-period EG ---")
    for label, chunk in split_by_periods(panel, ["2020-03-01", "2022-01-01"]):
        if len(chunk) < 120:
            print(f"{label}: skipped (too short: {len(chunk)} rows)")
            continue
        res = engle_granger_from_panel(
            chunk,
            y_ticker="EWA",
            x_ticker="EWC",
        )
        print(
            f"{label}: beta={res.beta:.4f}, ADF tau={res.adf.tau:.4f}, "
            f"coint@5%={res.is_cointegrated_5pct}, "
            f"ECM λ={res.ecm.get('lambda', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
