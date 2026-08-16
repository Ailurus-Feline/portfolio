"""Project constants: tickers, windows, costs, Z grid.

Locked defaults for the finished TS pipeline (EG → OU → backtest → OOS).
Costs are stylised 5 bps/leg, not broker-calibrated.
"""

from __future__ import annotations

from pathlib import Path

# Repository root: .../CQF/final  (parents: ts_pairs → src → final)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (y_ticker, x_ticker, role, economic story)
# Convention: first leg is the EG dependent variable y.
PAIRS: list[tuple[str, str, str, str]] = [
    ("EWA", "EWC", "main", "AU vs CA commodity-country ETFs (PDF example)"),
    ("XLE", "XOP", "control", "Energy sector vs oil & gas exploration ETF"),
    ("GLD", "GDX", "control", "Gold spot proxy vs gold miners"),
]

DEFAULT_START = "2019-01-01"
DEFAULT_END: str | None = None  # through latest available bar
MIN_BACKTEST_YEARS = 2
ROLLING_WINDOW_DAYS = 168  # ~8 months of trading days
ROLLING_STEP_DAYS = 12  # ~10–15 calendar days between re-estimates
Z_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
TRANSACTION_COST_BPS = 5.0  # per leg, charged on both legs of a position change
ADF_LAG = 1  # brief: EG residual ADF uses lag = 1
