"""Performance helpers: drawdown path, rolling Sharpe, VaR sketch.

Kept separate from ``backtest.py`` so the report can import plotting-friendly
series without re-running the book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Equity drawdown path: ``equity / cummax(equity) - 1``."""
    eq = equity.astype(float)
    return (eq / eq.cummax() - 1.0).rename("drawdown")


def rolling_sharpe(pnl: pd.Series, window: int = 63) -> pd.Series:
    """Annualised rolling Sharpe on daily P&L (√252 · mean / std)."""
    mu = pnl.rolling(window).mean()
    sd = pnl.rolling(window).std(ddof=1)
    out = np.sqrt(252.0) * mu / sd.replace(0.0, np.nan)
    return out.rename(f"rolling_sharpe_{window}d")


def historical_var(pnl: pd.Series, alpha: float = 0.05) -> float:
    """One-day historical VaR as a *positive* loss number (quantile of -pnl)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    return float(-np.quantile(pnl.dropna().to_numpy(), alpha))
