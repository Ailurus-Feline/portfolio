"""Unit tests for performance metric helpers."""

from __future__ import annotations

import pandas as pd

from ts_pairs.metrics import drawdown_series, historical_var, rolling_sharpe


def test_drawdown_and_var_smoke():
    equity = pd.Series([1.0, 1.1, 1.05, 1.2])
    dd = drawdown_series(equity)
    assert dd.min() < 0
    pnl = equity.pct_change().dropna()
    assert historical_var(pnl, 0.05) >= 0
    assert rolling_sharpe(pnl, window=2).notna().any()
