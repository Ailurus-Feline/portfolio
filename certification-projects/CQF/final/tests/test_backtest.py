"""Unit tests for the dollar-neutral pairs backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_pairs.backtest import run_pairs_backtest


def test_flat_positions_earn_nearly_zero_after_costs():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    panel = pd.DataFrame(
        {
            "y": np.linspace(10, 12, len(idx)),
            "x": np.linspace(20, 24, len(idx)),
        },
        index=idx,
    )
    pos = pd.Series(0.0, index=idx)
    bt = run_pairs_backtest(panel, pos, beta=0.5, z_entry=1.0, cost_bps=0.0)
    assert abs(bt.total_return) < 1e-12
    assert bt.n_trades == 0


def test_constant_long_tracks_spread_change():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    # y increases by 1 each day; x flat; beta=0 → spread_chg = dy
    panel = pd.DataFrame(
        {"y": [10.0, 11.0, 12.0, 13.0, 14.0], "x": [5.0, 5.0, 5.0, 5.0, 5.0]},
        index=idx,
    )
    pos = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx)
    bt = run_pairs_backtest(panel, pos, beta=0.0, z_entry=1.0, cost_bps=0.0)
    # gross_t = pos_{t-1} * dy_t / y_{t-1}; first bar has pos_lag=0 → 0
    # second bar: 1 * 1 / 10 = 0.1
    assert abs(bt.pnl.iloc[1] - 0.1) < 1e-12
    assert bt.n_trades == 1
