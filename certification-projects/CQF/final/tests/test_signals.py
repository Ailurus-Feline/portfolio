"""Unit tests for Z-score signal generation and Z* grid scan."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_pairs.ou_process import OUFit
from ts_pairs.signals import generate_positions, scan_z_grid


def _toy_ou() -> OUFit:
    return OUFit(
        mu=0.0,
        theta=0.1,
        half_life=np.log(2) / 0.1,
        sigma_eps=1.0,
        sigma_eq=1.0,
        ar_intercept=0.0,
        ar_slope=0.9,
        nobs=100,
        r_squared=0.8,
        dt=1.0,
    )


def test_generate_positions_enters_and_exits():
    # Residual path in σ units: 0 → 2.5 (short) → 0 (exit) → -2.5 (long) → 0
    z = pd.Series([0.0, 2.5, 2.0, 0.0, -2.5, -1.0, 0.0])
    pos = generate_positions(z, z_entry=2.0, z_exit=0.0)
    assert list(pos.astype(int)) == [0, -1, -1, 0, 1, 1, 0]


def test_scan_z_grid_monotonic_trade_count():
    rng = np.random.default_rng(0)
    # Mildly mean-reverting synthetic residual around 0.
    e = pd.Series(np.cumsum(rng.normal(scale=0.3, size=800)) * 0.0 + rng.normal(size=800))
    # Stronger: AR(1) residual
    vals = [0.0]
    for _ in range(799):
        vals.append(0.9 * vals[-1] + rng.normal())
    e = pd.Series(vals)

    scan = scan_z_grid(e, _toy_ou(), grid=[1.0, 1.5, 2.0, 2.5])
    trades = [r.n_trades for r in scan.results]
    # Wider bands should not *increase* trade count.
    assert trades == sorted(trades, reverse=True)
    assert scan.recommended_z in scan.grid
    assert not scan.to_frame().empty
