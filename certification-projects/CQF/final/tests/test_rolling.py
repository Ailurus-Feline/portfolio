"""Tests for rolling β re-estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_pairs.rolling import estimate_rolling_betas, run_rolling_beta_experiment


def _synthetic_panel(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    x = 40 + np.cumsum(rng.normal(scale=0.4, size=n))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.85 * e[t - 1] + rng.normal(scale=0.25)
    # Mild break in β halfway — rolling should move.
    beta = np.where(np.arange(n) < n // 2, 0.5, 0.8)
    y = 1.0 + beta * x + e
    panel = pd.DataFrame({"y": y, "x": x}, index=idx)
    panel.attrs["y_ticker"] = "Y"
    panel.attrs["x_ticker"] = "X"
    return panel


def test_estimate_rolling_betas_length():
    panel = _synthetic_panel()
    stamped = estimate_rolling_betas(panel, window=120, step=20)
    assert len(stamped) >= 5
    assert {"alpha", "beta", "half_life"}.issubset(stamped.columns)


def test_rolling_experiment_runs():
    out = run_rolling_beta_experiment(
        _synthetic_panel(), window=120, step=20, cost_bps=0.0, z_star=1.5
    )
    assert out.fixed.n_trades >= 0
    assert out.rolling.n_trades >= 0
    assert out.beta_path.notna().sum() > 50
