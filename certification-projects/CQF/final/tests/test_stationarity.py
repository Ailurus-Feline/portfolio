"""Tests for the I(1) screen and VAR lag helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_pairs.cointegration import adf_observed, integration_order_check
from ts_pairs.johansen import var_lag_stability


def test_adf_observed_rejects_white_noise():
    rng = np.random.default_rng(0)
    adf = adf_observed(pd.Series(rng.normal(size=400)), lag=1)
    assert adf.decide("5%")


def test_integration_order_random_walk_looks_I1():
    rng = np.random.default_rng(1)
    rw = pd.Series(np.cumsum(rng.normal(size=800)))
    chk = integration_order_check(rw, name="rw")
    assert chk["looks_I1"] is True


def test_var_lag_stability_runs():
    rng = np.random.default_rng(2)
    n = 400
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    x = 50 * np.exp(np.cumsum(rng.normal(scale=0.01, size=n)))
    y = 1.2 * x * np.exp(rng.normal(scale=0.01, size=n))
    panel = pd.DataFrame({"y": y, "x": x}, index=idx)
    out = var_lag_stability(panel, maxlags=4)
    assert out["aic_lag"] >= 1
    assert "stable" in out
