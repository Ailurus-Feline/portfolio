"""Unit tests for matrix OLS and ADF plumbing (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_pairs.cointegration import adf_lag1, add_intercept, engle_granger, ols_matrix


def test_ols_recovers_known_line():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 1.5 + 2.0 * x + rng.normal(scale=0.01, size=500)
    fit = ols_matrix(y, add_intercept(x))
    assert abs(fit.intercept - 1.5) < 0.05
    assert abs(fit.slope - 2.0) < 0.05
    assert fit.r_squared > 0.99


def test_adf_stationary_noise_rejects_unit_root():
    rng = np.random.default_rng(1)
    # White noise is stationary → tau should be very negative.
    noise = pd.Series(rng.normal(size=800))
    adf = adf_lag1(noise, lag=1)
    assert adf.decide("5%")


def test_eg_on_synthetic_cointegrated_pair():
    rng = np.random.default_rng(2)
    n = 1000
    # x ~ I(1); y = 0.5 + 1.2 x + stationary noise  → cointegrated by construction.
    shocks = rng.normal(size=n)
    x = pd.Series(np.cumsum(shocks), name="x")
    e = pd.Series(rng.normal(scale=0.5, size=n), name="e")
    y = 0.5 + 1.2 * x + e
    res = engle_granger(y, x, y_ticker="Y", x_ticker="X")
    assert abs(res.beta - 1.2) < 0.1
    assert res.adf.decide("5%")
    assert res.ecm["lambda"] < 0  # adjustment should pull y back
