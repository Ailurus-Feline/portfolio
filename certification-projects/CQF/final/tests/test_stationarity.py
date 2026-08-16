"""Tests for the I(1) screen (ADF / KPSS on levels vs differences)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_pairs.cointegration import adf_observed, integration_order_check


def test_adf_observed_rejects_white_noise():
    rng = np.random.default_rng(0)
    adf = adf_observed(pd.Series(rng.normal(size=400)), lag=1)
    assert adf.decide("5%")


def test_integration_order_random_walk_looks_I1():
    rng = np.random.default_rng(1)
    rw = pd.Series(np.cumsum(rng.normal(size=800)))
    chk = integration_order_check(rw, name="rw")
    assert chk["looks_I1"] is True
