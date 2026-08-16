"""Smoke tests for Johansen wrapper (synthetic cointegrated pair, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_pairs.johansen import johansen_pair


def test_johansen_detects_synthetic_coint():
    rng = np.random.default_rng(11)
    n = 1500
    # Positive prices: geometric random walk for x; y cointegrated in logs.
    x = 50 * np.exp(np.cumsum(rng.normal(scale=0.01, size=n)))
    # log y = 0.2 + 0.8 log x + stationary noise
    log_y = 0.2 + 0.8 * np.log(x) + rng.normal(scale=0.02, size=n)
    y = np.exp(log_y)
    res = johansen_pair(
        pd.Series(y),
        pd.Series(x),
        y_ticker="Y",
        x_ticker="X",
        k_ar_diff=1,
        use_log=True,
    )
    assert res.rank_trace_5pct >= 1
    # Normalised β ≈ [1, -0.8] for log y - 0.8 log x
    b = res.primary_beta
    assert abs(b[1] - (-0.8)) < 0.25
