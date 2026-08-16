"""Unit tests for OU / AR(1) residual fitting (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_pairs.ou_process import fit_ou_ar1, zscore_series


def _simulate_ou(rng: np.random.Generator, n: int = 2000, theta: float = 0.1, mu: float = 0.0, sigma: float = 1.0):
    """Euler discretisation of dX = θ(μ - X)dt + σ dW with dt=1."""
    x = np.zeros(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.normal()
    return pd.Series(x)


def test_ou_recovers_speed_order_of_magnitude():
    rng = np.random.default_rng(7)
    true_theta = 0.08
    # Longer path so the slow mean reversion has time to identify μ.
    series = _simulate_ou(rng, n=5000, theta=true_theta, mu=0.5, sigma=0.7)
    fit = fit_ou_ar1(series, dt=1.0)
    assert fit.is_mean_reverting
    # Discrete mapping is approximate; allow a loose band around truth.
    assert 0.04 < fit.theta < 0.14
    assert abs(fit.mu - 0.5) < 0.35
    assert fit.half_life > 0


def test_zscore_has_unit_scale_roughly():
    rng = np.random.default_rng(8)
    series = _simulate_ou(rng, theta=0.1, mu=0.0, sigma=1.0)
    fit = fit_ou_ar1(series)
    z = zscore_series(series, fit)
    assert abs(z.mean()) < 0.15
    assert 0.7 < z.std() < 1.3
