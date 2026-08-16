"""Ornstein–Uhlenbeck fitting on a cointegrating residual.

CQF brief — Engle–Granger \"Step 3\"
------------------------------------
After the residual ``e_t`` is obtained, we evaluate mean reversion so that
trade design can use bands ``μ_e ± Z σ_eq`` and exit when ``e_t`` returns
toward ``μ_e``.

Discrete AR(1) bridge (daily bars, Δt = 1 trading day)::

    e_t = a + b e_{t-1} + ε_t

If ``0 < b < 1``, map to OU parameters via::

    θ = -ln(b) / Δt
    μ = a / (1 - b)
    half-life = ln(2) / θ

Equilibrium residual volatility uses the stationary AR(1) formula::

    σ_eq = σ_ε / sqrt(1 - b²)

which is the long-run std of ``e`` under the fitted dynamics (and the
natural scale for Z-score entry thresholds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cointegration import add_intercept, ols_matrix


@dataclass(frozen=True)
class OUFit:
    """OU / AR(1) summary for one residual series."""

    mu: float  # long-run mean of the residual
    theta: float  # mean-reversion speed (1 / time)
    half_life: float  # trading days to close half the gap to μ
    sigma_eps: float  # innovation std of the AR(1) residual
    sigma_eq: float  # stationary std of e (entry-band scale)
    ar_intercept: float  # a in e_t = a + b e_{t-1} + ε
    ar_slope: float  # b
    nobs: int
    r_squared: float
    dt: float  # time step used in the θ mapping (1.0 = one trading day)

    @property
    def is_mean_reverting(self) -> bool:
        """True when the AR root is inside (0, 1) — classical OU regime."""
        return 0.0 < self.ar_slope < 1.0 and self.theta > 0.0

    def summary(self) -> str:
        return "\n".join(
            [
                "OU / AR(1) residual fit",
                f"  μ (equilibrium)     = {self.mu:.6f}",
                f"  θ (speed)           = {self.theta:.6f}",
                f"  half-life (days)    = {self.half_life:.2f}",
                f"  σ_eq (band scale)   = {self.sigma_eq:.6f}",
                f"  AR slope b          = {self.ar_slope:.6f}",
                f"  mean-reverting?     = {self.is_mean_reverting}",
                f"  nobs / R²           = {self.nobs} / {self.r_squared:.4f}",
            ]
        )


def fit_ou_ar1(
    residual: pd.Series | np.ndarray,
    *,
    dt: float = 1.0,
) -> OUFit:
    """Fit a discrete AR(1) and map it to OU parameters.

    Parameters
    ----------
    residual:
        Engle–Granger residual ``e_t`` (levels cointegrating error).
    dt:
        Time step in the same units you want for θ and half-life.
        Daily bars → ``dt=1.0`` yields half-life in *trading days*.

    Notes
    -----
    If ``b <= 0`` or ``b >= 1``, θ / half-life are set to ``nan``: the
    residual is not a classical stationary OU, and Z-band trading should
    be treated as exploratory at best.
    """
    e = pd.Series(residual, dtype=float).dropna()
    if e.shape[0] < 30:
        raise ValueError("Need at least ~30 residual observations for an AR(1) fit.")

    e_lag = e.shift(1)
    panel = pd.concat({"e": e, "e_lag": e_lag}, axis=1).dropna()
    y = panel["e"].to_numpy()
    x = panel["e_lag"].to_numpy()

    # Same matrix OLS used in the EG block — keeps the numerical story consistent.
    fit = ols_matrix(y, add_intercept(x))
    a = float(fit.intercept)
    b = float(fit.slope)
    sigma_eps = float(np.std(fit.resid, ddof=fit.k))

    if 0.0 < b < 1.0:
        theta = float(-np.log(b) / dt)
        half_life = float(np.log(2.0) / theta) if theta > 0 else np.nan
        mu = float(a / (1.0 - b))
        sigma_eq = float(sigma_eps / np.sqrt(1.0 - b**2))
    else:
        # Explosive / non-positive root: report AR coefficients, blank OU map.
        theta = np.nan
        half_life = np.nan
        mu = float(e.mean())
        sigma_eq = float(e.std(ddof=1))

    return OUFit(
        mu=mu,
        theta=theta,
        half_life=half_life,
        sigma_eps=sigma_eps,
        sigma_eq=sigma_eq,
        ar_intercept=a,
        ar_slope=b,
        nobs=fit.nobs,
        r_squared=fit.r_squared,
        dt=dt,
    )


def zscore_series(
    residual: pd.Series,
    ou: OUFit,
) -> pd.Series:
    """Standardise the residual with the fitted OU equilibrium moments.

    ``z_t = (e_t - μ) / σ_eq`` — entry when |z| exceeds Z*, exit near 0.
    """
    e = residual.astype(float)
    if ou.sigma_eq <= 0 or not np.isfinite(ou.sigma_eq):
        raise ValueError("σ_eq must be positive and finite to form Z-scores.")
    z = (e - ou.mu) / ou.sigma_eq
    return z.rename("zscore")
