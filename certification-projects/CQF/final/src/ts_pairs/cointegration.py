"""Stationarity tests and Engle–Granger procedure (matrix OLS + ADF).

CQF brief requirements touched here
-----------------------------------
* Recode regression estimation in *matrix form* (not ``statsmodels.OLS``
  as a black box for the cointegrating step).
* Engle–Granger Step 1: Augmented Dickey–Fuller on the residual with
  ``lag=1`` (project brief wording).
* Step 2: inspect the error-correction term in the ECM.

Critical values
---------------
Testing a *regression residual* for a unit root is not the same as a
vanilla ADF on an observed series: MacKinnon's cointegration response
surfaces apply. We compute the ADF τ-statistic ourselves and compare it
against asymptotic MacKinnon EG critical values (constant, no trend).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import ADF_LAG

# ---------------------------------------------------------------------------
# MacKinnon (approx. asymptotic) critical values for the Engle–Granger
# residual ADF, cointegrating regression with a constant, no deterministic
# trend, one explanatory I(1) regressor. Reject H0 (no coint) if tau < cv.
# Source: standard tables used in textbooks / MacKinnon response surfaces.
# ---------------------------------------------------------------------------
EG_CRIT_ASYMPTOTIC: dict[str, float] = {
    "1%": -3.90,
    "5%": -3.34,
    "10%": -3.04,
}

# Standard ADF critical values for an *observed* series (constant, no trend).
# These are milder than EG residual critical values — do not mix the two tables.
ADF_CRIT_OBSERVED: dict[str, float] = {
    "1%": -3.43,
    "5%": -2.86,
    "10%": -2.57,
}


# ---------------------------------------------------------------------------
# Matrix OLS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OLSResult:
    """Minimal OLS payload — enough for EG / ECM without a statsmodels fit object."""

    beta: np.ndarray  # shape (k,) including intercept if present
    fitted: np.ndarray
    resid: np.ndarray
    xtx_inv: np.ndarray
    nobs: int
    k: int
    r_squared: float
    se: np.ndarray  # coefficient standard errors (homoskedastic)

    @property
    def intercept(self) -> float:
        return float(self.beta[0])

    @property
    def slope(self) -> float:
        """First slope coefficient (β on x in the bivariate EG regression)."""
        if self.k < 2:
            raise ValueError("No slope coefficient in this specification.")
        return float(self.beta[1])


def ols_matrix(y: np.ndarray, X: np.ndarray) -> OLSResult:
    """Ordinary least squares via the normal equations.

    Solves ``β̂ = (X'X)^{-1} X'y`` with a symmetric positive-definite
    inverse (``numpy.linalg.inv``). For the project scale (T ~ a few
    thousand, k tiny) this is numerically fine; we deliberately avoid
    QR here so the report can quote the classical matrix formula.

    Parameters
    ----------
    y:
        Dependent variable, shape ``(n,)``.
    X:
        Design matrix, shape ``(n, k)`` — callers add the intercept column.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and X must share the same number of rows.")
    if X.ndim != 2:
        raise ValueError("X must be 2-D.")

    n, k = X.shape
    xtx = X.T @ X
    xty = X.T @ y
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ xty
    fitted = X @ beta
    resid = y - fitted

    # Homoskedastic variance estimator: σ² = e'e / (n - k)
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))

    ss_res = float(resid @ resid)
    ss_tot = float((y - y.mean()) @ (y - y.mean()))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return OLSResult(
        beta=beta,
        fitted=fitted,
        resid=resid,
        xtx_inv=xtx_inv,
        nobs=n,
        k=k,
        r_squared=r_squared,
        se=se,
    )


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Stack a column of ones in front of a 1-D or 2-D regressor array."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    ones = np.ones((x.shape[0], 1), dtype=float)
    return np.hstack([ones, x])


# ---------------------------------------------------------------------------
# Augmented Dickey–Fuller (lag = 1 by default, per brief)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ADFResult:
    """ADF regression summary on a single series (typically an EG residual)."""

    tau: float  # t-stat on the lagged level coefficient
    phi: float  # coefficient on y_{t-1}
    nobs: int
    lag: int
    critical_values: dict[str, float] = field(default_factory=lambda: dict(EG_CRIT_ASYMPTOTIC))
    regressor_names: tuple[str, ...] = ()

    def decide(self, level: str = "5%") -> bool:
        """Return True if we reject a unit root at ``level`` (i.e. residual looks stationary)."""
        if level not in self.critical_values:
            raise KeyError(f"Unknown level {level!r}; choose from {list(self.critical_values)}")
        return self.tau < self.critical_values[level]


def adf_lag1(
    series: np.ndarray | pd.Series,
    *,
    lag: int = ADF_LAG,
    critical_values: dict[str, float] | None = None,
) -> ADFResult:
    """Augmented Dickey–Fuller regression with a fixed lag order.

    Specification (constant, no trend)::

        Δy_t = a + φ y_{t-1} + Σ_{i=1}^{lag} γ_i Δy_{t-i} + ε_t

    The brief asks for ``lag=1`` on the EG residual; we keep ``lag`` as an
    argument so the report can show a small robustness check later.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")

    y = np.asarray(pd.Series(series).astype(float).dropna(), dtype=float)
    dy = np.diff(y)  # length n-1; dy[t] = y[t+1] - y[t] with 0-based t on y[:-1]

    # Align so that observation t uses y_t (level) and lagged differences.
    # After dropping ``lag`` bootstrap rows from the differenced sample:
    #   left-hand side: dy[lag:]
    #   level:          y[lag:-1]   (y_t sitting behind Δy_{t+1} … indexing care)
    #
    # Cleaner indexing with explicit time on the original series:
    # For t = lag+1, ..., n-1 (0-based on y):
    #   Δy_t = y[t] - y[t-1]
    #   regressors: 1, y[t-1], Δy_{t-1}, ..., Δy_{t-lag}
    n = y.shape[0]
    rows = []
    lhs = []
    for t in range(lag + 1, n):
        lhs.append(y[t] - y[t - 1])
        row = [1.0, y[t - 1]]
        for i in range(1, lag + 1):
            row.append(y[t - i] - y[t - i - 1])
        rows.append(row)

    if len(lhs) < 10:
        raise ValueError("Series too short for ADF regression.")

    Y = np.asarray(lhs, dtype=float)
    X = np.asarray(rows, dtype=float)
    fit = ols_matrix(Y, X)

    phi = float(fit.beta[1])
    tau = phi / float(fit.se[1]) if fit.se[1] > 0 else np.nan
    names = ("const", "y_lag") + tuple(f"dy_lag{i}" for i in range(1, lag + 1))

    return ADFResult(
        tau=tau,
        phi=phi,
        nobs=fit.nobs,
        lag=lag,
        critical_values=dict(critical_values or EG_CRIT_ASYMPTOTIC),
        regressor_names=names,
    )


# ---------------------------------------------------------------------------
# Engle–Granger procedure
# ---------------------------------------------------------------------------

@dataclass
class EngleGrangerResult:
    """Bundle for one EG run on a price pair (levels)."""

    y_ticker: str
    x_ticker: str
    alpha: float
    beta: float
    residual: pd.Series
    coint_ols: OLSResult
    adf: ADFResult
    ecm: dict[str, Any] = field(default_factory=dict)

    @property
    def is_cointegrated_5pct(self) -> bool:
        return self.adf.decide("5%")

    def summary(self) -> str:
        """Human-readable block for notebooks / logs."""
        adf = self.adf
        lines = [
            f"Engle–Granger: {self.y_ticker} ~ {self.x_ticker}",
            f"  cointegrating relation: y = {self.alpha:.4f} + {self.beta:.4f} x + e",
            f"  R^2 (levels)           = {self.coint_ols.r_squared:.4f}",
            f"  ADF(lag={adf.lag}) tau = {adf.tau:.4f}  "
            f"[crit 1%={adf.critical_values['1%']}, "
            f"5%={adf.critical_values['5%']}, "
            f"10%={adf.critical_values['10%']}]",
            f"  reject unit root @5%?  = {adf.decide('5%')}",
        ]
        if self.ecm:
            lines.append(
                f"  ECM: Δy = ... + λ e_{{t-1}} + ...;  "
                f"λ={self.ecm.get('lambda'):.4f}, t(λ)={self.ecm.get('lambda_tstat'):.4f}"
            )
        return "\n".join(lines)


def engle_granger(
    y: pd.Series,
    x: pd.Series,
    *,
    y_ticker: str = "y",
    x_ticker: str = "x",
    adf_lag: int = ADF_LAG,
    run_ecm: bool = True,
) -> EngleGrangerResult:
    """Two-step Engle–Granger (+ optional ECM 'Step 2' diagnostics).

    Step 1 — cointegrating regression on levels::

        y_t = α + β x_t + e_t

    then ADF(lag) on ``e_t``.

    Step 2 — error-correction sketch (bivariate)::

        Δy_t = c + λ e_{t-1} + γ Δx_t + u_t

    A significantly negative ``λ`` supports mean-reverting adjustment of
    ``y`` toward the long-run relation (the brief asks us to discuss the
    EC term across sub-periods later).
    """
    panel = pd.concat({"y": y.astype(float), "x": x.astype(float)}, axis=1).dropna()
    y_arr = panel["y"].to_numpy()
    x_arr = panel["x"].to_numpy()

    coint = ols_matrix(y_arr, add_intercept(x_arr))
    resid = pd.Series(coint.resid, index=panel.index, name="eg_residual")

    adf = adf_lag1(resid, lag=adf_lag)

    ecm_info: dict[str, Any] = {}
    if run_ecm:
        # Align Δy_t, Δx_t, e_{t-1}
        dy = panel["y"].diff()
        dx = panel["x"].diff()
        e_lag = resid.shift(1)
        ecm_panel = pd.concat({"dy": dy, "dx": dx, "e_lag": e_lag}, axis=1).dropna()
        Y = ecm_panel["dy"].to_numpy()
        X = add_intercept(ecm_panel[["e_lag", "dx"]].to_numpy())
        # Column order after intercept: e_lag, dx
        ecm_fit = ols_matrix(Y, X)
        lam = float(ecm_fit.beta[1])
        lam_t = lam / float(ecm_fit.se[1]) if ecm_fit.se[1] > 0 else np.nan
        ecm_info = {
            "lambda": lam,
            "lambda_tstat": lam_t,
            "gamma_dx": float(ecm_fit.beta[2]),
            "r_squared": ecm_fit.r_squared,
            "nobs": ecm_fit.nobs,
        }

    return EngleGrangerResult(
        y_ticker=y_ticker,
        x_ticker=x_ticker,
        alpha=coint.intercept,
        beta=coint.slope,
        residual=resid,
        coint_ols=coint,
        adf=adf,
        ecm=ecm_info,
    )


def engle_granger_from_panel(
    panel: pd.DataFrame,
    *,
    y_ticker: str | None = None,
    x_ticker: str | None = None,
    adf_lag: int = ADF_LAG,
    run_ecm: bool = True,
) -> EngleGrangerResult:
    """Run EG on an ``align_pair`` panel (columns ``y`` / ``x``)."""
    y_ticker = y_ticker or str(panel.attrs.get("y_ticker", "y"))
    x_ticker = x_ticker or str(panel.attrs.get("x_ticker", "x"))
    return engle_granger(
        panel["y"],
        panel["x"],
        y_ticker=y_ticker,
        x_ticker=x_ticker,
        adf_lag=adf_lag,
        run_ecm=run_ecm,
    )


def split_by_periods(
    panel: pd.DataFrame,
    breakpoints: list[str] | list[pd.Timestamp],
) -> list[tuple[str, pd.DataFrame]]:
    """Split a panel into contiguous dated slices for multi-period EG.

    Example::

        split_by_periods(panel, [\"2020-03-01\", \"2022-01-01\"])

    yields labelled slices ``[start, bp1)``, ``[bp1, bp2)``, ``[bp2, end]``.
    """
    bps = sorted(pd.to_datetime(breakpoints))
    edges = [panel.index.min(), *bps, panel.index.max() + pd.Timedelta(days=1)]
    slices: list[tuple[str, pd.DataFrame]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        chunk = panel.loc[(panel.index >= left) & (panel.index < right)]
        if chunk.empty:
            continue
        label = f"{left.date()} → {(right - pd.Timedelta(days=1)).date()}"
        slices.append((label, chunk))
    return slices


def adf_observed(
    series: np.ndarray | pd.Series,
    *,
    lag: int = ADF_LAG,
) -> ADFResult:
    """ADF on an observed series using *standard* (not EG) critical values."""
    return adf_lag1(series, lag=lag, critical_values=ADF_CRIT_OBSERVED)


def kpss_level(series: pd.Series) -> dict[str, float | bool]:
    """KPSS level test (H0: series is stationary around a constant).

    Complementary to ADF: failing to reject ADF *and* rejecting KPSS is the
    classic I(1) signature. Uses statsmodels (auxiliary test — listed as such
    in the numerical-methods table).
    """
    from statsmodels.tsa.stattools import kpss

    clean = pd.Series(series).astype(float).dropna()
    stat, pvalue, lags, crit = kpss(clean, regression="c", nlags="auto")
    return {
        "stat": float(stat),
        "pvalue": float(pvalue),
        "lags": int(lags),
        "crit_5pct": float(crit["5%"]),
        "reject_stationary_5pct": bool(stat > crit["5%"]),
    }


def integration_order_check(
    series: pd.Series,
    *,
    name: str = "series",
    lag: int = ADF_LAG,
) -> dict[str, object]:
    """Cheap I(1) screen: ADF/KPSS on levels vs first differences.

    Expected for ETF *prices*: levels look I(1); simple returns / diffs look I(0).
    """
    levels = pd.Series(series).astype(float).dropna()
    diffs = levels.diff().dropna()
    adf_l = adf_observed(levels, lag=lag)
    adf_d = adf_observed(diffs, lag=lag)
    kpss_l = kpss_level(levels)
    kpss_d = kpss_level(diffs)
    return {
        "name": name,
        "adf_level_tau": adf_l.tau,
        "adf_level_rejects_unitroot_5": adf_l.decide("5%"),
        "adf_diff_tau": adf_d.tau,
        "adf_diff_rejects_unitroot_5": adf_d.decide("5%"),
        "kpss_level_rejects_I0_5": kpss_l["reject_stationary_5pct"],
        "kpss_diff_rejects_I0_5": kpss_d["reject_stationary_5pct"],
        "looks_I1": (not adf_l.decide("5%")) and adf_d.decide("5%") and kpss_l["reject_stationary_5pct"],
    }
