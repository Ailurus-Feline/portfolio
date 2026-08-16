"""Johansen cointegration diagnostics (multivariate complement to EG).

Why this module exists
----------------------
Engle–Granger is residual-based and *asymmetric* (y on x). Johansen works
on a VECM and can detect cointegration rank without choosing a dependent
leg. The CQF brief encourages venturing into Johansen / VECM for depth.

Implementation note (numerical-methods table)
---------------------------------------------
We use ``statsmodels.tsa.vector_ar.vecm.coint_johansen`` for the eigenvalue
problem and trace / max-eigen critical values. Re-deriving the Johansen
reduced-rank MLE from scratch is outside the project's expected scope
(analogous to not re-coding a QP solver). We *do* own:

* data orientation (log-prices vs levels — we use log-prices by default),
* lag choice (AIC on levels VAR as a transparent default),
* interpretation of rank, β loadings, and a simple VECM adjustment readout.

Det_order
---------
``det_order=0`` → constant in the cointegrating relation (most common for
price levels / log-prices of ETFs without a linear trend in the CI vector).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass(frozen=True)
class JohansenResult:
    """Compact Johansen summary for a bivariate pair."""

    y_ticker: str
    x_ticker: str
    k_ar_diff: int  # lag order in differences (statsmodels naming)
    det_order: int
    trace_stat: np.ndarray  # shape (2,) for bivariate: r=0, r<=1 tests
    trace_crit_95: np.ndarray
    maxeig_stat: np.ndarray
    maxeig_crit_95: np.ndarray
    eigenvalues: np.ndarray
    beta: np.ndarray  # cointegrating vectors (columns), size (2, 2)
    used_log_prices: bool

    @property
    def rank_trace_5pct(self) -> int:
        """Smallest r such that we fail to reject H0: rank <= r (trace, 5%)."""
        rank = 0
        for i, (stat, crit) in enumerate(zip(self.trace_stat, self.trace_crit_95)):
            # i=0 tests r=0; i=1 tests r<=1. Reject → rank at least i+1.
            if stat > crit:
                rank = i + 1
            else:
                break
        return int(rank)

    @property
    def primary_beta(self) -> np.ndarray:
        """First cointegrating vector, normalised so the y-loading is 1."""
        b = self.beta[:, 0].astype(float).copy()
        if abs(b[0]) > 1e-12:
            b = b / b[0]
        return b

    def summary(self) -> str:
        b = self.primary_beta
        lines = [
            f"Johansen: [{self.y_ticker}, {self.x_ticker}]  "
            f"(log_prices={self.used_log_prices}, k_ar_diff={self.k_ar_diff})",
            f"  eigenvalues           = {np.array2string(self.eigenvalues, precision=4)}",
            f"  trace stats           = {np.array2string(self.trace_stat, precision=3)}",
            f"  trace 5% crit         = {np.array2string(self.trace_crit_95, precision=3)}",
            f"  rank by trace @5%     = {self.rank_trace_5pct}",
            f"  primary β (y-normed)  = [{b[0]:.4f}, {b[1]:.4f}]  "
            f"→  {self.y_ticker} + ({b[1]:.4f}) {self.x_ticker} ~ I(0)",
        ]
        return "\n".join(lines)


def _select_diff_lags(log_prices: pd.DataFrame, maxlags: int = 8) -> int:
    """Pick VECM difference lag via AIC on a levels VAR (k_ar = lag+1).

    ``coint_johansen`` expects ``k_ar_diff`` = number of *lags in differences*.
    A levels VAR of order p corresponds to k_ar_diff = p - 1.
    """
    from statsmodels.tsa.api import VAR

    # VAR on levels; AIC selects p >= 1.
    model = VAR(log_prices)
    # ic='aic' returns a fitted model with the selected lag.
    selected = model.fit(maxlags=maxlags, ic="aic")
    p = int(selected.k_ar)
    k_ar_diff = max(p - 1, 1)  # Johansen needs at least 1 diff lag in practice
    return k_ar_diff


def johansen_pair(
    y: pd.Series,
    x: pd.Series,
    *,
    y_ticker: str = "y",
    x_ticker: str = "x",
    k_ar_diff: int | None = None,
    det_order: int = 0,
    use_log: bool = True,
    maxlags: int = 8,
) -> JohansenResult:
    """Run Johansen on a bivariate ETF pair.

    Parameters
    ----------
    use_log:
        If True (default), take ``log`` of prices before the test — standard
        for equity/ETF pairs so β is closer to an elasticity.
    k_ar_diff:
        If None, choose via AIC on a levels VAR (see ``_select_diff_lags``).
    """
    panel = pd.concat({y_ticker: y.astype(float), x_ticker: x.astype(float)}, axis=1).dropna()
    if use_log:
        if (panel <= 0).any().any():
            raise ValueError("Log-price Johansen requires strictly positive prices.")
        data = np.log(panel)
    else:
        data = panel

    if k_ar_diff is None:
        k_ar_diff = _select_diff_lags(data, maxlags=maxlags)

    # statsmodels returns trace/maxeig stats ordered from r=0 upward.
    out = coint_johansen(data.to_numpy(), det_order=det_order, k_ar_diff=k_ar_diff)

    # statsmodels may return complex arrays with zero imaginary part — keep .real.
    return JohansenResult(
        y_ticker=y_ticker,
        x_ticker=x_ticker,
        k_ar_diff=k_ar_diff,
        det_order=det_order,
        trace_stat=np.asarray(out.lr1, dtype=float).real,
        trace_crit_95=np.asarray(out.cvt[:, 1], dtype=float).real,
        maxeig_stat=np.asarray(out.lr2, dtype=float).real,
        maxeig_crit_95=np.asarray(out.cvm[:, 1], dtype=float).real,
        eigenvalues=np.asarray(out.eig, dtype=float).real,
        beta=np.asarray(out.evec, dtype=float).real,
        used_log_prices=use_log,
    )


def johansen_from_panel(
    panel: pd.DataFrame,
    *,
    y_ticker: str | None = None,
    x_ticker: str | None = None,
    **kwargs,
) -> JohansenResult:
    """Convenience wrapper for ``align_pair`` panels (columns ``y`` / ``x``)."""
    y_ticker = y_ticker or str(panel.attrs.get("y_ticker", "y"))
    x_ticker = x_ticker or str(panel.attrs.get("x_ticker", "x"))
    return johansen_pair(
        panel["y"],
        panel["x"],
        y_ticker=y_ticker,
        x_ticker=x_ticker,
        **kwargs,
    )


def var_lag_stability(
    panel: pd.DataFrame,
    *,
    maxlags: int = 8,
) -> dict[str, object]:
    """VAR specification checks from the brief: AIC/BIC lag and eigenvalue stability.

    Run on *log-return* (stationary changes), not price levels. The brief notes
    that VAR stability/lag tests apply to structural models in differences;
    this project does not forecast returns — the check is diagnostic only.
    """
    from statsmodels.tsa.api import VAR

    log_ret = np.log(panel[["y", "x"]].astype(float)).diff().dropna()
    model = VAR(log_ret)
    aic = model.fit(maxlags=maxlags, ic="aic")
    bic = model.fit(maxlags=maxlags, ic="bic")
    # statsmodels ``roots`` are reciprocals of companion eigenvalues;
    # use the library stability check rather than rolling our own cutoff.
    companion = np.asarray(np.abs(aic.roots))
    # Companion modulus = 1/|root| when roots are the reciprocal representation.
    max_comp = float(np.max(1.0 / np.maximum(companion, 1e-12))) if companion.size else float("nan")
    return {
        "aic_lag": int(aic.k_ar),
        "bic_lag": int(bic.k_ar),
        "max_companion_modulus": max_comp,
        "stable": bool(aic.is_stable()),
        "nobs": int(aic.nobs),
    }
