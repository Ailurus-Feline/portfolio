"""Rolling Engle–Granger β re-estimation experiment.

CQF brief Part II §8
--------------------
Cointegration often assumes a stable β for 3–6 months. We re-estimate the
EG relation on a rolling window (default ~8 months of trading days) and
shift the window every ``ROLLING_STEP_DAYS`` bars, then compare a
*fixed-β* book to a *rolling-β* book on the same signal rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_pairs_backtest
from .cointegration import engle_granger
from .config import ROLLING_STEP_DAYS, ROLLING_WINDOW_DAYS, TRANSACTION_COST_BPS
from .ou_process import OUFit, fit_ou_ar1, zscore_series
from .signals import generate_positions, scan_z_grid


@dataclass(frozen=True)
class RollingBetaResult:
    """Fixed vs rolling hedge-ratio comparison."""

    window: int
    step: int
    z_star: float
    beta_path: pd.Series
    fixed: BacktestResult
    rolling: BacktestResult

    def summary(self) -> str:
        return "\n".join(
            [
                f"Rolling β experiment (window={self.window}, step={self.step}, Z*={self.z_star:g})",
                f"  β path: mean={self.beta_path.mean():.4f}, "
                f"std={self.beta_path.std():.4f}, "
                f"last={self.beta_path.iloc[-1]:.4f}",
                "  --- FIXED β (full-sample EG) ---",
                self.fixed.summary(),
                "  --- ROLLING β ---",
                self.rolling.summary(),
            ]
        )


def _fit_window(panel: pd.DataFrame) -> tuple[float, float, OUFit]:
    """Return (alpha, beta, OUFit) for one estimation window."""
    eg = engle_granger(panel["y"], panel["x"], run_ecm=False)
    ou = fit_ou_ar1(eg.residual)
    return eg.alpha, eg.beta, ou


def estimate_rolling_betas(
    panel: pd.DataFrame,
    *,
    window: int = ROLLING_WINDOW_DAYS,
    step: int = ROLLING_STEP_DAYS,
) -> pd.DataFrame:
    """Walk forward and store α, β, half-life at each re-estimation date.

    Parameters are stamped on the *last* date of each window and held until
    the next stamp (step bars later).
    """
    panel = panel.sort_index()
    rows: list[dict] = []
    i = window
    while i <= len(panel):
        chunk = panel.iloc[i - window : i]
        alpha, beta, ou = _fit_window(chunk)
        rows.append(
            {
                "date": chunk.index[-1],
                "alpha": alpha,
                "beta": beta,
                "half_life": ou.half_life,
                "sigma_eq": ou.sigma_eq,
                "mu": ou.mu,
            }
        )
        i += step

    if not rows:
        raise ValueError("Rolling window longer than the sample — shorten window.")
    return pd.DataFrame(rows).set_index("date").sort_index()


def _expand_param_path(
    index: pd.DatetimeIndex,
    stamped: pd.DataFrame,
) -> pd.DataFrame:
    """Forward-fill stamped parameters onto every trading day after warm-up."""
    path = stamped.reindex(index.union(stamped.index)).sort_index()
    path = path.ffill()
    return path.reindex(index)


def run_rolling_beta_experiment(
    panel: pd.DataFrame,
    *,
    window: int = ROLLING_WINDOW_DAYS,
    step: int = ROLLING_STEP_DAYS,
    cost_bps: float = TRANSACTION_COST_BPS,
    z_star: float | None = None,
) -> RollingBetaResult:
    """Compare fixed full-sample β vs rolling β with a shared Z* rule.

    Z* is chosen once on the *full-sample* residual of the fixed EG (so both
    legs share the same threshold). Rolling OU moments (μ, σ_eq) update with
    each window for the rolling book; the fixed book keeps full-sample OU.
    """
    panel = panel.sort_index()
    y_ticker = str(panel.attrs.get("y_ticker", "y"))
    x_ticker = str(panel.attrs.get("x_ticker", "x"))

    # --- Fixed benchmark -------------------------------------------------
    eg_full = engle_granger(
        panel["y"], panel["x"], y_ticker=y_ticker, x_ticker=x_ticker, run_ecm=False
    )
    ou_full = fit_ou_ar1(eg_full.residual)
    if z_star is None:
        z_star = scan_z_grid(eg_full.residual, ou_full).recommended_z

    z_fixed = zscore_series(eg_full.residual, ou_full)
    pos_fixed = generate_positions(z_fixed, z_entry=z_star)
    bt_fixed = run_pairs_backtest(
        panel, pos_fixed, beta=eg_full.beta, z_entry=z_star, cost_bps=cost_bps
    )

    # --- Rolling path ----------------------------------------------------
    stamped = estimate_rolling_betas(panel, window=window, step=step)
    path = _expand_param_path(panel.index, stamped)

    # Residual & z-score with time-varying α, β, μ, σ_eq (all lagged one step
    # in the backtest engine via position lag; parameters themselves are
    # known at close of the estimation window).
    resid = panel["y"] - path["alpha"] - path["beta"] * panel["x"]
    sigma = path["sigma_eq"].replace(0.0, np.nan)
    z_roll = ((resid - path["mu"]) / sigma).rename("zscore_rolling")
    pos_roll = generate_positions(z_roll.dropna(), z_entry=z_star)
    # Align positions back to full index (warmup → flat).
    pos_roll = pos_roll.reindex(panel.index).fillna(0.0)

    # Mark-to-market with *time-varying* β: customise spread change path.
    bt_roll = _backtest_with_beta_path(
        panel,
        pos_roll,
        beta_path=path["beta"],
        z_entry=z_star,
        cost_bps=cost_bps,
    )

    return RollingBetaResult(
        window=window,
        step=step,
        z_star=float(z_star),
        beta_path=path["beta"].rename("beta_rolling"),
        fixed=bt_fixed,
        rolling=bt_roll,
    )


def _backtest_with_beta_path(
    panel: pd.DataFrame,
    positions: pd.Series,
    *,
    beta_path: pd.Series,
    z_entry: float,
    cost_bps: float,
) -> BacktestResult:
    """Same economics as ``run_pairs_backtest``, but β_t is a series."""
    df = panel[["y", "x"]].astype(float).copy()
    pos = positions.reindex(df.index).fillna(0.0).astype(float)
    beta = beta_path.reindex(df.index).astype(float)

    dy = df["y"].diff()
    dx = df["x"].diff()
    # Use lagged β with lagged position (both known before today's move).
    beta_lag = beta.shift(1)
    pos_lag = pos.shift(1).fillna(0.0)
    spread_chg = dy - beta_lag * dx

    notional = df["y"].shift(1).abs() + beta_lag.abs() * df["x"].shift(1).abs()
    notional = notional.replace(0.0, np.nan)
    gross = pos_lag * spread_chg / notional

    dpos = pos.diff().fillna(pos.iloc[0])
    costs = 2.0 * dpos.abs() * (cost_bps / 1e4)
    pnl = gross - costs
    equity = (1.0 + pnl.fillna(0.0)).cumprod()
    daily = pnl.fillna(0.0)
    vol = float(daily.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * daily.mean() / vol) if vol > 0 else np.nan

    # Representative β for the summary line: mean of the path after warmup.
    beta_mean = float(beta.dropna().mean()) if beta.notna().any() else float("nan")

    from .backtest import _count_entries, _max_drawdown

    return BacktestResult(
        equity=equity.rename("equity"),
        pnl=pnl.rename("pnl"),
        positions=pos.rename("position"),
        costs=costs.rename("costs"),
        beta=beta_mean,
        z_entry=float(z_entry),
        total_return=float(equity.iloc[-1] - 1.0),
        sharpe=sharpe,
        max_drawdown=_max_drawdown(equity),
        n_trades=_count_entries(pos),
    )
