"""Systematic backtest for dollar-neutral pairs positions.

Given EG hedge ratio ``β`` and a position series in ``{-1, 0, +1}``
(from ``signals.generate_positions``), mark the spread on lagged signals
and scale by lagged gross notional so returns compound from equity = 1.

Spread definition
-----------------
At each bar the held residual mark-to-market uses price changes::

    Δs_t = Δy_t - β Δx_t
    r_t^{gross} = position_{t-1} * Δs_t / (|y_{t-1}| + |β| |x_{t-1}|)

We trade on the *previous* signal (no look-ahead). ``β`` is the EG slope
from the formation sample, or the latest rolling estimate when
``rolling.py`` re-fits the hedge.

Costs
-----
Each position change charges ``TRANSACTION_COST_BPS`` on *both* legs::

    cost = 2 * |Δposition| * (bps / 1e4)

``|Δposition|`` is 1 on enter/exit and 2 on a flip, so flips pay twice
— intentional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRANSACTION_COST_BPS


@dataclass(frozen=True)
class BacktestResult:
    """Paths and headline stats for one pair / one Z*."""

    equity: pd.Series
    pnl: pd.Series
    positions: pd.Series
    costs: pd.Series
    beta: float
    z_entry: float
    total_return: float
    sharpe: float  # annualised, using daily pnl / equity_0=1 style
    max_drawdown: float
    n_trades: int

    def summary(self) -> str:
        return "\n".join(
            [
                f"Backtest (β={self.beta:.4f}, Z={self.z_entry:g})",
                f"  total return     = {self.total_return:.2%}",
                f"  ann. Sharpe      = {self.sharpe:.3f}",
                f"  max drawdown     = {self.max_drawdown:.2%}",
                f"  n_trades         = {self.n_trades}",
            ]
        )


def _max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough drawdown as a negative fraction of equity."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _count_entries(positions: pd.Series) -> int:
    p = positions.fillna(0.0).to_numpy()
    return int(np.sum((p[1:] != 0) & (p[:-1] == 0)) + (1 if p[0] != 0 else 0))


def run_pairs_backtest(
    panel: pd.DataFrame,
    positions: pd.Series,
    *,
    beta: float,
    z_entry: float,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> BacktestResult:
    """Mark-to-market a dollar-neutral pairs book.

    Parameters
    ----------
    panel:
        Must contain columns ``y`` and ``x`` (adjusted prices).
    positions:
        Series aligned (or alignable) to ``panel.index`` with values in
        ``{-1, 0, +1}``.
    beta:
        Hedge ratio from the cointegrating regression ``y ~ x``.
    """
    df = panel[["y", "x"]].astype(float).copy()
    pos = positions.reindex(df.index).fillna(0.0).astype(float)

    dy = df["y"].diff()
    dx = df["x"].diff()
    # Spread change matching the EG residual in *price* space.
    spread_chg = dy - beta * dx

    # Trade on lagged position to avoid same-bar look-ahead.
    pos_lag = pos.shift(1).fillna(0.0)

    # Scale dollar spread P&L by lagged gross notional so returns are
    # dimensionless and compoundable from equity=1:
    #   notional ≈ |1|·y + |β|·x   (one unit of the y-leg hedged with β of x)
    notional = df["y"].shift(1).abs() + abs(beta) * df["x"].shift(1).abs()
    notional = notional.replace(0.0, np.nan)
    gross = pos_lag * spread_chg / notional

    # Transaction costs on position changes: bps charged on *each* leg, so
    # two legs → factor 2. Already a fraction of book size (unit notional).
    dpos = pos.diff().fillna(pos.iloc[0])
    costs = 2.0 * dpos.abs() * (cost_bps / 1e4)

    pnl = gross - costs
    equity = (1.0 + pnl.fillna(0.0)).cumprod()

    # Sharpe on daily pnl relative to unit starting capital (simple).
    daily = pnl.fillna(0.0)
    vol = float(daily.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * daily.mean() / vol) if vol > 0 else np.nan

    return BacktestResult(
        equity=equity.rename("equity"),
        pnl=pnl.rename("pnl"),
        positions=pos.rename("position"),
        costs=costs.rename("costs"),
        beta=float(beta),
        z_entry=float(z_entry),
        total_return=float(equity.iloc[-1] - 1.0),
        sharpe=sharpe,
        max_drawdown=_max_drawdown(equity),
        n_trades=_count_entries(pos),
    )
