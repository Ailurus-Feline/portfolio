"""Entry / exit rules and iterative Z* calibration.

CQF brief
---------
Do **not** assume ``Z = 1``. Scan a grid of thresholds and record how many
trades each level produces. Wider bands → fewer, larger excursions → higher
structural-break risk if the residual mean shifts.

Signal convention (dollar-neutral sketch on the EG residual)
------------------------------------------------------------
* ``z_t = (e_t - μ) / σ_eq``
* Enter **short spread** when ``z_t >= +Z``  (residual rich: short y / long x)
* Enter **long spread**  when ``z_t <= -Z``  (residual cheap: long y / short x)
* Exit when ``z_t`` crosses back through 0 (reversion to ``μ``), or optionally
  when ``|z_t|`` falls below a small exit buffer (default 0.0 = mean touch).

Positions are held constant between signals (no pyramiding). This module
only builds the position path and trade-count diagnostics; dollar P&L with
costs is marked in ``backtest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Z_GRID
from .ou_process import OUFit, zscore_series


@dataclass(frozen=True)
class SignalResult:
    """Position path and trade diagnostics for one Z threshold."""

    z: float
    positions: pd.Series  # +1 long spread, -1 short spread, 0 flat
    n_trades: int  # number of round-trip entries (flat → nonzero)
    avg_holding_days: float
    time_in_market: float  # fraction of bars with a nonzero position


@dataclass(frozen=True)
class ZStarScan:
    """Grid search over Z thresholds (brief: iterative choice of Z*)."""

    grid: list[float]
    results: list[SignalResult]
    recommended_z: float
    reason: str

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "Z": r.z,
                "n_trades": r.n_trades,
                "avg_holding_days": r.avg_holding_days,
                "time_in_market": r.time_in_market,
            }
            for r in self.results
        ]
        return pd.DataFrame(rows).set_index("Z")


def generate_positions(
    zscore: pd.Series,
    *,
    z_entry: float,
    z_exit: float = 0.0,
) -> pd.Series:
    """Build a causal position series from Z-scores.

    State machine
    -------------
    flat  --(|z|>=Z)-->  ± long/short
    in trade --(|z|<=z_exit with correct side)-->  flat

    Exit uses a *touch of the mean* by default (``z_exit=0``): for a long
    spread (entered at -Z) we flatten once ``z >= -z_exit``; for a short
    spread (entered at +Z) we flatten once ``z <= +z_exit``.
    """
    if z_entry <= 0:
        raise ValueError("z_entry must be positive.")
    if z_exit < 0:
        raise ValueError("z_exit must be >= 0.")

    z = zscore.astype(float).to_numpy()
    idx = zscore.index
    pos = np.zeros(len(z), dtype=float)
    state = 0.0  # current position

    for t in range(len(z)):
        zt = z[t]
        if not np.isfinite(zt):
            pos[t] = state
            continue

        if state == 0.0:
            if zt >= z_entry:
                state = -1.0  # residual rich → short the spread
            elif zt <= -z_entry:
                state = 1.0  # residual cheap → long the spread
        elif state > 0:
            # Long spread: exit when residual has reverted up to -z_exit (~0)
            if zt >= -z_exit:
                state = 0.0
        elif state < 0:
            # Short spread: exit when residual has reverted down to +z_exit (~0)
            if zt <= z_exit:
                state = 0.0

        pos[t] = state

    return pd.Series(pos, index=idx, name=f"position_Z{z_entry:g}")


def _count_round_trips(positions: pd.Series) -> tuple[int, float, float]:
    """Return (n_trades, avg_holding_days, time_in_market)."""
    p = positions.fillna(0.0).to_numpy()
    entries = 0
    hold_lengths: list[int] = []
    run = 0
    prev = 0.0

    for val in p:
        if prev == 0.0 and val != 0.0:
            entries += 1
            run = 1
        elif prev != 0.0 and val == prev:
            run += 1
        elif prev != 0.0 and val != prev:
            # Flatten or flip: close the previous run.
            hold_lengths.append(run)
            run = 1 if val != 0.0 else 0
            if prev != 0.0 and val != 0.0 and np.sign(prev) != np.sign(val):
                # Flip counts as a new entry as well.
                entries += 1
        prev = val

    if run > 0 and prev != 0.0:
        hold_lengths.append(run)

    avg_hold = float(np.mean(hold_lengths)) if hold_lengths else 0.0
    time_in = float(np.mean(p != 0.0)) if len(p) else 0.0
    return entries, avg_hold, time_in


def evaluate_z(
    residual: pd.Series,
    ou: OUFit,
    z_entry: float,
    *,
    z_exit: float = 0.0,
) -> SignalResult:
    """Fit-free evaluation: z-score from ``ou``, then build positions."""
    z = zscore_series(residual, ou)
    positions = generate_positions(z, z_entry=z_entry, z_exit=z_exit)
    n_trades, avg_hold, time_in = _count_round_trips(positions)
    return SignalResult(
        z=float(z_entry),
        positions=positions,
        n_trades=n_trades,
        avg_holding_days=avg_hold,
        time_in_market=time_in,
    )


def scan_z_grid(
    residual: pd.Series,
    ou: OUFit,
    *,
    grid: list[float] | None = None,
    z_exit: float = 0.0,
    target_trades_per_year: float = 4.0,
) -> ZStarScan:
    """Iterate Z upward/downward and pick a practical default Z*.

    Selection heuristic (documented for the report — not a black-box optimiser)
    --------------------------------------------------------------------------
    Prefer the smallest Z in the grid whose *annualised* trade count is at
    most ``target_trades_per_year``. If every level trades more often than
    that, fall back to the largest Z (widest band / fewest trades).

    This mirrors the brief's trade-off narrative: wider bands → fewer trades
    but fatter tails / break risk; we start conservative on frequency.
    """
    grid = list(grid) if grid is not None else list(Z_GRID)
    if not grid:
        raise ValueError("Z grid is empty.")

    years = max(len(residual.dropna()) / 252.0, 1e-9)
    results = [evaluate_z(residual, ou, z, z_exit=z_exit) for z in grid]

    # Pick Z whose annualised trade count is closest to the target.
    # (If every Z under-trades — slow OU — this still lands near a mid band
    # rather than collapsing to the noisiest threshold.)
    best = min(
        results,
        key=lambda r: abs(r.n_trades / years - target_trades_per_year),
    )
    ann = best.n_trades / years
    reason = (
        f"Z closest to {target_trades_per_year:g} trades/year "
        f"(Z={best.z:g}, realised {ann:.2f}/year, n={best.n_trades})"
    )

    return ZStarScan(
        grid=grid,
        results=results,
        recommended_z=float(best.z),
        reason=reason,
    )
