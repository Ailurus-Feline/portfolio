"""Train/test (time-split) validation for the pairs pipeline.

CQF brief Part II §7
--------------------
Use a scikit-learn-inspired split that respects *time* (no shuffled CV).
Formation (train) estimates α, β, OU moments, and Z*; the trade period
(test) applies those frozen parameters — the honest check against the
in-sample smoke results.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestResult, run_pairs_backtest
from .cointegration import engle_granger
from .ou_process import fit_ou_ar1, zscore_series
from .signals import generate_positions, scan_z_grid


@dataclass(frozen=True)
class TrainTestBundle:
    """Frozen formation parameters + separate train/test backtests."""

    split_date: pd.Timestamp
    train_frac: float
    alpha: float
    beta: float
    z_star: float
    half_life: float
    train: BacktestResult
    test: BacktestResult
    z_scan_table: pd.DataFrame
    z_reason: str

    def summary(self) -> str:
        return "\n".join(
            [
                f"Train/test split @ {self.split_date.date()} "
                f"(train_frac={self.train_frac:.2f})",
                f"  frozen α={self.alpha:.4f}, β={self.beta:.4f}, "
                f"Z*={self.z_star:g}, half-life={self.half_life:.1f}d",
                f"  Z* rule: {self.z_reason}",
                "  --- TRAIN ---",
                self.train.summary(),
                "  --- TEST  ---",
                self.test.summary(),
            ]
        )


def time_split_index(index: pd.DatetimeIndex, train_frac: float = 0.7) -> pd.Timestamp:
    """Return the first timestamp that belongs to the *test* segment."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1).")
    if len(index) < 50:
        raise ValueError("Need a longer sample for a meaningful time split.")
    cut = int(len(index) * train_frac)
    cut = min(max(cut, 1), len(index) - 1)
    return pd.Timestamp(index[cut])


def _residual_from_params(
    panel: pd.DataFrame,
    *,
    alpha: float,
    beta: float,
) -> pd.Series:
    """Out-of-sample residual using *frozen* cointegrating coefficients."""
    e = panel["y"].astype(float) - alpha - beta * panel["x"].astype(float)
    return e.rename("eg_residual_frozen")


def run_train_test_backtest(
    panel: pd.DataFrame,
    *,
    y_ticker: str = "y",
    x_ticker: str = "x",
    train_frac: float = 0.7,
    cost_bps: float | None = None,
) -> TrainTestBundle:
    """Fit on train, trade on train *and* test with frozen parameters.

    The train backtest is reported for diagnostics only (still in-sample for
    that segment). The test backtest is the decision-relevant number.
    """
    from .config import TRANSACTION_COST_BPS

    if cost_bps is None:
        cost_bps = TRANSACTION_COST_BPS

    panel = panel.sort_index()
    split = time_split_index(panel.index, train_frac=train_frac)
    train_panel = panel.loc[panel.index < split]
    test_panel = panel.loc[panel.index >= split]
    if len(train_panel) < 100 or len(test_panel) < 50:
        raise ValueError(
            f"Split @ {split.date()} yields train={len(train_panel)}, "
            f"test={len(test_panel)}; widen the sample or adjust train_frac."
        )

    y_ticker = str(panel.attrs.get("y_ticker", y_ticker))
    x_ticker = str(panel.attrs.get("x_ticker", x_ticker))

    eg = engle_granger(
        train_panel["y"],
        train_panel["x"],
        y_ticker=y_ticker,
        x_ticker=x_ticker,
    )
    ou = fit_ou_ar1(eg.residual)
    scan = scan_z_grid(eg.residual, ou)

    def _segment_bt(seg: pd.DataFrame) -> BacktestResult:
        resid = _residual_from_params(seg, alpha=eg.alpha, beta=eg.beta)
        # Re-use train OU (μ, σ_eq) for z-scores — no peeking at test moments.
        z = zscore_series(resid, ou)
        pos = generate_positions(z, z_entry=scan.recommended_z, z_exit=0.0)
        return run_pairs_backtest(
            seg,
            pos,
            beta=eg.beta,
            z_entry=scan.recommended_z,
            cost_bps=cost_bps,
        )

    return TrainTestBundle(
        split_date=split,
        train_frac=train_frac,
        alpha=eg.alpha,
        beta=eg.beta,
        z_star=scan.recommended_z,
        half_life=ou.half_life,
        train=_segment_bt(train_panel),
        test=_segment_bt(test_panel),
        z_scan_table=scan.to_frame(),
        z_reason=scan.reason,
    )
