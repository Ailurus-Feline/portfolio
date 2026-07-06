"""Alpha combination methods for v4 (equal-weight, IC-weighted, Ridge, Decision Tree)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.alpha_signals import (
    build_breadth_alpha_signal,
    build_market_base,
    load_sentiment_composite,
)
from src.config import IC_PRIMARY_HORIZON, V2_BEST_EMA, V4_ALPHA_NAMES
from src.diagnostics import add_forward_returns, rank_ic


@dataclass(frozen=True)
class CombinationSpec:
    """One v4 combination configuration."""

    method: str
    ridge_lambda: float | None = None
    ic_horizon: int | None = None
    tree_max_depth: int | None = None
    tree_min_samples_leaf: int | None = None
    weights: tuple[float, ...] | None = None


def spec_cache_key(spec: CombinationSpec) -> str:
    """Stable cache key for fitted combination specs."""
    return (
        f"{spec.method}|{spec.ridge_lambda}|{spec.ic_horizon}|"
        f"{spec.tree_max_depth}|{spec.tree_min_samples_leaf}"
    )


def _alpha_column(alpha: str) -> str:
    return f"{alpha}_z"


def build_v4_alpha_matrix() -> pd.DataFrame:
    """Merge v4 candidate alphas into one dated panel with forward returns."""
    base = build_market_base()
    sentiment = load_sentiment_composite()

    panel = sentiment.loc[:, ["date", "split", "index_close"]].copy()
    panel["sentiment_composite_z"] = sentiment["signal_z"].to_numpy()

    for alpha in ("advance_decline", "positive_return", "index_momentum"):
        signal = build_breadth_alpha_signal(base, alpha, ema_span=V2_BEST_EMA)
        panel = panel.merge(
            signal.loc[:, ["date", "signal_z"]].rename(columns={"signal_z": _alpha_column(alpha)}),
            on="date",
            how="inner",
        )

    panel = add_forward_returns(panel).dropna(subset=[_alpha_column(a) for a in V4_ALPHA_NAMES])
    return panel.sort_values("date").reset_index(drop=True)


def _alpha_matrix_values(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, [_alpha_column(alpha) for alpha in V4_ALPHA_NAMES]].to_numpy(dtype=float)


def _train_xy(train: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    x_train = _alpha_matrix_values(train)
    y_train = train[f"fwd_ret_{horizon}d"].to_numpy(dtype=float)
    mask = np.isfinite(y_train)
    return x_train[mask], y_train[mask]


def equal_weight_score(frame: pd.DataFrame) -> pd.Series:
    """Equal-weight average of standardized alpha z-scores."""
    values = _alpha_matrix_values(frame)
    return pd.Series(values.mean(axis=1), index=frame.index, name="composite_raw")


def ic_weights(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Non-negative IC weights fit on the train split only."""
    weights: list[float] = []
    target = train[f"fwd_ret_{horizon}d"]
    for alpha in V4_ALPHA_NAMES:
        signal = train[_alpha_column(alpha)]
        ic = rank_ic(signal, target)
        weights.append(max(float(ic), 0.0) if not np.isnan(ic) else 0.0)

    total = float(sum(weights))
    if total <= 0.0:
        return np.full(len(V4_ALPHA_NAMES), 1.0 / len(V4_ALPHA_NAMES))
    return np.asarray(weights, dtype=float) / total


def ic_weighted_score(frame: pd.DataFrame, train: pd.DataFrame, horizon: int) -> tuple[pd.Series, np.ndarray]:
    """IC-weighted linear combination using train-split weights only."""
    weights = ic_weights(train, horizon)
    values = _alpha_matrix_values(frame)
    score = values @ weights
    return pd.Series(score, index=frame.index, name="composite_raw"), weights


def fit_ridge_coefficients(x_train: np.ndarray, y_train: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression without intercept: beta = (X'X + lambda I)^-1 X'y."""
    xtx = x_train.T @ x_train
    penalty = lam * np.eye(x_train.shape[1])
    return np.linalg.solve(xtx + penalty, x_train.T @ y_train)


def ridge_score(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    lam: float,
    horizon: int = IC_PRIMARY_HORIZON,
) -> tuple[pd.Series, np.ndarray]:
    """Ridge combination fit on train, applied to all rows."""
    x_train, y_train = _train_xy(train, horizon)
    if len(y_train) < x_train.shape[1] + 1:
        raise ValueError("Insufficient train rows for ridge fit.")

    beta = fit_ridge_coefficients(x_train, y_train, lam)
    score = _alpha_matrix_values(frame) @ beta
    return pd.Series(score, index=frame.index, name="composite_raw"), beta


def tree_score(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    max_depth: int,
    min_samples_leaf: int,
    horizon: int = IC_PRIMARY_HORIZON,
) -> tuple[pd.Series, np.ndarray]:
    """Decision tree combination fit on train only (PPT baseline: depth=3, leaf=100)."""
    x_train, y_train = _train_xy(train, horizon)
    if len(y_train) < max(min_samples_leaf * 2, 50):
        raise ValueError("Insufficient train rows for tree fit.")

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
    )
    model.fit(x_train, y_train)
    score = model.predict(_alpha_matrix_values(frame))
    return pd.Series(score, index=frame.index, name="composite_raw"), model.feature_importances_


def standardize_composite(
    composite_raw: pd.Series,
    train: pd.DataFrame,
) -> pd.Series:
    """Z-score composite signal using train-split mean and std only."""
    train_values = composite_raw.loc[train.index]
    mean = float(train_values.mean())
    std = float(train_values.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(0.0, index=composite_raw.index, name="signal_z")
    return pd.Series((composite_raw - mean) / std, index=composite_raw.index, name="signal_z")


def build_composite_signal(
    frame: pd.DataFrame,
    spec: CombinationSpec,
) -> tuple[pd.DataFrame, CombinationSpec]:
    """Return a backtest-ready signal frame with standardized composite signal_z."""
    train = frame.loc[frame["split"] == "train"]
    meta = spec

    if spec.method == "equal_weight":
        raw = equal_weight_score(frame)
        weights = tuple(1.0 / len(V4_ALPHA_NAMES) for _ in V4_ALPHA_NAMES)
        meta = CombinationSpec(method=spec.method, weights=weights)
    elif spec.method == "ic_weighted":
        horizon = spec.ic_horizon or IC_PRIMARY_HORIZON
        raw, weights = ic_weighted_score(frame, train, horizon)
        meta = CombinationSpec(method=spec.method, ic_horizon=horizon, weights=tuple(weights))
    elif spec.method == "ridge":
        lam = spec.ridge_lambda if spec.ridge_lambda is not None else 1.0
        horizon = spec.ic_horizon or IC_PRIMARY_HORIZON
        raw, beta = ridge_score(frame, train, lam, horizon=horizon)
        meta = CombinationSpec(method=spec.method, ridge_lambda=lam, ic_horizon=horizon, weights=tuple(beta))
    elif spec.method == "decision_tree":
        depth = spec.tree_max_depth if spec.tree_max_depth is not None else 3
        leaf = spec.tree_min_samples_leaf if spec.tree_min_samples_leaf is not None else 100
        horizon = spec.ic_horizon or IC_PRIMARY_HORIZON
        raw, importances = tree_score(frame, train, depth, leaf, horizon=horizon)
        meta = CombinationSpec(
            method=spec.method,
            ic_horizon=horizon,
            tree_max_depth=depth,
            tree_min_samples_leaf=leaf,
            weights=tuple(importances),
        )
    else:
        raise ValueError(f"Unknown combination method: {spec.method!r}")

    signal_z = standardize_composite(raw, train)
    out = frame.loc[:, ["date", "split", "index_close"]].copy()
    out["signal_z"] = signal_z
    out["composite_raw"] = raw
    return out, meta
