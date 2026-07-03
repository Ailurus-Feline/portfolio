"""Composite sentiment, smoothing, and regime labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_WEIGHTS,
    EMA_SPAN,
    Z_OVERCOOL,
    Z_OVERHEAT,
)

BREADTH_COLUMNS = ("new_high_low_net", "above_ma", "positive_return")
PCT_COLUMNS = tuple(f"{column}_pct" for column in BREADTH_COLUMNS)


@dataclass(frozen=True)
class SentimentParams:
    """Configurable sentiment construction parameters."""

    weights: tuple[float, float, float] = DEFAULT_WEIGHTS
    ema_span: int = EMA_SPAN

    def __post_init__(self) -> None:
        if len(self.weights) != 3:
            raise ValueError("weights must contain exactly three values.")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("weights must sum to 1.")
        if self.ema_span < 2:
            raise ValueError("ema_span must be at least 2.")


def default_sentiment_params() -> SentimentParams:
    """Return the v1 default parameter set."""
    return SentimentParams(weights=DEFAULT_WEIGHTS, ema_span=EMA_SPAN)


def expanding_percentile_rank(series: pd.Series) -> pd.Series:
    """
    Map values to an expanding historical percentile in [0, 100].

    Uses only past and current observations to avoid look-ahead bias.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)

    history: list[float] = []
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        history.append(value)
        arr = np.asarray(history, dtype=float)
        out[i] = (np.sum(arr < value) + 0.5 * np.sum(arr == value)) / len(arr) * 100.0

    return pd.Series(out, index=series.index, name=f"{series.name}_pct")


def prepare_percentile_features(indicators: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank the three breadth metrics once for grid-search reuse."""
    result = indicators.copy()
    for column in BREADTH_COLUMNS:
        result[f"{column}_pct"] = expanding_percentile_rank(indicators[column])
    return result


def assign_regime_labels(z: pd.Series) -> pd.Series:
    """Map z-scores to overheated, overcooled, or neutral labels."""
    regime = pd.Series(pd.NA, index=z.index, dtype="object")
    valid = z.notna()
    regime.loc[valid & (z > Z_OVERHEAT)] = "overheated"
    regime.loc[valid & (z < Z_OVERCOOL)] = "overcooled"
    regime.loc[valid & regime.isna()] = "neutral"
    return regime


def composite_from_percentiles(
    features: pd.DataFrame,
    params: SentimentParams,
) -> pd.DataFrame:
    """Build sentiment columns from precomputed percentile features."""
    result = features.copy()
    weight_array = np.asarray(params.weights, dtype=float)
    result["sentiment_raw"] = result.loc[:, PCT_COLUMNS].to_numpy() @ weight_array
    result["sentiment_slow"] = result["sentiment_raw"].ewm(span=params.ema_span, adjust=False).mean()

    expanding_mean = result["sentiment_slow"].expanding(min_periods=2).mean()
    expanding_std = result["sentiment_slow"].expanding(min_periods=2).std()
    result["sentiment_z"] = (result["sentiment_slow"] - expanding_mean) / expanding_std
    result["regime"] = assign_regime_labels(result["sentiment_z"])
    return result


def composite_sentiment(
    indicators: pd.DataFrame,
    params: SentimentParams | None = None,
) -> pd.DataFrame:
    """Percentile-rank breadth metrics and combine them with configurable weights."""
    if params is None:
        params = default_sentiment_params()
    features = prepare_percentile_features(indicators)
    return composite_from_percentiles(features, params)


def select_analysis_rows(sentiment: pd.DataFrame) -> pd.DataFrame:
    """Return rows where sentiment and index data are both available."""
    mask = sentiment["sentiment_z"].notna() & sentiment["index_close"].notna()
    return sentiment.loc[mask].reset_index(drop=True)


def merge_with_index(sentiment: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Attach index levels and log index for plotting."""
    merged = sentiment.merge(index_df, on="date", how="left")
    merged["index_close"] = merged["close"]
    merged["index_log"] = np.log(merged["index_close"])
    return merged.drop(columns=["close"])
