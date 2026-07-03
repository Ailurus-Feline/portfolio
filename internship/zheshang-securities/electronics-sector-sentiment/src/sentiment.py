"""Composite sentiment, smoothing, and regime labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    EMA_SPAN,
    Z_OVERCOOL,
    Z_OVERHEAT,
)


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


def composite_sentiment(indicators: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank the three breadth metrics and combine them with equal weights."""
    result = indicators.copy()

    for column in ("new_high_low_net", "above_ma", "positive_return"):
        result[f"{column}_pct"] = expanding_percentile_rank(indicators[column])

    pct_columns = [f"{col}_pct" for col in ("new_high_low_net", "above_ma", "positive_return")]
    result["sentiment_raw"] = result[pct_columns].mean(axis=1)

    result["sentiment_slow"] = result["sentiment_raw"].ewm(span=EMA_SPAN, adjust=False).mean()

    expanding_mean = result["sentiment_slow"].expanding(min_periods=2).mean()
    expanding_std = result["sentiment_slow"].expanding(min_periods=2).std()
    result["sentiment_z"] = (result["sentiment_slow"] - expanding_mean) / expanding_std

    result["regime"] = "neutral"
    result.loc[result["sentiment_z"] > Z_OVERHEAT, "regime"] = "overheated"
    result.loc[result["sentiment_z"] < Z_OVERCOOL, "regime"] = "overcooled"
    return result


def merge_with_index(sentiment: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Attach index levels and log index for plotting."""
    merged = sentiment.merge(index_df, on="date", how="left")
    merged["index_log"] = np.log(merged["close"])
    merged = merged.rename(columns={"close": "index_close"})
    return merged
