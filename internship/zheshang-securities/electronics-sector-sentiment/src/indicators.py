"""Cross-sectional breadth indicators."""

from __future__ import annotations

import pandas as pd

from src.config import ROLLING_HIGH_LOW, ROLLING_MA, ROLLING_RETURN


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise ratio with NaN when the denominator is zero."""
    out = numerator / denominator
    return out.where(denominator > 0)


def new_high_low_net_ratio(close: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """
    (# 60-day highs - # 60-day lows) / valid stocks.

    A new high means today's close equals the maximum close over the last
    60 trading days (inclusive). A new low is defined symmetrically.
    """
    rolling_max = close.rolling(ROLLING_HIGH_LOW, min_periods=ROLLING_HIGH_LOW).max()
    rolling_min = close.rolling(ROLLING_HIGH_LOW, min_periods=ROLLING_HIGH_LOW).min()

    is_new_high = (close >= rolling_max) & valid
    is_new_low = (close <= rolling_min) & valid

    n_valid = valid.sum(axis=1)
    return _safe_ratio(is_new_high.sum(axis=1) - is_new_low.sum(axis=1), n_valid)


def above_ma_ratio(close: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Share of valid stocks whose close is above their own 120-day moving average."""
    ma = close.rolling(ROLLING_MA, min_periods=ROLLING_MA).mean()
    is_above = (close > ma) & valid
    n_valid = valid.sum(axis=1)
    return _safe_ratio(is_above.sum(axis=1), n_valid)


def positive_return_ratio(close: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Share of valid stocks with a positive 20-day simple return."""
    ret_20 = close / close.shift(ROLLING_RETURN) - 1.0
    is_positive = (ret_20 > 0) & valid
    n_valid = valid.sum(axis=1)
    return _safe_ratio(is_positive.sum(axis=1), n_valid)


def compute_breadth_indicators(
    close_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Return daily raw breadth sub-indicators."""
    indicators = pd.DataFrame(index=close_panel.index)
    indicators["new_high_low_net"] = new_high_low_net_ratio(close_panel, valid_panel)
    indicators["above_ma"] = above_ma_ratio(close_panel, valid_panel)
    indicators["positive_return"] = positive_return_ratio(close_panel, valid_panel)
    indicators["valid_stocks"] = valid_panel.sum(axis=1)
    return indicators
