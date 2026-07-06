"""Build standardized alpha signal panels for v3 backtests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import INDEX_MOMENTUM_LOOKBACK, V2_BEST_EMA, V2_BEST_SENTIMENT_CSV, WEIGHT_PRESETS
from src.diagnostics import attach_splits
from src.data_loader import load_market_data
from src.indicators import compute_breadth_indicators
from src.sentiment import (
    SentimentParams,
    assign_regime_labels,
    build_signal_z,
    composite_from_percentiles,
    merge_with_index,
    prepare_percentile_features,
    select_analysis_rows,
)

ALPHA_NAMES: tuple[str, ...] = (
    "sentiment_composite",
    "new_high_low_net",
    "above_ma",
    "positive_return",
    "advance_decline",
    "index_momentum",
)

SENTIMENT_ALPHAS: frozenset[str] = frozenset(
    {
        "sentiment_composite",
        "new_high_low_net",
        "above_ma",
        "positive_return",
        "advance_decline",
    }
)


def _analysis_frame(indicators: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Align indicator rows with the v1/v2 analysis timeline."""
    features = prepare_percentile_features(indicators).reset_index(names="date")
    v2_params = SentimentParams(weights=WEIGHT_PRESETS["balanced_momentum"], ema_span=V2_BEST_EMA)
    timeline = attach_splits(
        select_analysis_rows(
            merge_with_index(composite_from_percentiles(features, v2_params), index_df)
        )
    )
    base = merge_with_index(features, index_df)
    base = base.merge(timeline[["date", "split"]], on="date", how="inner")
    return base.sort_values("date").reset_index(drop=True)


def _signal_from_raw_column(base: pd.DataFrame, column: str, ema_span: int = V2_BEST_EMA) -> pd.DataFrame:
    z = build_signal_z(base[column], ema_span=ema_span)
    frame = base.loc[:, ["date", "split", "index_close"]].copy()
    frame["signal_z"] = z
    frame["regime"] = assign_regime_labels(z)
    return frame.dropna(subset=["signal_z", "index_close"]).reset_index(drop=True)


def load_sentiment_composite(path: Path = V2_BEST_SENTIMENT_CSV) -> pd.DataFrame:
    """Load the v2-selected composite sentiment signal."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python3 -m src.v2_main` first.")
    frame = pd.read_csv(path, parse_dates=["date"])
    out = frame.loc[:, ["date", "split", "index_close", "sentiment_z", "regime"]].rename(
        columns={"sentiment_z": "signal_z"}
    )
    return out.dropna(subset=["signal_z", "index_close"]).reset_index(drop=True)


def build_market_base() -> pd.DataFrame:
    """Shared indicator panel aligned to the analysis timeline."""
    market = load_market_data()
    indicators = compute_breadth_indicators(
        close_panel=market["close_panel"],
        valid_panel=market["valid_panel"],
    )
    return _analysis_frame(indicators, market["index"])


def build_index_momentum_signal(base: pd.DataFrame) -> pd.DataFrame:
    """Index 60-day return z-score (auxiliary / trend factor)."""
    ret = base["index_close"].pct_change(INDEX_MOMENTUM_LOOKBACK)
    return _signal_from_raw_column(
        base.assign(index_momentum=ret),
        "index_momentum",
        ema_span=V2_BEST_EMA,
    )


def build_all_alpha_signals() -> dict[str, pd.DataFrame]:
    """Return all v3 alpha signal frames keyed by alpha name."""
    signals: dict[str, pd.DataFrame] = {
        "sentiment_composite": load_sentiment_composite(),
    }

    base = build_market_base()
    for column in ("new_high_low_net", "above_ma", "positive_return", "advance_decline"):
        signals[column] = _signal_from_raw_column(base, column)

    signals["index_momentum"] = build_index_momentum_signal(base)
    return signals


def build_breadth_alpha_signal(
    base: pd.DataFrame,
    column: str,
    ema_span: int = V2_BEST_EMA,
) -> pd.DataFrame:
    """Build one breadth or momentum alpha with a custom EMA span."""
    if column == "index_momentum":
        ret = base["index_close"].pct_change(INDEX_MOMENTUM_LOOKBACK)
        return _signal_from_raw_column(base.assign(index_momentum=ret), "index_momentum", ema_span=ema_span)
    return _signal_from_raw_column(base, column, ema_span=ema_span)
