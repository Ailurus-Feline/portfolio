"""Entry point for the v1 electronics sector sentiment pipeline."""

from __future__ import annotations

import pandas as pd

from src.config import FIGURES_DIR, OUTPUT_CSV, OUTPUT_DIR, OUTPUT_FIGURE, PROCESSED_DATA_DIR
from src.data_loader import load_market_data
from src.indicators import compute_breadth_indicators
from src.plot import plot_sentiment_vs_index
from src.sentiment import composite_sentiment, merge_with_index


def run_pipeline() -> pd.DataFrame:
    """Load data, compute sentiment, save CSV and figure."""
    market = load_market_data()

    indicators = compute_breadth_indicators(
        close_panel=market["close_panel"],
        valid_panel=market["valid_panel"],
    )
    indicators = indicators.reset_index(names="date")

    sentiment = composite_sentiment(indicators)
    result = merge_with_index(sentiment, market["index"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_CSV, index=False)
    plot_sentiment_vs_index(result, OUTPUT_FIGURE)

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_FIGURE}")
    print(f"Rows: {len(result):,}; plot rows: {result['sentiment_z'].notna().sum():,}")
    return result


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
