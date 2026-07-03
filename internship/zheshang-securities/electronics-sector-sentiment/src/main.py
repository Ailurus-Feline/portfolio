"""Entry point for the v1 electronics sector sentiment pipeline."""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR, V1_FIGURES_DIR, V1_OUTPUT_CSV, V1_OUTPUT_DIR, V1_OUTPUT_FIGURE
from src.data_loader import load_market_data
from src.indicators import compute_breadth_indicators
from src.plot import plot_sentiment_vs_index
from src.sentiment import composite_sentiment, merge_with_index, select_analysis_rows


def run_pipeline() -> pd.DataFrame:
    """Load data, compute sentiment, save CSV and figure."""
    market = load_market_data()

    indicators = compute_breadth_indicators(
        close_panel=market["close_panel"],
        valid_panel=market["valid_panel"],
    ).reset_index(names="date")

    sentiment = composite_sentiment(indicators)
    result = select_analysis_rows(merge_with_index(sentiment, market["index"]))

    V1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    V1_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(V1_OUTPUT_CSV, index=False)
    plot_sentiment_vs_index(result, V1_OUTPUT_FIGURE)

    print(f"Wrote {V1_OUTPUT_CSV}")
    print(f"Wrote {V1_OUTPUT_FIGURE}")
    print(f"Rows: {len(result):,}")
    return result


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
