"""Project constants for the sentiment pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

V1_OUTPUT_DIR = OUTPUT_DIR / "v1"
V1_FIGURES_DIR = V1_OUTPUT_DIR / "figures"
V2_OUTPUT_DIR = OUTPUT_DIR / "v2"
V2_FIGURES_DIR = V2_OUTPUT_DIR / "figures"

INDEX_FILE = RAW_DATA_DIR / "index_801080.csv"
HISTORY_FILE = RAW_DATA_DIR / "constituents_history.csv"
CALENDAR_FILE = RAW_DATA_DIR / "trading_calendar.csv"
PRICE_FILE_PATTERN = "prices_daily_*.csv"

# Breadth indicator lookbacks (trading days).
ROLLING_HIGH_LOW = 60
ROLLING_MA = 120
ROLLING_RETURN = 20
MIN_HISTORY_DAYS = 120

# Sentiment construction (v1 defaults).
EMA_SPAN = 90
DEFAULT_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
Z_OVERHEAT = 1.0
Z_OVERCOOL = -1.0

TRADING_STATUS_OK = "交易"

# v1 deliverables.
V1_OUTPUT_CSV = V1_OUTPUT_DIR / "sentiment_daily.csv"
V1_OUTPUT_FIGURE = V1_FIGURES_DIR / "sentiment_vs_index.png"

# v2 grid search and diagnostics.
V2_GRID_RESULTS = V2_OUTPUT_DIR / "grid_results.csv"
V2_BEST_CONFIG = V2_OUTPUT_DIR / "best_config.yaml"
V2_FINAL_REPORT = V2_OUTPUT_DIR / "final_report.txt"
V2_BEST_SENTIMENT_CSV = V2_OUTPUT_DIR / "sentiment_daily_best.csv"
V2_IC_BY_SPLIT_FIGURE = V2_FIGURES_DIR / "ic_by_split.png"
V2_GRID_TOP_FIGURE = V2_FIGURES_DIR / "grid_top_configs.png"
V2_SELECTION_FIGURE = V2_FIGURES_DIR / "selection_candidates.png"
V2_EVENT_STUDY_FIGURE = V2_FIGURES_DIR / "event_study.png"
V2_QUINTILE_FIGURE = V2_FIGURES_DIR / "quintile_forward_returns.png"
V2_BEST_SENTIMENT_FIGURE = V2_FIGURES_DIR / "sentiment_vs_index_best.png"
V2_V1_V2_COMPARISON_FIGURE = V2_FIGURES_DIR / "v1_v2_zscore_comparison.png"
V2_SUMMARY_FIGURE = V2_FIGURES_DIR / "v2_summary.png"

TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
FINAL_RATIO = 0.2

IC_HORIZONS = (20, 60)
IC_PRIMARY_HORIZON = 20
IC_SECONDARY_HORIZON = 60
IC_SCORE_WEIGHTS = {20: 0.6, 60: 0.4}

EMA_GRID = (60, 90, 120)
WEIGHT_PRESETS: dict[str, tuple[float, float, float]] = {
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "high_low_heavy": (0.5, 0.3, 0.2),
    "ma_heavy": (0.2, 0.5, 0.3),
    "momentum_heavy": (0.2, 0.3, 0.5),
    "balanced_trend": (0.25, 0.5, 0.25),
    "balanced_momentum": (0.25, 0.25, 0.5),
}

TOP_K_TRAIN = 5
GRID_TOP_N = 10
