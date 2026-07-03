"""Project constants for the v1 sentiment pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

INDEX_FILE = RAW_DATA_DIR / "index_801080.csv"
HISTORY_FILE = RAW_DATA_DIR / "constituents_history.csv"
CALENDAR_FILE = RAW_DATA_DIR / "trading_calendar.csv"
PRICE_FILE_PATTERN = "prices_daily_*.csv"

# Breadth indicator lookbacks (trading days).
ROLLING_HIGH_LOW = 60
ROLLING_MA = 120
ROLLING_RETURN = 20
MIN_HISTORY_DAYS = 120

# Sentiment construction.
EMA_SPAN = 90
SUB_INDICATOR_WEIGHT = 1 / 3
Z_OVERHEAT = 1.0
Z_OVERCOOL = -1.0

TRADING_STATUS_OK = "交易"

OUTPUT_CSV = OUTPUT_DIR / "sentiment_daily.csv"
OUTPUT_FIGURE = FIGURES_DIR / "sentiment_vs_index.png"
