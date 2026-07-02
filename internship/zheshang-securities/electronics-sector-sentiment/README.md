# Electronics Sector Sentiment Index

A quantitative sentiment indicator for the Shenwan **Electronics** sector (index code **801080**), built from market-breadth signals across constituent stocks and visualized against the log-transformed sector index.

## Overview

This project constructs a composite sentiment score from three breadth-based sub-indicators, smooths it with an EMA, converts it to a z-score, and maps overheated/overcooled regimes onto the sector index price line. An optional anomaly layer flags limit-up events among the top market-cap constituents.

**Universe:** Shenwan Electronics constituent stocks (count varies with index rebalancing; ~300+ in recent years)  
**History:** Daily closes from **2008** onward  
**Benchmark:** Shenwan Electronics Index (801080)

## Methodology

### 1. Breadth Sub-Indicators

Three daily market-width metrics measure internal sector strength:

| Indicator | Definition |
|-----------|------------|
| **New High / Low Net Ratio** | `(# stocks at 60-day high − # at 60-day low) / valid stocks` |
| **Above MA Ratio** | `% of stocks with close > own 120-day moving average` |
| **Up Probability** | `% of stocks with positive 20-day return` |

"Valid stocks" on a given day are constituents with a closing price and sufficient history for the longest lookback window (120 days).

### 2. Standardization & Composite

Each sub-indicator is converted to a **historical percentile rank (0–100)** using an expanding window (no look-ahead bias). The three ranks are combined with **equal weights (⅓ each)** to form the raw sentiment value.

### 3. Smoothing & Z-Score

- **Slow line:** EMA(90) on the raw sentiment series. EMA is preferred over SMA at the same window because it lags less while preserving similar smoothness.
- **Z-score:** Deviation of the slow line from its own expanding historical mean, in units of expanding historical standard deviation.

### 4. Regime Labels

| Condition | Label | Index line color |
|-----------|-------|------------------|
| `z > +1` | Overheated | Green |
| `z < −1` | Overcooled | Red |
| otherwise | Neutral | Blue |

Regime labels are projected onto the **log-transformed** Shenwan Electronics Index (`ln(close)`) for visual comparison.

### 5. Anomaly Detection (Optional Overlay)

In addition to the sentiment signal, flag **anomaly days** when any of the **top 3 stocks by total market cap** hits a limit-up (涨停):

- Main board: ~10% daily limit
- STAR Market / ChiNext (e.g. 688256 Cambricon): ~20% daily limit

Board-specific thresholds are applied when classifying limit-up events.

## Project Structure

```
electronics-sector-sentiment/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # Source data
│   └── processed/        # Intermediate outputs
├── src/
│   ├── data_loader.py    # Load & clean price / market-cap data
│   ├── indicators.py     # Three breadth sub-indicators
│   ├── sentiment.py      # Percentile ranks, composite, EMA, z-score
│   ├── anomaly.py        # Top-3 market-cap limit-up detection
│   ├── plot.py           # Dual-axis chart with regime coloring
│   └── main.py           # End-to-end pipeline entry point
└── output/
    ├── figures/          # Generated charts
    └── sentiment_daily.csv
```

## Data Requirements

### Required datasets

| Dataset | Fields | Notes |
|---------|--------|-------|
| **Constituent daily prices** | `date`, `symbol`, `close` | All Shenwan Electronics constituents; from 2008-01-01 |
| **Sector index** | `date`, `close` | 801080 (Shenwan Electronics Index) |
| **Market capitalization** | `date`, `symbol`, `market_cap` | Total market cap; for daily top-3 ranking and anomaly detection |
| **Stock metadata** | `symbol`, `board` | Main board / STAR / ChiNext — required for limit-up thresholds (10% vs 20%) |

### Strongly recommended

| Dataset | Fields | Why |
|---------|--------|-----|
| **Daily returns or prev close** | `date`, `symbol`, `pct_chg` or `pre_close` | Simplifies limit-up detection; can be derived from `close` if absent |
| **Suspension / tradability flag** | `date`, `symbol`, `is_trading` | Exclude suspended names from "valid stocks" |
| **Historical constituent membership** | `symbol`, `in_date`, `out_date` | Reduces survivorship bias when stocks enter or leave the index |

### Derived in pipeline (no extra download if prices are available)

- 60-day high / low flags  
- 120-day moving average  
- 20-day return sign  
- Expanding percentile ranks, EMA(90), z-score  

### Suggested raw file layout

```
data/raw/
├── index_801080.csv               # sector index daily close
├── constituents_electronics.csv   # symbol, name, board
├── prices_daily.csv               # date, symbol, close [, pre_close, pct_chg]
├── market_cap_daily.csv           # date, symbol, market_cap
└── constituents_history.csv       # optional: symbol, in_date, out_date
```

Raw data files are excluded from version control. See `.gitignore`.

### Effective start date

With a 120-day lookback, breadth indicators become meaningful roughly **120 trading days after 2008-01-01** (~mid-2008). Index and z-score series follow the same warmup logic.

## Setup

```bash
cd electronics-sector-sentiment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Place raw data under data/raw/, then run the full pipeline
python -m src.main
```

Outputs:

- `output/sentiment_daily.csv` — daily sentiment, z-score, and regime labels
- `output/figures/sentiment_vs_index.png` — dual-axis chart

## Output Chart

The main chart has:

- **Left Y-axis:** `ln(Shenwan Electronics Index)`
- **Right Y-axis:** Sentiment slow-line z-score (standard deviations)
- **Black line:** Z-score with reference lines at 0, ±1
- **Colored index line:** Blue (neutral), green (overheated), red (overcooled)
- **Anomaly markers (optional):** Highlight days when a top-3 market-cap stock limit-ups

## Design Decisions

| Topic | Default choice | Rationale |
|-------|----------------|-----------|
| Target sector | Electronics (801080) | Confirmed project scope |
| History start | 2008 | Confirmed project scope |
| Sub-indicator weights | Equal (⅓ each) | Spec requirement |
| Percentile window | Expanding | No future data; full history context |
| Z-score window | Expanding mean & std | Consistent with percentile logic |
| Constituent changes | Static current list (v1) | Simpler; survivorship bias acknowledged |
| Smoothing | EMA(90) | Lower lag than SMA(90) at similar smoothness |

## Roadmap

- [x] Phase 0: Repo setup (`requirements.txt`, `.gitignore`, directory layout)
- [ ] Phase 1: Data ingestion & cleaning (Electronics, from 2008)
- [ ] Phase 2: Breadth sub-indicators
- [ ] Phase 3: Percentile standardization & composite
- [ ] Phase 4: EMA smoothing & z-score
- [ ] Phase 5: Anomaly detection (top-3 limit-up)
- [ ] Phase 6: Visualization
- [ ] Phase 7: End-to-end pipeline & GitHub publish

## License

Personal / internship project. No long-term maintenance planned.
