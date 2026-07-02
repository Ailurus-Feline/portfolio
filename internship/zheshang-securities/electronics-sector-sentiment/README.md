# Electronics Sector Sentiment Index

A market-breadth sentiment indicator for the Shenwan Electronics sector (index **801080**). The index aggregates three breadth signals from constituent stocks, smooths them into a slow-moving sentiment line, and compares the result against the log-transformed Shenwan Electronics Index to highlight overheated and overcooled regimes.

## What This Project Does

The pipeline builds a daily sentiment series for the electronics sector using data from **2008** onward. It:

1. Computes three breadth sub-indicators from constituent stock prices
2. Standardizes each sub-indicator to a 0–100 historical percentile
3. Combines them into a single raw sentiment score (equal weights)
4. Applies an EMA(90) slow line and converts it to a z-score
5. Labels overheated (`z > +1`) and overcooled (`z < −1`) periods on the sector index chart
6. Optionally flags anomaly days when a top-3 market-cap constituent hits limit-up

The main output is a dual-axis chart: log sector index on the left, sentiment z-score on the right, with the index line colored by regime.

## Methodology

### Breadth Sub-Indicators

| Indicator | Formula |
|-----------|---------|
| **New High / Low Net Ratio** | `(# at 60-day high − # at 60-day low) / valid stocks` |
| **Above MA Ratio** | `% with close above own 120-day moving average` |
| **Up Probability** | `% with positive 20-day return` |

A stock counts as *valid* on a given day if it has a closing price and at least 120 days of price history.

### Composite Sentiment

Each sub-indicator is mapped to its **expanding historical percentile (0–100)** without look-ahead bias. The three percentiles are averaged with **equal weight (⅓ each)** to produce the raw sentiment value.

### Slow Line and Z-Score

- **Slow line:** EMA(90) of the raw sentiment series
- **Z-score:** `(slow line − expanding mean) / expanding standard deviation`

EMA is used instead of SMA at the same window because it achieves similar smoothness with less lag.

### Regime Labels

| Z-score | Regime | Index line color |
|---------|--------|------------------|
| `> +1` | Overheated | Green |
| `< −1` | Overcooled | Red |
| otherwise | Neutral | Blue |

### Anomaly Overlay

On any day when one of the **top 3 constituents by total market cap** hits limit-up:

| Board | Limit-up threshold |
|-------|-------------------|
| Main board | ~10% |
| STAR Market / ChiNext | ~20% |

## Data

The pipeline expects the following inputs under `data/raw/`:

| File | Contents |
|------|----------|
| `index_801080.csv` | Sector index daily close (`date`, `close`) |
| `prices_daily.csv` | Constituent daily close (`date`, `symbol`, `close`) |
| `market_cap_daily.csv` | Daily total market cap (`date`, `symbol`, `market_cap`) |
| `constituents_electronics.csv` | Stock list with board type (`symbol`, `name`, `board`) |

Optional but useful:

| File | Contents |
|------|----------|
| `constituents_history.csv` | Constituent entry/exit dates (`symbol`, `in_date`, `out_date`) |

All price data should start from **2008-01-01**. Because the longest lookback is 120 trading days, breadth indicators become stable around **mid-2008**.

Raw data is not included in this repository.

## Quick Start

```bash
cd electronics-sector-sentiment
pip install -r requirements.txt

# Place CSV files in data/raw/, then:
python -m src.main
```

### Outputs

| Path | Description |
|------|-------------|
| `output/sentiment_daily.csv` | Daily sentiment, z-score, and regime labels |
| `output/figures/sentiment_vs_index.png` | Dual-axis chart |

## Project Layout

```
electronics-sector-sentiment/
├── README.md
├── requirements.txt
├── data/raw/           # input CSVs
├── data/processed/     # intermediate tables
├── src/                # pipeline modules
└── output/             # results and figures
```

## Development Phases

The project is organized in three stages. **v1 alone satisfies the core internship requirement** (sentiment index construction and visualization). v2 and v3 are extensions.

### v1 — Baseline Sentiment Index *(current focus)*

Build the daily sentiment indicator with fixed parameters and produce the main deliverables:

- Equal weights (⅓ each) across the three breadth sub-indicators
- EMA(90) slow line and expanding-window z-score
- Dual-axis chart (log index + sentiment z-score, regime-colored index line)
- Optional limit-up anomaly overlay for top-3 market-cap constituents
- `sentiment_daily.csv` output

### v2 — Indicator Parameter Tuning

Optimize **indicator construction** parameters only — no trading rules, no PnL. Candidate knobs include sub-indicator weights and EMA span (grid search). Parameter quality is evaluated via signal diagnostics such as forward-return event studies and information coefficient (IC), with out-of-sample validation to limit overfitting.

### v3 — Strategy & Backtest

Layer trading rules on top of the v2 sentiment series. Grid search over strategy parameters with objectives prioritized in order: **return → turnover → margin**. Deliverables include a PnL / equity curve and out-of-sample performance summary.

## Assumptions (v1 defaults)

- **Sector:** Shenwan Electronics (801080)
- **Start year:** 2008
- **Sub-indicator weights:** Equal (⅓ each)
- **EMA span:** 90
- **Percentile & z-score windows:** Expanding (no future data)
- **Constituent list:** Static snapshot; historical index changes are not yet modeled

## Status

**v1** is in progress. Core pipeline modules (`src/`) are under development. The repository currently contains project documentation, dependencies, and directory scaffolding.

## License

Internship research project. All rights reserved.
