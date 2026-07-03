# Electronics Sector Sentiment Index

A market-breadth sentiment indicator for the Shenwan Electronics sector (index **801080**). The index aggregates three breadth signals from constituent stocks, smooths them into a slow-moving sentiment line, and compares the result against the log-transformed Shenwan Electronics Index to highlight overheated and overcooled regimes.

## What This Project Does

The pipeline builds a daily sentiment series for the electronics sector using data from **2008** onward. It:

1. Computes three breadth sub-indicators from constituent stock closes
2. Standardizes each sub-indicator to a 0–100 expanding historical percentile
3. Combines the three percentiles into a single raw sentiment score (weighted)
4. Applies an EMA slow line and converts it to an expanding z-score
5. Labels overheated (`z > +1`) and overcooled (`z < −1`) periods on the sector index chart

The main v1 output is a dual-axis chart: log sector index on the left, sentiment z-score on the right, with the index line colored by regime.

## Three Sentiment Sub-Indicators

All three indicators are computed **cross-sectionally** each day: among stocks that are valid constituents on that date. A stock is *valid* when it belongs to the sector, has `close > 0`, is actively trading (`is_trading == "交易"`), and has at least **120** trading days of price history.

Only **close** prices are used for breadth construction.

### 1. New High / Low Net Ratio (`new_high_low_net`)

**What it measures:** How many stocks are breaking out vs breaking down.

| Item | Definition |
|------|------------|
| New high | Today's close equals the maximum close over the past **60** trading days (inclusive) |
| New low | Today's close equals the minimum close over the past **60** trading days (inclusive) |
| Formula | `(# new highs − # new lows) / valid stocks` |

**Interpretation:** Positive values mean more stocks at 60-day highs than lows → broad participation in an uptrend. Negative values mean more stocks at lows → broad weakness.

### 2. Above MA Ratio (`above_ma`)

**What it measures:** What share of the sector is in an intermediate uptrend.

| Item | Definition |
|------|------------|
| Moving average | Each stock's own **120-day** simple moving average of close |
| Formula | `(# stocks with close > 120-day MA) / valid stocks` |

**Interpretation:** High values mean most constituents are trading above their trend → sector-wide strength. Low values mean most are below trend → sector-wide weakness.

### 3. Positive Return Ratio (`positive_return`)

**What it measures:** Short-term momentum breadth across the sector.

| Item | Definition |
|------|------------|
| Return window | **20-day** simple return: `close / close_20d_ago − 1` |
| Formula | `(# stocks with 20-day return > 0) / valid stocks` |

**Interpretation:** High values mean most stocks have positive recent momentum. Low values mean most have lost ground over the past month.

### From Sub-Indicators to Composite Sentiment

Each raw sub-indicator is mapped to its **expanding historical percentile (0–100)** without look-ahead bias. The three percentiles are combined with configurable weights:

```
sentiment_raw = w1 × new_high_low_pct + w2 × above_ma_pct + w3 × positive_return_pct
```

**v1 defaults:** equal weights `(⅓, ⅓, ⅓)` and `EMA(90)`.

**v2 selected:** `balanced_momentum` weights `(0.25, 0.25, 0.50)` and `EMA(60)` — see `output/v2/best_config.yaml`.

### Slow Line and Z-Score

- **Slow line:** EMA of `sentiment_raw`
- **Z-score:** `(slow line − expanding mean) / expanding standard deviation`

EMA is used instead of SMA at the same window because it achieves similar smoothness with less lag.

### Regime Labels

| Z-score | Regime | Index line color | Trading intuition |
|---------|--------|------------------|-------------------|
| `> +1` | Overheated | Green | Sentiment elevated → consider reducing exposure |
| `< −1` | Overcooled | Red | Sentiment depressed → consider adding exposure |
| otherwise | Neutral | Blue | No extreme signal |

## Data

The pipeline expects the following inputs under `data/raw/`:

| File | Contents |
|------|----------|
| `index_801080.csv` | Sector index daily close |
| `prices_daily_*.csv` | Constituent daily prices (`date`, `symbol`, `close`, `is_trading`, …) |
| `constituents_history.csv` | Constituent entry/exit dates (`symbol`, `in_date`, `out_date`) |
| `trading_calendar.csv` | Trading dates |

Optional for later stages:

| File | Contents |
|------|----------|
| `constituents_electronics.csv` | Current constituent snapshot |
| `risk_free_rate.csv` | CN 10Y yield (v3 backtest) |
| `backtest_config.yaml` | v3 capital and cost assumptions |

All price data should start from **2008-01-01**. Because the longest lookback is 120 trading days, breadth indicators become stable around **mid-2008**.

## Quick Start

```bash
cd electronics-sector-sentiment
pip install -r requirements.txt

# Place CSV files in data/raw/, then:
python3 -m src.main      # v1 baseline
python3 -m src.v2_main   # v2 grid search + diagnostics
```

## Outputs

Results are written under `output/v1/` and `output/v2/`.

### v1 — Baseline Sentiment Index

Run: `python3 -m src.main`

| Path | Description |
|------|-------------|
| `output/v1/sentiment_daily.csv` | Daily breadth raw values, percentiles, `sentiment_raw`, `sentiment_slow`, `sentiment_z`, `regime`, index levels |
| `output/v1/figures/sentiment_vs_index.png` | Dual-axis chart: log index (regime-colored) + sentiment z-score |

Default parameters: equal weights `(⅓, ⅓, ⅓)`, EMA span **90**.

### v2 — Parameter Tuning & Signal Diagnostics

Run: `python3 -m src.v2_main`

Uses a chronological **6-2-2** split (train / valid / final) and **Rank IC** as the primary metric. Grid search over weight presets × EMA `{60, 90, 120}`.

| Path | Description |
|------|-------------|
| `output/v2/grid_results.csv` | All parameter combinations with Train / Valid / Final IC metrics |
| `output/v2/best_config.yaml` | Selected weights and EMA (locked before final evaluation) |
| `output/v2/final_report.txt` | Text summary of splits, IC, and event-study results |
| `output/v2/sentiment_daily_best.csv` | Daily sentiment series using the selected v2 parameters |
| `output/v2/figures/v2_summary.png` | Four-panel overview (start here) |
| `output/v2/figures/ic_by_split.png` | Rank IC on train / valid / final |
| `output/v2/figures/event_study.png` | Mean forward return by regime (overcooled / neutral / overheated) |
| `output/v2/figures/quintile_forward_returns.png` | Mean forward return by sentiment quintile (Q1 low → Q5 high) |
| `output/v2/figures/grid_top_configs.png` | Top configs ranked by valid IC score |
| `output/v2/figures/selection_candidates.png` | Train top-5 pool ranked by valid IC; green = selected |
| `output/v2/figures/sentiment_vs_index_best.png` | Sentiment vs index chart with v2 best parameters |
| `output/v2/figures/v1_v2_zscore_comparison.png` | Overlay of v1 vs v2 z-scores and their difference |

**v2 weight presets** (defined in `src/config.py`):

| Preset | New high/low | Above MA | Momentum | Notes |
|--------|-------------|----------|----------|-------|
| `equal` | 0.33 | 0.33 | 0.33 | v1 default |
| `high_low_heavy` | 0.50 | 0.30 | 0.20 | Emphasizes breakout breadth |
| `ma_heavy` | 0.20 | 0.50 | 0.30 | Emphasizes trend position |
| `momentum_heavy` | 0.20 | 0.30 | 0.50 | Emphasizes 20-day momentum |
| `balanced_trend` | 0.25 | 0.50 | 0.25 | Trend-focused balance |
| `balanced_momentum` | 0.25 | 0.25 | 0.50 | **v2 selected** — equal non-momentum split, momentum half |

## Project Layout

```
electronics-sector-sentiment/
├── README.md
├── requirements.txt
├── data/raw/              # input CSVs
├── data/processed/        # intermediate tables
├── src/
│   ├── main.py            # v1 entry point
│   ├── v2_main.py         # v2 grid search entry point
│   ├── config.py          # paths, defaults, v2 grid
│   ├── data_loader.py
│   ├── indicators.py      # three breadth sub-indicators
│   ├── sentiment.py       # percentile, composite, z-score, regime
│   ├── diagnostics.py     # IC, event study, v2 figures
│   └── plot.py            # sentiment vs index chart
└── output/
    ├── v1/                # v1 CSV + figures
    └── v2/                # v2 CSV, config, report, figures
```

## Development Phases

The project is organized in three stages. **v1 alone satisfies the core internship requirement** (sentiment index construction and visualization). v2 and v3 are extensions.

### v1 — Baseline Sentiment Index *(done)*

Fixed parameters, daily sentiment CSV, and dual-axis chart.

### v2 — Indicator Parameter Tuning *(done)*

Grid search over sub-indicator weights and EMA span. Evaluated via Rank IC (primary), event study, and quintile forward returns on a 6-2-2 chronological split. No trading rules, no PnL.

### v3 — Strategy & Backtest *(planned)*

Layer trading rules on top of the v2 sentiment series (`backtest_config.yaml`). Objectives prioritized: **return → turnover → margin**. Deliverables include a cumulative return / equity curve and out-of-sample performance summary.

## Assumptions

- **Sector:** Shenwan Electronics (801080)
- **Start year:** 2008
- **v1 weights:** Equal `(⅓, ⅓, ⅓)`; **v2 selected:** `(0.25, 0.25, 0.50)`, EMA **60**
- **Percentile & z-score windows:** Expanding (no future data)
- **Constituent membership:** Historical changes modeled via `constituents_history.csv`

## Status

**v1** and **v2** pipelines are implemented and runnable. **v3** backtest is planned.

## License

Internship research project. All rights reserved.
