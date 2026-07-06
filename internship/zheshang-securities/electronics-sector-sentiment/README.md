# Electronics Sector Sentiment Index

Market-breadth sentiment indicator and backtest for the Shenwan Electronics sector (index **801080**). The project builds sentiment signals from constituent stocks, tunes parameters via Rank IC, backtests long-only trading rules, and tests multi-alpha combinations.

**Final deployable strategy:** see [`FINAL_STRATEGY.yaml`](FINAL_STRATEGY.yaml) — single-alpha `advance_decline` with optimized thresholds (+1297% full sample, +707% excess vs buy & hold).

---

## Quick Start

```bash
cd electronics-sector-sentiment
pip install -r requirements.txt

# Place CSV files in data/raw/, then run in order:
python3 -m src.main              # v1 — baseline sentiment index
python3 -m src.v2_main           # v2 — parameter tuning (Rank IC)
python3 -m src.v3_main           # v3 — backtest + champion
python3 -m src.v4_main           # v4 — alpha combination (optional)

# Optional: v3 parameter grid (4810 configs, ~7 min)
python3 -m src.v3_optimize_main
```

Raw data is not included. Required inputs under `data/raw/`: `index_801080.csv`, `prices_daily_*.csv`, `constituents_history.csv`, `trading_calendar.csv`, `backtest_config.yaml`.

---

## Final Trading Strategy

| Item | Value |
|------|-------|
| **Signal** | `advance_decline` — (advancers − decliners) / valid stocks |
| **Smoothing** | Expanding percentile → EMA(60) → expanding z-score |
| **Buy** | z < **−1.0** → 100% long |
| **Sell** | z > **+1.5** → cash (0%) |
| **Between** | Maintain position (hold after buy until overheated) |
| **Execution** | T-day close signal → T+1 position |
| **Direction** | Long-only on index 801080 |

**Backtest (with costs):** full sample +1297% (−60% max DD, 23 trades); final 20% holdout +271% (+28.5% vs B&H). PnL chart: `output/final_pnl_vs_benchmark.png`.

v4 tested equal-weight, IC-weighted, Ridge, and Decision Tree combinations — **none beat this strategy**. Artifacts: `output/v3/champion/`.

---

## Pipeline Overview

```
v1  Sentiment index construction + chart
      ↓
v2  Weight / EMA grid search (Rank IC, no PnL)
      ↓
v3  Alpha backtest → champion (advance_decline)
      ↓
v4  Multi-alpha combination → confirms v3 champion
```

| Phase | Purpose | Entry point |
|-------|---------|-------------|
| v1 | Baseline composite sentiment | `src.main` |
| v2 | Tune indicator weights & EMA | `src.v2_main` |
| v3 | Backtest alphas, lock champion | `src.v3_main` |
| v4 | Combine alphas (PPT methods) | `src.v4_main` |

Evaluation uses chronological **6-2-2** splits (train 60% / valid 20% / final 20%). v2 selects on valid IC; v3/v4 select trading thresholds on valid; final split evaluated once.

---

## Output Directory Layout

All outputs are regenerable (gitignored except `.gitkeep`).

```
output/
├── v1/                              # Baseline sentiment index
│   ├── sentiment_daily.csv
│   └── figures/sentiment_vs_index.png
│
├── v2/                              # Parameter tuning
│   ├── grid_results.csv
│   ├── best_config.yaml
│   ├── final_report.txt
│   ├── sentiment_daily_best.csv
│   └── figures/                     # IC, event study, grid charts
│
├── v3/                              # Strategy backtest
│   ├── best_config.yaml             # Champion parameters
│   ├── certified_alphas.yaml        # Alpha certification for v4
│   ├── final_report.txt
│   ├── champion/                    # ★ Deployable strategy artifacts
│   │   ├── backtest_daily.csv
│   │   ├── performance_summary.csv
│   │   ├── report.txt
│   │   └── figures/
│   │       ├── equity_curve.png     # Cumulative return (%)
│   │       └── pnl_vs_benchmark.png # NAV vs buy & hold
│   ├── exploratory/                 # 6 alphas × 3 rules (default ±1)
│   │   ├── matrix_results.csv
│   │   └── alphas/{alpha}/{rule}/
│   └── optimization/                # Optional grid (v3_optimize_main)
│       ├── grid_results.csv
│       ├── top_configs.csv
│       ├── report.txt
│       └── figures/
│
└── v4/                              # Alpha combination
    ├── alpha_matrix.csv
    ├── combination_results.csv
    ├── best_config.yaml             # Reference only (v3 wins)
    ├── final_report.txt
    ├── methods/{method}/{config}/   # Best per method
    └── sensitivity/grid_results.csv
```

---

## Sentiment Sub-Indicators

All breadth metrics are computed **cross-sectionally** each day among valid constituents (`close > 0`, actively trading, ≥120 days history). Only **close** prices are used.

### 1. New High / Low Net (`new_high_low_net`)

60-day new highs minus new lows, divided by valid stocks.

### 2. Above MA Ratio (`above_ma`)

Share of stocks with `close > 120-day MA`.

### 3. Positive Return Ratio (`positive_return`)

Share of stocks with positive 20-day return.

### 4. Advance / Decline (`advance_decline`) — **final signal**

`(advancers − decliners) / valid stocks` on each day. Used as the deployable trading alpha in v3.

### Composite Sentiment (v1/v2)

```
sentiment_raw = w1·new_high_low_pct + w2·above_ma_pct + w3·positive_return_pct
```

| Version | Weights | EMA |
|---------|---------|-----|
| v1 default | (⅓, ⅓, ⅓) | 90 |
| **v2 selected** | (0.25, 0.25, 0.50) `balanced_momentum` | 60 |

Z-score: expanding mean/std of EMA-smoothed percentile. Regimes: overheated z>+1, overcooled z<−1.

---

## v2 Weight Presets

| Preset | New high/low | Above MA | Momentum |
|--------|-------------|----------|----------|
| `equal` | 0.33 | 0.33 | 0.33 |
| `balanced_momentum` | 0.25 | 0.25 | 0.50 | **selected** |
| `ma_heavy` | 0.20 | 0.50 | 0.30 |
| `momentum_heavy` | 0.20 | 0.30 | 0.50 |

---

## Project Layout

```
electronics-sector-sentiment/
├── FINAL_STRATEGY.yaml          # Locked deployable strategy
├── README.md
├── requirements.txt
├── data/raw/                    # Input CSVs (gitignored)
├── src/
│   ├── main.py                  # v1
│   ├── v2_main.py               # v2
│   ├── v3_main.py               # v3 matrix + champion
│   ├── v3_optimize_main.py      # v3 parameter grid
│   ├── v4_main.py               # v4 combination
│   ├── alpha_signals.py         # v3 alpha panels
│   ├── alpha_combination.py     # v4 combination methods
│   ├── backtest.py              # Long-only backtest engine
│   ├── strategy_rules.py        # Position rules
│   ├── indicators.py            # Breadth indicators
│   ├── sentiment.py             # Composite, z-score, regime
│   ├── diagnostics.py           # IC, event study
│   ├── data_loader.py
│   ├── config.py
│   └── plot.py
└── output/                      # v1–v4 results (gitignored)
```

---

## Assumptions

- **Sector:** Shenwan Electronics (801080), data from **2008**
- **No look-ahead:** expanding percentiles and z-scores
- **Costs:** 1M capital, 2 bp commission/side, 0.1% stamp tax on sells (`backtest_config.yaml`)
- **Membership:** historical constituent changes via `constituents_history.csv`

## Status

v1–v4 pipelines implemented and finalized. Deploy **v3 champion** (`advance_decline`, buy z<−1, sell z>+1.5).

## License

Internship research project. All rights reserved.
