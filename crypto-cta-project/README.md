# Crypto CTA Strategy

This project uses a single reusable script to cover both trend-following and
factor-research workflows.

The main entry point is [crypto-cta-project/crypto_cta_strategy.py](crypto-cta-project/crypto_cta_strategy.py).

## Scope

- Market: spot crypto.
- Primary symbols: BTC/USDT, ETH/USDT.
- Extended symbol scenario: SOL/USDT.
- Bar frequency: 1h.

## Trend Workflow

1. Download OHLCV via CCXT (demo fallback when unavailable).
2. Clean/validate bars (sort, deduplicate, numeric conversion, missing-bar check).
3. Build MA signals with one-bar lag (anti-look-ahead).
4. Backtest with turnover-based transaction costs.
5. Run scenario analyses (window sweep, long-only vs long-short, fee sensitivity, extended symbols).

## Factor Workflow

1. Build future return label: `future_ret_1h`.
2. Construct factors:
   - Momentum: 4w, 12w, 26w, 52w
   - Reversal: 1w, 2w
   - Volatility/autocorrelation: 4w, 12w, 26w, vol-change, return-autocorr
   - Technical factors: volume-price trend, range position, RSI, Bollinger position, MACD signal spread, distance-to-MA, MFI, STC, DMI/ADX
3. Compute IC table:
   - Pearson IC
   - Clipped Pearson IC
   - Spearman IC
4. Evaluate rolling IC stability.
5. Monetize via historical quantiles (80/20 rule by default):
   - Thresholds use only historical factor values (`shift(1)` + rolling quantiles).
   - PnL includes turnover-based fee deduction.
6. Export baseline selected-factor metrics and sensitivity outputs.
7. Batch analyze Top-N factors (default Top-3 and Top-5 subsets from ranking):
   - Per-factor rolling IC, equity, and sensitivity plots
   - Cross-factor comparison plots and consolidated tables

## Design Guarantees

- No look-ahead in signal/threshold logic.
- Historical-only quantile thresholds.
- Fee-adjusted PnL.
- IC + rolling IC + monetization + metrics + sensitivity in one run.

## Requirement Coverage

- Workflow A (trend baseline): data acquisition, cleaning, MA signal, backtest, metrics, multi-symbol summary, and result persistence.
- Workflow A extensions: MA window sweep, long-only vs long-short, fee sensitivity, and symbol extension.
- Workflow B (factor research): target construction, multi-factor IC test, rolling IC stability, quantile monetization, metrics, and sensitivity test.
- Workflow B Top-N extension: each selected factor has rolling IC/equity/sensitivity outputs and comparison plots.

## Outputs

### CSV Artifacts

Saved under [crypto-cta-project/results/csv](crypto-cta-project/results/csv):

- `<SYMBOL>_1h.csv`
- `<SYMBOL>_1h_ma_backtest.csv`
- `ma_strategy_summary.csv`
- `window_sweep.csv`
- `long_only_vs_long_short.csv`
- `fee_sensitivity.csv`
- `extended_symbol_summary.csv`
- `factor_dataset.csv`
- `factor_rolling_ic.csv`
- `factor_ic_table.csv`
- `factor_selection.csv`
- `factor_backtest_metrics.csv`
- `factor_sensitivity.csv`
- `factor_quantile_bt_core.csv`
- `factor_quantile_bt_full.csv`
- `factor_top_factors.csv`
- `factor_top3_factors.csv`
- `factor_top5_factors.csv`
- `factor_top_rolling_ic.csv`
- `factor_top_metrics.csv`
- `factor_top_sensitivity.csv`
- `factor_top_equity_curves.csv`

### Parquet Artifacts

Saved under [crypto-cta-project/data/clean](crypto-cta-project/data/clean):

- `<SYMBOL>_1h.parquet`

### Raw Data Artifacts

Saved under [crypto-cta-project/data/raw](crypto-cta-project/data/raw):

- `<SYMBOL>_1h_raw.csv` (downloaded exchange data)
- `<SYMBOL>_1h_raw_demo.csv` (fallback synthetic data)

### Figure Artifacts

Saved under [crypto-cta-project/results/figures](crypto-cta-project/results/figures):

- `baseline_btc_price_ma.png`
- `baseline_btc_equity.png`
- `factor_ic_comparison.png`
- `factor_rolling_ic.png`
- `factor_equity_curve.png`
- `factor_sensitivity_sharpe.png`
- `factor_top_ranked_ic.png`
- `factor_top_rolling_ic_compare.png`
- `factor_top_equity_compare.png`
- `factor_top_sensitivity_compare.png`
- `factor_top_<factor>_rolling_ic.png`
- `factor_top_<factor>_equity_curve.png`
- `factor_top_<factor>_sensitivity_sharpe.png`

## Running

From project root:

```bash
python crypto_cta_strategy.py
```

If exchange access is blocked, the script automatically falls back to deterministic demo OHLCV so the full workflow still completes.

## Notes

- This is a research/teaching framework, not a production trading engine.
- Prioritize correctness (timestamp handling, lagging, fee modeling) before profitability.
