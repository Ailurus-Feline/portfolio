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
2. Construct factors (momentum, volume-price, range position, RSI).
3. Compute IC table:
	- Pearson IC
	- Clipped Pearson IC
	- Spearman IC
4. Evaluate rolling IC stability.
5. Monetize via historical quantiles (80/20 rule by default):
	- Thresholds use only historical factor values (`shift(1)` + rolling quantiles).
	- PnL includes turnover-based fee deduction.
6. Export metrics and sensitivity outputs for notebook/report conclusions.

## Design Guarantees
- No look-ahead in signal/threshold logic.
- Historical-only quantile thresholds.
- Fee-adjusted PnL.
- IC + rolling IC + monetization + metrics + sensitivity in one run.

## Outputs

### Data Artifacts
Saved under [crypto-cta-project/data/clean](crypto-cta-project/data/clean):
- `class2_ic_table.csv`

### Report Artifacts
Saved under [crypto-cta-project/results](crypto-cta-project/results):
- `class2_backtest_metrics.csv`
- `class2_sensitivity.csv`
- `class2_quantile_bt_core.csv`
- `ma_strategy_summary.csv`
- `<SYMBOL>_1h_ma_backtest.csv`
- `window_sweep.csv`
- `long_only_vs_long_short.csv`
- `fee_sensitivity.csv`
- `extended_symbol_summary.csv`

## Running
From project root:

```bash
python crypto_cta_strategy.py
```

If exchange access is blocked, the script automatically falls back to deterministic demo OHLCV so the full workflow still completes.

## Notes
- This is a research/teaching framework, not a production trading engine.
- Prioritize correctness (timestamp handling, lagging, fee modeling) before profitability.
