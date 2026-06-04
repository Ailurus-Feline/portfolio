# Crypto CTA Strategy

This project implements a compact end-to-end crypto CTA research workflow based on moving-average trend following.

## Strategy Scope
- Market: spot crypto
- Primary symbols: BTC/USDT, ETH/USDT
- Extended symbol scenario: SOL/USDT
- Bar frequency: 1h
- Signal family: dual moving-average trend following

## Research Pipeline
1. Download OHLCV bars from exchange via CCXT.
2. Fall back to realistic demo data when download is unavailable.
3. Clean and validate data (sort, deduplicate, numeric conversion, missing-bar check).
4. Build MA-based trading signals with one-bar lag to avoid look-ahead bias.
5. Run backtests with turnover-based transaction costs.
6. Compute core metrics and export results.
7. Run scenario analyses for robustness checks.

## Core Components
- Exchange/data access: `make_exchange`, `fetch_ohlcv_loop`, `download_or_demo`
- Data quality: `clean_ohlcv`, `data_quality_report`
- Signal/backtest: `add_ma_signal`, `add_ma_signal_long_only`, `backtest_signal`
- Evaluation: `performance_summary`, `max_drawdown`
- Visualization: `plot_price_and_mas`, `plot_equity`
- Workflow entry points: `run_baseline_workflow`, `run_strategy_scenarios`, `main`

## Scenario Analyses Included
- Moving-average window sweep across multiple fast/slow pairs
- Long-short versus long-only signal design
- Transaction cost sensitivity across fee levels
- Multi-symbol rerun with an additional asset

## Output Artifacts
Generated files are saved under `results/`:
- `ma_strategy_summary.csv`
- `<SYMBOL>_1h_ma_backtest.csv`
- `window_sweep.csv`
- `long_only_vs_long_short.csv`
- `fee_sensitivity.csv`
- `extended_symbol_summary.csv`

## Running
From the project root:

```bash
python crypto_cta_strategy.py
```

If your environment does not have exchange access, the script automatically switches to demo OHLCV data so the full pipeline can still run.

## Notes
- This is a research baseline, not a production trading system.
- Prioritize experiment correctness (data integrity, lagging, and cost modeling) before focusing on profitability.
