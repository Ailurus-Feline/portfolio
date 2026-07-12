# Crypto CTA Strategy

Single-script pipeline for Class 1 (trend), Class 2 (factor research), and Class 3 (model combination).

Entry point: [crypto_cta_strategy.py](crypto_cta_strategy.py)

## Scope

- Market: spot crypto
- Symbols: BTC/USDT, ETH/USDT (+ SOL/USDT in scenarios)
- Bar frequency: 1h

## Workflows

### Class 1 — Trend baseline

Download/clean OHLCV → MA signal (one-bar lag) → backtest → scenario analyses (window sweep, long-only vs long-short, fee sensitivity).

### Class 2 — Factor research

Build alpha matrix → IC table → rolling IC → quantile monetization → Top-N factor batch analysis.

### Class 3 — Model combination

Combine Class-2 Top factors (equal-weight / IC-weight / linear / ridge / tree) → unified `signal_backtest` → train/valid/test metrics → calendar returns (month/quarter/year) → sensitivity grid for all methods.

## Output layout

```
results/
  class1_trend/
    csv/       # MA backtests, scenario tables, cleaned symbol exports
    figures/   # baseline price/MA and equity plots
  class2_factor/
    csv/       # factor dataset, IC tables, quantile backtests, Top-N tables
    figures/   # IC, rolling IC, equity, sensitivity plots
  class3_combo/
    csv/       # summary, period metrics, sensitivity, calendar returns
    csv/backtests/   # per-horizon/per-method signal backtest series
    figures/   # combo equity curves and sensitivity heatmaps
data/
  raw/         # downloaded or demo OHLCV
  clean/       # parquet clean bars
```

## Running

```bash
python crypto_cta_strategy.py
```

If exchange access is blocked, the script falls back to deterministic demo OHLCV.

## Notes

- Research/teaching framework, not a production trading engine.
- Train/valid/test split: fit on train, diagnose on valid (sensitivity), evaluate once on test.
