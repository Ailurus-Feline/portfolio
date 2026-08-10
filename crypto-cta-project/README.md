# Crypto CTA Strategy

Single-script pipeline for Class 1–4:

1. Trend baseline
2. Factor research
3. Model combination
4. Risk layer (TP/SL + multi-asset allocation)

Entry point: [crypto_cta_strategy.py](crypto_cta_strategy.py)

## Scope

- Market: spot crypto
- Symbols: BTC/USDT, ETH/USDT (+ SOL/USDT in Class 1 scenarios)
- Bar frequency: 1h

## Workflows

### Class 1 — Trend baseline

Download/clean OHLCV → MA signal (one-bar lag) → backtest → scenario analyses (window sweep, long-only vs long-short, fee sensitivity).

### Class 2 — Factor research

Build alpha matrix → IC table → rolling IC → quantile monetization → Top-N factor batch analysis.

### Class 3 — Model combination

Combine Class-2 Top factors (equal-weight / IC-weight / linear / ridge / tree) → unified `signal_backtest` → train/valid/test metrics → calendar returns (month/quarter/year) → sensitivity grid for all methods.

### Class 4 — Risk layer

Build on Class-3 best combo (`ridge` / `1h` by default):

1. **Exit rules** on OHLC bars
   - `none`: model position only
   - `fixed`: +2% TP / -1% SL
   - `atr`: ATR(24) × multipliers
   - `time`: force flat after 24 bars
   - `trailing`: 10% trailing stop from running extreme
   - Same-bar TP+SL ambiguity: **SL first** (conservative)
2. Compare with/without TP/SL metrics on train/valid/test
3. **Multi-asset risk allocation** on BTC + ETH
   - equal capital
   - Sharpe weighting
   - risk target (`1/vol`)
   - mean-variance optimization (MVO)
   - Weights fit on train sleeve returns only; test is evaluation-only

## Output layout

```
results/
  class1_trend/
    csv/
    figures/
  class2_factor/
    csv/
    figures/
  class3_combo/
    csv/
    csv/backtests/
    figures/
  class4_risk/
    csv/       # exit metrics, portfolio weights/metrics, per-mode backtests, summary
    figures/   # exit-rule equity and portfolio equity comparisons
data/
  raw/
  clean/
```

### Class 4 key artifacts

- `risk_exit_metrics.csv` — single-asset metrics by exit mode and period
- `risk_<SYMBOL>_<mode>_backtest.csv` — bar-level pnl/position/exit_reason
- `risk_portfolio_weights.csv` — train-fitted weights by sleeve/method
- `risk_portfolio_metrics.csv` — portfolio train/valid/test metrics
- `risk_summary.csv` — compact test-period overview
- `risk_*_exit_equity.png` / `risk_portfolio_equity_*.png`

## Running

```bash
python crypto_cta_strategy.py
```

If exchange access is blocked, the script falls back to deterministic demo OHLCV.

## Notes

- Research/teaching framework, not a production trading engine.
- Train/valid/test: fit on train, diagnose on valid, evaluate once on test.
- TP/SL can look better after tuning; treat exit parameters as risk controls, not free alpha knobs.
- Portfolio goal is risk budgeting / concentration control, not equal dollar bets by default.
