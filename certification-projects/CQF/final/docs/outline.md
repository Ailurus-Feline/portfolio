# TS project outline (locked)

## Pairs
- A (main): EWA–EWC — commodity-country ETFs (brief example)
- B: XLE–XOP — energy sector vs E&P
- C: GLD–GDX — gold vs gold miners

## Stack
- Python only
- Core modules under `src/ts_pairs/`
- `report/TS_REPORT.md` is the graded write-up

## Method (from the brief)
### Part I
1. Matrix-form regression (own OLS)
2. Engle–Granger (ADF lag=1 on residual; EC-term analysis; multi-period)
3. Step 3: OU / mean-reversion (theta, half-life) → entry at μ±Zσ, exit at μ
4. Iterate Z* (not default Z=1); N_trades vs Z table
5. Structural-break discussion
6. Johansen / VECM (depth add-on)
7. Study 2–3 pairs

### Part II
6. Systematic backtest: drawdowns, rolling Sharpe; discuss plots
7. Train/test (time-aware) validation
8. Rolling re-estimation (8m window, shift ~12d)

## Report TOC
0. Cover / abstract
1. Introduction
2. Methodology
3. Numerical methods inventory
4. Results — EWA–EWC
5. Results — XLE–XOP / GLD–GDX
6. Discussion
7. Conclusion
References
