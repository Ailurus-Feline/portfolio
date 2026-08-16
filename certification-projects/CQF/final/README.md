# Pairs Trading: Cointegration, Validation, and Stress Tests

Statistical-arbitrage study on three U.S.-listed ETF pairs. The stack is built to look like a research workflow: test whether a spread is I(0), size the hedge, trade mean reversion, then **invalidate** the book out of sample.

**Main result:** in-sample EWA–EWC looks tradable (Sharpe ~0.6); a frozen 70/30 time split does not (test Sharpe ~−0.4). The leak survives the full \(Z\) grid and zero costs. A naive \(\beta=1\) residual is not stationary. That is the point of the project — not a claimed live edge.

## Universe

| Pair | Role | Economic link |
|------|------|----------------|
| EWA–EWC | Primary | Australia vs Canada country ETFs (commodity / risk-on) |
| XLE–XOP | Control | Energy sector vs upstream E&P |
| GLD–GDX | Control | Gold vs gold miners |

## Method

1. **I(1) screen** — ADF + KPSS on levels vs differences  
2. **Engle–Granger** — matrix OLS, residual ADF (lag 1), ECM \(\lambda\)  
3. **Johansen** — trace rank on log-prices; VAR lag (AIC/BIC) and stability on log-returns  
4. **OU residual** — \(\theta\), half-life, \(\sigma_{\mathrm{eq}}\), \(Z\)-scores  
5. **Signals** — grid-search \(Z^*\) (not fixed at 1); dollar-neutral book with costs  
6. **Validation** — chronological 70/30 train/test; 8-month rolling \(\beta\)  
7. **Stress** — \(Z\) and cost grids, EG vs naive vs Johansen hedge, rolling ADF gate  

Write-up: `report/TS_REPORT.md` (HTML: `report/TS_REPORT.html`).

## How to run

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src pytest tests/ -q
PYTHONPATH=src python scripts/smoke_all_pairs.py
PYTHONPATH=src python scripts/export_sensitivity.py
```

20 unit tests. Optional exporters: `scripts/export_exam_tables.py`, `export_figures_*.py`, `export_report_html.py`. Outputs go to `figures/` and `results/`.

## Layout

```text
src/ts_pairs/     numerical library (OLS, EG, Johansen, OU, backtest, OOS, stress)
scripts/          smoke runs and figure/table exporters
tests/            20 tests
report/           analytical write-up
figures/ results/ plots and CSV tables
```
