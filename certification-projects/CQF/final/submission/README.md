# Submission staging (CQF Final — TS)

Deadline: **Tuesday 18 August 2026, 23:59 BST**. Candidate: **Mao Yikai**.

## Portal files (two only)

| File | Local path |
|------|------------|
| FILE 1 report | `TS Mao Yikai REPORT.pdf` |
| FILE 2 zip | `TS Mao Yikai CODE.zip` |

Do **not** upload `CODE.zip`, `FinalProject.zip`, `Final Project Declaration.pdf`, or loose `.py` files.

## What goes in the zip

Put the *project contents* in, not the whole git repo and not junk:

```text
TS Mao Yikai CODE.zip
├── TS Mao Yikai Declaration.pdf
├── TS Mao Yikai REPORT.pdf      # brief asks the zip to include the converted PDF
├── README.md
├── requirements.txt
├── src/ts_pairs/
├── scripts/
├── tests/
├── report/                      # TS_REPORT.md (+ html twin optional)
├── figures/                     # png assets used in the report
└── results/                     # csv tables used in the report
```

**Leave out:** `.venv/`, `__pycache__/`, `.pytest_cache/`, `data/raw/` caches,
`docs/PENDING_COMMITS.md`, empty `notebooks/`, `.git/`.

## Build PDF

```bash
PYTHONPATH=src python scripts/export_report_html.py --pdf
```
