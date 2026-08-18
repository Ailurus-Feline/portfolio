# Analytical report

Graded manuscript: **`TS_REPORT.md`**.

HTML twin and paged PDF (no `file://` header):

```bash
PYTHONPATH=src python scripts/export_report_html.py --pdf
```

That writes `report/TS_REPORT.html` and `submission/TS Mao Yikai REPORT.pdf`.
Do not Print-to-PDF from Markdown; Python-Markdown would eat `\( ... \)` and
leave raw TeX.
