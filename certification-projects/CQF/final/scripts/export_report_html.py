"""Export HTML (and optionally PDF) from ``report/TS_REPORT.md``.

Markdown treats ``\\(`` as an escaped parenthesis, so math is extracted
*before* Markdown, converted to MathML, then Chrome prints a PDF *without*
the ``file://`` header. Page numbers are stamped afterwards.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "report" / "TS_REPORT.md"
HTML_PATH = ROOT / "report" / "TS_REPORT.html"
PDF_PATH = ROOT / "submission" / "TS Mao Yikai REPORT.pdf"

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

_DISPLAY: list[str] = []
_INLINE: list[str] = []


def _rewrite_image_paths(md: str) -> str:
    def repl(match: re.Match[str]) -> str:
        alt, path = match.group(1), match.group(2)
        fig = (ROOT / "report" / path).resolve()
        return f"![{alt}]({fig.as_uri()})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)


def _protect_math(md: str) -> str:
    """Replace LaTeX delimiters with placeholders Markdown will not eat."""
    _DISPLAY.clear()
    _INLINE.clear()

    def disp(match: re.Match[str]) -> str:
        _DISPLAY.append(match.group(1).strip())
        return f"@@DISP{len(_DISPLAY) - 1}@@"

    def inl(match: re.Match[str]) -> str:
        _INLINE.append(match.group(1).strip())
        return f"@@INL{len(_INLINE) - 1}@@"

    md = re.sub(r"\\\[(.+?)\\\]", disp, md, flags=re.S)
    md = re.sub(r"\\\((.+?)\\\)", inl, md, flags=re.S)
    return md


def _to_mathml(tex: str, *, display: bool) -> str:
    from latex2mathml.converter import convert

    display_attr = "block" if display else "inline"
    try:
        inner = convert(tex)
    except Exception:
        inner = f"<mtext>{html.escape(tex)}</mtext>"
        inner = f'<math xmlns="http://www.w3.org/1998/Math/MathML">{inner}</math>'
        return inner
    # latex2mathml already wraps <math>; force display style.
    if inner.startswith("<math"):
        inner = re.sub(
            r"<math([^>]*)>",
            f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="{display_attr}">',
            inner,
            count=1,
        )
    tag = "div" if display else "span"
    cls = "math-display" if display else "math-inline"
    return f"<{tag} class=\"{cls}\">{inner}</{tag}>"


def _restore_math(body: str) -> str:
    def disp(match: re.Match[str]) -> str:
        return _to_mathml(_DISPLAY[int(match.group(1))], display=True)

    def inl(match: re.Match[str]) -> str:
        return _to_mathml(_INLINE[int(match.group(1))], display=False)

    body = re.sub(r"@@DISP(\d+)@@", disp, body)
    body = re.sub(r"@@INL(\d+)@@", inl, body)
    return body


def render_html() -> Path:
    text = _rewrite_image_paths(MD_PATH.read_text(encoding="utf-8"))
    text = _protect_math(text)

    import markdown  # type: ignore

    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    body = _restore_math(body)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>CQF TS Final Project Report</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 900px;
           margin: 2rem auto; padding: 0 1rem; line-height: 1.45; color: #222; }}
    code, pre {{ font-family: Menlo, Consolas, monospace; font-size: 0.9em; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92em; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.55rem; text-align: left; }}
    img {{ max-width: 100%; height: auto; }}
    h1, h2, h3 {{ color: #111; }}
    .math-display {{ margin: 1rem 0; text-align: center; overflow-x: auto; }}
    math {{ font-family: 'STIX Two Math', 'Cambria Math', 'Times New Roman', serif; }}
    @page {{ margin: 18mm 16mm 22mm 16mm; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_PATH.write_text(doc, encoding="utf-8")
    return HTML_PATH


def _stamp_page_numbers(pdf_path: Path) -> None:
    """Draw 'n / N' at the bottom centre without Chrome's file:// footer."""
    import matplotlib.pyplot as plt
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    if n_pages == 0:
        return
    box = reader.pages[0].mediabox
    width_pt = float(box.width)
    height_pt = float(box.height)

    writer = PdfWriter()
    tmp_overlay = pdf_path.with_suffix(".overlay.pdf")

    overlays = []
    for i in range(n_pages):
        fig = plt.figure(figsize=(width_pt / 72.0, height_pt / 72.0))
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(
            0.5,
            0.018,
            f"{i + 1} / {n_pages}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
            transform=ax.transAxes,
        )
        buf = tmp_overlay.with_name(f"overlay_{i}.pdf")
        fig.savefig(buf, format="pdf", transparent=True)
        plt.close(fig)
        overlays.append(buf)

    overlay_reader_pages = [PdfReader(str(p)).pages[0] for p in overlays]
    for page, overlay in zip(reader.pages, overlay_reader_pages):
        page.merge_page(overlay)
        writer.add_page(page)

    with pdf_path.open("wb") as fh:
        writer.write(fh)
    for p in overlays:
        p.unlink(missing_ok=True)


def render_pdf(html_path: Path = HTML_PATH, pdf_path: Path = PDF_PATH) -> Path:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if not CHROME.exists():
        raise RuntimeError(f"Google Chrome not found at {CHROME}")

    uri = html_path.resolve().as_uri()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_pdf = Path(tmp.name)

    cmd = [
        str(CHROME),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={tmp_pdf}",
        uri,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    tmp_pdf.replace(pdf_path)
    _stamp_page_numbers(pdf_path)
    return pdf_path


def main() -> None:
    html_path = render_html()
    print("Wrote", html_path)
    if "--pdf" in sys.argv:
        out = render_pdf()
        print("Wrote", out)


if __name__ == "__main__":
    main()
