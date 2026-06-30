#!/usr/bin/env python3
"""Generate the FitRank approach deck PDF from markdown source."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deck" / "Redrob_FitRank_Approach.md"
OUTPUT = ROOT / "deck" / "Redrob_FitRank_Approach.pdf"
SCREENSHOT = ROOT / "deck" / "assets" / "demo_screenshot.png"

METRIC_FILES = {
    "audit": ROOT / "outputs" / "audit_report.json",
    "eval": ROOT / "outputs" / "eval_report.json",
    "sensitivity": ROOT / "outputs" / "sensitivity_report.json",
}


def _load_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name, path in METRIC_FILES.items():
        if path.exists():
            metrics[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            metrics[name] = {}
    return metrics


def _lookup(data: object, path: str) -> str:
    """Resolve dotted path like audit.top_5.0.candidate_id from nested dicts/lists."""
    current: object = data
    for part in path.split("."):
        if isinstance(current, dict):
            if part in current:
                current = current[part]
                continue
            return f"[{path}]"
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return f"[{path}]"
            if 0 <= index < len(current):
                current = current[index]
                continue
            return f"[{path}]"
        return f"[{path}]"
    if current is None:
        return "n/a"
    if isinstance(current, float):
        if path.endswith("_pct") or "pct" in path:
            sign = "+" if current >= 0 else ""
            return f"{sign}{current:.1f}%"
        return f"{current:.4f}" if abs(current) < 10 else f"{current:.2f}"
    return str(current)


def _inject_metrics(text: str, metrics: dict[str, object]) -> str:
    def replacer(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        namespace, _, path = key.partition(".")
        if namespace in metrics and path:
            return _lookup(metrics[namespace], path)
        return f"[{key}]"

    return re.sub(r"\{\{([^}]+)\}\}", replacer, text)


def _ascii_safe(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def _truncate(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _wrap_line(text: str, width: int = 95) -> list[str]:
    if len(text) <= width:
        return [text]
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if length + extra > width and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra
    if current:
        lines.append(" ".join(current))
    return lines or [text[:width]]


def _safe_multi_cell(pdf: FPDF, height: float, text: str, font_name: str = "Helvetica", font_style: str = "", size: int = 10) -> None:
    pdf.set_font(font_name, font_style, size)
    for chunk in _wrap_line(_truncate(text)):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, height, _ascii_safe(chunk))


def _parse_slides(markdown: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"^## Slide \d+:\s*(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))
    slides: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        slides.append((title, body))
    return slides


def _render_body(pdf: FPDF, body: str, metrics: dict[str, object]) -> None:
    body = _inject_metrics(body, metrics)
    in_code = False
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not line.strip():
            pdf.ln(2)
            continue
        if line.strip() == "---":
            pdf.ln(2)
            continue
        if line.startswith("!["):
            continue

        if in_code:
            _safe_multi_cell(pdf, 3.5, line, font_name="Courier", size=7)
            continue

        if line.startswith("|"):
            _safe_multi_cell(pdf, 4.5, line, size=9)
            continue

        if line.startswith("**") and line.endswith("**"):
            _safe_multi_cell(pdf, 5, line.strip("*"), font_style="B", size=11)
            continue

        if line.startswith("- "):
            _safe_multi_cell(pdf, 5, f"  - {line[2:]}")
            continue

        if re.match(r"^\d+\.\s", line):
            _safe_multi_cell(pdf, 5, f"  {line}")
            continue

        _safe_multi_cell(pdf, 5, line)


class FitRankDeckPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"FitRank Approach Deck  |  Page {self.page_no()}/{{nb}}", align="C")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Deck source not found: {SOURCE}")

    metrics = _load_metrics()
    slides = _parse_slides(SOURCE.read_text(encoding="utf-8"))
    if not slides:
        raise ValueError("No slides found in deck markdown")

    pdf = FitRankDeckPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    for index, (title, body) in enumerate(slides, start=1):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 10, _ascii_safe(title))
        pdf.ln(2)
        _render_body(pdf, body, metrics)

        if index == 9 and SCREENSHOT.exists():
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Live demo screenshot", ln=True)
            usable_width = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.image(str(SCREENSHOT), w=min(usable_width, 170))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
