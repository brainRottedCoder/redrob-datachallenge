#!/usr/bin/env python3
"""Generate the FitRank approach deck PDF from markdown source."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deck" / "Redrob_FitRank_Approach.md"
OUTPUT = ROOT / "deck" / "Redrob_FitRank_Approach.pdf"

SLIDES = [
    "FitRank - Recruiter-Trustable AI Ranking",
    "Problem: keyword filters fail on adversarial candidate profiles",
    "Data Insights: traps, templates, ~900 genuine ML practitioners",
    "Core Idea: profile coherence beats keyword stuffing",
    "Architecture: 10-phase offline scoring pipeline",
    "Differentiators: coherence, templates, skill trust, Redrob signals",
    "Scoring Model: JD fit, career evidence, coherence, trust, availability",
    "Examples: Senior NLP Engineer ranks above HR Manager honeypot",
    "Results: validator pass, <5 min runtime, audit report",
    "Redrob Vision: intelligent discovery product feature",
]


def main() -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for index, slide in enumerate(SLIDES, start=1):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, f"Slide {index}", ln=True)
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, slide.encode("ascii", "ignore").decode("ascii"))
        if SOURCE.exists() and index == 1:
            pdf.ln(4)
            pdf.set_font("Helvetica", size=10)
            snippet = SOURCE.read_text(encoding="utf-8")[:500]
            pdf.multi_cell(0, 6, snippet.encode("ascii", "ignore").decode("ascii"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
