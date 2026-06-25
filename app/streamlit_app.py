"""Streamlit demo for FitRank shortlist exploration."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import streamlit as st

from fitrank.jd_parser import parse_jd
from fitrank.loader import load_sample
from fitrank.ranker import rank_candidates
from fitrank.reasoning import build_reasoning

JD_PATH = Path("data/job_description.txt")

st.set_page_config(page_title="FitRank Demo", layout="wide")
st.title("FitRank — Candidate Shortlist Demo")

jd_text = st.text_area(
    "Job description",
    value=JD_PATH.read_text(encoding="utf-8") if JD_PATH.exists() else "",
    height=200,
)

if st.button("Rank candidates"):
    role = parse_jd(jd_text)
    st.session_state["ranked"] = rank_candidates(load_sample(), role, top_n=20)

ranked = st.session_state.get("ranked", [])
for candidate, components, _ in ranked:
    st.subheader(f"{candidate.profile.current_title} ({candidate.candidate_id})")
    st.progress(min(max(components.final_score, 0.0), 1.0))
    st.write(build_reasoning(candidate, components))
    cols = st.columns(5)
    cols[0].metric("JD Fit", f"{components.jd_fit:.2f}")
    cols[1].metric("Career", f"{components.career_evidence:.2f}")
    cols[2].metric("Coherence", f"{components.coherence:.2f}")
    cols[3].metric("Trust", f"{components.platform_trust:.2f}")
    cols[4].metric("Availability", f"{components.availability:.2f}")

if ranked:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, (candidate, components, _) in enumerate(ranked, start=1):
        writer.writerow(
            [candidate.candidate_id, rank, components.final_score, build_reasoning(candidate, components)]
        )
    st.download_button("Export CSV", buffer.getvalue(), file_name="fitrank_shortlist.csv", mime="text/csv")

if st.checkbox("Show trap examples"):
    st.warning("Trap profiles: non-ML titles with stuffed AI skills and incoherent career templates.")
    traps = [
        c
        for c in load_sample()
        if c.profile.current_title in {"HR Manager", "Project Manager", "Content Writer"}
    ][:2]
    for trap in traps:
        st.write(f"- {trap.candidate_id}: {trap.profile.current_title}")
