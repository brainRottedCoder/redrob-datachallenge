"""Streamlit demo for FitRank shortlist exploration."""

from __future__ import annotations

import streamlit as st

from fitrank.jd_parser import parse_jd
from fitrank.loader import load_sample
from fitrank.ranker import rank_candidates
from fitrank.reasoning import build_reasoning

st.set_page_config(page_title="FitRank Demo", layout="wide")
st.title("FitRank — Candidate Shortlist Demo")

jd_text = st.text_area(
    "Job description",
    value=open("data/job_description.txt", encoding="utf-8").read(),
    height=200,
)
if st.button("Rank candidates"):
    role = parse_jd(jd_text)
    ranked = rank_candidates(load_sample(), role, top_n=20)
    for candidate, components, _ in ranked:
        st.subheader(f"{candidate.profile.current_title} ({candidate.candidate_id})")
        st.progress(components.final_score)
        st.write(build_reasoning(candidate, components))
        cols = st.columns(5)
        cols[0].metric("JD Fit", f"{components.jd_fit:.2f}")
        cols[1].metric("Career", f"{components.career_evidence:.2f}")
        cols[2].metric("Coherence", f"{components.coherence:.2f}")
        cols[3].metric("Trust", f"{components.platform_trust:.2f}")
        cols[4].metric("Availability", f"{components.availability:.2f}")

if st.checkbox("Show trap examples"):
    st.warning("Trap profiles: non-ML titles with stuffed AI skills and incoherent career templates.")
