"""Streamlit demo for FitRank shortlist exploration."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import plotly.express as px
import streamlit as st

from fitrank.jd_parser import parse_jd
from fitrank.loader import load_candidates, load_sample
from fitrank.ranker import rank_candidates
from fitrank.reasoning import build_reasoning
from fitrank.title_gate import TitleDomain, classify_title

JD_PATH = Path("data/job_description.txt")
CANDIDATES_PATH = Path("data/candidates.jsonl")

st.set_page_config(page_title="FitRank Demo", layout="wide")
st.title("FitRank — Candidate Shortlist Demo")

jd_text = st.text_area(
    "Job description",
    value=JD_PATH.read_text(encoding="utf-8") if JD_PATH.exists() else "",
    height=200,
)

with st.sidebar:
    st.header("Filters")
    domain_filter = st.multiselect(
        "Title domain",
        options=[TitleDomain.ML_AI, TitleDomain.AI_ADJACENT, TitleDomain.SOFTWARE, TitleDomain.NON_TECH],
        default=[TitleDomain.ML_AI, TitleDomain.AI_ADJACENT],
    )
    country_filter = st.text_input("Country (leave empty for all)", value="")
    min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.05)

if st.button("Rank candidates"):
    role = parse_jd(jd_text)
    if CANDIDATES_PATH.exists():
        candidates = load_candidates(CANDIDATES_PATH, validate=False)
    else:
        candidates = load_sample()
    st.session_state["ranked"] = rank_candidates(candidates, role, top_n=100)

ranked = st.session_state.get("ranked", [])

if ranked:
    filtered = [
        (candidate, components, score)
        for candidate, components, score in ranked
        if classify_title(candidate.profile.current_title) in domain_filter
        and (not country_filter or country_filter.lower() in candidate.profile.country.lower())
        and score.final_score >= min_score
    ]

    st.subheader("Score distribution")
    scores = [score.final_score for _, _, score in ranked]
    fig = px.histogram(
        x=scores,
        nbins=20,
        labels={"x": "Final score"},
        title="Top-100 score distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"Showing {len(filtered)} of {len(ranked)} ranked candidates")
    for candidate, components, score in filtered:
        domain = classify_title(candidate.profile.current_title)
        badge = {
            TitleDomain.ML_AI: "🟢 ML_AI",
            TitleDomain.AI_ADJACENT: "🟡 AI_ADJACENT",
            TitleDomain.SOFTWARE: "🔵 SOFTWARE",
            TitleDomain.NON_TECH: "🔴 NON_TECH",
        }.get(domain, "⚪ UNKNOWN")

        st.subheader(f"{candidate.profile.current_title} {badge} ({candidate.candidate_id})")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final score", f"{score.final_score:.4f}")
        col2.metric("ML tenure", f"{components.ml_tenure_years:.1f} yrs")
        col3.metric("Total exp", f"{candidate.profile.years_of_experience:.1f} yrs")
        col4.metric("Honeypot", "Yes" if components.is_honeypot else "No")

        st.progress(min(max(score.final_score, 0.0), 1.0), text="Final score")
        tenure_ratio = components.ml_tenure_years / max(candidate.profile.years_of_experience, 1.0)
        st.progress(min(max(tenure_ratio, 0.0), 1.0), text="ML tenure / total experience")
        st.write(build_reasoning(candidate, components, ml_tenure=components.ml_tenure_years))

        with st.expander("Penalty breakdown"):
            p = components.penalties_detail
            penalties = [
                ("Template mismatch", p.template_mismatch),
                ("Skill inflation", p.skill_inflation),
                ("Shallow boilerplate", p.shallow_boilerplate),
                ("Expert zero-endorse", p.expert_zero_endorse),
                ("Title/skill mismatch", p.non_ml_title_high_skill),
                ("Consulting only", p.consulting_only),
                ("Pure research", p.pure_research),
                ("CV without NLP", p.cv_without_nlp),
                ("Salary inverted", p.salary_inverted),
                ("Experience gap", p.experience_gap),
                ("Job hopper", p.job_hopper),
                ("Seniority mismatch", p.seniority_mismatch),
            ]
            penalty_data = [{"Penalty": name, "Value": f"{val:.2f}"} for name, val in penalties if val > 0.01]
            if penalty_data:
                st.table(penalty_data)
            else:
                st.write("No significant penalties")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, (candidate, components, score) in enumerate(ranked, start=1):
        writer.writerow(
            [
                candidate.candidate_id,
                rank,
                score.final_score,
                build_reasoning(candidate, components, ml_tenure=components.ml_tenure_years),
            ]
        )
    st.download_button(
        "Export full 100-candidate submission",
        buffer.getvalue(),
        file_name="fitrank_submission.csv",
        mime="text/csv",
    )

if st.checkbox("Show trap examples"):
    st.warning("Trap profiles: non-ML titles with stuffed AI skills and incoherent career templates.")
    traps = [
        c
        for c in load_sample()
        if c.profile.current_title in {"HR Manager", "Project Manager", "Content Writer"}
    ][:2]
    for trap in traps:
        st.write(f"- {trap.candidate_id}: {trap.profile.current_title}")
