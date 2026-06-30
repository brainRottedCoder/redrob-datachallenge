"""FitRank judge-ready Streamlit demo: Rank, Compare, Trap Detector."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import plotly.express as px
import streamlit as st

from app.demo_components import (
    export_audit_json,
    render_candidate_card,
    render_comparison_delta_table,
    render_comparison_radar,
    render_role_profile_sidebar,
    render_trap_card,
)
from eval.baseline_ranker import score_candidate_baseline
from fitrank.embedder import encode_jd, load_candidate_embeddings
from fitrank.jd_parser import parse_jd
from fitrank.loader import load_candidates, load_sample
from fitrank.llm_reasoning import (
    DEFAULT_CACHE_PATH,
    DEFAULT_MODEL_PATH,
    LlamaReasoningEngine,
    ReasoningCache,
    ReasoningStats,
    is_llama_available,
    jd_fingerprint,
    resolve_reasoning,
)
from fitrank.ranker import load_capability_vectors, rank_candidates, score_candidate
from fitrank.reasoning import build_reasoning
from fitrank.title_gate import TitleDomain, classify_title

JD_PATH = Path("data/job_description.txt")
CANDIDATES_PATH = Path("data/candidates.jsonl")
EMBEDDINGS_PATH = Path("outputs/candidate_embeddings.npy")
IDS_PATH = Path("outputs/candidate_ids.json")

st.set_page_config(page_title="FitRank — Recruiter Intelligence Demo", layout="wide")
st.title("FitRank — Recruiter Intelligence Demo")
st.caption("JD-agnostic profile coherence ranking · offline by default")

default_jd = JD_PATH.read_text(encoding="utf-8") if JD_PATH.exists() else ""
jd_text = st.text_area("Job description", value=default_jd, height=220)

with st.sidebar:
    st.header("Controls")
    use_ai_reasoning = st.toggle(
        "AI reasoning (local LLM)",
        value=False,
        help="Optional local GGUF model; defaults to structured rule-based reasoning.",
    )
    if use_ai_reasoning and not is_llama_available(DEFAULT_MODEL_PATH):
        st.caption("LLM unavailable — using template reasoning.")
    ranking_mode = st.selectbox("Ranking mode", ["auto", "heuristic", "learned"], index=0)
    calibrate = st.checkbox("Calibrate scores (0.05–0.95)", value=False)
    domain_filter = st.multiselect(
        "Title domain filter",
        options=[TitleDomain.ML_AI, TitleDomain.AI_ADJACENT, TitleDomain.SOFTWARE, TitleDomain.NON_TECH],
        default=[TitleDomain.ML_AI, TitleDomain.AI_ADJACENT, TitleDomain.SOFTWARE, TitleDomain.NON_TECH],
    )
    country_filter = st.text_input("Country (empty = all)", value="")
    min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.05)
    st.divider()
    render_role_profile_sidebar(parse_jd(jd_text))

if st.button("Rank candidates", type="primary"):
    role = parse_jd(jd_text)
    with st.spinner("Ranking candidates…"):
        try:
            if CANDIDATES_PATH.exists():
                candidates = load_candidates(CANDIDATES_PATH, validate=False)
            else:
                st.warning("`data/candidates.jsonl` not found — using sample data.")
                candidates = load_sample()
            if EMBEDDINGS_PATH.exists() and IDS_PATH.exists():
                candidate_embeddings = load_candidate_embeddings(EMBEDDINGS_PATH, IDS_PATH)
            else:
                candidate_embeddings = load_capability_vectors()
            jd_embedding = encode_jd(jd_text)
            ranked = rank_candidates(
                candidates,
                role,
                top_n=100,
                jd_text=jd_text,
                jd_embedding=jd_embedding,
                candidate_embeddings=candidate_embeddings,
                ranking_mode=ranking_mode,
            )
            if calibrate:
                from fitrank.calibrator import calibrate_scores

                ranked = calibrate_scores(ranked)
            st.session_state["ranked"] = ranked
            st.session_state["role_profile"] = role
            st.session_state["jd_hash"] = jd_fingerprint(jd_text, role)
        except Exception as exc:
            st.error(f"Ranking failed: {exc}")

ranked: list = st.session_state.get("ranked", [])
role_profile = st.session_state.get("role_profile", parse_jd(jd_text))
jd_hash = st.session_state.get("jd_hash", "")

reasoning_cache = ReasoningCache(DEFAULT_CACHE_PATH) if use_ai_reasoning else None
llm_engine = (
    LlamaReasoningEngine(DEFAULT_MODEL_PATH)
    if use_ai_reasoning and is_llama_available(DEFAULT_MODEL_PATH)
    else None
)
reasoning_stats = ReasoningStats() if use_ai_reasoning else None


def _display_reasoning(candidate, components):
    template = build_reasoning(
        candidate, components,
        ml_tenure=components.ml_tenure_years,
        role_profile=role_profile,
    )
    if not use_ai_reasoning:
        return template, "Structured"
    text = resolve_reasoning(
        candidate, components, mode="llm", role_profile=role_profile,
        jd_hash=jd_hash, cache=reasoning_cache, engine=llm_engine,
        model_path=DEFAULT_MODEL_PATH, ml_tenure=components.ml_tenure_years,
        stats=reasoning_stats, max_new=100,
    )
    badge = "AI" if text != template else "Structured"
    return text, badge


if not ranked:
    st.info("Paste a JD and click **Rank candidates** to begin.")
    st.stop()

filtered = [
    (candidate, components, score)
    for candidate, components, score in ranked
    if classify_title(candidate.profile.current_title) in domain_filter
    and (not country_filter or country_filter.lower() in candidate.profile.country.lower())
    and score.final_score >= min_score
]

tab_rank, tab_compare, tab_traps = st.tabs(["Rank", "Compare", "Trap Detector"])

with tab_rank:
    st.subheader("Score distribution")
    scores = [score.final_score for _, _, score in ranked]
    st.plotly_chart(
        px.histogram(x=scores, nbins=20, labels={"x": "Final score"}, title="Top-100 scores"),
        use_container_width=True,
    )

    st.write(f"Showing {min(25, len(filtered))} of {len(filtered)} filtered ({len(ranked)} ranked)")
    for rank, (candidate, components, score) in enumerate(filtered[:25], start=1):
        render_candidate_card(candidate, components, score, role_profile, rank=rank)
        if use_ai_reasoning:
            reasoning_text, badge = _display_reasoning(candidate, components)
            st.caption(f"AI reasoning ({badge}): {reasoning_text}")

    st.divider()
    col_csv, col_json = st.columns(2)
    with col_csv:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r, (candidate, components, score) in enumerate(ranked, start=1):
            text, _ = _display_reasoning(candidate, components)
            writer.writerow([candidate.candidate_id, r, score.final_score, text])
        st.download_button("Export CSV (top 100)", buffer.getvalue(), "fitrank_submission.csv", "text/csv")
    with col_json:
        st.download_button(
            "Export JSON audit",
            export_audit_json(role_profile, ranked),
            "fitrank_audit.json",
            "application/json",
        )

with tab_compare:
    st.subheader("Side-by-side comparison")
    labels = [f"#{i} {c.profile.current_title}" for i, (c, _, _) in enumerate(ranked, start=1)]
    default_b = min(9, len(ranked) - 1)
    idx_a = st.selectbox("Candidate A", range(len(ranked)), format_func=lambda i: labels[i], index=0)
    idx_b = st.selectbox("Candidate B", range(len(ranked)), format_func=lambda i: labels[i], index=default_b)
    cand_a, comp_a, score_a = ranked[idx_a]
    cand_b, comp_b, score_b = ranked[idx_b]

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"**A:** {cand_a.profile.current_title}")
        st.metric("Final score", f"{score_a.final_score:.4f}")
    with col_right:
        st.markdown(f"**B:** {cand_b.profile.current_title}")
        st.metric("Final score", f"{score_b.final_score:.4f}")

    st.plotly_chart(render_comparison_radar(cand_a, cand_b, comp_a, comp_b), use_container_width=True)
    st.table(render_comparison_delta_table(comp_a, comp_b))

with tab_traps:
    st.subheader("Trap Detector")
    if st.toggle("Show profiles excluded from top 100", value=True):
        with st.spinner("Scanning trap profiles…"):
            pool_source = (
                load_candidates(CANDIDATES_PATH, validate=False)
                if CANDIDATES_PATH.exists()
                else load_sample()
            )
            ranked_ids = {c.candidate_id for c, _, _ in ranked}
            trap_pool = [
                c for c in pool_source
                if c.candidate_id not in ranked_ids
                and classify_title(c.profile.current_title) in {TitleDomain.NON_TECH, TitleDomain.AI_ADJACENT}
            ][:50]
            if not trap_pool:
                trap_pool = [c for c in pool_source if c.candidate_id not in ranked_ids][-50:]

            trap_scored = []
            for candidate in trap_pool:
                components, _ = score_candidate(candidate, role_profile, ranking_mode=ranking_mode)
                baseline = score_candidate_baseline(candidate).keyword_score / 20.0
                trap_scored.append((candidate, components, baseline))
            trap_scored.sort(key=lambda x: x[1].final_score)

        st.write(f"Showing {min(15, len(trap_scored))} low-scoring trap-like profiles")
        for candidate, components, baseline in trap_scored[:15]:
            render_trap_card(candidate, components, role_profile, baseline_score=baseline)
