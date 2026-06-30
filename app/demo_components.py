"""Reusable Streamlit UI helpers for the FitRank judge demo."""

from __future__ import annotations

import json
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from fitrank.jd_parser import role_profile_to_dict
from fitrank.models import Candidate, CandidateScore, RoleProfile
from fitrank.ranker import ComponentScores
from fitrank.reasoning import build_reasoning, build_trap_explanation
from fitrank.title_gate import TitleDomain, classify_title

COMPONENT_LABELS = [
    ("JD Fit", "jd_fit"),
    ("Career", "career_evidence"),
    ("Coherence", "coherence"),
    ("Trust", "platform_trust"),
    ("Availability", "availability"),
]


def _domain_badge(domain: TitleDomain) -> str:
    return {
        TitleDomain.ML_AI: "ML_AI",
        TitleDomain.AI_ADJACENT: "AI_ADJACENT",
        TitleDomain.SOFTWARE: "SOFTWARE",
        TitleDomain.NON_TECH: "NON_TECH / trap",
    }.get(domain, "UNKNOWN")


def render_role_profile_sidebar(role_profile: RoleProfile) -> None:
    """Show parsed JD preferences in the sidebar."""
    st.subheader("Parsed RoleProfile")
    st.write(f"**Domain:** {role_profile.domain}")
    st.write(f"**Seniority:** {role_profile.seniority}")
    st.write(f"**Min experience:** {role_profile.min_experience_years:.0f} yrs")
    st.write(f"**Work mode:** {role_profile.preferred_work_mode}")

    prefs = role_profile.prefs
    st.caption("JD preferences")
    st.write(f"- Penalize consulting-only: `{prefs.penalize_consulting_only}`")
    st.write(f"- Penalize pure research: `{prefs.penalize_pure_research}`")
    st.write(f"- Penalize domain mismatch: `{prefs.penalize_domain_mismatch}`")
    st.write(f"- Values production exp: `{prefs.values_production_experience}`")
    if prefs.preferred_locations:
        st.write(f"- Locations: {', '.join(prefs.preferred_locations)}")
    else:
        st.write("- Locations: *(none — neutral bonus)*")
    if prefs.required_evidence_keywords:
        st.write(f"- Evidence keywords: {', '.join(prefs.required_evidence_keywords[:6])}")

    if role_profile.target_titles:
        st.write("**Target titles:** " + ", ".join(role_profile.target_titles[:4]))


def render_component_bar_chart(components: ComponentScores, title: str = "Component scores") -> go.Figure:
    """Horizontal bar chart of the five positive scoring components."""
    labels = [label for label, _ in COMPONENT_LABELS]
    values = [getattr(components, attr) for _, attr in COMPONENT_LABELS]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#1abc9c"],
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 1.05], title="Score"),
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def render_candidate_card(
    candidate: Candidate,
    components: ComponentScores,
    score: CandidateScore,
    role_profile: RoleProfile,
    rank: int | None = None,
) -> None:
    """Render a single ranked candidate card with chart and reasoning."""
    domain = classify_title(candidate.profile.current_title)
    badge = _domain_badge(domain)
    prefix = f"#{rank} " if rank is not None else ""
    trap_tag = " [TRAP]" if components.is_honeypot else ""

    st.markdown(f"### {prefix}{candidate.profile.current_title} `{badge}`{trap_tag}")
    st.caption(candidate.candidate_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final score", f"{score.final_score:.4f}")
    c2.metric("ML tenure", f"{components.ml_tenure_years:.1f} yrs")
    c3.metric("Total exp", f"{candidate.profile.years_of_experience:.1f} yrs")
    c4.metric("Penalties", f"{components.penalties:.2f}")

    st.plotly_chart(
        render_component_bar_chart(components),
        use_container_width=True,
        key=f"bar_{candidate.candidate_id}_{rank}",
    )

    with st.expander("Why this candidate?"):
        st.write(
            build_reasoning(
                candidate,
                components,
                ml_tenure=components.ml_tenure_years,
                role_profile=role_profile,
            )
        )
        assessments = candidate.redrob_signals.skill_assessment_scores
        if assessments:
            top = sorted(assessments.items(), key=lambda x: x[1], reverse=True)[:3]
            st.write("Top assessments: " + ", ".join(f"{k}={v:.0f}" for k, v in top))

    with st.expander("Penalty breakdown"):
        p = components.penalties_detail
        penalty_rows = [
            ("Template mismatch", p.template_mismatch),
            ("Skill inflation", p.skill_inflation),
            ("Shallow boilerplate", p.shallow_boilerplate),
            ("Expert zero-endorse", p.expert_zero_endorse),
            ("Title/skill mismatch", p.non_ml_title_high_skill),
            ("Consulting only", p.consulting_only),
            ("Pure research", p.pure_research),
            ("Domain mismatch", p.domain_mismatch),
            ("Salary inverted", p.salary_inverted),
            ("Experience gap", p.experience_gap),
            ("Job hopper", p.job_hopper),
            ("Seniority mismatch", p.seniority_mismatch),
        ]
        active = [{"Penalty": name, "Value": f"{val:.2f}"} for name, val in penalty_rows if val > 0.01]
        if active:
            st.table(active)
        else:
            st.write("No significant penalties")


def render_comparison_radar(
    cand_a: Candidate,
    cand_b: Candidate,
    components_a: ComponentScores,
    components_b: ComponentScores,
) -> go.Figure:
    """Overlay radar chart for two candidates' component scores."""
    categories = [label for label, _ in COMPONENT_LABELS]
    values_a = [getattr(components_a, attr) for _, attr in COMPONENT_LABELS]
    values_b = [getattr(components_b, attr) for _, attr in COMPONENT_LABELS]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_a + [values_a[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=cand_a.profile.current_title[:30],
            line_color="#3498db",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=values_b + [values_b[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=cand_b.profile.current_title[:30],
            line_color="#e74c3c",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def render_comparison_delta_table(
    components_a: ComponentScores,
    components_b: ComponentScores,
) -> list[dict[str, Any]]:
    """Build delta rows for side-by-side comparison."""
    rows = []
    for label, attr in COMPONENT_LABELS:
        a_val = getattr(components_a, attr)
        b_val = getattr(components_b, attr)
        rows.append({
            "Component": label,
            "Candidate A": round(a_val, 3),
            "Candidate B": round(b_val, 3),
            "Delta (A−B)": round(a_val - b_val, 3),
        })
    rows.append({
        "Component": "Final score",
        "Candidate A": round(components_a.final_score, 3),
        "Candidate B": round(components_b.final_score, 3),
        "Delta (A−B)": round(components_a.final_score - components_b.final_score, 3),
    })
    return rows


def render_trap_card(
    candidate: Candidate,
    components: ComponentScores,
    role_profile: RoleProfile,
    baseline_score: float | None = None,
) -> None:
    """Render a trap/honeypot profile explanation card."""
    st.markdown(f"#### {candidate.profile.current_title} — `{candidate.candidate_id}`")
    st.markdown(build_trap_explanation(candidate, components, role_profile))
    if baseline_score is not None:
        st.info(
            f"Keyword baseline would score this higher ({baseline_score:.2f}) "
            f"than FitRank ({components.final_score:.3f}) — classic trap pattern."
        )
    st.divider()


def export_audit_json(
    role_profile: RoleProfile,
    ranked: list[tuple[Candidate, ComponentScores, CandidateScore]],
) -> str:
    """Serialize RoleProfile + top-N audit for JSON download."""
    audit = {
        "role_profile": role_profile_to_dict(role_profile),
        "top_candidates": [
            {
                "rank": i,
                "candidate_id": c.candidate_id,
                "title": c.profile.current_title,
                "final_score": score.final_score,
                "is_honeypot": comp.is_honeypot,
                "components": {
                    "jd_fit": comp.jd_fit,
                    "career_evidence": comp.career_evidence,
                    "coherence": comp.coherence,
                    "platform_trust": comp.platform_trust,
                    "availability": comp.availability,
                    "penalties": comp.penalties,
                },
                "reasoning": build_reasoning(c, comp, ml_tenure=comp.ml_tenure_years, role_profile=role_profile),
            }
            for i, (c, comp, score) in enumerate(ranked, start=1)
        ],
    }
    return json.dumps(audit, indent=2)
