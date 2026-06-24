"""Score aggregation and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from fitrank.career_analyzer import analyze_career
from fitrank.coherence import compute_coherence
from fitrank.models import Candidate, CandidateScore, RoleProfile
from fitrank.penalties import compute_penalties
from fitrank.signals import PlatformTrustScore, compute_platform_trust
from fitrank.skill_trust import extract_ml_skills
from fitrank.title_gate import classify_title, title_jd_match


@dataclass
class ComponentScores:
    jd_fit: float
    career_evidence: float
    coherence: float
    platform_trust: float
    availability: float
    penalties: float
    final_score: float
    is_honeypot: bool


def load_weights(path: str | Path = "config/weights.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def score_candidate(
    candidate: Candidate,
    role_profile: RoleProfile,
    weights: dict | None = None,
) -> tuple[ComponentScores, CandidateScore]:
    weights = weights or load_weights()
    career = analyze_career(candidate)
    title_domain = classify_title(candidate.profile.current_title)
    coherence = compute_coherence(candidate, career, title_domain)
    penalties = compute_penalties(candidate, career, coherence, role_profile)
    platform = compute_platform_trust(candidate, role_profile)

    title_match = title_jd_match(candidate.profile.current_title, role_profile)
    capability_match = _capability_match(candidate, role_profile)
    education = _education_relevance(candidate)
    jd_fit = (
        0.40 * title_match
        + 0.30 * capability_match
        + 0.20 * platform.assessment_jd_overlap
        + 0.10 * education
    )

    career_score = (
        0.50 * career.all_career_ml_depth_norm
        + 0.35 * career.current_role_ml_depth_norm
        + 0.15 * career.career_momentum_norm
    )

    raw = (
        weights["jd_fit"] * jd_fit
        + weights["career_evidence"] * career_score
        + weights["coherence"] * coherence.coherence_score
        + weights["platform_trust"] * platform.platform_trust
        + weights["availability"] * platform.availability_score
        - penalties.total_penalty
    )
    if coherence.is_honeypot:
        raw *= 0.25
    final_score = max(0.0, min(1.0, raw))

    components = ComponentScores(
        jd_fit=round(jd_fit, 4),
        career_evidence=round(career_score, 4),
        coherence=coherence.coherence_score,
        platform_trust=platform.platform_trust,
        availability=platform.availability_score,
        penalties=penalties.total_penalty,
        final_score=round(final_score, 4),
        is_honeypot=coherence.is_honeypot,
    )
    candidate_score = CandidateScore(
        candidate_id=candidate.candidate_id,
        jd_fit=components.jd_fit,
        career_evidence=components.career_evidence,
        coherence=components.coherence,
        platform_trust=components.platform_trust,
        availability=components.availability,
        penalties=components.penalties,
        final_score=components.final_score,
    )
    return components, candidate_score


def rank_candidates(
    candidates,
    role_profile: RoleProfile,
    weights: dict | None = None,
    top_n: int = 100,
) -> list[tuple[Candidate, ComponentScores, CandidateScore]]:
    scored: list[tuple[Candidate, ComponentScores, CandidateScore]] = []
    for candidate in candidates:
        components, candidate_score = score_candidate(candidate, role_profile, weights)
        scored.append((candidate, components, candidate_score))

    scored.sort(key=lambda item: (-item[2].final_score, item[0].candidate_id))
    return scored[:top_n]


def _capability_match(candidate: Candidate, role_profile: RoleProfile) -> float:
    if not role_profile.required_capabilities:
        return 0.0
    corpus = " ".join(
        [candidate.profile.summary]
        + [skill.name for skill in candidate.skills]
        + [entry.description for entry in candidate.career_history]
    ).lower()
    matched = sum(1 for cap in role_profile.required_capabilities if cap.lower() in corpus)
    return matched / len(role_profile.required_capabilities)


def _education_relevance(candidate: Candidate) -> float:
    if not candidate.education:
        return 0.0
    relevant = 0
    for edu in candidate.education:
        field = edu.field_of_study.lower()
        if any(token in field for token in ("computer", "ai", "machine", "data", "software")):
            relevant += 1
        if edu.tier in {"tier_1", "tier_2"}:
            relevant += 0.5
    return min(relevant / len(candidate.education), 1.0)
