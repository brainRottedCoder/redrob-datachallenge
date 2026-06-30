"""Heuristic feature extraction for learned and rule-based ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fitrank.career_analyzer import analyze_career
from fitrank.coherence import compute_coherence
from fitrank.component_scores import ComponentScores
from fitrank.config_loader import load_weights as _load_weights
from fitrank.embedder import (
    compute_sparse_capability_match,
    semantic_match_from_precomputed,
)
from fitrank.models import Candidate, RoleProfile
from fitrank.penalties import compute_penalties
from fitrank.signals import compute_platform_trust, load_normalization_constants
from fitrank.skill_trust import analyze_skills
from fitrank.title_gate import TitleDomain, classify_title, title_jd_match

# 12 core features used by the LightGBM learned ranker.
FEATURE_NAMES: list[str] = [
    "title_jd_match",
    "capability_match",
    "career_ml_depth",
    "current_role_ml_depth",
    "coherence_score",
    "skill_trust_ratio",
    "platform_trust",
    "availability_score",
    "penalties_total",
    "assessment_jd_overlap",
    "career_momentum",
    "ml_tenure_years",
]


@dataclass(frozen=False)
class RankingFeatures:
    """Interpretable feature vector for one candidate.

    Core fields (used by the learned ranker) are required. Additional fields are
    optional and used by ablation/sensitivity analysis and reasoning.
    """
    # Core fields (must match FEATURE_NAMES order).
    title_jd_match: float = 0.0
    capability_match: float = 0.0
    career_ml_depth: float = 0.0
    current_role_ml_depth: float = 0.0
    coherence_score: float = 0.0
    skill_trust_ratio: float = 0.0
    platform_trust: float = 0.0
    availability_score: float = 0.0
    penalties_total: float = 0.0
    assessment_jd_overlap: float = 0.0
    career_momentum: float = 0.0
    ml_tenure_years: float = 0.0

    # Additional fields for ablation / reasoning / analysis.
    capability_match_sparse: float = 0.0
    capability_match_dense: float = 0.0
    template_coherence: float = 0.0
    title_domain_confirmation: float = 0.0
    skill_career_alignment: float = 0.0
    assessment_avg: float = 0.0
    github_norm: float = 0.0
    engagement_composite: float = 0.0
    verification_score: float = 0.0
    trusted_skill_ratio: float = 0.0
    skill_inflation_risk: float = 0.0
    total_penalty: float = 0.0
    total_years_experience: float = 0.0
    domain_ml_ai: float = 0.0
    domain_ai_adjacent: float = 0.0
    domain_software: float = 0.0
    domain_non_tech: float = 0.0
    all_career_ml_depth_norm: float = 0.0
    current_role_ml_depth_norm: float = 0.0
    career_momentum_norm: float = 0.0
    open_source_bonus: float = 0.0
    is_honeypot: float = 0.0
    education_relevance: float = 0.0

    _feature_order: list[str] = field(default_factory=lambda: list(FEATURE_NAMES), repr=False)

    def to_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}


_EDUCATION_RELEVANT_FIELDS = {
    "computer science",
    "artificial intelligence",
    "machine learning",
    "data science",
    "software engineering",
    "statistics",
    "mathematics",
    "data engineering",
    "information technology",
    "computer engineering",
    "electrical engineering",
}


def load_weights(path: str | Path = "config/weights.yaml") -> dict:
    return _load_weights(path)


def extract_ranking_features(
    candidate: Candidate,
    role_profile: RoleProfile,
    weights: dict | None = None,
    norms: dict[str, float] | None = None,
    jd_embedding: np.ndarray | None = None,
    candidate_embeddings: dict[str, np.ndarray] | None = None,
) -> tuple[RankingFeatures, ComponentScores, float]:
    """Extract ML features, component scores, and heuristic final score."""
    weights = weights or load_weights()
    norms = norms or load_normalization_constants()

    career = analyze_career(candidate)
    title_domain = classify_title(candidate.profile.current_title)
    coherence = compute_coherence(candidate, career, title_domain, role_profile)
    penalties = compute_penalties(candidate, career, coherence, role_profile, weights)
    platform = compute_platform_trust(candidate, role_profile, weights=weights, norms=norms)
    skill_trust = analyze_skills(candidate.skills, role_profile)

    title_match = title_jd_match(candidate.profile.current_title, role_profile)
    capability_match = _capability_match(
        candidate, role_profile, jd_embedding, candidate_embeddings
    )
    education = _education_relevance(candidate)

    jd_parts = weights.get("jd_fit_components", {})
    jd_fit = (
        jd_parts.get("title_jd_match", 0.40) * title_match
        + jd_parts.get("capability_match", 0.30) * capability_match
        + jd_parts.get("assessment_jd_overlap", 0.20) * platform.assessment_jd_overlap
        + jd_parts.get("education_relevance", 0.10) * education
    )

    career_parts = weights.get("career_evidence_components", {})
    open_source_bonus = _open_source_bonus(candidate)
    career_score = min(
        career_parts.get("all_career_ml_depth", 0.50) * career.all_career_ml_depth_norm
        + career_parts.get("current_role_ml_depth", 0.35) * career.current_role_ml_depth_norm
        + career_parts.get("career_momentum", 0.15) * career.career_momentum_norm
        + 0.05 * open_source_bonus,
        1.0,
    )

    heuristic_score = compute_heuristic_final_score(
        weights,
        jd_fit,
        career_score,
        coherence.coherence_score,
        platform.platform_trust,
        platform.availability_score,
        penalties.total_penalty,
        coherence.is_honeypot,
    )

    domain_one_hot = {
        TitleDomain.ML_AI: (1.0, 0.0, 0.0, 0.0),
        TitleDomain.AI_ADJACENT: (0.0, 1.0, 0.0, 0.0),
        TitleDomain.SOFTWARE: (0.0, 0.0, 1.0, 0.0),
        TitleDomain.NON_TECH: (0.0, 0.0, 0.0, 1.0),
    }.get(title_domain, (0.0, 0.0, 0.0, 1.0))

    features = RankingFeatures(
        title_jd_match=round(title_match, 4),
        capability_match=round(capability_match, 4),
        career_ml_depth=round(career.all_career_ml_depth_norm, 4),
        current_role_ml_depth=round(career.current_role_ml_depth_norm, 4),
        coherence_score=coherence.coherence_score,
        skill_trust_ratio=round(skill_trust.trusted_skill_ratio, 4),
        platform_trust=platform.platform_trust,
        availability_score=platform.availability_score,
        penalties_total=penalties.total_penalty,
        assessment_jd_overlap=platform.assessment_jd_overlap,
        career_momentum=round(career.career_momentum_norm, 4),
        ml_tenure_years=career.years_of_ml_experience,
        # Additional fields.
        capability_match_sparse=round(capability_match, 4),
        capability_match_dense=0.0,
        template_coherence=coherence.template_coherence,
        title_domain_confirmation=coherence.title_domain_confirmation,
        skill_career_alignment=coherence.skill_career_alignment,
        assessment_avg=platform.assessment_avg,
        github_norm=platform.github_norm,
        engagement_composite=platform.engagement_composite,
        verification_score=platform.verification_score,
        trusted_skill_ratio=skill_trust.trusted_skill_ratio,
        skill_inflation_risk=skill_trust.skill_inflation_risk,
        total_penalty=penalties.total_penalty,
        total_years_experience=candidate.profile.years_of_experience,
        domain_ml_ai=domain_one_hot[0],
        domain_ai_adjacent=domain_one_hot[1],
        domain_software=domain_one_hot[2],
        domain_non_tech=domain_one_hot[3],
        all_career_ml_depth_norm=career.all_career_ml_depth_norm,
        current_role_ml_depth_norm=career.current_role_ml_depth_norm,
        career_momentum_norm=career.career_momentum_norm,
        open_source_bonus=open_source_bonus,
        is_honeypot=float(coherence.is_honeypot),
        education_relevance=education,
    )

    components = ComponentScores(
        jd_fit=round(jd_fit, 4),
        career_evidence=round(career_score, 4),
        coherence=coherence.coherence_score,
        platform_trust=platform.platform_trust,
        availability=platform.availability_score,
        penalties=penalties.total_penalty,
        final_score=round(heuristic_score, 4),
        is_honeypot=coherence.is_honeypot,
        penalties_detail=penalties,
        ml_tenure_years=career.years_of_ml_experience,
    )
    return features, components, heuristic_score


def compute_heuristic_final_score(
    weights: dict,
    jd_fit: float,
    career_score: float,
    coherence_score: float,
    platform_trust: float,
    availability_score: float,
    penalties_total: float,
    is_honeypot: bool,
    honeypot_penalty_threshold: float = 0.30,
) -> float:
    raw = (
        weights["jd_fit"] * jd_fit
        + weights["career_evidence"] * career_score
        + weights["coherence"] * coherence_score
        + weights["platform_trust"] * platform_trust
        + weights["availability"] * availability_score
        - penalties_total
    )
    if is_honeypot and penalties_total > honeypot_penalty_threshold:
        raw *= 0.25
    return max(0.0, min(1.0, raw))


def _capability_match(
    candidate: Candidate,
    role_profile: RoleProfile,
    jd_embedding: np.ndarray | None,
    candidate_embeddings: dict[str, np.ndarray] | None,
) -> float:
    if candidate_embeddings is not None and jd_embedding is not None:
        return semantic_match_from_precomputed(
            candidate.candidate_id, jd_embedding, candidate_embeddings
        )
    return compute_sparse_capability_match(candidate, role_profile)


def _education_relevance(candidate: Candidate) -> float:
    if not candidate.education:
        return 0.0
    relevant = 0.0
    for edu in candidate.education:
        field = edu.field_of_study.lower().strip()
        if any(
            relevant_field == field or field.startswith(relevant_field)
            for relevant_field in _EDUCATION_RELEVANT_FIELDS
        ):
            relevant += 1.0
        if edu.tier in {"tier_1", "tier_2"}:
            relevant += 0.5
    return min(relevant / len(candidate.education), 1.0)


def _open_source_corpus(candidate: Candidate) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for text in [
        candidate.profile.summary,
        candidate.profile.headline,
        *[entry.description for entry in candidate.career_history],
    ]:
        for sentence in text.split("."):
            key = sentence.strip().lower()[:80]
            if key and key not in seen:
                seen.add(key)
                parts.append(sentence)
    return " ".join(parts).lower()


def _open_source_bonus(candidate: Candidate) -> float:
    corpus = _open_source_corpus(candidate)
    open_source_terms = [
        "open source",
        "open-source",
        "github contributions",
        "contributed to",
        "maintained",
        "published on github",
        "open sourced",
        "public repo",
        "pypi",
        "huggingface hub",
        "hf model hub",
    ]
    matched = sum(1 for term in open_source_terms if term in corpus)
    return min(matched / 2.0, 1.0)
