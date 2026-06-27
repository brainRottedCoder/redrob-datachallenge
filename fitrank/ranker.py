"""Score aggregation and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import heapq

from fitrank.career_analyzer import analyze_career
from fitrank.coherence import compute_coherence
from fitrank.config_loader import load_weights as _load_weights
from fitrank.models import Candidate, CandidateScore, RoleProfile
from fitrank.penalties import PenaltyScore, compute_penalties
from fitrank.embedder import (
    compute_semantic_capability_match,
    load_candidate_vectors,
    semantic_match_from_precomputed,
)
from fitrank.signals import compute_platform_trust, load_normalization_constants
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
    penalties_detail: PenaltyScore
    ml_tenure_years: float = 0.0


def load_weights(path: str | Path = "config/weights.yaml") -> dict:
    return _load_weights(path)


def score_candidate(
    candidate: Candidate,
    role_profile: RoleProfile,
    weights: dict | None = None,
    norms: dict[str, float] | None = None,
    capability_vectors: dict[str, dict[str, float]] | None = None,
) -> tuple[ComponentScores, CandidateScore]:
    weights = weights or load_weights()
    norms = norms or load_normalization_constants()
    career = analyze_career(candidate)
    title_domain = classify_title(candidate.profile.current_title)
    coherence = compute_coherence(candidate, career, title_domain)
    penalties = compute_penalties(candidate, career, coherence, role_profile, weights)
    platform = compute_platform_trust(candidate, role_profile, weights=weights, norms=norms)

    jd_parts = weights.get("jd_fit_components", {})
    title_match = title_jd_match(candidate.profile.current_title, role_profile)
    capability_match = _capability_match(candidate, role_profile, capability_vectors)
    education = _education_relevance(candidate)
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

    raw = (
        weights["jd_fit"] * jd_fit
        + weights["career_evidence"] * career_score
        + weights["coherence"] * coherence.coherence_score
        + weights["platform_trust"] * platform.platform_trust
        + weights["availability"] * platform.availability_score
        - penalties.total_penalty
    )
    # Apply the honeypot multiplier only when structural incoherence is confirmed by
    # explicit penalty signals, to avoid over-penalizing data-quality edge cases.
    if coherence.is_honeypot and penalties.total_penalty > 0.30:
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
        penalties_detail=penalties,
        ml_tenure_years=career.years_of_ml_experience,
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
    norms: dict[str, float] | None = None,
    top_n: int = 100,
    capability_vectors: dict[str, dict[str, float]] | None = None,
) -> list[tuple[Candidate, ComponentScores, CandidateScore]]:
    weights = weights or load_weights()
    norms = norms or load_normalization_constants()
    heap: list[tuple[tuple[float, str], Candidate, ComponentScores, CandidateScore]] = []
    for candidate in candidates:
        components, candidate_score = score_candidate(
            candidate, role_profile, weights, norms, capability_vectors
        )
        key = (candidate_score.final_score, candidate.candidate_id)
        if len(heap) < top_n:
            heapq.heappush(heap, (key, candidate, components, candidate_score))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, candidate, components, candidate_score))

    ranked = sorted(heap, key=lambda item: (-item[0][0], item[0][1]))
    return [(item[1], item[2], item[3]) for item in ranked]


def _capability_match(
    candidate: Candidate,
    role_profile: RoleProfile,
    capability_vectors: dict[str, dict[str, float]] | None = None,
) -> float:
    """Semantic overlap between candidate text and JD capabilities."""
    if capability_vectors is not None:
        return semantic_match_from_precomputed(
            candidate.candidate_id, role_profile, capability_vectors
        )
    return compute_semantic_capability_match(candidate, role_profile)


DEFAULT_VECTORS_PATH = Path("outputs/candidate_vectors.jsonl")


def load_capability_vectors(path: str | Path | None = None) -> dict[str, dict[str, float]] | None:
    """Load pre-computed vectors when the cache file exists."""
    source = Path(path) if path else DEFAULT_VECTORS_PATH
    if not source.exists():
        return None
    return load_candidate_vectors(source)


_EDUCATION_RELEVANT_FIELDS = {
    "computer science", "artificial intelligence", "machine learning", "data science",
    "software engineering", "statistics", "mathematics", "data engineering",
    "information technology", "computer engineering", "electrical engineering",
}


def _education_relevance(candidate: Candidate) -> float:
    if not candidate.education:
        return 0.0
    relevant = 0.0
    for edu in candidate.education:
        field = edu.field_of_study.lower().strip()
        if any(relevant_field == field or field.startswith(relevant_field) for relevant_field in _EDUCATION_RELEVANT_FIELDS):
            relevant += 1.0
        if edu.tier in {"tier_1", "tier_2"}:
            relevant += 0.5
    return min(relevant / len(candidate.education), 1.0)


def _open_source_corpus(candidate: Candidate) -> str:
    """Build the open-source search corpus once per candidate, deduplicating sentences."""
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
    """Small bonus for explicit open-source / GitHub contribution language."""
    corpus = _open_source_corpus(candidate)
    open_source_terms = [
        "open source", "open-source", "github contributions", "contributed to",
        "maintained", "published on github", "open sourced", "public repo",
        "pypi", "huggingface hub", "hf model hub",
    ]
    matched = sum(1 for term in open_source_terms if term in corpus)
    return min(matched / 2.0, 1.0)
