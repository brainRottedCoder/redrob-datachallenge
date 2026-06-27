"""Redrob platform signal scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from fitrank.constants import INDIAN_TIER1
from fitrank.models import Candidate, RoleProfile

DEFAULT_NORMALIZATION = {
    "profile_views_received_30d": 120,
    "search_appearance_30d": 45,
    "saved_by_recruiters_30d": 18,
}

# Reference date for recency calculation (dataset spans roughly 2024-2026).
_RECENCY_CUTOFF = datetime(2025, 12, 1)


@dataclass
class PlatformTrustScore:
    assessment_avg: float
    assessment_jd_overlap: float
    github_norm: float
    engagement_composite: float
    verification_score: float
    availability_score: float
    platform_trust: float


def load_normalization_constants(path: str | Path = "config/normalization.yaml") -> dict[str, float]:
    source = Path(path)
    if not source.exists():
        return DEFAULT_NORMALIZATION
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    return {**DEFAULT_NORMALIZATION, **data}


def compute_platform_trust(
    candidate: Candidate,
    role_profile: RoleProfile,
    norms: dict[str, float] | None = None,
    weights: dict | None = None,
) -> PlatformTrustScore:
    from fitrank.config_loader import load_weights

    weights = weights or load_weights()
    trust_parts = weights.get("platform_trust_components", {})
    norms = norms or load_normalization_constants()
    signals = candidate.redrob_signals

    assessments = signals.skill_assessment_scores
    assessment_avg = (
        sum(assessments.values()) / len(assessments) / 100.0 if assessments else 0.0
    )
    assessment_overlap = _assessment_jd_overlap(assessments, role_profile.required_capabilities)
    github = (
        0.0
        if signals.github_activity_score < 0
        else signals.github_activity_score / 100.0
    )
    engagement = _engagement_composite(signals, norms)
    verification = (
        int(signals.verified_email)
        + int(signals.verified_phone)
        + int(signals.linkedin_connected)
    ) / 3.0
    availability = _availability_score(candidate, role_profile)

    certification_bonus = _certification_bonus(candidate)

    platform_trust = (
        trust_parts.get("assessment_avg", 0.35) * assessment_avg
        + trust_parts.get("github", 0.20) * github
        + trust_parts.get("engagement", 0.30) * engagement
        + trust_parts.get("verification", 0.15) * verification
        + 0.05 * certification_bonus
    )

    return PlatformTrustScore(
        assessment_avg=round(assessment_avg, 4),
        assessment_jd_overlap=round(assessment_overlap, 4),
        github_norm=round(github, 4),
        engagement_composite=round(engagement, 4),
        verification_score=round(verification, 4),
        availability_score=round(availability, 4),
        platform_trust=round(platform_trust, 4),
    )


def _availability_score(candidate: Candidate, role_profile: RoleProfile) -> float:
    """Availability and recruitability score.

    Combines open-to-work, notice period, relocation, work-mode fit,
    location fit, platform recency, and recruiter responsiveness.
    All weights sum to 1.0.
    """
    signals = candidate.redrob_signals

    work_mode_match = _work_mode_match(
        signals.preferred_work_mode,
        role_profile.preferred_work_mode,
    )
    otw = float(signals.open_to_work_flag)
    notice = 1.0 - min(signals.notice_period_days, 150) / 150.0
    relocate = float(signals.willing_to_relocate)
    location_bonus = _location_bonus(candidate)
    recency = _recency_score(signals)
    response = _response_score(signals)

    availability = (
        0.20 * otw
        + 0.20 * notice
        + 0.10 * relocate
        + 0.10 * work_mode_match
        + 0.15 * location_bonus
        + 0.15 * recency
        + 0.10 * response
    )
    return max(0.0, min(1.0, availability))


def _location_bonus(candidate: Candidate) -> float:
    """Small bonus for candidates in India/Tier-1 cities or willing to relocate."""
    signals = candidate.redrob_signals
    location = candidate.profile.location.lower()
    country = candidate.profile.country.lower()

    if signals.willing_to_relocate:
        return 0.8
    if country in {"india", "in"}:
        if any(city in location for city in INDIAN_TIER1):
            return 1.0
        return 0.5
    return 0.0


def _recency_score(signals) -> float:
    """Active candidates score higher."""
    try:
        last_active = datetime.strptime(signals.last_active_date, "%Y-%m-%d")
        if last_active >= _RECENCY_CUTOFF:
            return 1.0
        # Linear decay over 6 months before cutoff.
        days_since = (_RECENCY_CUTOFF - last_active).days
        return max(0.0, 1.0 - days_since / 180.0)
    except (ValueError, TypeError):
        return 0.5


def _response_score(signals) -> float:
    """Recruiter responsiveness and application activity."""
    response_rate = max(0.0, min(signals.recruiter_response_rate, 1.0))
    # Faster response is better; cap at 72 hours.
    avg_response = signals.avg_response_time_hours
    response_speed = max(0.0, 1.0 - min(avg_response, 72.0) / 72.0)
    # Recent application activity indicates job-seeking behavior.
    application_activity = min(signals.applications_submitted_30d / 5.0, 1.0)
    return 0.50 * response_rate + 0.25 * response_speed + 0.25 * application_activity


def _assessment_jd_overlap(
    assessments: dict[str, float],
    capabilities: list[str],
) -> float:
    if not assessments or not capabilities:
        return 0.0
    capability_lower = {cap.lower() for cap in capabilities}
    matched = 0
    for skill_name, score in assessments.items():
        if score < 60:
            continue
        name_lower = skill_name.lower()
        if any(_word_boundary_contains(name_lower, cap) for cap in capability_lower):
            matched += 1
    return matched / len(capabilities)


def _word_boundary_contains(text: str, substring: str) -> bool:
    """Return True if substring appears as a whole word in text."""
    import re

    pattern = r"(?<![a-z0-9])" + re.escape(substring) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _engagement_composite(signals, norms: dict[str, float]) -> float:
    saved = min(signals.saved_by_recruiters_30d / norms["saved_by_recruiters_30d"], 1.0)
    views = min(signals.profile_views_received_30d / norms["profile_views_received_30d"], 1.0)
    search = min(signals.search_appearance_30d / norms["search_appearance_30d"], 1.0)
    interview = max(0.0, min(signals.interview_completion_rate, 1.0))
    offer = 0.0 if signals.offer_acceptance_rate < 0 else min(signals.offer_acceptance_rate, 1.0)
    return (
        0.35 * saved
        + 0.15 * views
        + 0.10 * search
        + 0.25 * interview
        + 0.15 * offer
    )


def _certification_bonus(candidate) -> float:
    """Tiered bonus for relevant AI/ML/cloud/MLOps certifications."""
    if not candidate.certifications:
        return 0.0

    high_tier = {
        "aws machine learning", "azure ai", "google cloud professional machine learning",
        "tensorflow developer", "pytorch developer", "kubernetes", "mlops", "aws certified",
        "databricks", "hugging face", "nvidia deep learning", "google cloud professional data engineer",
    }
    general_tier = {
        "machine learning", "deep learning", "data science", "artificial intelligence",
        "aws", "azure", "gcp", "google cloud", "tensorflow", "pytorch",
        "docker", "mongodb", "python", "scala", "statistics",
    }
    high_score = 0.0
    general_score = 0.0
    for cert in candidate.certifications:
        name = cert.name.lower()
        if any(kw in name for kw in high_tier):
            high_score += 1.0
        elif any(kw in name for kw in general_tier):
            general_score += 1.0
    return min(0.5 * high_score + 0.25 * general_score, 1.0)


def _work_mode_match(candidate_mode: str, jd_mode: str) -> float:
    if jd_mode == "flexible" or not jd_mode:
        return 1.0
    if candidate_mode == jd_mode:
        return 1.0
    if jd_mode == "hybrid" and candidate_mode in {"remote", "onsite", "flexible"}:
        return 0.7
    if jd_mode == "remote" and candidate_mode == "flexible":
        return 0.8
    return 0.3
