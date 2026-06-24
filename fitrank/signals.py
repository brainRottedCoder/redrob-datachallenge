"""Redrob platform signal scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from fitrank.models import Candidate, RoleProfile

DEFAULT_NORMALIZATION = {
    "profile_views_received_30d": 120,
    "search_appearance_30d": 45,
    "saved_by_recruiters_30d": 18,
}


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
) -> PlatformTrustScore:
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
    availability = (
        0.40 * float(signals.open_to_work_flag)
        + 0.30 * (1.0 - min(signals.notice_period_days, 150) / 150.0)
        + 0.30 * float(signals.willing_to_relocate)
    )

    platform_trust = (
        0.35 * assessment_avg
        + 0.20 * github
        + 0.30 * engagement
        + 0.15 * verification
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


def _assessment_jd_overlap(
    assessments: dict[str, float],
    capabilities: list[str],
) -> float:
    if not assessments or not capabilities:
        return 0.0
    capability_lower = {cap.lower() for cap in capabilities}
    matched = 0
    for skill_name, score in assessments.items():
        if score >= 60 and any(cap in skill_name.lower() for cap in capability_lower):
            matched += 1
    return matched / len(capabilities)


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
