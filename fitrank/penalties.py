"""Penalty computation for incoherent or inflated profiles."""

from __future__ import annotations

from dataclasses import dataclass

from fitrank.career_analyzer import CareerEvidence
from fitrank.coherence import CoherenceScore
from fitrank.models import Candidate, RoleProfile
from fitrank.skill_trust import analyze_skills, extract_ml_skills
from fitrank.title_gate import TitleDomain, classify_title


@dataclass
class PenaltyScore:
    template_mismatch: float
    skill_inflation: float
    shallow_boilerplate: float
    expert_zero_endorse: float
    non_ml_title_high_skill: float
    total_penalty: float


def compute_penalties(
    candidate: Candidate,
    career_evidence: CareerEvidence,
    coherence: CoherenceScore,
    role_profile: RoleProfile,
) -> PenaltyScore:
    skill_trust = analyze_skills(candidate.skills, role_profile)
    title_domain = classify_title(candidate.profile.current_title)

    template_mismatch = 1.0 if coherence.template_coherence == 0.0 else 0.0
    shallow = min(career_evidence.shallow_ai_count / 3.0, 1.0)
    expert_zero = 1.0 if any(
        skill.proficiency == "expert" and skill.endorsements == 0 for skill in candidate.skills
    ) else 0.0
    non_ml_high = (
        1.0
        if title_domain == TitleDomain.NON_TECH and len(extract_ml_skills(candidate.skills)) >= 7
        else 0.0
    )

    total = (
        0.25 * template_mismatch
        + 0.20 * skill_trust.skill_inflation_risk
        + 0.15 * shallow
        + 0.10 * expert_zero
        + 0.15 * non_ml_high
    )
    if coherence.is_honeypot:
        total = min(0.70, total + 0.25)

    return PenaltyScore(
        template_mismatch=template_mismatch,
        skill_inflation=skill_trust.skill_inflation_risk,
        shallow_boilerplate=shallow,
        expert_zero_endorse=expert_zero,
        non_ml_title_high_skill=non_ml_high,
        total_penalty=min(total, 0.70),
    )
