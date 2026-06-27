"""Penalty computation for incoherent or inflated profiles."""

from __future__ import annotations

from dataclasses import dataclass

from fitrank.career_analyzer import CareerEvidence
from fitrank.coherence import CoherenceScore
from fitrank.constants import (
    CONSULTING_FIRMS,
    CV_SPEECH_ROBOTICS_TITLES,
    PHD_TITLE_PATTERNS,
    RESEARCH_ONLY_TITLES,
)
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
    consulting_only: float
    pure_research: float
    cv_without_nlp: float
    salary_inverted: float
    experience_gap: float
    job_hopper: float
    seniority_mismatch: float
    total_penalty: float


def _consulting_only_penalty(candidate: Candidate) -> float:
    """Penalty if all career history is at consulting/services firms."""
    if not candidate.career_history:
        return 0.0
    for entry in candidate.career_history:
        company = entry.company.lower()
        if not any(firm in company for firm in CONSULTING_FIRMS):
            return 0.0
    return 1.0


def _is_research_title(title: str) -> bool:
    """Check whether the title is a research-only title."""
    lowered = title.lower()
    if any(rt in lowered for rt in RESEARCH_ONLY_TITLES):
        return True
    return any(lowered.startswith(pattern) or lowered.endswith(pattern) for pattern in PHD_TITLE_PATTERNS)


def _pure_research_penalty(candidate: Candidate, career_evidence: CareerEvidence) -> float:
    """Penalty for research-only titles without production deployment evidence."""
    if not _is_research_title(candidate.profile.current_title):
        return 0.0
    production_terms = [
        "deployment", "serving", "production", "monitoring", "ab test", "a/b test",
        "online", "production model", "model serving", "inference serving",
    ]
    career_text = " ".join(entry.description.lower() for entry in candidate.career_history)
    if any(term in career_text for term in production_terms):
        return 0.0
    return 1.0


def _cv_without_nlp_penalty(candidate: Candidate, career_evidence: CareerEvidence) -> float:
    """Penalty for CV/speech/robotics titles without NLP/IR evidence."""
    title = candidate.profile.current_title.lower()
    if not any(cv in title for cv in CV_SPEECH_ROBOTICS_TITLES):
        return 0.0
    nlp_ir_terms = [
        "nlp", "natural language", "retrieval", "ranking", "search", "transformer",
        "bert", "llm", "rag", "semantic search", "embedding",
    ]
    career_text = " ".join(entry.description.lower() for entry in candidate.career_history)
    if any(term in career_text for term in nlp_ir_terms):
        return 0.0
    return 1.0


def _salary_inverted_penalty(candidate: Candidate) -> float:
    """Penalty for impossible salary ranges (min > max)."""
    salary = candidate.redrob_signals.expected_salary_range_inr_lpa
    if salary.min > salary.max:
        return 1.0
    return 0.0


def _experience_gap_penalty(candidate: Candidate, role_profile: RoleProfile) -> float:
    """Soft penalty for candidates well below the JD's experience requirement."""
    min_exp = max(role_profile.min_experience_years - 1.0, 0.0)
    if candidate.profile.years_of_experience < min_exp:
        return 1.0
    return 0.0


def _job_hopper_penalty(candidate: Candidate) -> float:
    """Penalty for very short average tenure across many roles."""
    history = candidate.career_history
    if len(history) < 4:
        return 0.0
    total_months = sum(entry.duration_months for entry in history)
    avg_months = total_months / len(history)
    if avg_months < 8:
        return 1.0
    return 0.0


def _seniority_mismatch_penalty(
    candidate: Candidate,
    career_evidence,
    role_profile: RoleProfile,
) -> float:
    """Soft penalty when JD expects senior experience but candidate lacks ML tenure.

    Only fires when role_profile.seniority is 'senior' AND the candidate's computed
    ML-specific experience is below the threshold, not just total years of experience.
    """
    if role_profile.seniority != "senior":
        return 0.0
    ml_years = getattr(career_evidence, "years_of_ml_experience", 0.0)
    if ml_years < 3.0:
        return 1.0
    return 0.0


def compute_penalties(
    candidate: Candidate,
    career_evidence: CareerEvidence,
    coherence: CoherenceScore,
    role_profile: RoleProfile,
    weights: dict | None = None,
) -> PenaltyScore:
    from fitrank.config_loader import load_weights

    weights = weights or load_weights()
    penalty_weights = weights.get("penalty_weights", {})
    skill_trust = analyze_skills(candidate.skills, role_profile)
    title_domain = classify_title(candidate.profile.current_title)

    template_mismatch = 1.0 if (
        coherence.template_coherence < 0.1
        and career_evidence.template_domain != "unknown"
    ) else 0.0
    shallow = min(career_evidence.shallow_ai_count / 3.0, 1.0)
    expert_zero = 1.0 if any(
        skill.proficiency == "expert" and skill.endorsements == 0 for skill in candidate.skills
    ) else 0.0
    ml_skills = extract_ml_skills(candidate.skills)
    non_ml_high = 1.0 if (
        title_domain in {TitleDomain.NON_TECH, TitleDomain.AI_ADJACENT}
        and len(ml_skills) >= (9 if title_domain == TitleDomain.AI_ADJACENT else 7)
    ) else 0.0

    consulting_only = _consulting_only_penalty(candidate)
    pure_research = _pure_research_penalty(candidate, career_evidence)
    cv_without_nlp = _cv_without_nlp_penalty(candidate, career_evidence)
    salary_inverted = _salary_inverted_penalty(candidate)
    experience_gap = _experience_gap_penalty(candidate, role_profile)
    job_hopper = _job_hopper_penalty(candidate)
    seniority_mismatch = _seniority_mismatch_penalty(candidate, career_evidence, role_profile)

    total = (
        penalty_weights.get("template_mismatch", 0.20) * template_mismatch
        + penalty_weights.get("skill_inflation", 0.15) * skill_trust.skill_inflation_risk
        + penalty_weights.get("shallow_boilerplate", 0.10) * shallow
        + penalty_weights.get("expert_zero_endorse", 0.08) * expert_zero
        + penalty_weights.get("non_ml_title_high_skill", 0.10) * non_ml_high
        + penalty_weights.get("consulting_only", 0.07) * consulting_only
        + penalty_weights.get("pure_research", 0.07) * pure_research
        + penalty_weights.get("cv_without_nlp", 0.07) * cv_without_nlp
        + penalty_weights.get("salary_inverted", 0.07) * salary_inverted
        + penalty_weights.get("experience_gap", 0.05) * experience_gap
        + penalty_weights.get("job_hopper", 0.04) * job_hopper
        + penalty_weights.get("seniority_mismatch", 0.04) * seniority_mismatch
    )
    if coherence.is_honeypot:
        total = min(0.70, total + 0.25)

    return PenaltyScore(
        template_mismatch=template_mismatch,
        skill_inflation=skill_trust.skill_inflation_risk,
        shallow_boilerplate=shallow,
        expert_zero_endorse=expert_zero,
        non_ml_title_high_skill=non_ml_high,
        consulting_only=consulting_only,
        pure_research=pure_research,
        cv_without_nlp=cv_without_nlp,
        salary_inverted=salary_inverted,
        experience_gap=experience_gap,
        job_hopper=job_hopper,
        seniority_mismatch=seniority_mismatch,
        total_penalty=min(total, 0.70),
    )
