"""Human-readable per-candidate reasoning strings."""

from __future__ import annotations

from fitrank.career_analyzer import DEEP_ML_PATTERNS
from fitrank.constants import (
    CONSULTING_FIRMS,
    CV_SPEECH_ROBOTICS_TITLES,
    INDIAN_TIER1,
    ML_KEYWORDS,
    RESEARCH_ONLY_TITLES,
)
from fitrank.models import Candidate
from fitrank.ranker import ComponentScores


# Keywords pulled from the deep ML pattern list for reasoning extraction.
_ML_KEYWORDS = ML_KEYWORDS


def _find_current_role_evidence(candidate: Candidate) -> list[str]:
    """Extract concrete ML/IR keywords from the current role description."""
    current_roles = [entry for entry in candidate.career_history if entry.is_current]
    if not current_roles:
        return []
    text = current_roles[-1].description.lower()
    found = []
    for keyword in _ML_KEYWORDS:
        if keyword.lower() in text and keyword not in found:
            found.append(keyword)
        if len(found) >= 4:
            break
    return found


def _top_assessments(candidate: Candidate) -> list[tuple[str, float]]:
    """Return top 2 skill assessments that are relevant to the JD."""
    assessments = candidate.redrob_signals.skill_assessment_scores
    if not assessments:
        return []
    ml_assessments = [
        (name, score)
        for name, score in assessments.items()
        if any(kw.lower() in name.lower() for kw in _ML_KEYWORDS)
    ]
    ml_assessments.sort(key=lambda x: x[1], reverse=True)
    return ml_assessments[:2]


def _describe_availability(candidate: Candidate) -> list[str]:
    """Describe availability signals."""
    signals = candidate.redrob_signals
    notes = []
    if signals.open_to_work_flag:
        notes.append("OTW")
    if signals.notice_period_days <= 30:
        notes.append("≤30d notice")
    elif signals.notice_period_days <= 60:
        notes.append(f"{signals.notice_period_days}d notice")
    else:
        notes.append(f"{signals.notice_period_days}d notice")
    if signals.willing_to_relocate:
        notes.append("reloc")
    return notes


def _describe_location(candidate: Candidate) -> str | None:
    """Return a short location note if India/Tier-1."""
    location = candidate.profile.location.lower()
    country = candidate.profile.country.lower()
    if country in {"india", "in"}:
        for city in INDIAN_TIER1:
            if city in location:
                return city.title()
        return "India"
    return None


def _estimate_ml_tenure(candidate: Candidate) -> float:
    """Estimate years of ML-relevant experience from career history."""
    total_ml_months = 0.0
    for entry in candidate.career_history:
        text = (entry.title + " " + entry.description).lower()
        if any(pattern.search(text) for pattern in DEEP_ML_PATTERNS):
            total_ml_months += entry.duration_months
    return round(total_ml_months / 12.0, 1)


def _describe_penalty_reasons(components: ComponentScores) -> list[str]:
    """Return explicit, human-readable penalty reasons."""
    p = components.penalties_detail
    reasons: list[str] = []
    if components.is_honeypot:
        reasons.append("honeypot")
    if p.template_mismatch >= 0.5:
        reasons.append("template mismatch")
    if p.skill_inflation >= 0.5:
        reasons.append("skill inflation")
    if p.shallow_boilerplate >= 0.5:
        reasons.append("shallow boilerplate")
    if p.expert_zero_endorse >= 0.5:
        reasons.append("expert zero-endorse")
    if p.non_ml_title_high_skill >= 0.5:
        reasons.append("title/skill mismatch")
    if p.consulting_only >= 0.5:
        reasons.append("consulting-only")
    if p.pure_research >= 0.5:
        reasons.append("pure research")
    if p.cv_without_nlp >= 0.5:
        reasons.append("CV/speech without NLP/IR")
    if p.salary_inverted >= 0.5:
        reasons.append("inverted salary")
    if p.experience_gap >= 0.5:
        reasons.append("experience gap")
    if p.job_hopper >= 0.5:
        reasons.append("job hopper")
    return reasons


def build_reasoning(
    candidate: Candidate,
    components: ComponentScores,
) -> str:
    """Build a concise, candidate-specific reasoning string."""
    title = candidate.profile.current_title
    exp = candidate.profile.years_of_experience

    fragments: list[str] = []

    role_evidence = _find_current_role_evidence(candidate)
    if role_evidence:
        fragments.append(f"current role: {','.join(role_evidence[:3])}")
    else:
        fragments.append("no strong ML evidence in current role")

    assessments = _top_assessments(candidate)
    if assessments:
        assess_str = ", ".join(f"{name}={score:.0f}" for name, score in assessments)
        fragments.append(f"assessments {assess_str}")
    else:
        fragments.append("no assessments")

    gh = candidate.redrob_signals.github_activity_score
    if gh is not None and gh >= 0:
        fragments.append(f"GH {gh:.0f}")
    else:
        fragments.append("no GH")

    ml_tenure = _estimate_ml_tenure(candidate)
    avail = _describe_availability(candidate)
    location = _describe_location(candidate)
    exp_avail = f"{exp:.1f}yrs total/{ml_tenure:.1f}yrs ML"
    if location:
        exp_avail += f" {location}"
    if avail:
        exp_avail += f" ({'/'.join(avail)})"
    fragments.append(exp_avail)

    penalty_reasons = _describe_penalty_reasons(components)
    if penalty_reasons:
        fragments.append(f"| {','.join(penalty_reasons)} ({components.penalties:.2f})")
    elif components.penalties >= 0.05:
        fragments.append(f"| minor penalties ({components.penalties:.2f})")

    scores = (
        f"JD={components.jd_fit:.2f} Career={components.career_evidence:.2f} "
        f"Coh={components.coherence:.2f} Trust={components.platform_trust:.2f}"
    )

    evidence_line = "; ".join(fragments)
    return f"{title} | {scores} | {evidence_line}"
