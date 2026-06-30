"""Human-readable per-candidate reasoning strings."""

from __future__ import annotations

from dataclasses import dataclass

from fitrank.constants import ML_KEYWORDS
from fitrank.models import Candidate, RoleProfile
from fitrank.ranker import ComponentScores


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


def _describe_location(candidate: Candidate, role_profile: RoleProfile | None = None) -> str | None:
    """Return a short location note when JD or profile indicates geography."""
    location = candidate.profile.location.lower()
    country = candidate.profile.country.lower()
    preferred = role_profile.prefs.preferred_locations if role_profile else []

    if preferred:
        for loc in preferred:
            if loc in location or loc in country:
                return loc.title()
        if country in {"india", "in"} and "india" in preferred:
            return "India"
        return None

    if country in {"india", "in"}:
        return candidate.profile.location.split(",")[0].strip().title() or "India"
    return None


def _domain_mismatch_label(role_profile: RoleProfile | None) -> str:
    if role_profile and role_profile.prefs.domain:
        domain = role_profile.prefs.domain.replace("_", " ")
        return f"domain mismatch ({domain})"
    return "domain mismatch"


def _describe_penalty_reasons(
    components: ComponentScores,
    role_profile: RoleProfile | None = None,
) -> list[str]:
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
    if p.domain_mismatch >= 0.5:
        reasons.append(_domain_mismatch_label(role_profile))
    if p.salary_inverted >= 0.5:
        reasons.append("inverted salary")
    if p.experience_gap >= 0.5:
        reasons.append("experience gap")
    if p.job_hopper >= 0.5:
        reasons.append("job hopper")
    if p.seniority_mismatch >= 0.5:
        reasons.append("seniority gap")
    return reasons


@dataclass(frozen=True)
class ReasoningEvidence:
    """Structured evidence shared by template and LLM reasoning paths."""

    candidate_id: str
    title: str
    years_of_experience: float
    ml_tenure_years: float
    jd_fit: float
    career_evidence: float
    coherence: float
    platform_trust: float
    penalties: float
    is_honeypot: bool
    role_evidence: list[str]
    assessments: list[tuple[str, float]]
    github_score: float | None
    availability_notes: list[str]
    location: str | None
    penalty_reasons: list[str]
    score_line: str
    fragments: list[str]


def _build_score_line(components: ComponentScores) -> str:
    scores = (
        f"JD={components.jd_fit:.2f} Career={components.career_evidence:.2f} "
        f"Coh={components.coherence:.2f} Trust={components.platform_trust:.2f}"
    )
    if components.is_honeypot:
        scores += " [HONEYPOT ×0.25]"
    return scores


def _gather_evidence(
    candidate: Candidate,
    components: ComponentScores,
    ml_tenure: float | None = None,
    role_profile: RoleProfile | None = None,
) -> ReasoningEvidence:
    """Collect candidate-specific facts for template or LLM reasoning."""
    title = candidate.profile.current_title
    exp = candidate.profile.years_of_experience
    ml_tenure_val = ml_tenure if ml_tenure is not None else components.ml_tenure_years

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
    github_score: float | None = gh if gh is not None and gh >= 0 else None
    if github_score is not None:
        fragments.append(f"GH {github_score:.0f}")
    else:
        fragments.append("no GH")

    avail = _describe_availability(candidate)
    location = _describe_location(candidate, role_profile)
    exp_avail = f"{exp:.1f}yrs total/{ml_tenure_val:.1f}yrs ML"
    if location:
        exp_avail += f" {location}"
    if avail:
        exp_avail += f" ({'/'.join(avail)})"
    fragments.append(exp_avail)

    penalty_reasons = _describe_penalty_reasons(components, role_profile)
    if penalty_reasons:
        fragments.append(f"| {','.join(penalty_reasons)} ({components.penalties:.2f})")
    elif components.penalties >= 0.05:
        fragments.append(f"| minor penalties ({components.penalties:.2f})")

    return ReasoningEvidence(
        candidate_id=candidate.candidate_id,
        title=title,
        years_of_experience=exp,
        ml_tenure_years=ml_tenure_val,
        jd_fit=components.jd_fit,
        career_evidence=components.career_evidence,
        coherence=components.coherence,
        platform_trust=components.platform_trust,
        penalties=components.penalties,
        is_honeypot=components.is_honeypot,
        role_evidence=role_evidence,
        assessments=assessments,
        github_score=github_score,
        availability_notes=avail,
        location=location,
        penalty_reasons=penalty_reasons,
        score_line=_build_score_line(components),
        fragments=fragments,
    )


def build_trap_explanation(
    candidate: Candidate,
    components: ComponentScores,
    role_profile: RoleProfile,
) -> str:
    """Structured explanation for trap/honeypot profiles in the demo."""
    lines: list[str] = []
    title = candidate.profile.current_title
    lines.append(f"**{title}** ({candidate.candidate_id})")

    penalty_reasons = _describe_penalty_reasons(components, role_profile)
    if penalty_reasons:
        lines.append("Penalties: " + ", ".join(penalty_reasons))
    else:
        lines.append("Penalties: none significant")

    if components.is_honeypot:
        lines.append(
            "Coherence failed: profile flagged as honeypot "
            f"(score={components.coherence:.2f})"
        )
    elif components.coherence < 0.3:
        lines.append(f"Low coherence score ({components.coherence:.2f})")

    if role_profile.prefs.penalize_domain_mismatch and components.penalties_detail.domain_mismatch >= 0.5:
        keywords = ", ".join(role_profile.prefs.required_evidence_keywords[:5])
        lines.append(f"JD expects evidence: {keywords or 'general ML'}")

    lines.append(
        f"FitRank score={components.final_score:.3f} "
        f"(JD={components.jd_fit:.2f}, Career={components.career_evidence:.2f}, "
        f"Coh={components.coherence:.2f})"
    )
    return "\n\n".join(lines)


def build_reasoning(
    candidate: Candidate,
    components: ComponentScores,
    ml_tenure: float | None = None,
    role_profile: RoleProfile | None = None,
) -> str:
    """Build a concise, candidate-specific reasoning string."""
    evidence = _gather_evidence(candidate, components, ml_tenure, role_profile)
    evidence_line = "; ".join(evidence.fragments)
    return f"{evidence.title} | {evidence.score_line} | {evidence_line}"
