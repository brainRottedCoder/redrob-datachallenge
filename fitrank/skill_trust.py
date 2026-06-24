"""ML skill trust analysis."""

from __future__ import annotations

from dataclasses import dataclass

from fitrank.coherence import _is_ml_skill
from fitrank.models import RoleProfile, Skill


@dataclass
class SkillTrustScore:
    trusted_ml_skills_count: int
    all_ml_skills_count: int
    trusted_skill_ratio: float
    skill_inflation_risk: float


def analyze_skills(skills: list[Skill], role_profile: RoleProfile) -> SkillTrustScore:
    del role_profile  # reserved for JD-conditioned weighting in later iterations
    ml_skills = extract_ml_skills(skills)
    trusted = [skill for skill in ml_skills if _is_trusted(skill)]
    inflation = _skill_inflation_risk(ml_skills)
    ratio = len(trusted) / max(1, len(ml_skills))
    return SkillTrustScore(
        trusted_ml_skills_count=len(trusted),
        all_ml_skills_count=len(ml_skills),
        trusted_skill_ratio=ratio,
        skill_inflation_risk=inflation,
    )


def extract_ml_skills(skills: list[Skill]) -> list[Skill]:
    return [skill for skill in skills if _is_ml_skill(skill.name)]


def _is_trusted(skill: Skill) -> bool:
    return skill.duration_months >= 12 and skill.endorsements > 0


def _skill_inflation_risk(ml_skills: list[Skill]) -> float:
    if not ml_skills:
        return 0.0
    inflated = [
        skill
        for skill in ml_skills
        if skill.proficiency in {"advanced", "expert"} and skill.duration_months < 6
    ]
    return len(inflated) / len(ml_skills)
