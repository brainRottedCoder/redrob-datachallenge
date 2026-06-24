"""Phase 7 skill trust and penalty tests."""

from fitrank.models import RoleProfile, Skill
from fitrank.skill_trust import analyze_skills


def test_t7_1_trusted_skill_pass():
    skills = [Skill("PyTorch", "advanced", 12, 36)]
    trust = analyze_skills(skills, RoleProfile())
    assert trust.trusted_ml_skills_count == 1


def test_t7_3_trusted_skill_fail_endorse():
    skills = [Skill("PyTorch", "advanced", 0, 36)]
    trust = analyze_skills(skills, RoleProfile())
    assert trust.trusted_ml_skills_count == 0


def test_t7_5_skill_inflation_high():
    skills = [
        Skill("PyTorch", "advanced", 1, 3),
        Skill("NLP", "expert", 1, 2),
        Skill("RAG", "advanced", 1, 4),
        Skill("LLM", "advanced", 1, 5),
        Skill("LoRA", "expert", 1, 1),
    ]
    trust = analyze_skills(skills, RoleProfile())
    assert trust.skill_inflation_risk >= 0.8
