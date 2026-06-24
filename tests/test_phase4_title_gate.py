"""Phase 4 title gate tests (PRD T4.1-T4.12)."""

from __future__ import annotations

import pytest

from fitrank.loader import load_candidates
from fitrank.models import RoleProfile
from fitrank.title_gate import (
    TitleDomain,
    classify_title,
    title_domain_bonus,
    title_jd_match_targets,
)


@pytest.fixture
def ml_engineer_role() -> RoleProfile:
    return RoleProfile(target_titles=["ML Engineer"])


def test_t4_1_ml_title_classification():
    assert classify_title("Senior NLP Engineer") == TitleDomain.ML_AI


def test_t4_2_data_scientist():
    assert classify_title("Data Scientist") == TitleDomain.ML_AI


def test_t4_3_hr_manager():
    assert classify_title("HR Manager") == TitleDomain.NON_TECH


def test_t4_4_accountant():
    assert classify_title("Accountant") == TitleDomain.NON_TECH


def test_t4_5_software_engineer():
    assert classify_title("Full Stack Developer") == TitleDomain.SOFTWARE


def test_t4_6_ai_specialist_trap():
    assert classify_title("AI Specialist") == TitleDomain.AI_ADJACENT


def test_t4_7_computer_vision():
    assert classify_title("Computer Vision Engineer") == TitleDomain.ML_AI


def test_t4_8_jd_match_exact(ml_engineer_role):
    score = title_jd_match_targets("ML Engineer", ml_engineer_role.target_titles)
    assert score >= 0.9


def test_t4_9_jd_match_near(ml_engineer_role):
    score = title_jd_match_targets(
        "Machine Learning Engineer",
        ml_engineer_role.target_titles,
    )
    assert score >= 0.6


def test_t4_10_jd_match_miss(ml_engineer_role):
    score = title_jd_match_targets("HR Manager", ml_engineer_role.target_titles)
    assert score <= 0.1


def test_t4_11_domain_bonus_correctness():
    assert title_domain_bonus(TitleDomain.ML_AI) == 1.0
    assert title_domain_bonus(TitleDomain.AI_ADJACENT) == 0.5
    assert title_domain_bonus(TitleDomain.SOFTWARE) == 0.3
    assert title_domain_bonus(TitleDomain.NON_TECH) == 0.0


def test_t4_12_all_candidates_classified(candidates_path):
    for candidate in load_candidates(candidates_path):
        domain = classify_title(candidate.profile.current_title)
        assert domain is not None
        assert isinstance(domain, TitleDomain)
