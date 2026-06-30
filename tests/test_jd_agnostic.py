"""JD-agnostic scoring: penalties and coherence follow parsed JD preferences."""

from __future__ import annotations

import pytest

from fitrank.career_analyzer import CareerEvidence
from fitrank.coherence import compute_coherence, title_domain_confirmation_score
from fitrank.jd_parser import parse_jd
from fitrank.models import Candidate, CareerEntry, JDPrefs, Profile, RedrobSignals, RoleProfile, SalaryRange, Skill
from fitrank.penalties import compute_penalties
from fitrank.title_gate import TitleDomain


def _signals() -> RedrobSignals:
    return RedrobSignals(
        80, "2025-12-01", "2025-12-01", True, 1, 1, 0.5, 1.0, {}, 1, 1, 30,
        SalaryRange(10, 20), "remote", True, -1, 1, 1, 0.5, -1, True, False, False,
    )


def _cv_candidate() -> Candidate:
    return Candidate(
        "CAND_CV",
        Profile("A", "h", "s", "Pune", "IN", 6.0, "Computer Vision Engineer", "Co", "201-500", "Tech"),
        [
            CareerEntry(
                "Co", "Computer Vision Engineer", "2020-01-01", None, 48, True,
                "Tech", "201-500",
                "object detection using YOLO and OpenCV for autonomous driving pipelines",
            )
        ],
        [],
        [Skill("OpenCV", "advanced", 2, 12)],
        _signals(),
    )


def _consulting_candidate() -> Candidate:
    return Candidate(
        "CAND_CONS",
        Profile("A", "h", "s", "Pune", "IN", 8.0, "ML Engineer", "Co", "201-500", "Tech"),
        [
            CareerEntry("TCS", "ML Engineer", "2018-01-01", "2020-01-01", 24, False, "Svc", "1000+", "ML at TCS"),
            CareerEntry("Infosys", "Senior ML Engineer", "2020-01-01", None, 48, True, "Svc", "1000+", "ML at Infosys"),
        ],
        [],
        [Skill("PyTorch", "advanced", 3, 24)],
        _signals(),
    )


def _research_candidate() -> Candidate:
    return Candidate(
        "CAND_PHD",
        Profile("A", "h", "s", "Pune", "IN", 7.0, "Research Scientist", "Co", "201-500", "Tech"),
        [
            CareerEntry(
                "Lab", "Research Scientist", "2018-01-01", None, 60, True,
                "Research", "50-200", "published papers on transformer architectures",
            )
        ],
        [],
        [Skill("PyTorch", "expert", 1, 12)],
        _signals(),
    )


NLP_JD = (
    "Senior NLP Engineer in Pune. Retrieval, ranking, transformers, RAG, embeddings. "
    "Production deployment required. Not consulting-only backgrounds."
)
CV_JD = (
    "Computer Vision Engineer in Bangalore. Object detection, OpenCV, CNN, YOLO. "
    "We need computer vision expertise."
)
CONSULTING_JD = (
    "Senior ML Engineer with consulting background welcome. Client-facing services "
    "experience valued. PyTorch and deployment."
)
RESEARCH_JD = (
    "Research Scientist with PhD and publications. Research background valued. "
    "Postdoc experience welcome. NLP and transformers."
)


@pytest.mark.parametrize(
    "jd_text,candidate_fn,penalty_attr,expect_penalty",
    [
        (NLP_JD, _cv_candidate, "domain_mismatch", True),
        (CV_JD, _cv_candidate, "domain_mismatch", False),
        (NLP_JD, _consulting_candidate, "consulting_only", True),
        (CONSULTING_JD, _consulting_candidate, "consulting_only", False),
        (NLP_JD, _research_candidate, "pure_research", True),
        (RESEARCH_JD, _research_candidate, "pure_research", False),
    ],
)
def test_conditional_penalty_matrix(jd_text, candidate_fn, penalty_attr, expect_penalty):
    role = parse_jd(jd_text)
    candidate = candidate_fn()
    career = CareerEvidence(5, 3, 0.3, "ml_work", 0)
    coherence = compute_coherence(candidate, career, TitleDomain.ML_AI, role)
    penalties = compute_penalties(candidate, career, coherence, role)
    fired = getattr(penalties, penalty_attr) >= 0.5
    assert fired == expect_penalty


def test_cv_jd_disables_domain_mismatch_penalty_flag():
    profile = parse_jd(CV_JD)
    assert profile.prefs.penalize_domain_mismatch is False
    assert profile.domain == "computer_vision"


def test_consulting_jd_disables_consulting_penalty_flag():
    profile = parse_jd(CONSULTING_JD)
    assert profile.prefs.penalize_consulting_only is False


def test_research_jd_disables_pure_research_penalty_flag():
    profile = parse_jd(RESEARCH_JD)
    assert profile.prefs.penalize_pure_research is False
    assert profile.prefs.values_production_experience is False


def test_jd_extracts_preferred_locations():
    profile = parse_jd("Senior ML Engineer based in Pune/Noida, India. Hybrid role.")
    assert "pune" in profile.prefs.preferred_locations
    assert "noida" in profile.prefs.preferred_locations


def test_cv_title_full_confirmation_under_cv_jd():
    role = parse_jd(CV_JD)
    candidate = _cv_candidate()
    career = CareerEvidence(5, 3, 0.3, "ml_work", 0)
    score = title_domain_confirmation_score(candidate, TitleDomain.ML_AI, career, role)
    assert score == pytest.approx(1.0)


def test_cv_title_reduced_confirmation_under_nlp_jd():
    role = parse_jd(NLP_JD)
    candidate = _cv_candidate()
    career = CareerEvidence(5, 3, 0.3, "ml_work", 0)
    score = title_domain_confirmation_score(candidate, TitleDomain.ML_AI, career, role)
    assert score == pytest.approx(0.3)


def test_empty_locations_neutral_bonus():
    from fitrank.signals import _location_bonus

    role = RoleProfile(prefs=JDPrefs(preferred_locations=[]))
    candidate = _cv_candidate()
    assert _location_bonus(candidate, role) == pytest.approx(0.5)
