"""Additional tests for new signals, penalties, and improvements introduced in round 3."""

from __future__ import annotations

import pytest

from fitrank.career_analyzer import (
    CareerEvidence,
    analyze_career,
    score_career_momentum,
    _years_of_ml_experience,
)
from fitrank.coherence import compute_coherence
from fitrank.models import (
    Candidate,
    CareerEntry,
    Profile,
    RedrobSignals,
    RoleProfile,
    SalaryRange,
    Skill,
)
from fitrank.penalties import compute_penalties
from fitrank.signals import compute_platform_trust
from fitrank.title_gate import TitleDomain, classify_title


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signals(**kwargs):
    defaults = dict(
        profile_completeness_score=80,
        signup_date="2024-01-01",
        last_active_date="2025-12-15",
        open_to_work_flag=True,
        profile_views_received_30d=10,
        applications_submitted_30d=2,
        recruiter_response_rate=0.7,
        avg_response_time_hours=4.0,
        skill_assessment_scores={},
        connection_count=100,
        endorsements_received=5,
        notice_period_days=30,
        expected_salary_range_inr_lpa=SalaryRange(20, 40),
        preferred_work_mode="remote",
        willing_to_relocate=True,
        github_activity_score=50,
        search_appearance_30d=5,
        saved_by_recruiters_30d=3,
        interview_completion_rate=0.8,
        offer_acceptance_rate=0.6,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )
    defaults.update(kwargs)
    return RedrobSignals(**defaults)


def _profile(title="ML Engineer", yoe=5.0):
    return Profile("A", "h", "s", "Bangalore", "India", yoe, title, "Co", "201-500", "Tech")


def _entry(**kwargs):
    defaults = dict(
        company="Co",
        title="ML Eng",
        start_date="2022-01-01",
        end_date=None,
        duration_months=24,
        is_current=True,
        industry="Tech",
        company_size="201-500",
        description="",
    )
    defaults.update(kwargs)
    return CareerEntry(**defaults)


def _candidate(title="ML Engineer", yoe=5.0, history=None, skills=None, **signal_kwargs):
    return Candidate(
        "CAND_TEST001",
        _profile(title, yoe),
        history or [],
        [],
        skills or [],
        _signals(**signal_kwargs),
    )


# ---------------------------------------------------------------------------
# Title gate — new patterns from round 2
# ---------------------------------------------------------------------------


def test_llm_engineer_is_ml_ai():
    assert classify_title("LLM Engineer") == TitleDomain.ML_AI


def test_senior_llm_engineer_is_ml_ai():
    assert classify_title("Senior LLM Engineer") == TitleDomain.ML_AI


def test_genai_engineer_is_ml_ai():
    assert classify_title("GenAI Engineer") == TitleDomain.ML_AI


def test_generative_ai_engineer_is_ml_ai():
    assert classify_title("Generative AI Engineer") == TitleDomain.ML_AI


def test_data_science_lead_is_ml_ai():
    assert classify_title("Data Science Lead") == TitleDomain.ML_AI


def test_ranking_engineer_is_ml_ai():
    assert classify_title("Ranking Engineer") == TitleDomain.ML_AI


def test_recommendation_engineer_is_ml_ai():
    assert classify_title("Recommendation Engineer") == TitleDomain.ML_AI


def test_search_scientist_is_ml_ai():
    assert classify_title("Search Scientist") == TitleDomain.ML_AI


def test_nlp_scientist_is_ml_ai():
    assert classify_title("NLP Scientist") == TitleDomain.ML_AI


# ---------------------------------------------------------------------------
# Career analyzer — years_of_ml_experience
# ---------------------------------------------------------------------------


def test_years_of_ml_experience_genuine():
    history = [
        _entry(
            description="fine-tuned LLaMA-2 using LoRA; deployed BentoML model serving",
            duration_months=24,
            is_current=True,
        ),
        _entry(
            description="built RAG pipeline with FAISS and embedding retrieval",
            duration_months=12,
            is_current=False,
            end_date="2023-01-01",
        ),
    ]
    result = _years_of_ml_experience(history)
    assert result == pytest.approx(3.0)


def test_years_of_ml_experience_no_ml_roles():
    history = [
        _entry(description="managed customer support tickets", duration_months=24),
        _entry(description="sales enterprise cloud software", duration_months=12, is_current=False),
    ]
    result = _years_of_ml_experience(history)
    assert result == 0.0


def test_years_of_ml_experience_in_career_evidence():
    """CareerEvidence.years_of_ml_experience is populated correctly."""
    candidate = _candidate(
        history=[_entry(description="built RAG pipeline and fine-tuned LLM using PyTorch", duration_months=24)]
    )
    evidence = analyze_career(candidate)
    assert evidence.years_of_ml_experience >= 1.5


# ---------------------------------------------------------------------------
# Career momentum — unknown company sizes excluded
# ---------------------------------------------------------------------------


def test_momentum_ignores_unknown_company_size():
    """Entries with no company_size should be excluded from slope calculation."""
    history = [
        _entry(company_size="11-50", is_current=False, end_date="2021-01-01", duration_months=12),
        _entry(company_size="unknown_size", is_current=False, end_date="2022-01-01", duration_months=12),
        _entry(company_size="5001-10000", is_current=True, duration_months=24),
    ]
    # Should only use the two known sizes; should be positive momentum
    result = score_career_momentum(history)
    assert result > 0


def test_momentum_all_unknown_returns_zero():
    history = [
        _entry(company_size="xyz", is_current=False, end_date="2021-01-01", duration_months=12),
        _entry(company_size="abc", is_current=True, duration_months=12),
    ]
    assert score_career_momentum(history) == 0.0


# ---------------------------------------------------------------------------
# Penalties — experience_gap
# ---------------------------------------------------------------------------


def test_experience_gap_penalty_fires_below_threshold():
    candidate = _candidate(yoe=1.5)
    career = CareerEvidence(2, 1, 0.0, "ml_work", 0)
    coherence = compute_coherence(candidate, career, TitleDomain.ML_AI)
    # JD requires 5+ years; 1.5 < (5 - 1) = 4 → penalty should fire
    role = RoleProfile(min_experience_years=5.0)
    penalties = compute_penalties(candidate, career, coherence, role)
    assert penalties.experience_gap == 1.0


def test_experience_gap_penalty_clear_above_threshold():
    candidate = _candidate(yoe=6.0)
    career = CareerEvidence(10, 5, 0.2, "ml_work", 0)
    coherence = compute_coherence(candidate, career, TitleDomain.ML_AI)
    role = RoleProfile(min_experience_years=5.0)
    penalties = compute_penalties(candidate, career, coherence, role)
    assert penalties.experience_gap == 0.0


# ---------------------------------------------------------------------------
# Penalties — job_hopper
# ---------------------------------------------------------------------------


def test_job_hopper_penalty_fires_short_tenure():
    history = [
        _entry(duration_months=4, is_current=False, end_date="2020-05-01"),
        _entry(duration_months=5, is_current=False, end_date="2021-06-01"),
        _entry(duration_months=3, is_current=False, end_date="2022-09-01"),
        _entry(duration_months=6, is_current=True),
    ]
    candidate = _candidate(history=history)
    career = analyze_career(candidate)
    coherence = compute_coherence(candidate, career, TitleDomain.ML_AI)
    penalties = compute_penalties(candidate, career, coherence, RoleProfile())
    assert penalties.job_hopper == 1.0


def test_job_hopper_penalty_clear_stable_tenure():
    history = [
        _entry(duration_months=24, is_current=False, end_date="2020-01-01"),
        _entry(duration_months=18, is_current=False, end_date="2022-01-01"),
        _entry(duration_months=12, is_current=False, end_date="2023-01-01"),
        _entry(duration_months=12, is_current=True),
    ]
    candidate = _candidate(history=history)
    career = analyze_career(candidate)
    coherence = compute_coherence(candidate, career, TitleDomain.ML_AI)
    penalties = compute_penalties(candidate, career, coherence, RoleProfile())
    assert penalties.job_hopper == 0.0


def test_job_hopper_penalty_skips_few_roles():
    """Less than 4 roles should not trigger job_hopper."""
    history = [
        _entry(duration_months=2, is_current=False, end_date="2020-03-01"),
        _entry(duration_months=2, is_current=True),
    ]
    candidate = _candidate(history=history)
    career = analyze_career(candidate)
    coherence = compute_coherence(candidate, career, TitleDomain.ML_AI)
    penalties = compute_penalties(candidate, career, coherence, RoleProfile())
    assert penalties.job_hopper == 0.0


# ---------------------------------------------------------------------------
# Signals — tiered certification bonus
# ---------------------------------------------------------------------------


def _cert_candidate(cert_names: list[str]):
    from fitrank.models import Certification

    certs = [Certification(name=n, issuer="org", year=2024) for n in cert_names]
    return Candidate(
        "CAND_CERT",
        _profile(),
        [],
        [],
        [],
        _signals(),
        certifications=certs,
    )


def test_high_tier_cert_bonus_larger_than_general():
    from fitrank.signals import _certification_bonus

    high_candidate = _cert_candidate(["AWS Machine Learning Specialty"])
    general_candidate = _cert_candidate(["Python Programming Certificate"])
    high_bonus = _certification_bonus(high_candidate)
    general_bonus = _certification_bonus(general_candidate)
    assert high_bonus > general_bonus


def test_no_certs_zero_bonus():
    from fitrank.signals import _certification_bonus

    candidate = _cert_candidate([])
    assert _certification_bonus(candidate) == 0.0


def test_cert_bonus_capped_at_one():
    from fitrank.signals import _certification_bonus

    # Lots of high-tier certs — should still cap at 1.0
    candidate = _cert_candidate([
        "AWS Machine Learning Specialty",
        "Azure AI Engineer Associate",
        "Databricks Certified ML Professional",
        "NVIDIA Deep Learning Institute",
        "TensorFlow Developer Certificate",
    ])
    assert _certification_bonus(candidate) <= 1.0


# ---------------------------------------------------------------------------
# Signals — word boundary assessment overlap
# ---------------------------------------------------------------------------


def test_assessment_overlap_word_boundary():
    from fitrank.signals import _assessment_jd_overlap

    # "ir" (information retrieval abbreviation) must NOT match "Twitter"
    assessments = {"Twitter API": 80.0}
    capabilities = ["ir"]
    overlap = _assessment_jd_overlap(assessments, capabilities)
    assert overlap == 0.0


def test_assessment_overlap_exact_match():
    from fitrank.signals import _assessment_jd_overlap

    assessments = {"PyTorch": 85.0, "NLP": 70.0}
    capabilities = ["pytorch", "nlp"]
    overlap = _assessment_jd_overlap(assessments, capabilities)
    assert overlap == pytest.approx(1.0)


def test_assessment_overlap_low_score_excluded():
    from fitrank.signals import _assessment_jd_overlap

    # Score < 60 should not count
    assessments = {"PyTorch": 55.0}
    capabilities = ["pytorch"]
    overlap = _assessment_jd_overlap(assessments, capabilities)
    assert overlap == 0.0


# ---------------------------------------------------------------------------
# Coherence — penalized ML titles (CV without NLP)
# ---------------------------------------------------------------------------


def test_cv_without_nlp_gets_reduced_confirmation():
    from fitrank.coherence import title_domain_confirmation_score
    from fitrank.jd_parser import parse_jd

    nlp_role = parse_jd("Senior NLP Engineer. Retrieval, ranking, transformers, RAG, embeddings.")
    candidate = _candidate(
        title="Computer Vision Engineer",
        history=[_entry(description="object detection using YOLO and OpenCV for autonomous driving")],
    )
    career = CareerEvidence(3, 2, 0.0, "ml_work", 0)
    score = title_domain_confirmation_score(candidate, TitleDomain.ML_AI, career, nlp_role)
    assert score == pytest.approx(0.3)


def test_cv_with_nlp_gets_full_confirmation():
    from fitrank.coherence import title_domain_confirmation_score
    from fitrank.jd_parser import parse_jd

    nlp_role = parse_jd("Senior NLP Engineer. Retrieval, ranking, transformers, RAG, embeddings.")
    candidate = _candidate(
        title="Computer Vision Engineer",
        history=[
            _entry(
                description="built vision-language model using transformer and LLM with RAG retrieval; "
                            "semantic search and BERT-based embeddings"
            )
        ],
    )
    career = CareerEvidence(6, 4, 0.0, "ml_work", 0)
    score = title_domain_confirmation_score(candidate, TitleDomain.ML_AI, career, nlp_role)
    assert score == pytest.approx(1.0)


def test_cv_jd_gives_full_confirmation_without_nlp_evidence():
    from fitrank.coherence import title_domain_confirmation_score
    from fitrank.jd_parser import parse_jd

    cv_role = parse_jd("Computer Vision Engineer. Object detection, OpenCV, CNN, YOLO.")
    candidate = _candidate(
        title="Computer Vision Engineer",
        history=[_entry(description="object detection using YOLO and OpenCV")],
    )
    career = CareerEvidence(3, 2, 0.0, "ml_work", 0)
    score = title_domain_confirmation_score(candidate, TitleDomain.ML_AI, career, cv_role)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Coherence — data_science template domain compatibility
# ---------------------------------------------------------------------------


def test_data_science_template_ml_ai_domain_compatible():
    from fitrank.coherence import template_coherence_score

    score = template_coherence_score("data_science", TitleDomain.ML_AI)
    assert score == pytest.approx(0.9)


def test_data_science_template_non_tech_incompatible():
    from fitrank.coherence import template_coherence_score

    score = template_coherence_score("data_science", TitleDomain.NON_TECH)
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Availability score normalization
# ---------------------------------------------------------------------------


def test_availability_score_bounded():
    """Availability score must never exceed 1.0 for any valid candidate."""
    candidate = _candidate(
        open_to_work_flag=True,
        notice_period_days=0,
        willing_to_relocate=True,
        preferred_work_mode="remote",
        interview_completion_rate=1.0,
        offer_acceptance_rate=1.0,
        recruiter_response_rate=1.0,
        avg_response_time_hours=0.0,
        applications_submitted_30d=10,
        last_active_date="2025-12-15",
    )
    platform = compute_platform_trust(candidate, RoleProfile())
    assert platform.availability_score <= 1.0
    assert platform.availability_score >= 0.0
