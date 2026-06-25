"""Phase 5 career analyzer tests."""

from fitrank.career_analyzer import (
    analyze_career,
    classify_template_domain,
    score_career_momentum,
)
from fitrank.models import Candidate, CareerEntry, Profile, RedrobSignals, SalaryRange, Skill


def _profile(title="ML Engineer"):
    return Profile(
        anonymized_name="A",
        headline="h",
        summary="I've been curious about AI and experimenting with ChatGPT",
        location="X",
        country="IN",
        years_of_experience=5.0,
        current_title=title,
        current_company="Co",
        current_company_size="201-500",
        current_industry="Tech",
    )


def _signals():
    return RedrobSignals(
        profile_completeness_score=80,
        signup_date="2024-01-01",
        last_active_date="2024-06-01",
        open_to_work_flag=True,
        profile_views_received_30d=1,
        applications_submitted_30d=1,
        recruiter_response_rate=0.5,
        avg_response_time_hours=1.0,
        skill_assessment_scores={},
        connection_count=1,
        endorsements_received=1,
        notice_period_days=30,
        expected_salary_range_inr_lpa=SalaryRange(10, 20),
        preferred_work_mode="remote",
        willing_to_relocate=True,
        github_activity_score=-1,
        search_appearance_30d=1,
        saved_by_recruiters_30d=1,
        interview_completion_rate=0.5,
        offer_acceptance_rate=-1,
        verified_email=True,
        verified_phone=False,
        linkedin_connected=False,
    )


def _entry(**kwargs):
    defaults = dict(
        company="Co",
        title="Eng",
        start_date="2020-01-01",
        end_date=None,
        duration_months=24,
        is_current=True,
        industry="Tech",
        company_size="201-500",
        description="",
    )
    defaults.update(kwargs)
    return CareerEntry(**defaults)


def test_t5_1_deep_ml_depth_genuine():
    candidate = Candidate(
        "CAND_0000001",
        _profile(),
        [
            _entry(
                description="fine-tuned LLaMA-2 using LoRA and QLoRA; deployed via BentoML and PyTorch"
            )
        ],
        [],
        [],
        _signals(),
    )
    evidence = analyze_career(candidate)
    assert evidence.all_career_ml_depth >= 5


def test_t5_2_shallow_only_zero_depth():
    candidate = Candidate(
        "CAND_0000002",
        _profile(),
        [_entry(description="curious about AI, ChatGPT tools")],
        [],
        [],
        _signals(),
    )
    assert analyze_career(candidate).all_career_ml_depth == 0


def test_t5_4_support_template():
    desc = "Customer support team lead at a SaaS product. Managed a team of 8 support agents"
    assert classify_template_domain(desc) == "support"


def test_t5_7_positive_momentum():
    history = [
        _entry(company_size="11-50", is_current=False, end_date="2022-01-01"),
        _entry(company_size="201-500", is_current=False, end_date="2024-01-01"),
        _entry(company_size="5001-10000", is_current=True),
    ]
    assert score_career_momentum(history) > 0


def test_t5_3_current_role_focus():
    from fitrank.models import Candidate, CareerEntry, Profile, RedrobSignals, SalaryRange

    candidate = Candidate(
        "CAND_0000101",
        Profile("A", "h", "s", "L", "IN", 5.0, "ML Engineer", "Co", "201-500", "Tech"),
        [
            CareerEntry("Old", "ML Eng", "2018-01-01", "2023-01-01", 60, False, "Tech", "201-500",
                        "fine-tuned LLaMA using LoRA and PyTorch with FAISS retrieval"),
            CareerEntry("Now", "Manager", "2023-01-01", None, 24, True, "Tech", "201-500",
                        "Customer support team lead at a SaaS product."),
        ],
        [],
        [],
        RedrobSignals(
            80, "2024-01-01", "2024-06-01", True, 1, 1, 0.5, 1.0, {}, 1, 1, 30,
            SalaryRange(10, 20), "remote", True, -1, 1, 1, 0.5, -1, True, False, False,
        ),
    )
    evidence = analyze_career(candidate)
    assert evidence.current_role_ml_depth == 0
    assert evidence.all_career_ml_depth > 0


def test_t5_6_ml_template():
    desc = "Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA"
    assert classify_template_domain(desc) == "ml_work"


def test_t5_10_shallow_boilerplate_count():
    candidate = Candidate("CAND_0000003", _profile(), [_entry()], [], [], _signals())
    assert analyze_career(candidate).shallow_ai_count >= 1
