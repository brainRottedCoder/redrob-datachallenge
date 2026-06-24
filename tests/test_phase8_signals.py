"""Phase 8 platform signal tests."""

from fitrank.models import Candidate, Profile, RedrobSignals, RoleProfile, SalaryRange
from fitrank.signals import compute_platform_trust


def _candidate(signals: RedrobSignals) -> Candidate:
    return Candidate(
        "CAND_0000001",
        Profile("A", "h", "s", "L", "IN", 5.0, "ML Engineer", "Co", "201-500", "Tech"),
        [],
        [],
        [],
        signals,
    )


def _base_signals(**kwargs):
    defaults = dict(
        profile_completeness_score=80,
        signup_date="2024-01-01",
        last_active_date="2024-06-01",
        open_to_work_flag=True,
        profile_views_received_30d=10,
        applications_submitted_30d=1,
        recruiter_response_rate=0.5,
        avg_response_time_hours=1.0,
        skill_assessment_scores={"PyTorch": 85, "NLP": 70},
        connection_count=1,
        endorsements_received=1,
        notice_period_days=0,
        expected_salary_range_inr_lpa=SalaryRange(10, 20),
        preferred_work_mode="remote",
        willing_to_relocate=True,
        github_activity_score=75,
        search_appearance_30d=3,
        saved_by_recruiters_30d=5,
        interview_completion_rate=0.9,
        offer_acceptance_rate=0.8,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )
    defaults.update(kwargs)
    return RedrobSignals(**defaults)


def test_t8_1_assessment_average():
    platform = compute_platform_trust(_candidate(_base_signals()), RoleProfile())
    assert abs(platform.assessment_avg - 0.775) < 0.01


def test_t8_2_no_assessments():
    platform = compute_platform_trust(
        _candidate(_base_signals(skill_assessment_scores={})),
        RoleProfile(),
    )
    assert platform.assessment_avg == 0.0


def test_t8_3_github_present():
    platform = compute_platform_trust(_candidate(_base_signals()), RoleProfile())
    assert platform.github_norm == 0.75
