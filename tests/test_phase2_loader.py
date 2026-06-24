"""Phase 2 loader tests (PRD T2.1-T2.9)."""

from __future__ import annotations

import json
import re
import tracemalloc

import pytest

from fitrank.loader import load_candidates, load_sample
from fitrank.models import Candidate, CandidateScore


def test_t2_1_load_sample(sample_path):
    candidates = list(load_candidates(sample_path))
    with sample_path.open(encoding="utf-8") as handle:
        expected_count = len(json.load(handle))
    assert len(candidates) == expected_count
    assert all(isinstance(candidate, Candidate) for candidate in candidates)


def test_t2_2_required_field_presence(sample_path):
    candidate = next(iter(load_candidates(sample_path)))
    assert candidate.candidate_id
    assert candidate.profile
    assert candidate.career_history
    assert candidate.skills is not None
    assert candidate.redrob_signals


def test_t2_3_type_correctness(sample_path):
    candidate = next(iter(load_candidates(sample_path)))
    assert isinstance(candidate.profile.years_of_experience, float)


def test_t2_4_malformed_line_handling(sample_path, tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    good_line = next(iter(load_candidates(sample_path)))
    jsonl_path = tmp_path / "mixed.jsonl"
    jsonl_path.write_text(
        json.dumps({"candidate_id": "bad"}) + "\n"
        + json.dumps(_candidate_to_dict(good_line)) + "\n",
        encoding="utf-8",
    )

    loaded = list(load_candidates(jsonl_path))
    assert len(loaded) == 1
    assert any("Skipping malformed candidate" in record.message for record in caplog.records)


def test_t2_5_career_history_ordering(sample_path):
    candidate = next(iter(load_candidates(sample_path)))
    dates = [entry.start_date for entry in candidate.career_history]
    assert dates == sorted(dates)
    current_entries = [entry for entry in candidate.career_history if entry.is_current]
    assert current_entries
    assert current_entries[-1].end_date is None


def test_t2_6_signals_completeness_and_minus_one():
    from fitrank.models import RedrobSignals, SalaryRange

    signals = RedrobSignals(
        profile_completeness_score=80.0,
        signup_date="2024-01-01",
        last_active_date="2024-06-01",
        open_to_work_flag=True,
        profile_views_received_30d=10,
        applications_submitted_30d=2,
        recruiter_response_rate=0.5,
        avg_response_time_hours=4.0,
        skill_assessment_scores={},
        connection_count=100,
        endorsements_received=5,
        notice_period_days=30,
        expected_salary_range_inr_lpa=SalaryRange(min=10.0, max=20.0),
        preferred_work_mode="remote",
        willing_to_relocate=True,
        github_activity_score=-1.0,
        search_appearance_30d=3,
        saved_by_recruiters_30d=1,
        interview_completion_rate=0.8,
        offer_acceptance_rate=-1.0,
        verified_email=True,
        verified_phone=False,
        linkedin_connected=True,
    )
    assert isinstance(signals.github_activity_score, float)
    assert signals.github_activity_score == -1.0


def test_t2_7_generator_memory(candidates_path):
    tracemalloc.start()
    count = 0
    for _ in load_candidates(candidates_path):
        count += 1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert count == 100_000
    assert peak < 2 * 1024**3


def test_t2_8_candidate_score_defaults():
    score = CandidateScore(candidate_id="CAND_0000001")
    assert score.jd_fit == 0.0
    assert score.career_evidence == 0.0
    assert score.coherence == 0.0
    assert score.platform_trust == 0.0
    assert score.availability == 0.0
    assert score.penalties == 0.0
    assert score.final_score == 0.0


def test_t2_9_candidate_id_format(candidates_path):
    pattern = re.compile(r"^CAND_[0-9]{7}$")
    for index, candidate in enumerate(load_candidates(candidates_path)):
        assert pattern.match(candidate.candidate_id), candidate.candidate_id
        if index >= 999:
            break


def test_load_sample_helper():
    candidates = load_sample()
    assert candidates
    assert isinstance(candidates[0], Candidate)


def _candidate_to_dict(candidate: Candidate) -> dict:
    """Round-trip helper for malformed-line fixture."""
    return {
        "candidate_id": candidate.candidate_id,
        "profile": {
            "anonymized_name": candidate.profile.anonymized_name,
            "headline": candidate.profile.headline,
            "summary": candidate.profile.summary,
            "location": candidate.profile.location,
            "country": candidate.profile.country,
            "years_of_experience": candidate.profile.years_of_experience,
            "current_title": candidate.profile.current_title,
            "current_company": candidate.profile.current_company,
            "current_company_size": candidate.profile.current_company_size,
            "current_industry": candidate.profile.current_industry,
        },
        "career_history": [
            {
                "company": entry.company,
                "title": entry.title,
                "start_date": entry.start_date,
                "end_date": entry.end_date,
                "duration_months": entry.duration_months,
                "is_current": entry.is_current,
                "industry": entry.industry,
                "company_size": entry.company_size,
                "description": entry.description,
            }
            for entry in candidate.career_history
        ],
        "education": [
            {
                "institution": entry.institution,
                "degree": entry.degree,
                "field_of_study": entry.field_of_study,
                "start_year": entry.start_year,
                "end_year": entry.end_year,
                "grade": entry.grade,
                "tier": entry.tier,
            }
            for entry in candidate.education
        ],
        "skills": [
            {
                "name": skill.name,
                "proficiency": skill.proficiency,
                "endorsements": skill.endorsements,
                "duration_months": skill.duration_months,
            }
            for skill in candidate.skills
        ],
        "certifications": [
            {
                "name": cert.name,
                "issuer": cert.issuer,
                "year": cert.year,
            }
            for cert in candidate.certifications
        ],
        "languages": [
            {
                "language": lang.language,
                "proficiency": lang.proficiency,
            }
            for lang in candidate.languages
        ],
        "redrob_signals": {
            "profile_completeness_score": candidate.redrob_signals.profile_completeness_score,
            "signup_date": candidate.redrob_signals.signup_date,
            "last_active_date": candidate.redrob_signals.last_active_date,
            "open_to_work_flag": candidate.redrob_signals.open_to_work_flag,
            "profile_views_received_30d": candidate.redrob_signals.profile_views_received_30d,
            "applications_submitted_30d": candidate.redrob_signals.applications_submitted_30d,
            "recruiter_response_rate": candidate.redrob_signals.recruiter_response_rate,
            "avg_response_time_hours": candidate.redrob_signals.avg_response_time_hours,
            "skill_assessment_scores": candidate.redrob_signals.skill_assessment_scores,
            "connection_count": candidate.redrob_signals.connection_count,
            "endorsements_received": candidate.redrob_signals.endorsements_received,
            "notice_period_days": candidate.redrob_signals.notice_period_days,
            "expected_salary_range_inr_lpa": {
                "min": candidate.redrob_signals.expected_salary_range_inr_lpa.min,
                "max": candidate.redrob_signals.expected_salary_range_inr_lpa.max,
            },
            "preferred_work_mode": candidate.redrob_signals.preferred_work_mode,
            "willing_to_relocate": candidate.redrob_signals.willing_to_relocate,
            "github_activity_score": candidate.redrob_signals.github_activity_score,
            "search_appearance_30d": candidate.redrob_signals.search_appearance_30d,
            "saved_by_recruiters_30d": candidate.redrob_signals.saved_by_recruiters_30d,
            "interview_completion_rate": candidate.redrob_signals.interview_completion_rate,
            "offer_acceptance_rate": candidate.redrob_signals.offer_acceptance_rate,
            "verified_email": candidate.redrob_signals.verified_email,
            "verified_phone": candidate.redrob_signals.verified_phone,
            "linkedin_connected": candidate.redrob_signals.linkedin_connected,
        },
    }
