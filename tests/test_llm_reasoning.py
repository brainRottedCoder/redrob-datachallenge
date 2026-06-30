"""Tests for optional local LLM reasoning."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fitrank.component_scores import ComponentScores
from fitrank.llm_reasoning import (
    ReasoningCache,
    ReasoningStats,
    build_llm_prompt,
    build_reasoning_context,
    cache_key,
    ensure_valid_reasoning,
    is_llama_available,
    jd_fingerprint,
    resolve_reasoning,
)
from fitrank.models import Candidate, Profile, RedrobSignals, SalaryRange
from fitrank.penalties import PenaltyScore
from fitrank.reasoning import ReasoningEvidence, _gather_evidence, build_reasoning


def _penalty_detail(total: float = 0.0) -> PenaltyScore:
    return PenaltyScore(
        template_mismatch=0.0,
        skill_inflation=0.0,
        shallow_boilerplate=0.0,
        expert_zero_endorse=0.0,
        non_ml_title_high_skill=0.0,
        consulting_only=0.0,
        pure_research=0.0,
        domain_mismatch=0.0,
        salary_inverted=0.0,
        experience_gap=0.0,
        job_hopper=0.0,
        seniority_mismatch=0.0,
        total_penalty=total,
    )


def _components(**kwargs) -> ComponentScores:
    defaults = dict(
        jd_fit=0.85,
        career_evidence=0.58,
        coherence=0.79,
        platform_trust=0.85,
        availability=0.7,
        penalties=0.0,
        final_score=0.74,
        is_honeypot=False,
        penalties_detail=_penalty_detail(),
        ml_tenure_years=7.7,
    )
    defaults.update(kwargs)
    return ComponentScores(**defaults)


def _candidate(candidate_id: str = "CAND_0000001") -> Candidate:
    profile = Profile(
        "A",
        "ML engineer headline",
        "summary",
        "Bangalore, India",
        "India",
        7.8,
        "Senior AI Engineer",
        "Co",
        "201-500",
        "Tech",
    )
    signals = RedrobSignals(
        profile_completeness_score=80,
        signup_date="2024-01-01",
        last_active_date="2025-12-15",
        open_to_work_flag=True,
        profile_views_received_30d=10,
        applications_submitted_30d=2,
        recruiter_response_rate=0.7,
        avg_response_time_hours=4.0,
        skill_assessment_scores={"LoRA": 87.0, "PEFT": 86.0},
        connection_count=100,
        endorsements_received=5,
        notice_period_days=45,
        expected_salary_range_inr_lpa=SalaryRange(20, 40),
        preferred_work_mode="remote",
        willing_to_relocate=True,
        github_activity_score=83.0,
        search_appearance_30d=5,
        saved_by_recruiters_30d=3,
        interview_completion_rate=0.8,
        offer_acceptance_rate=0.6,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )
    return Candidate(candidate_id, profile, [], [], [], signals)


def test_gather_evidence_matches_template_fields():
    candidate = _candidate()
    components = _components()
    evidence = _gather_evidence(candidate, components)
    template = build_reasoning(candidate, components)
    assert evidence.title in template
    assert evidence.score_line in template
    assert "LoRA" in template or "assessments" in template


def test_jd_fingerprint_stable():
    a = jd_fingerprint("same jd text")
    b = jd_fingerprint("same jd text")
    c = jd_fingerprint("different jd")
    assert a == b
    assert a != c


def test_cache_roundtrip(tmp_path: Path):
    cache_path = tmp_path / "cache.jsonl"
    cache = ReasoningCache(cache_path)
    key = "abc123"
    cache.put(key, "Senior AI Engineer | JD=0.85 | Strong ML fit.", candidate_id="CAND_1", jd_hash="jd1", model="test")
    reloaded = ReasoningCache(cache_path)
    assert reloaded.get(key) == "Senior AI Engineer | JD=0.85 | Strong ML fit."


def test_cache_hit_skips_llm(tmp_path: Path):
    candidate = _candidate()
    components = _components()
    jd_hash = "testjd"
    key = cache_key(jd_hash, candidate.candidate_id, components)
    cache_path = tmp_path / "cache.jsonl"
    cache = ReasoningCache(cache_path)
    cached_text = "Senior AI Engineer | JD=0.85 Career=0.58 Coh=0.79 Trust=0.85 | cached reasoning prose"
    cache.put(key, cached_text, candidate_id=candidate.candidate_id, jd_hash=jd_hash, model="mock")

    stats = ReasoningStats()
    engine = MagicMock()
    result = resolve_reasoning(
        candidate,
        components,
        mode="llm",
        jd_hash=jd_hash,
        cache=cache,
        engine=engine,
        stats=stats,
    )
    assert result == cached_text
    assert stats.cache_hits == 1
    assert stats.llm_generated == 0
    engine.generate.assert_not_called()


def test_unavailable_model_falls_back_to_template():
    candidate = _candidate()
    components = _components()
    result = resolve_reasoning(
        candidate,
        components,
        mode="llm",
        model_path=Path("nonexistent/model.gguf"),
    )
    assert result == build_reasoning(candidate, components)
    assert "JD=0.85" in result


def test_template_mode_matches_build_reasoning():
    candidate = _candidate()
    components = _components()
    assert resolve_reasoning(candidate, components, mode="template") == build_reasoning(
        candidate, components
    )


def test_ensure_valid_reasoning_prepends_scores():
    candidate = _candidate()
    components = _components()
    evidence = build_reasoning_context(candidate, components)
    result = ensure_valid_reasoning(
        "Strong production ML background with embedding and RAG experience.",
        candidate,
        components,
        evidence,
    )
    assert re.search(r"JD=0\.85", result)
    assert "Senior AI Engineer" in result
    assert len(result) >= 20


def test_ensure_valid_reasoning_honeypot_label():
    candidate = _candidate()
    components = _components(is_honeypot=True)
    evidence = build_reasoning_context(candidate, components)
    result = ensure_valid_reasoning(
        "Senior AI Engineer | JD=0.85 Career=0.58 Coh=0.79 Trust=0.85 | suspicious profile",
        candidate,
        components,
        evidence,
    )
    assert "[HONEYPOT" in result


def test_mock_engine_generates_valid_reasoning(tmp_path: Path):
    candidate = _candidate("CAND_0000002")
    components = _components()
    evidence = build_reasoning_context(candidate, components)

    engine = MagicMock()
    engine.generate.return_value = (
        "Strong fit for production ML with verified LoRA assessments and 7.7 years ML tenure."
    )

    stats = ReasoningStats()
    result = resolve_reasoning(
        candidate,
        components,
        mode="llm",
        jd_hash="jdhash",
        cache=ReasoningCache(tmp_path / "c.jsonl"),
        engine=engine,
        model_path=Path("mock.gguf"),
        stats=stats,
    )
    assert re.search(r"JD=0\.85", result)
    assert len(result) >= 20
    assert stats.llm_generated == 1
    engine.generate.assert_called_once()


def test_build_llm_prompt_includes_candidate_id():
    candidate = _candidate("CAND_0000099")
    components = _components()
    evidence: ReasoningEvidence = build_reasoning_context(candidate, components)
    prompt = build_llm_prompt(evidence)
    assert "CAND_0000099" in prompt
    assert "Do not invent" in prompt


def test_max_new_zero_uses_cache_or_template(tmp_path: Path):
    candidate = _candidate()
    components = _components()
    stats = ReasoningStats()
    result = resolve_reasoning(
        candidate,
        components,
        mode="llm",
        cache=ReasoningCache(tmp_path / "empty.jsonl"),
        engine=MagicMock(),
        stats=stats,
        max_new=0,
    )
    assert result == build_reasoning(candidate, components)
    assert stats.template_fallbacks == 1


@pytest.mark.parametrize("model_exists", [False])
def test_is_llama_available_without_model(tmp_path: Path, model_exists: bool):
    assert is_llama_available(tmp_path / "missing.gguf") is False
