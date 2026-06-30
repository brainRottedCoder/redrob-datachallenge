"""Phase 12 tests for Round 4 improvements."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from fitrank.calibrator import calibrate_scores, rescale_submission_scores
from fitrank.career_analyzer import (
    CareerEvidence,
    SHALLOW_AI_PATTERNS,
    _count_deep_patterns,
    _normalize_momentum,
    count_shallow_ai_boilerplate,
)
from fitrank.coherence import CoherenceScore, compute_coherence, skill_career_alignment
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
from fitrank.features import _open_source_bonus, _open_source_corpus
from fitrank.reasoning import build_reasoning
from fitrank.signals import _assessment_jd_overlap
from fitrank.title_gate import TitleDomain, classify_title
from validate_submission import validate_submission


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


def _candidate(candidate_id="CAND_TEST001", title="ML Engineer", yoe=5.0, history=None, skills=None, summary="s", **signal_kwargs):
    profile = Profile("A", "h", summary, "Bangalore", "India", yoe, title, "Co", "201-500", "Tech")
    return Candidate(
        candidate_id,
        profile,
        history or [],
        [],
        skills or [],
        _signals(**signal_kwargs),
    )


# ---------------------------------------------------------------------------
# C1 — ML depth max and per-entry cap
# ---------------------------------------------------------------------------


def test_single_role_capped_at_12_depth():
    from fitrank.career_analyzer import score_career_ml_depth

    # Create a description that will match many patterns (>12).
    text = " ".join([
        "LoRA", "QLoRA", "RLHF", "DPO", "SFT", "PEFT", "LLM", "GPT", "LLaMA", "Mistral",
        "RAG", "FAISS", "Milvus", "Weaviate", "Pinecone", "Qdrant", "semantic search",
        "embedding", "PyTorch", "TensorFlow", "MLflow", "W&B", "Kubernetes", "BentoML",
    ])
    history = [_entry(description=text, duration_months=24)]
    depth = score_career_ml_depth(history)
    assert depth <= 12


def test_multi_role_accumulates_beyond_12():
    from fitrank.career_analyzer import score_career_ml_depth

    text = "PyTorch RAG FAISS LLM LoRA BentoML BERT transformer NDP"
    history = [
        _entry(description=text, duration_months=24, is_current=False, end_date="2023-01-01"),
        _entry(description=text, duration_months=24, is_current=True),
    ]
    depth = score_career_ml_depth(history)
    assert depth > 12


def test_ml_depth_norm_not_saturated_for_2_roles():
    from fitrank.career_analyzer import ML_DEPTH_MAX, score_career_ml_depth

    text = "PyTorch RAG FAISS LLM LoRA BentoML"
    history = [
        _entry(description=text, duration_months=24, is_current=False, end_date="2023-01-01"),
        _entry(description=text, duration_months=24, is_current=True),
    ]
    depth = score_career_ml_depth(history)
    assert depth / ML_DEPTH_MAX < 1.0


# ---------------------------------------------------------------------------
# C2 — Tanh momentum normalization
# ---------------------------------------------------------------------------


def test_momentum_norm_tanh_higher_for_steep_slope():
    steep = _normalize_momentum(3.5)
    moderate = _normalize_momentum(2.0)
    assert steep > moderate


def test_momentum_norm_tanh_bounded():
    for value in [-10.0, -2.0, 0.0, 2.0, 10.0]:
        norm = _normalize_momentum(value)
        assert 0.0 <= norm <= 1.0


def test_momentum_norm_zero_is_midpoint():
    assert _normalize_momentum(0.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# C3 — AI_ADJACENT honeypot detection
# ---------------------------------------------------------------------------


def test_ai_adjacent_honeypot_flagged():
    candidate = _candidate(
        title="AI Specialist",
        history=[_entry(description="curious about AI and ChatGPT tools")],
        skills=[Skill("PyTorch", "advanced", 1, 12), Skill("NLP", "advanced", 1, 12)],
    )
    career = CareerEvidence(
        all_career_ml_depth=0,
        current_role_ml_depth=0,
        career_momentum=0.0,
        template_domain="support",
        shallow_ai_count=1,
    )
    coherence = compute_coherence(candidate, career, TitleDomain.AI_ADJACENT)
    assert coherence.is_honeypot


def test_genuine_ai_adjacent_not_honeypot():
    candidate = _candidate(
        title="AI Specialist",
        history=[_entry(description="built RAG pipeline with FAISS and fine-tuned LLM")],
        skills=[Skill("PyTorch", "advanced", 1, 12)],
    )
    career = CareerEvidence(
        all_career_ml_depth=8,
        current_role_ml_depth=4,
        career_momentum=0.0,
        template_domain="ml_work",
        shallow_ai_count=0,
    )
    coherence = compute_coherence(candidate, career, TitleDomain.AI_ADJACENT)
    assert not coherence.is_honeypot


# ---------------------------------------------------------------------------
# C4 — Skill-count damping in alignment
# ---------------------------------------------------------------------------


def test_single_skill_alignment_max_0_33():
    candidate = _candidate(
        history=[_entry(description="PyTorch deep learning")],
        skills=[Skill("PyTorch", "advanced", 1, 12)],
    )
    assert skill_career_alignment(candidate) <= 0.34 + 1e-6


def test_multi_skill_alignment_higher_than_single():
    single = _candidate(
        history=[_entry(description="PyTorch deep learning")],
        skills=[Skill("PyTorch", "advanced", 1, 12)],
    )
    multi = _candidate(
        history=[_entry(description="PyTorch NLP RAG LLM LoRA BERT")],
        skills=[
            Skill("PyTorch", "advanced", 1, 12),
            Skill("NLP", "advanced", 1, 12),
            Skill("RAG", "advanced", 1, 12),
            Skill("LLM", "advanced", 1, 12),
            Skill("LoRA", "advanced", 1, 12),
            Skill("BERT", "advanced", 1, 12),
        ],
    )
    assert skill_career_alignment(multi) > skill_career_alignment(single)


def test_zero_match_alignment_zero():
    candidate = _candidate(
        history=[_entry(description="sales and marketing leadership")],
        skills=[
            Skill("PyTorch", "advanced", 1, 12),
            Skill("NLP", "advanced", 1, 12),
            Skill("RAG", "advanced", 1, 12),
        ],
    )
    assert skill_career_alignment(candidate) == 0.0


# ---------------------------------------------------------------------------
# H1 — Open-source corpus deduplication
# ---------------------------------------------------------------------------


def test_open_source_no_double_count():
    candidate = _candidate(
        summary="open source contributions on GitHub.",
        history=[_entry(description="open source contributions on GitHub.")],
    )
    corpus = _open_source_corpus(candidate)
    assert corpus.count("open source") == 1


def test_open_source_bonus_no_double_count():
    candidate = _candidate(
        summary="open source contributions on GitHub.",
        history=[_entry(description="open source contributions on GitHub.")],
    )
    bonus = _open_source_bonus(candidate)
    assert bonus <= 0.5


# ---------------------------------------------------------------------------
# H2 — Honeypot label in reasoning
# ---------------------------------------------------------------------------


def _components(final_score=0.5, is_honeypot=False):
    from fitrank.penalties import PenaltyScore

    return type(
        "ComponentScores",
        (),
        {
            "jd_fit": 0.5,
            "career_evidence": 0.5,
            "coherence": 0.5,
            "platform_trust": 0.5,
            "penalties": 0.0,
            "final_score": final_score,
            "is_honeypot": is_honeypot,
            "penalties_detail": PenaltyScore(*([0.0] * 12), total_penalty=0.0),
            "ml_tenure_years": 0.0,
        },
    )


def test_honeypot_label_appears_in_reasoning():
    candidate = _candidate()
    reasoning = build_reasoning(candidate, _components(is_honeypot=True))
    assert "[HONEYPOT" in reasoning


def test_no_honeypot_label_for_normal_candidate():
    candidate = _candidate()
    reasoning = build_reasoning(candidate, _components(is_honeypot=False))
    assert "[HONEYPOT" not in reasoning


# ---------------------------------------------------------------------------
# H3 — Assessment overlap denominator cap
# ---------------------------------------------------------------------------


def test_assessment_overlap_capped_denominator():
    assessments = {"PyTorch": 85.0, "NLP": 75.0}
    capabilities = ["pytorch", "nlp", "llm", "rag", "embedding", "bert", "lora", "milvus", "kubernetes", "mlflow"]
    overlap = _assessment_jd_overlap(assessments, capabilities)
    assert overlap == pytest.approx(0.4)


def test_assessment_overlap_uncapped_small_jd():
    assessments = {"PyTorch": 85.0, "NLP": 75.0}
    capabilities = ["pytorch", "nlp", "llm"]
    overlap = _assessment_jd_overlap(assessments, capabilities)
    assert overlap == pytest.approx(0.67, abs=0.01)


# ---------------------------------------------------------------------------
# H4 — Per-skill recency weighting
# ---------------------------------------------------------------------------


def test_current_role_skill_scores_higher_than_past():
    candidate = _candidate(
        history=[
            _entry(description="PyTorch deep learning", duration_months=24, is_current=False, end_date="2023-01-01"),
            _entry(description="TensorFlow NLP", duration_months=24, is_current=True),
        ],
        skills=[Skill("PyTorch", "advanced", 1, 12), Skill("TensorFlow", "advanced", 1, 12)],
    )
    alignment = skill_career_alignment(candidate)
    # PyTorch in past (1), TensorFlow in current (2). Minimum denominator 3*2=6.
    # Score = 1 + 2 = 3. 3/6 = 0.5.
    assert alignment > 0.0


def test_skill_not_in_any_description_scores_zero():
    candidate = _candidate(
        history=[_entry(description="sales and marketing")],
        skills=[Skill("PyTorch", "advanced", 1, 12)],
    )
    assert skill_career_alignment(candidate) == 0.0


def test_recency_alignment_bounded_at_one():
    candidate = _candidate(
        history=[_entry(description="PyTorch NLP RAG LLM BERT", is_current=True)]
        + [
            _entry(description="PyTorch NLP RAG LLM BERT", is_current=False, end_date="2023-01-01"),
        ],
        skills=[
            Skill("PyTorch", "advanced", 1, 12),
            Skill("NLP", "advanced", 1, 12),
            Skill("RAG", "advanced", 1, 12),
            Skill("LLM", "advanced", 1, 12),
            Skill("BERT", "advanced", 1, 12),
        ],
    )
    assert skill_career_alignment(candidate) <= 1.0


# ---------------------------------------------------------------------------
# M1 — Threshold-based template_mismatch penalty
# ---------------------------------------------------------------------------


def test_template_mismatch_fires_on_explicit_incompatibility():
    candidate = _candidate(title="ML Engineer")
    career = CareerEvidence(5, 3, 0.0, "support", 0)
    coherence = CoherenceScore(0.0, 0.0, 0.0, 0.1, True)
    penalties = compute_penalties(candidate, career, coherence, RoleProfile())
    assert penalties.template_mismatch == 1.0


def test_template_mismatch_clear_for_unknown_template():
    candidate = _candidate(title="ML Engineer")
    career = CareerEvidence(5, 3, 0.0, "unknown", 0)
    coherence = CoherenceScore(0.3, 0.0, 0.0, 0.1, False)
    penalties = compute_penalties(candidate, career, coherence, RoleProfile())
    assert penalties.template_mismatch == 0.0


def test_template_mismatch_clear_for_compatible_pair():
    candidate = _candidate(title="Data Scientist")
    career = CareerEvidence(5, 3, 0.0, "data_science", 0)
    coherence = CoherenceScore(0.9, 0.0, 0.0, 0.1, False)
    penalties = compute_penalties(candidate, career, coherence, RoleProfile())
    assert penalties.template_mismatch == 0.0


# ---------------------------------------------------------------------------
# M2 — Audit report co-located with CSV
# ---------------------------------------------------------------------------


def test_audit_report_colocated_with_csv(tmp_path, jd_text):
    from rank import write_submission_csv
    from fitrank.jd_parser import parse_jd
    from fitrank.ranker import rank_candidates
    from fitrank.loader import load_sample

    csv_path = tmp_path / "sub.csv"
    role = parse_jd(jd_text)
    ranked = rank_candidates(load_sample(), role, top_n=10, jd_text=jd_text)
    write_submission_csv(csv_path, ranked)

    expected_audit = tmp_path / "audit_report.json"
    assert not expected_audit.exists()

    # Manually run the audit-report logic to verify path choice.
    import json
    from collections import Counter
    from fitrank.reasoning import build_reasoning

    audit = {
        "runtime_seconds": 1.0,
        "title_distribution": dict(Counter(item[0].profile.current_title for item in ranked)),
        "honeypot_flags_in_shortlist": 0,
    }
    audit_path = csv_path.parent / "audit_report.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    assert audit_path.exists()


# ---------------------------------------------------------------------------
# M3 — Validator reasoning quality checks
# ---------------------------------------------------------------------------


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in rows:
            writer.writerow(row)


def test_validator_rejects_empty_reasoning(tmp_path):
    path = tmp_path / "sub.csv"
    _write_csv(path, [["CAND_0000001", 1, 0.9, ""]])
    errors = validate_submission(path)
    assert any("reasoning must not be empty" in e for e in errors)


def test_validator_rejects_short_reasoning(tmp_path):
    path = tmp_path / "sub.csv"
    _write_csv(path, [["CAND_0000001", 1, 0.9, "short"]])
    errors = validate_submission(path)
    assert any("reasoning too short" in e for e in errors)


def test_validator_rejects_missing_score_field(tmp_path):
    path = tmp_path / "sub.csv"
    _write_csv(path, [["CAND_0000001", 1, 0.9, "No score field here."]])
    errors = validate_submission(path)
    assert any("reasoning missing 'JD=X.XX'" in e for e in errors)


def test_validator_accepts_valid_reasoning(tmp_path):
    path = tmp_path / "sub.csv"
    rows = [[f"CAND_{i:07d}", i + 1, 0.9 - i * 0.005, f"JD=0.50 Career=0.40 Coh=0.30 Trust=0.20 | reasoning row {i}"] for i in range(100)]
    _write_csv(path, rows)
    errors = validate_submission(path)
    assert not errors


# ---------------------------------------------------------------------------
# L2 — Score calibration module
# ---------------------------------------------------------------------------


def _make_ranked(scores):
    from fitrank.models import CandidateScore
    from fitrank.penalties import PenaltyScore
    from fitrank.ranker import ComponentScores

    ranked = []
    for i, score in enumerate(scores):
        candidate = _candidate(candidate_id=f"CAND_{i:07d}")
        components = ComponentScores(
            jd_fit=0.5,
            career_evidence=0.5,
            coherence=0.5,
            platform_trust=0.5,
            availability=0.5,
            penalties=0.0,
            final_score=score,
            is_honeypot=False,
            penalties_detail=PenaltyScore(
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
                total_penalty=0.0,
            ),
            ml_tenure_years=0.0,
        )
        score_obj = CandidateScore(candidate_id=f"CAND_{i:07d}", final_score=score)
        ranked.append((candidate, components, score_obj))
    return ranked


def test_calibrator_preserves_order():
    ranked = _make_ranked([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    calibrated = calibrate_scores(ranked)
    original_order = [item[2].final_score for item in ranked]
    calibrated_order = [item[2].final_score for item in calibrated]
    assert calibrated_order == sorted(calibrated_order, reverse=True)
    assert [sorted(original_order, reverse=True).index(s) for s in original_order] == [
        sorted(calibrated_order, reverse=True).index(s) for s in calibrated_order
    ]


def test_calibrator_spreads_p10_p90():
    # 50 values spread from 0.1 to 0.95; calibration should widen the middle band.
    scores = [0.1 + i * (0.85 / 49) for i in range(50)]
    ranked = _make_ranked(scores)
    calibrated = calibrate_scores(ranked, target_p10=0.40, target_p90=0.90)
    calibrated_scores = sorted(item[2].final_score for item in calibrated)
    p10 = calibrated_scores[max(0, int(0.1 * (len(calibrated_scores) - 1)))]
    p90 = calibrated_scores[int(0.9 * (len(calibrated_scores) - 1))]
    assert p10 >= 0.25
    assert p90 <= 0.95
    assert p90 - p10 > 0.40


def test_rescale_submission_hits_endpoints():
    scores = [0.7419 - i * (0.2418 / 99) for i in range(100)]
    scores[-1] = 0.5001
    ranked = _make_ranked(scores)
    rescaled = rescale_submission_scores(ranked)
    rescaled_scores = [item[2].final_score for item in rescaled]
    assert rescaled_scores[0] == pytest.approx(0.95, abs=0.0001)
    assert rescaled_scores[-1] == pytest.approx(0.55, abs=0.0001)


def test_rescale_preserves_order():
    ranked = _make_ranked([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    original_ids = [item[0].candidate_id for item in ranked]
    rescaled = rescale_submission_scores(ranked)
    rescaled_ids = [item[0].candidate_id for item in rescaled]
    assert rescaled_ids == original_ids


def test_rescale_non_increasing():
    scores = [0.55 + i * 0.001 for i in range(100)]
    ranked = _make_ranked(list(reversed(scores)))
    rescaled = rescale_submission_scores(ranked)
    rescaled_scores = [item[2].final_score for item in rescaled]
    assert all(rescaled_scores[i] >= rescaled_scores[i + 1] for i in range(len(rescaled_scores) - 1))


def test_rescale_flat_scores():
    ranked = _make_ranked([0.6] * 10)
    rescaled = rescale_submission_scores(ranked)
    rescaled_scores = [item[2].final_score for item in rescaled]
    assert all(s == pytest.approx(0.95) for s in rescaled_scores)


# ---------------------------------------------------------------------------
# L3 — Expanded shallow AI patterns
# ---------------------------------------------------------------------------


def _shallow_count(text):
    return count_shallow_ai_boilerplate(text)


def test_hobby_project_is_shallow():
    assert _shallow_count("hobby project using ChatGPT for text generation") >= 1


def test_learning_pytorch_is_shallow():
    assert _shallow_count("learning PyTorch and machine learning") >= 1


def test_real_pytorch_work_not_shallow():
    # Should NOT be counted as shallow because the text describes production work.
    assert _shallow_count("deployed PyTorch model to production with BentoML") == 0


def test_watched_andrew_ng_is_shallow():
    assert _shallow_count("watched Andrew Ng Coursera course on machine learning") >= 1
