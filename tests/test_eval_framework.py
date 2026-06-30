"""Tests for the local evaluation and validation framework."""

from __future__ import annotations

from eval.baseline_ranker import BASELINE_KEYWORDS, keyword_count, score_candidate_baseline
from eval.ground_truth import GroundTruth, label_candidate
from eval.metrics import (
    compute_all_metrics,
    honeypot_exclusion_rate,
    ndcg_at_k,
    precision_at_k,
    title_diversity,
)
from fitrank.loader import load_candidates
from fitrank.models import Candidate


def _make_ground_truth() -> dict[str, GroundTruth]:
    return {
        "CAND_0000001": GroundTruth(
            candidate_id="CAND_0000001",
            label="relevant_2",
            tier="relevant_2",
            is_honeypot=False,
            relevance=2.0,
            title_domain="ML_AI",
            career_ml_depth=8,
            ml_skill_count=10,
        ),
        "CAND_0000002": GroundTruth(
            candidate_id="CAND_0000002",
            label="relevant_1",
            tier="relevant_1",
            is_honeypot=False,
            relevance=1.0,
            title_domain="ML_AI",
            career_ml_depth=3,
            ml_skill_count=6,
        ),
        "CAND_0000003": GroundTruth(
            candidate_id="CAND_0000003",
            label="honeypot",
            tier="honeypot",
            is_honeypot=True,
            relevance=0.0,
            title_domain="NON_TECH",
            career_ml_depth=0,
            ml_skill_count=9,
        ),
        "CAND_0000004": GroundTruth(
            candidate_id="CAND_0000004",
            label="irrelevant",
            tier="irrelevant",
            is_honeypot=False,
            relevance=0.0,
            title_domain="NON_TECH",
            career_ml_depth=0,
            ml_skill_count=1,
        ),
    }


def test_ndcg_perfect_ranking():
    ground_truth = _make_ground_truth()
    ranked_ids = ["CAND_0000001", "CAND_0000002", "CAND_0000003", "CAND_0000004"]
    assert ndcg_at_k(ranked_ids, ground_truth, k=4) == 1.0


def test_ndcg_worst_ranking():
    ground_truth = _make_ground_truth()
    perfect = ["CAND_0000001", "CAND_0000002", "CAND_0000003", "CAND_0000004"]
    worst = ["CAND_0000004", "CAND_0000003", "CAND_0000002", "CAND_0000001"]
    assert ndcg_at_k(worst, ground_truth, k=4) < ndcg_at_k(perfect, ground_truth, k=4)


def test_precision_all_genuine():
    ground_truth = _make_ground_truth()
    ranked_ids = ["CAND_0000001", "CAND_0000002"]
    assert precision_at_k(ranked_ids, ground_truth, k=2) == 1.0


def test_precision_all_irrelevant():
    ground_truth = _make_ground_truth()
    ranked_ids = ["CAND_0000003", "CAND_0000004"]
    assert precision_at_k(ranked_ids, ground_truth, k=2) == 0.0


def test_honeypot_exclusion_none_excluded():
    ground_truth = _make_ground_truth()
    ranked_ids = ["CAND_0000003", "CAND_0000001"]
    assert honeypot_exclusion_rate(ranked_ids, ground_truth, k=2) == 0.0


def test_honeypot_exclusion_all_excluded():
    ground_truth = _make_ground_truth()
    ranked_ids = ["CAND_0000001", "CAND_0000002"]
    assert honeypot_exclusion_rate(ranked_ids, ground_truth, k=2) == 1.0


def test_title_diversity_entropy():
    class Profile:
        current_title = "ML Engineer"

    class CandidateStub:
        profile = Profile()

    candidates_map = {
        "CAND_0000001": CandidateStub(),
        "CAND_0000002": CandidateStub(),
    }
    diversity = title_diversity(["CAND_0000001", "CAND_0000002"], candidates_map, k=2)
    assert diversity["unique_title_count"] == 1
    assert diversity["entropy"] == 0.0


def test_compute_all_metrics_structure():
    ground_truth = _make_ground_truth()
    ranked_ids = ["CAND_0000001", "CAND_0000002", "CAND_0000003", "CAND_0000004"]
    metrics = compute_all_metrics(ranked_ids, ground_truth)
    assert "@10" in metrics["ndcg"]
    assert "@100" in metrics["precision"]
    assert metrics["genuine_in_top_100"] == 2


def test_baseline_keyword_count():
    text = "machine learning pytorch nlp rag transformers"
    assert keyword_count(text) >= 5


def test_baseline_scores_sample_candidate(sample_path):
    candidates = list(load_candidates(sample_path))
    assert candidates
    result = score_candidate_baseline(candidates[0])
    assert result.keyword_score >= 0
    assert result.candidate_id.startswith("CAND_")


def test_label_candidate_returns_ground_truth(sample_path):
    candidate: Candidate = next(iter(load_candidates(sample_path)))
    label = label_candidate(candidate)
    assert label.candidate_id == candidate.candidate_id
    assert label.label in {
        "relevant_2",
        "relevant_1",
        "ai_adjacent",
        "honeypot",
        "irrelevant",
    }
    assert label.relevance >= 0.0


def test_baseline_keyword_list_not_empty():
    assert len(BASELINE_KEYWORDS) >= 40
