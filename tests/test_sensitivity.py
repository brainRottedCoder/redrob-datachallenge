"""Tests for sensitivity analysis and rank stability metrics."""

from __future__ import annotations

import random

from eval.ground_truth import GroundTruth
from eval.metrics import ndcg_at_k
from eval.sensitivity import (
    CachedCandidate,
    apply_feature_ablation,
    heuristic_score,
    jaccard_at_k,
    overlap_at_k,
    perturb_weights,
    rank_from_cache,
    rank_stability,
    run_ablation,
    run_bootstrap,
)
from fitrank.component_scores import ComponentScores
from fitrank.features import RankingFeatures
from fitrank.penalties import PenaltyScore


def _penalty_detail(total: float = 0.0, skill_inflation: float = 0.0) -> PenaltyScore:
    return PenaltyScore(
        template_mismatch=0.0,
        skill_inflation=skill_inflation,
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


def _components(
    candidate_id: str,
    *,
    jd_fit: float = 0.5,
    career: float = 0.5,
    coherence: float = 0.5,
    trust: float = 0.5,
    penalties: float = 0.0,
    skill_inflation: float = 0.0,
) -> ComponentScores:
    return ComponentScores(
        jd_fit=jd_fit,
        career_evidence=career,
        coherence=coherence,
        platform_trust=trust,
        availability=0.5,
        penalties=penalties,
        final_score=0.0,
        is_honeypot=False,
        penalties_detail=_penalty_detail(penalties, skill_inflation),
        ml_tenure_years=3.0,
    )


def _cached_rows() -> list[CachedCandidate]:
    rows = [
        ("CAND_A", 0.9, 0.8, 0.85, 0.7),
        ("CAND_B", 0.7, 0.9, 0.6, 0.8),
        ("CAND_C", 0.5, 0.5, 0.5, 0.5),
        ("CAND_D", 0.2, 0.2, 0.2, 0.2),
    ]
    cached: list[CachedCandidate] = []
    for candidate_id, jd_fit, career, coherence, trust in rows:
        cached.append(
            CachedCandidate(
                candidate_id=candidate_id,
                components=_components(
                    candidate_id,
                    jd_fit=jd_fit,
                    career=career,
                    coherence=coherence,
                    trust=trust,
                ),
                features=RankingFeatures(coherence_score=coherence, trusted_skill_ratio=0.5),
            )
        )
    return cached


def _ground_truth() -> dict[str, GroundTruth]:
    return {
        "CAND_A": GroundTruth(
            candidate_id="CAND_A",
            label="relevant_2",
            tier="relevant_2",
            is_honeypot=False,
            relevance=2.0,
            title_domain="ML_AI",
            career_ml_depth=8,
            ml_skill_count=10,
        ),
        "CAND_B": GroundTruth(
            candidate_id="CAND_B",
            label="relevant_1",
            tier="relevant_1",
            is_honeypot=False,
            relevance=1.0,
            title_domain="ML_AI",
            career_ml_depth=4,
            ml_skill_count=6,
        ),
        "CAND_C": GroundTruth(
            candidate_id="CAND_C",
            label="irrelevant",
            tier="irrelevant",
            is_honeypot=False,
            relevance=0.0,
            title_domain="SOFTWARE",
            career_ml_depth=1,
            ml_skill_count=2,
        ),
        "CAND_D": GroundTruth(
            candidate_id="CAND_D",
            label="honeypot",
            tier="honeypot",
            is_honeypot=True,
            relevance=0.0,
            title_domain="NON_TECH",
            career_ml_depth=0,
            ml_skill_count=9,
        ),
    }


def test_perturb_weights_within_bounds():
    base = {
        "jd_fit": 0.25,
        "career_evidence": 0.30,
        "coherence": 0.20,
        "platform_trust": 0.15,
        "availability": 0.10,
    }
    rng = random.Random(0)
    for _ in range(20):
        perturbed = perturb_weights(base, rng, pct=0.10)
        for key, original in base.items():
            assert original * 0.9 <= perturbed[key] <= original * 1.1


def test_rank_stability_identical_rankings():
    ids = ["A", "B", "C", "D"]
    stability = rank_stability(ids, ids, k=3)
    assert stability["overlap"] == 3
    assert stability["jaccard"] == 1.0
    assert stability["kendall_tau"] == 1.0


def test_jaccard_and_overlap():
    base = ["A", "B", "C", "D"]
    other = ["A", "B", "X", "Y"]
    assert overlap_at_k(base, other, k=3) == 2
    assert jaccard_at_k(base, other, k=3) == round(2 / 4, 4)


def test_bootstrap_produces_stability_metrics():
    cached = _cached_rows()
    weights = {
        "jd_fit": 0.25,
        "career_evidence": 0.30,
        "coherence": 0.20,
        "platform_trust": 0.15,
        "availability": 0.10,
    }
    result = run_bootstrap(cached, weights, samples=5, seed=1, top_n=4)
    assert result["samples"] == 5
    assert 0 <= result["top20_mean_overlap"] <= 20
    assert result["top100_mean_jaccard"] >= 0.0


def test_ablation_reduces_ndcg_for_coherence_removal():
    cached = _cached_rows()
    weights = {
        "jd_fit": 0.25,
        "career_evidence": 0.30,
        "coherence": 0.20,
        "platform_trust": 0.15,
        "availability": 0.10,
    }
    ground_truth = _ground_truth()
    result = run_ablation(cached, ground_truth, {}, weights, top_n=4)

    baseline = result["heuristic"]["baseline_ndcg@100"]
    coherence_ndcg = result["heuristic"]["coherence"]["ndcg@100"]
    assert coherence_ndcg <= baseline


def test_feature_ablation_zeros_target_features():
    features = RankingFeatures(
        coherence_score=0.8,
        template_coherence=0.7,
        trusted_skill_ratio=0.6,
        assessment_avg=0.9,
    )
    ablated = apply_feature_ablation(features, "coherence")
    assert ablated.coherence_score == 0.0
    assert ablated.template_coherence == 0.0
    assert ablated.trusted_skill_ratio == 0.6


def test_rank_from_cache_orders_by_heuristic_score():
    cached = _cached_rows()
    weights = {
        "jd_fit": 0.25,
        "career_evidence": 0.30,
        "coherence": 0.20,
        "platform_trust": 0.15,
        "availability": 0.10,
    }
    ranked = rank_from_cache(
        cached,
        lambda row: heuristic_score(row.components, weights),
        top_n=3,
    )
    assert ranked[0] == "CAND_A"
    assert len(ranked) == 3


def test_ndcg_baseline_beats_random_on_synthetic_labels():
    ground_truth = _ground_truth()
    good = ["CAND_A", "CAND_B", "CAND_C", "CAND_D"]
    bad = ["CAND_D", "CAND_C", "CAND_B", "CAND_A"]
    assert ndcg_at_k(good, ground_truth, k=4) > ndcg_at_k(bad, ground_truth, k=4)
