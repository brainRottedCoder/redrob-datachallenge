"""Sensitivity analysis and ablation utilities for FitRank.

These functions operate on cached candidate rows (ComponentScores + RankingFeatures)
so that ablation and bootstrap experiments are fast and deterministic.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from eval.ground_truth import GroundTruth
from eval.metrics import compute_all_metrics, ndcg_at_k
from fitrank.component_scores import ComponentScores
from fitrank.features import RankingFeatures

COMPONENTS = ["jd_fit", "career_evidence", "coherence", "platform_trust", "availability"]


@dataclass(frozen=True)
class CachedCandidate:
    candidate_id: str
    components: ComponentScores
    features: RankingFeatures


def heuristic_score(components: ComponentScores, weights: dict) -> float:
    """Compute the weighted heuristic score from component scores."""
    raw = (
        weights.get("jd_fit", 0.0) * components.jd_fit
        + weights.get("career_evidence", 0.0) * components.career_evidence
        + weights.get("coherence", 0.0) * components.coherence
        + weights.get("platform_trust", 0.0) * components.platform_trust
        + weights.get("availability", 0.0) * components.availability
        - components.penalties
    )
    if components.is_honeypot and components.penalties > 0.30:
        raw *= 0.25
    return max(0.0, min(1.0, raw))


def perturb_weights(weights: dict, rng: random.Random, pct: float = 0.10) -> dict:
    """Perturb each component weight by ±pct.

    Note: weights are not re-normalized here so that unit tests can verify the
    perturbation bounds directly. Callers that need normalized weights should do
    so after calling this function.
    """
    perturbed = dict(weights)
    for k in COMPONENTS:
        perturbed[k] = weights[k] * rng.uniform(1.0 - pct, 1.0 + pct)
    return perturbed


def overlap_at_k(a: list[str], b: list[str], k: int) -> int:
    """Count of shared ids in the top-k of two rankings."""
    return len(set(a[:k]) & set(b[:k]))


def jaccard_at_k(a: list[str], b: list[str], k: int) -> float:
    """Jaccard similarity of the top-k id sets of two rankings."""
    set_a = set(a[:k])
    set_b = set(b[:k])
    union = set_a | set_b
    if not union:
        return 0.0
    return round(len(set_a & set_b) / len(union), 4)


def _kendall_tau(a: list[str], b: list[str], k: int) -> float:
    """Simple Kendall tau over the top-k of two rankings (concordant pairs)."""
    ids = [x for x in a[:k] if x in set(b[:k])]
    if len(ids) < 2:
        return 1.0
    pos_in_b = {x: i for i, x in enumerate(b[:k])}
    concordant = 0
    discordant = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if pos_in_b[ids[i]] < pos_in_b[ids[j]]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return round((concordant - discordant) / total, 4)


def rank_stability(a: list[str], b: list[str], k: int) -> dict[str, Any]:
    """Return overlap, jaccard, and kendall_tau for two rankings at top-k."""
    return {
        "overlap": overlap_at_k(a, b, k),
        "jaccard": jaccard_at_k(a, b, k),
        "kendall_tau": _kendall_tau(a, b, k),
    }


def rank_from_cache(
    cached_rows: list[CachedCandidate],
    score_fn: Callable[[CachedCandidate], float],
    top_n: int = 100,
) -> list[str]:
    """Rank cached candidates using a provided scoring function."""
    scored = [(score_fn(row), row.candidate_id) for row in cached_rows]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [candidate_id for _, candidate_id in scored[:top_n]]


def apply_feature_ablation(features: RankingFeatures, component: str) -> RankingFeatures:
    """Return a copy of features with fields belonging to ``component`` zeroed out."""
    ablated = copy.copy(features)
    mapping = {
        "jd_fit": ["title_jd_match", "capability_match", "capability_match_sparse", "capability_match_dense"],
        "career_evidence": ["career_ml_depth", "all_career_ml_depth_norm", "current_role_ml_depth_norm", "career_momentum"],
        "coherence": ["coherence_score", "template_coherence", "title_domain_confirmation", "skill_career_alignment"],
        "platform_trust": ["platform_trust", "assessment_avg", "github_norm", "engagement_composite", "verification_score"],
        "availability": ["availability_score"],
    }
    for field in mapping.get(component, []):
        if hasattr(ablated, field):
            setattr(ablated, field, 0.0)
    return ablated


def run_bootstrap(
    cached_rows: list[CachedCandidate],
    weights: dict,
    samples: int = 20,
    seed: int = 42,
    top_n: int = 100,
) -> dict[str, Any]:
    """Measure ranking stability under random weight perturbations."""
    rng = random.Random(seed)
    base = rank_from_cache(
        cached_rows,
        lambda row: heuristic_score(row.components, weights),
        top_n=top_n,
    )
    top20_overlaps: list[int] = []
    top100_jaccards: list[float] = []

    for _ in range(samples):
        perturbed = perturb_weights(weights, rng)
        total = sum(perturbed[k] for k in COMPONENTS)
        if total > 0:
            perturbed = {k: v / total for k, v in perturbed.items()}
        perturbed_rank = rank_from_cache(
            cached_rows,
            lambda row: heuristic_score(row.components, perturbed),
            top_n=top_n,
        )
        top20_overlaps.append(overlap_at_k(base, perturbed_rank, 20))
        top100_jaccards.append(jaccard_at_k(base, perturbed_rank, 100))

    return {
        "samples": samples,
        "top20_mean_overlap": round(sum(top20_overlaps) / len(top20_overlaps), 4),
        "top20_min_overlap": min(top20_overlaps),
        "top20_max_overlap": max(top20_overlaps),
        "top100_mean_jaccard": round(sum(top100_jaccards) / len(top100_jaccards), 4),
    }


def run_ablation(
    cached_rows: list[CachedCandidate],
    ground_truth: dict[str, GroundTruth],
    model: Any,
    weights: dict,
    top_n: int = 100,
) -> dict[str, Any]:
    """Run component ablation using cached heuristic scores."""
    base_ids = rank_from_cache(
        cached_rows,
        lambda row: heuristic_score(row.components, weights),
        top_n=top_n,
    )
    baseline_ndcg = ndcg_at_k(base_ids, ground_truth, k=top_n)

    result: dict[str, Any] = {
        "heuristic": {
            "baseline_ndcg@100": round(baseline_ndcg, 4),
        }
    }

    for component in COMPONENTS:
        ablated_weights = copy.deepcopy(weights)
        ablated_weights[component] = 0.0
        ids = rank_from_cache(
            cached_rows,
            lambda row: heuristic_score(row.components, ablated_weights),
            top_n=top_n,
        )
        ndcg = ndcg_at_k(ids, ground_truth, k=top_n)
        overlap = len(set(base_ids[:20]) & set(ids[:20])) / 20.0
        result["heuristic"][component] = {
            "ndcg@100": round(ndcg, 4),
            "drop": round(baseline_ndcg - ndcg, 4),
            "top_20_overlap": round(overlap, 4),
        }
    return result


def build_headline(report: dict[str, Any]) -> str:
    """Build a one-line summary of a sensitivity/ablation report."""
    baseline = report.get("heuristic", {}).get("baseline_ndcg@100", 0.0)
    return f"Baseline NDCG@100 = {baseline:.4f}"


def run_sensitivity_analysis(
    cached_rows: list[CachedCandidate],
    ground_truth: dict[str, GroundTruth],
    weights: dict,
    top_n: int = 100,
) -> dict[str, Any]:
    """Convenience wrapper that runs both ablation and bootstrap."""
    report = run_ablation(cached_rows, ground_truth, None, weights, top_n=top_n)
    report["bootstrap"] = run_bootstrap(cached_rows, weights, top_n=top_n)
    report["headline"] = build_headline(report)
    return report
