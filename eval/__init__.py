"""Local evaluation and validation framework for FitRank."""

from eval.ground_truth import GroundTruth, build_ground_truth, load_ground_truth, save_ground_truth
from eval.metrics import (
    compute_all_metrics,
    honeypot_exclusion_rate,
    ndcg_at_k,
    precision_at_k,
    title_diversity,
)
from eval.sensitivity import (
    build_headline,
    jaccard_at_k,
    overlap_at_k,
    perturb_weights,
    rank_stability,
    run_ablation,
    run_bootstrap,
    run_sensitivity_analysis,
)

__all__ = [
    "GroundTruth",
    "build_ground_truth",
    "load_ground_truth",
    "save_ground_truth",
    "compute_all_metrics",
    "ndcg_at_k",
    "precision_at_k",
    "honeypot_exclusion_rate",
    "title_diversity",
    "build_headline",
    "jaccard_at_k",
    "overlap_at_k",
    "perturb_weights",
    "rank_stability",
    "run_ablation",
    "run_bootstrap",
    "run_sensitivity_analysis",
]
