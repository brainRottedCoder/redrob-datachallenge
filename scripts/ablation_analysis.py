#!/usr/bin/env python3
"""Ablation and sensitivity analysis for FitRank.

Measures how much each scoring component contributes to the final ranking by
zeroing out one component at a time and comparing NDCG against the synthetic
ground-truth labels. Also reports top-20 stability under ±10% weight perturbation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.compare import _generate_synthetic_labels, _ndcg, _load_ground_truth
from fitrank.embedder import encode_jd, load_candidate_embeddings
from fitrank.jd_parser import parse_jd
from fitrank.loader import load_candidates
from fitrank.ranker import rank_candidates, load_weights
from fitrank.signals import load_normalization_constants

COMPONENTS = ["jd_fit", "career_evidence", "coherence", "platform_trust", "availability"]


def _rank_with_weight_override(
    candidates_path: Path,
    jd_path: Path,
    weights: dict,
    top_n: int = 100,
) -> list[str]:
    jd_text = jd_path.read_text(encoding="utf-8")
    role_profile = parse_jd(jd_text)
    candidate_embeddings = load_candidate_embeddings(
        "outputs/candidate_embeddings.npy", "outputs/candidate_ids.json"
    )
    jd_embedding = encode_jd(jd_text)

    ranked = rank_candidates(
        load_candidates(candidates_path, validate=False),
        role_profile,
        weights=weights,
        top_n=top_n,
        jd_text=jd_text,
        jd_embedding=jd_embedding,
        candidate_embeddings=candidate_embeddings,
        ranking_mode="heuristic",  # ablation runs on heuristic for interpretability
    )
    return [item[0].candidate_id for item in ranked]


def _ablation_ndcg(
    candidates_path: Path,
    jd_path: Path,
    base_weights: dict,
    labels: dict[str, int],
    top_n: int = 100,
) -> dict[str, Any]:
    base_ids = _rank_with_weight_override(candidates_path, jd_path, base_weights, top_n)
    base_ndcg = _ndcg([float(labels.get(cid, 0)) for cid in base_ids], [labels.get(cid, 0) for cid in base_ids], top_n)

    results: dict[str, Any] = {
        "full_model": {"ndcg@100": round(base_ndcg, 4)},
    }

    for component in COMPONENTS:
        ablated = copy.deepcopy(base_weights)
        ablated[component] = 0.0
        ids = _rank_with_weight_override(candidates_path, jd_path, ablated, top_n)
        ndcg = _ndcg([float(labels.get(cid, 0)) for cid in ids], [labels.get(cid, 0) for cid in ids], top_n)
        overlap = len(set(base_ids[:20]) & set(ids[:20])) / 20.0
        results[f"remove_{component}"] = {
            "ndcg@100": round(ndcg, 4),
            "drop_vs_full": round(base_ndcg - ndcg, 4),
            "top_20_overlap": round(overlap, 4),
        }

    return results


def _sensitivity_analysis(
    candidates_path: Path,
    jd_path: Path,
    base_weights: dict,
    top_n: int = 100,
    perturbations: int = 20,
) -> dict[str, Any]:
    base_ids = _rank_with_weight_override(candidates_path, jd_path, base_weights, top_n)
    rng = np.random.default_rng(42)
    overlaps: list[float] = []

    for _ in range(perturbations):
        perturbed = copy.deepcopy(base_weights)
        for component in COMPONENTS:
            perturbed[component] *= rng.uniform(0.9, 1.1)
        # Normalize so weights sum to 1.0.
        total = sum(perturbed[c] for c in COMPONENTS)
        for component in COMPONENTS:
            perturbed[component] /= total
        ids = _rank_with_weight_override(candidates_path, jd_path, perturbed, top_n)
        overlap = len(set(base_ids[:20]) & set(ids[:20])) / 20.0
        overlaps.append(overlap)

    return {
        "top_20_overlap_mean": round(float(np.mean(overlaps)), 4),
        "top_20_overlap_min": round(float(np.min(overlaps)), 4),
        "top_20_overlap_max": round(float(np.max(overlaps)), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FitRank ablation and sensitivity analysis")
    parser.add_argument("--candidates", default=str(ROOT / "data" / "candidates.jsonl"))
    parser.add_argument("--jd", default=str(ROOT / "data" / "job_description.txt"))
    parser.add_argument("--ground-truth", default=str(ROOT / "outputs" / "ground_truth.jsonl"))
    parser.add_argument("--weights", default=str(ROOT / "config" / "weights.yaml"))
    parser.add_argument("--report", default=str(ROOT / "outputs" / "ablation_report.json"))
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    jd_path = Path(args.jd)
    weights = load_weights(args.weights)
    norms = load_normalization_constants()

    if not Path(args.ground_truth).exists():
        candidates = list(load_candidates(candidates_path, validate=False))
        jd_text = jd_path.read_text(encoding="utf-8")
        role_profile = parse_jd(jd_text)
        jd_embedding = encode_jd(jd_text)
        candidate_embeddings = load_candidate_embeddings(
            "outputs/candidate_embeddings.npy", "outputs/candidate_ids.json"
        )
        labels = _generate_synthetic_labels(candidates, role_profile, weights, norms, jd_embedding, candidate_embeddings)
        Path(args.ground_truth).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.ground_truth).open("w", encoding="utf-8") as handle:
            for cid, label in labels.items():
                handle.write(json.dumps({"candidate_id": cid, "label": label}) + "\n")
    else:
        labels = _load_ground_truth(Path(args.ground_truth))

    ablation = _ablation_ndcg(candidates_path, jd_path, weights, labels, args.top_n)
    sensitivity = _sensitivity_analysis(candidates_path, jd_path, weights, args.top_n)

    report = {
        "ablation": ablation,
        "sensitivity": sensitivity,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
