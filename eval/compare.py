"""Evaluation framework: baseline comparison + NDCG metrics.

Because the Redrob challenge dataset has no public labels, we generate weak
synthetic labels from the heuristic ranker (top 5% positive, bottom 20% negative)
and compute ranking quality metrics against those labels. This lets us prove
that FitRank improves over a naive keyword-counting baseline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fitrank.calibrator import rescale_submission_scores
from fitrank.embedder import encode_jd, load_candidate_embeddings
from fitrank.features import extract_ranking_features
from fitrank.jd_parser import parse_jd
from fitrank.loader import load_candidates
from fitrank.models import Candidate, RoleProfile
from fitrank.pseudo_labels import (
    GENUINE_PROBE_ID,
    KNOWN_TRAP_IDS,
    assign_pseudo_label,
    percentile_threshold,
)
from fitrank.ranker import rank_candidates
from fitrank.signals import load_normalization_constants
from fitrank.skill_trust import extract_ml_skills


def _ndcg(scores: list[float], labels: list[int], k: int | None = None) -> float:
    """Compute Normalized Discounted Cumulative Gain for a ranked list.

    ``scores`` are the model-predicted relevance scores used to sort the list.
    ``labels`` are the ground-truth relevance labels in the same order as ``scores``.
    """
    if k is None:
        k = len(scores)
    k = min(k, len(scores))
    if k == 0:
        return 0.0

    def dcg(vals: list[float]) -> float:
        return sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(vals[:k]))

    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    sorted_labels = [label for _, label in paired]
    ideal = sorted(labels, reverse=True)
    ideal_dcg = dcg(ideal)
    actual_dcg = dcg(sorted_labels)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


import math  # noqa: E402


def _precision_at_k(ranked_ids: list[str], positive_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in positive_ids)
    return hits / k


def _recall_at_k(ranked_ids: list[str], positive_ids: set[str], k: int) -> float:
    if not positive_ids:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in positive_ids)
    return hits / len(positive_ids)


def _generate_synthetic_labels(
    candidates: list[Candidate],
    role_profile: RoleProfile,
    weights: dict,
    norms: dict[str, float],
    jd_embedding: np.ndarray | None,
    candidate_embeddings: dict[str, np.ndarray] | None,
) -> dict[str, int]:
    """Generate positive/negative labels from heuristic scores."""
    scored: list[tuple[str, float, bool]] = []
    for candidate in candidates:
        _, _, heuristic_score = extract_ranking_features(
            candidate,
            role_profile,
            weights=weights,
            norms=norms,
            jd_embedding=jd_embedding,
            candidate_embeddings=candidate_embeddings,
        )
        is_honeypot = candidate.candidate_id in KNOWN_TRAP_IDS
        scored.append((candidate.candidate_id, heuristic_score, is_honeypot))

    scores = [s for _, s, _ in scored]
    pos_threshold = percentile_threshold(scores, 0.95)
    neg_threshold = percentile_threshold(scores, 0.20)

    labels: dict[str, int] = {}
    for cid, score, is_honeypot in scored:
        label = assign_pseudo_label(
            score, pos_threshold, neg_threshold, is_honeypot=is_honeypot, candidate_id=cid
        )
        if label is not None:
            labels[cid] = label
    return labels


def _keyword_baseline_rank(
    candidates: list[Candidate],
    role_profile: RoleProfile,
) -> list[tuple[str, float]]:
    """Naive baseline: count AI skill matches against JD capabilities."""
    capabilities = {cap.lower() for cap in role_profile.required_capabilities}
    scored: list[tuple[str, float]] = []
    for candidate in candidates:
        ml_skills = extract_ml_skills(candidate.skills)
        matches = sum(
            1 for skill in ml_skills
            if any(cap in skill.name.lower() for cap in capabilities)
        )
        # Tie-break by total ML skills (bad, but intentionally naive).
        score = matches + 0.01 * len(ml_skills)
        scored.append((candidate.candidate_id, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def _load_ranked_ids(csv_path: Path) -> list[str]:
    ids: list[str] = []
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ids.append(row["candidate_id"])
    return ids


def _build_ground_truth_file(
    candidates_path: Path,
    jd_path: Path,
    weights: dict,
    norms: dict[str, float],
    output_path: Path,
) -> dict[str, int]:
    """Generate and save synthetic labels for reproducible evaluation."""
    candidates = list(load_candidates(candidates_path, validate=False))
    jd_text = jd_path.read_text(encoding="utf-8")
    role_profile = parse_jd(jd_text)
    jd_embedding = encode_jd(jd_text)
    candidate_embeddings = load_candidate_embeddings(
        "outputs/candidate_embeddings.npy", "outputs/candidate_ids.json"
    )

    labels = _generate_synthetic_labels(
        candidates, role_profile, weights, norms, jd_embedding, candidate_embeddings
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for cid, label in labels.items():
            handle.write(json.dumps({"candidate_id": cid, "label": label}) + "\n")
    return labels


def _load_ground_truth(ground_truth_path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with ground_truth_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            labels[record["candidate_id"]] = int(record["label"])
    return labels


def run_comparison(
    candidates_path: Path,
    jd_path: Path,
    ground_truth_path: Path,
    baseline_output: Path,
    fitrank_output: Path,
    report_path: Path,
    weights_path: Path | None = None,
    top_n: int = 100,
) -> dict[str, Any]:
    """Run baseline vs. FitRank comparison and write a JSON report."""
    from fitrank.config_loader import load_weights

    weights = load_weights(weights_path) if weights_path else load_weights()
    norms = load_normalization_constants()

    # Generate ground truth if missing.
    if not ground_truth_path.exists():
        _build_ground_truth_file(candidates_path, jd_path, weights, norms, ground_truth_path)
    labels = _load_ground_truth(ground_truth_path)
    positive_ids = {cid for cid, label in labels.items() if label == 1}
    negative_ids = {cid for cid, label in labels.items() if label == 0}

    # Run baseline keyword ranker and write its CSV.
    candidates = list(load_candidates(candidates_path, validate=False))
    jd_text = jd_path.read_text(encoding="utf-8")
    role_profile = parse_jd(jd_text)
    baseline_ranked = _keyword_baseline_rank(candidates, role_profile)[:top_n]

    baseline_output.parent.mkdir(parents=True, exist_ok=True)
    with baseline_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score) in enumerate(baseline_ranked, start=1):
            writer.writerow([cid, rank, round(score, 4), "baseline keyword count"])

    baseline_ids = [cid for cid, _ in baseline_ranked]
    fitrank_ids = _load_ranked_ids(fitrank_output)

    baseline_labels = [labels.get(cid, 0) for cid in baseline_ids]
    fitrank_rows = list(csv.DictReader(fitrank_output.open(encoding="utf-8")))
    fitrank_ids = [r["candidate_id"] for r in fitrank_rows]
    fitrank_scores = [float(r["score"]) for r in fitrank_rows]
    fitrank_labels = [labels.get(cid, 0) for cid in fitrank_ids]
    baseline_scores = [score for _, score in baseline_ranked]

    metrics = {
        "baseline": {
            "ndcg@10": round(_ndcg(baseline_scores, baseline_labels, 10), 4),
            "ndcg@100": round(_ndcg(baseline_scores, baseline_labels, 100), 4),
            "precision@100": round(_precision_at_k(baseline_ids, positive_ids, 100), 4),
            "recall@100": round(_recall_at_k(baseline_ids, positive_ids, 100), 4),
            "traps_in_top_100": sorted(set(baseline_ids) & KNOWN_TRAP_IDS),
            "probe_rank": next((i + 1 for i, cid in enumerate(baseline_ids) if cid == GENUINE_PROBE_ID), None),
        },
        "fitrank": {
            "ndcg@10": round(_ndcg(fitrank_scores, fitrank_labels, 10), 4),
            "ndcg@100": round(_ndcg(fitrank_scores, fitrank_labels, 100), 4),
            "precision@100": round(_precision_at_k(fitrank_ids, positive_ids, 100), 4),
            "recall@100": round(_recall_at_k(fitrank_ids, positive_ids, 100), 4),
            "traps_in_top_100": sorted(set(fitrank_ids) & KNOWN_TRAP_IDS),
            "probe_rank": next((i + 1 for i, cid in enumerate(fitrank_ids) if cid == GENUINE_PROBE_ID), None),
        },
        "label_counts": {
            "positive": len(positive_ids),
            "negative": len(negative_ids),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FitRank against a keyword baseline")
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--jd", default="data/job_description.txt")
    parser.add_argument("--ground-truth", default="outputs/ground_truth.jsonl")
    parser.add_argument("--baseline-out", default="outputs/baseline_submission.csv")
    parser.add_argument("--fitrank-out", default="outputs/submission.csv")
    parser.add_argument("--report", default="outputs/eval_report.json")
    parser.add_argument("--weights", default="config/weights.yaml")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    from fitrank.config_loader import load_weights

    weights = load_weights(args.weights)
    norms = load_normalization_constants()

    if not Path(args.ground_truth).exists():
        _build_ground_truth_file(
            Path(args.candidates), Path(args.jd), weights, norms, Path(args.ground_truth)
        )

    metrics = run_comparison(
        candidates_path=Path(args.candidates),
        jd_path=Path(args.jd),
        ground_truth_path=Path(args.ground_truth),
        baseline_output=Path(args.baseline_out),
        fitrank_output=Path(args.fitrank_out),
        report_path=Path(args.report),
        weights_path=Path(args.weights),
        top_n=args.top_n,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
