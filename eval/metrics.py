"""Ranking evaluation metrics for FitRank vs baseline."""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

from eval.ground_truth import GroundTruth, LABEL_RELEVANCE


IRRELEVANT_LABELS = frozenset({"irrelevant", "honeypot"})


def relevance_score(ground_truth: dict[str, GroundTruth], candidate_id: str) -> float:
    entry = ground_truth.get(candidate_id)
    if entry is None:
        return 0.0
    return entry.relevance


def _dcg_at_k(relevances: list[float], k: int) -> float:
    total = 0.0
    for index, rel in enumerate(relevances[:k], start=1):
        total += rel / math.log2(index + 1)
    return total


def _idcg_at_k(ground_truth: dict[str, GroundTruth], k: int) -> float:
    ideal = sorted((entry.relevance for entry in ground_truth.values()), reverse=True)
    return _dcg_at_k(ideal, k)


def ndcg_at_k(
    ranked_ids: list[str],
    ground_truth: dict[str, GroundTruth],
    k: int,
) -> float:
    relevances = [relevance_score(ground_truth, candidate_id) for candidate_id in ranked_ids]
    dcg = _dcg_at_k(relevances, k)
    idcg = _idcg_at_k(ground_truth, k)
    if idcg <= 0:
        return 0.0
    return round(dcg / idcg, 4)


def precision_at_k(
    ranked_ids: list[str],
    ground_truth: dict[str, GroundTruth],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    relevant = sum(
        1
        for candidate_id in top
        if ground_truth.get(candidate_id) and ground_truth[candidate_id].label not in IRRELEVANT_LABELS
    )
    return round(relevant / k, 4)


def honeypot_exclusion_rate(
    ranked_ids: list[str],
    ground_truth: dict[str, GroundTruth],
    k: int = 100,
) -> float:
    known_honeypots = {
        candidate_id for candidate_id, entry in ground_truth.items() if entry.is_honeypot
    }
    if not known_honeypots:
        return 1.0
    in_shortlist = set(ranked_ids[:k]) & known_honeypots
    excluded = len(known_honeypots) - len(in_shortlist)
    return round(excluded / len(known_honeypots), 4)


def honeypots_in_shortlist(
    ranked_ids: list[str],
    ground_truth: dict[str, GroundTruth],
    k: int = 100,
) -> int:
    known_honeypots = {
        candidate_id for candidate_id, entry in ground_truth.items() if entry.is_honeypot
    }
    return len(set(ranked_ids[:k]) & known_honeypots)


def title_diversity(
    ranked_ids: list[str],
    candidates_map: dict[str, Any],
    k: int = 100,
) -> dict[str, Any]:
    titles: list[str] = []
    for candidate_id in ranked_ids[:k]:
        candidate = candidates_map.get(candidate_id)
        if candidate is None:
            continue
        title = getattr(getattr(candidate, "profile", None), "current_title", None)
        if title:
            titles.append(title)

    counts = Counter(titles)
    total = len(titles) or 1
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log2(probability)

    return {
        "title_counts": dict(counts.most_common()),
        "unique_title_count": len(counts),
        "entropy": round(entropy, 4),
    }


def load_ranked_ids_from_csv(path: str | Path) -> list[str]:
    rows: list[tuple[int, str]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["rank"]), row["candidate_id"].strip()))
    rows.sort(key=lambda item: item[0])
    return [candidate_id for _, candidate_id in rows]


def compute_all_metrics(
    ranked_ids: list[str],
    ground_truth: dict[str, GroundTruth],
    candidates_map: dict[str, Any] | None = None,
    k_values: tuple[int, ...] = (10, 100),
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "ndcg": {f"@{k}": ndcg_at_k(ranked_ids, ground_truth, k) for k in k_values},
        "precision": {f"@{k}": precision_at_k(ranked_ids, ground_truth, k) for k in k_values},
        "honeypot_exclusion_rate": honeypot_exclusion_rate(ranked_ids, ground_truth, k=100),
        "honeypots_in_top_100": honeypots_in_shortlist(ranked_ids, ground_truth, k=100),
        "genuine_in_top_100": sum(
            1
            for candidate_id in ranked_ids[:100]
            if ground_truth.get(candidate_id)
            and ground_truth[candidate_id].label in LABEL_RELEVANCE
            and LABEL_RELEVANCE[ground_truth[candidate_id].label] > 0.0
        ),
        "title_diversity": title_diversity(ranked_ids, candidates_map or {}, k=100)
        if candidates_map
        else {"unique_title_count": 0, "entropy": 0.0},
    }
    return metrics


def percent_improvement(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return round(((candidate - baseline) / baseline) * 100, 1)
