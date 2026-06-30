#!/usr/bin/env python3
"""Audit a FitRank submission CSV with ground-truth-aware metrics."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.ground_truth import GENUINE_LABELS, load_ground_truth
from eval.metrics import (
    compute_all_metrics,
    load_ranked_ids_from_csv,
    percent_improvement,
)


def audit_submission(
    submission_path: Path,
    ground_truth_path: Path,
    baseline_path: Path | None = None,
) -> dict:
    ranked_ids = load_ranked_ids_from_csv(submission_path)
    ground_truth = load_ground_truth(ground_truth_path) if ground_truth_path.exists() else {}

    rows = list(csv.DictReader(submission_path.open(encoding="utf-8")))
    scores = [float(row["score"]) for row in rows]
    titles = Counter(row["reasoning"].split(" | ")[0] for row in rows)

    report: dict = {
        "rows": len(rows),
        "title_distribution": dict(titles.most_common(15)),
        "score_distribution": {
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
            "mean": round(statistics.mean(scores), 4) if scores else 0.0,
            "median": round(statistics.median(scores), 4) if scores else 0.0,
        },
    }

    if ground_truth:
        metrics = compute_all_metrics(ranked_ids, ground_truth)
        report["genuine_in_top_100"] = metrics["genuine_in_top_100"]
        report["honeypots_in_top_100"] = metrics["honeypots_in_top_100"]
        report["honeypot_exclusion_rate"] = metrics["honeypot_exclusion_rate"]
        report["ndcg@10"] = metrics["ndcg"]["@10"]
        report["ndcg@100"] = metrics["ndcg"]["@100"]
        report["precision@100"] = metrics["precision"]["@100"]

    if baseline_path and baseline_path.exists() and ground_truth:
        baseline_ids = load_ranked_ids_from_csv(baseline_path)
        baseline_metrics = compute_all_metrics(baseline_ids, ground_truth)
        report["baseline_comparison"] = {
            "ndcg@100": baseline_metrics["ndcg"]["@100"],
            "precision@100": baseline_metrics["precision"]["@100"],
            "honeypot_exclusion_rate": baseline_metrics["honeypot_exclusion_rate"],
            "ndcg@100_improvement_pct": percent_improvement(
                baseline_metrics["ndcg"]["@100"], report["ndcg@100"]
            ),
            "precision@100_improvement_pct": percent_improvement(
                baseline_metrics["precision"]["@100"], report["precision@100"]
            ),
        }

    return report


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/audit_shortlist.py outputs/submission.csv [ground_truth.jsonl]")
        return 1

    submission_path = Path(sys.argv[1])
    ground_truth_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/ground_truth.jsonl")
    )
    baseline_path = Path("outputs/baseline_submission.csv")

    report = audit_submission(submission_path, ground_truth_path, baseline_path)

    print(f"Rows: {report['rows']}")
    print("Title distribution:")
    for title, count in list(report["title_distribution"].items())[:10]:
        print(f"  {title}: {count}")

    print("Score distribution:")
    for key, value in report["score_distribution"].items():
        print(f"  {key}: {value}")

    if "genuine_in_top_100" in report:
        print(f"Genuine ML candidates in top 100: {report['genuine_in_top_100']}")
        print(f"Honeypots in top 100: {report['honeypots_in_top_100']}")
        print(f"Honeypot exclusion rate: {report['honeypot_exclusion_rate']:.2%}")
        print(f"NDCG@10: {report['ndcg@10']}")
        print(f"NDCG@100: {report['ndcg@100']}")
        print(f"Precision@100: {report['precision@100']}")

    if "baseline_comparison" in report:
        print("Baseline comparison:")
        for key, value in report["baseline_comparison"].items():
            print(f"  {key}: {value}")

    audit_path = submission_path.parent / "audit_shortlist_report.json"
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
