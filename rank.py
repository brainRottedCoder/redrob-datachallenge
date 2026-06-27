#!/usr/bin/env python3
"""FitRank CLI — rank candidates against a job description."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from fitrank.calibrator import calibrate_scores
from fitrank.jd_parser import parse_jd, save_role_profile
from fitrank.loader import load_candidates
from fitrank.ranker import load_capability_vectors, load_weights, rank_candidates
from fitrank.reasoning import build_reasoning


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank candidates for a job description")
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--jd", default="data/job_description.txt")
    parser.add_argument("--out", default="outputs/submission.csv")
    parser.add_argument("--weights", default="config/weights.yaml")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--calibrate", action="store_true", help="Calibrate scores for multi-JD analysis (do NOT use for challenge submission).")
    args = parser.parse_args()

    start = time.time()
    jd_text = Path(args.jd).read_text(encoding="utf-8")
    role_profile = parse_jd(jd_text)
    save_role_profile(role_profile)

    weights = load_weights(args.weights)
    capability_vectors = load_capability_vectors()
    ranked = rank_candidates(
        load_candidates(args.candidates, validate=False),
        role_profile,
        weights=weights,
        top_n=args.top_n,
        capability_vectors=capability_vectors,
    )
    ranked = ranked[: args.top_n]

    if args.calibrate:
        ranked = calibrate_scores(ranked)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_submission_csv(output_path, ranked)

    audit = {
        "runtime_seconds": round(time.time() - start, 2),
        "title_distribution": dict(Counter(item[0].profile.current_title for item in ranked)),
        "known_traps_excluded": [],
        "honeypot_flags_in_shortlist": sum(1 for _, comp, _ in ranked if comp.is_honeypot),
        "score_histogram": {
            "min": min(item[2].final_score for item in ranked),
            "max": max(item[2].final_score for item in ranked),
            "mean": round(sum(item[2].final_score for item in ranked) / len(ranked), 4),
        },
        "top_5": [
            {
                "candidate_id": item[0].candidate_id,
                "title": item[0].profile.current_title,
                "final_score": item[2].final_score,
                "reasoning": build_reasoning(item[0], item[1], ml_tenure=item[1].ml_tenure_years),
            }
            for item in ranked[:5]
        ],
    }
    audit_path = Path(args.out).parent / "audit_report.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, default=lambda o: dict(o)),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "validate_submission.py", str(output_path)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
    )
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        return result.returncode

    print(f"Ranked {len(ranked)} candidates in {audit['runtime_seconds']}s")
    return 0


def _write_submission_csv(path: Path, ranked) -> None:
    if not ranked:
        raise ValueError("No candidates ranked")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (candidate, components, score_obj) in enumerate(ranked, start=1):
            writer.writerow(
                [
                    candidate.candidate_id,
                    rank,
                    round(score_obj.final_score, 4),
                    build_reasoning(candidate, components, ml_tenure=components.ml_tenure_years),
                ]
            )


if __name__ == "__main__":
    raise SystemExit(main())
