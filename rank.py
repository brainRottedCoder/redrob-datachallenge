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

from fitrank.calibrator import calibrate_scores, rescale_submission_scores
from fitrank.embedder import encode_jd, load_candidate_embeddings
from fitrank.jd_parser import parse_jd, save_role_profile
from fitrank.loader import load_candidates
from fitrank.llm_reasoning import (
    DEFAULT_CACHE_PATH,
    DEFAULT_MODEL_PATH,
    LlamaReasoningEngine,
    ReasoningCache,
    ReasoningMode,
    ReasoningStats,
    jd_fingerprint,
    resolve_reasoning,
)
from fitrank.ranker import load_weights, rank_candidates
from fitrank.reasoning import build_reasoning


def write_submission_csv(
    path: Path,
    ranked,
    *,
    role_profile=None,
    jd_hash: str = "",
    reasoning_mode: ReasoningMode = "template",
    reasoning_cache: ReasoningCache | None = None,
    llm_engine: LlamaReasoningEngine | None = None,
    llm_model: Path = DEFAULT_MODEL_PATH,
    reasoning_max_new: int = 100,
    reasoning_stats: ReasoningStats | None = None,
) -> None:
    """Write ranked candidates to submission CSV (shared by rank.py and eval)."""
    if not ranked:
        raise ValueError("No candidates ranked")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (candidate, components, score_obj) in enumerate(ranked, start=1):
            if reasoning_mode == "template":
                reasoning = build_reasoning(
                    candidate,
                    components,
                    ml_tenure=components.ml_tenure_years,
                    role_profile=role_profile,
                )
            else:
                reasoning = resolve_reasoning(
                    candidate,
                    components,
                    mode=reasoning_mode,
                    role_profile=role_profile,
                    jd_hash=jd_hash,
                    cache=reasoning_cache,
                    engine=llm_engine,
                    model_path=llm_model,
                    ml_tenure=components.ml_tenure_years,
                    stats=reasoning_stats,
                    max_new=reasoning_max_new,
                )
            writer.writerow(
                [
                    candidate.candidate_id,
                    rank,
                    round(score_obj.final_score, 4),
                    reasoning,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank candidates for a job description")
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--jd", default="data/job_description.txt")
    parser.add_argument("--out", default="outputs/submission.csv")
    parser.add_argument("--weights", default="config/weights.yaml")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate scores to a 0.05-0.95 range for stronger discrimination.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run baseline comparison and write outputs/eval_report.json after ranking.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run weight perturbation bootstrap and ablation analysis after ranking.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "learned", "heuristic"),
        default="auto",
        help="Ranking mode: learned model (auto if present), heuristic only, or require learned.",
    )
    parser.add_argument("--model", default="models/fitrank_lgb.txt", help="Path to trained LightGBM model.")
    parser.add_argument(
        "--reasoning",
        choices=("template", "llm", "auto"),
        default="template",
        help="Reasoning mode: template (default), local LLM, or auto when model present.",
    )
    parser.add_argument(
        "--llm-model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to GGUF model for --reasoning llm|auto.",
    )
    parser.add_argument(
        "--reasoning-cache",
        default=str(DEFAULT_CACHE_PATH),
        help="JSONL cache for LLM-generated reasoning strings.",
    )
    parser.add_argument(
        "--reasoning-max-new",
        type=int,
        default=100,
        help="Max fresh LLM calls per run (0 = cache-only).",
    )
    args = parser.parse_args()

    start = time.time()
    jd_text = Path(args.jd).read_text(encoding="utf-8")
    role_profile = parse_jd(jd_text)
    save_role_profile(role_profile)

    weights = load_weights(args.weights)
    candidate_embeddings = load_candidate_embeddings(
        "outputs/candidate_embeddings.npy",
        "outputs/candidate_ids.json",
    )
    # Only encode JD when pre-computed candidate embeddings exist; otherwise use sparse matching
    # and avoid batch-encoding 100K profiles on the fly (exceeds 5-minute budget).
    jd_embedding = encode_jd(jd_text) if candidate_embeddings else None
    ranked = rank_candidates(
        load_candidates(args.candidates, validate=False),
        role_profile,
        weights=weights,
        top_n=args.top_n,
        jd_text=jd_text,
        jd_embedding=jd_embedding,
        candidate_embeddings=candidate_embeddings,
        ranking_mode=args.mode,
        model_path=args.model,
    )
    ranked = ranked[: args.top_n]

    if args.calibrate:
        ranked = calibrate_scores(ranked, target_p10=0.40, target_p90=0.90)
    else:
        ranked = rescale_submission_scores(ranked)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jd_hash = jd_fingerprint(jd_text, role_profile)
    reasoning_mode: ReasoningMode = args.reasoning
    reasoning_cache = (
        ReasoningCache(args.reasoning_cache) if reasoning_mode != "template" else None
    )
    llm_engine = None
    if reasoning_mode in ("llm", "auto"):
        from fitrank.llm_reasoning import is_llama_available

        llm_model = Path(args.llm_model)
        if is_llama_available(llm_model):
            llm_engine = LlamaReasoningEngine(llm_model)

    reasoning_stats = ReasoningStats() if reasoning_mode != "template" else None
    write_submission_csv(
        output_path,
        ranked,
        role_profile=role_profile,
        jd_hash=jd_hash,
        reasoning_mode=reasoning_mode,
        reasoning_cache=reasoning_cache,
        llm_engine=llm_engine,
        llm_model=Path(args.llm_model),
        reasoning_max_new=args.reasoning_max_new,
        reasoning_stats=reasoning_stats,
    )

    def _reasoning_for_audit(item):
        candidate, components, _ = item
        if reasoning_mode == "template":
            return build_reasoning(
                candidate,
                components,
                ml_tenure=components.ml_tenure_years,
                role_profile=role_profile,
            )
        return resolve_reasoning(
            candidate,
            components,
            mode=reasoning_mode,
            role_profile=role_profile,
            jd_hash=jd_hash,
            cache=reasoning_cache,
            engine=llm_engine,
            model_path=Path(args.llm_model),
            ml_tenure=components.ml_tenure_years,
            max_new=0,
        )

    audit = {
        "runtime_seconds": round(time.time() - start, 2),
        "title_distribution": dict(Counter(item[0].profile.current_title for item in ranked)),
        "ranking_mode": args.mode,
        "model_path": args.model,
        "reasoning_mode": reasoning_mode,
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
                "reasoning": _reasoning_for_audit(item),
            }
            for item in ranked[:5]
        ],
    }
    if reasoning_stats is not None:
        audit["llm_cache_hits"] = reasoning_stats.cache_hits
        audit["llm_generated"] = reasoning_stats.llm_generated
        audit["llm_template_fallbacks"] = reasoning_stats.template_fallbacks
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

    if args.eval:
        from eval.compare import run_comparison

        run_comparison(
            candidates_path=Path(args.candidates),
            jd_path=Path(args.jd),
            ground_truth_path=Path("outputs/ground_truth.jsonl"),
            baseline_output=Path("outputs/baseline_submission.csv"),
            fitrank_output=output_path,
            report_path=Path("outputs/eval_report.json"),
            weights_path=Path(args.weights),
            top_n=args.top_n,
        )

    if args.sensitivity:
        from eval.sensitivity import run_sensitivity_analysis

        report = run_sensitivity_analysis(
            candidates_path=Path(args.candidates),
            jd_path=Path(args.jd),
            ground_truth_path=Path("outputs/ground_truth.jsonl"),
            report_path=Path("outputs/sensitivity_report.json"),
            weights_path=Path(args.weights),
            model_path=Path(args.model),
        )
        print(report["headline"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
