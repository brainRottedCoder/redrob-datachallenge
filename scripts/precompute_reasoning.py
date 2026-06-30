#!/usr/bin/env python3
"""Pre-compute local LLM reasoning for the top-100 shortlist."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fitrank.embedder import encode_jd, load_candidate_embeddings
from fitrank.jd_parser import parse_jd
from fitrank.loader import load_candidates
from fitrank.llm_reasoning import (
    DEFAULT_CACHE_PATH,
    DEFAULT_MODEL_PATH,
    LlamaReasoningEngine,
    ReasoningCache,
    ReasoningStats,
    is_llama_available,
    jd_fingerprint,
    resolve_reasoning,
)
from fitrank.ranker import load_weights, rank_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-compute LLM reasoning for top-100 shortlist")
    parser.add_argument("--candidates", default=str(ROOT / "data" / "candidates.jsonl"))
    parser.add_argument("--jd", default=str(ROOT / "data" / "job_description.txt"))
    parser.add_argument("--weights", default=str(ROOT / "config" / "weights.yaml"))
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--mode", choices=("auto", "learned", "heuristic"), default="auto")
    parser.add_argument("--ranker-model", default=str(ROOT / "models" / "fitrank_lgb.txt"))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not is_llama_available(model_path):
        print(f"Local LLM unavailable. Install requirements-llm.txt and place GGUF at {model_path}")
        return 1

    start = time.time()
    jd_text = Path(args.jd).read_text(encoding="utf-8")
    role_profile = parse_jd(jd_text)
    jd_hash = jd_fingerprint(jd_text, role_profile)
    weights = load_weights(args.weights)
    candidate_embeddings = load_candidate_embeddings(
        str(ROOT / "outputs" / "candidate_embeddings.npy"),
        str(ROOT / "outputs" / "candidate_ids.json"),
    )
    jd_embedding = encode_jd(jd_text)

    ranked = rank_candidates(
        load_candidates(args.candidates, validate=False),
        role_profile,
        weights=weights,
        top_n=args.top_n,
        jd_text=jd_text,
        jd_embedding=jd_embedding,
        candidate_embeddings=candidate_embeddings,
        ranking_mode=args.mode,
        model_path=args.ranker_model,
    )
    ranked = ranked[: args.top_n]

    cache = ReasoningCache(args.cache)
    engine = LlamaReasoningEngine(model_path)
    stats = ReasoningStats()
    generated = 0

    for candidate, components, _ in ranked:
        before = stats.llm_generated
        resolve_reasoning(
            candidate,
            components,
            mode="llm",
            role_profile=role_profile,
            jd_hash=jd_hash,
            cache=cache,
            engine=engine,
            model_path=model_path,
            ml_tenure=components.ml_tenure_years,
            stats=stats,
            max_new=100,
        )
        if stats.llm_generated > before:
            generated += 1
            print(f"  generated {generated}/{args.top_n}: {candidate.candidate_id}")

    elapsed = round(time.time() - start, 2)
    print(f"Cache: {args.cache} ({len(cache)} entries)")
    print(f"Cache hits: {stats.cache_hits}, new: {stats.llm_generated}, fallbacks: {stats.template_fallbacks}")
    print(f"Completed in {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
