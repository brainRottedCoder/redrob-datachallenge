#!/usr/bin/env python3
"""Pre-compute dense semantic embeddings for all candidates."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fitrank.embedder import precompute_candidate_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-compute candidate semantic embeddings")
    parser.add_argument("--candidates", default=str(ROOT / "data" / "candidates.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-dir", default=str(ROOT / "models" / "all-MiniLM-L6-v2"))
    args = parser.parse_args()

    start = time.time()
    embeddings_path, ids_path, manifest_path = precompute_candidate_embeddings(
        args.candidates,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
    )
    elapsed = round(time.time() - start, 2)
    print(f"Pre-computed embeddings written to {embeddings_path}")
    print(f"Candidate ids written to {ids_path}")
    print(f"Manifest written to {manifest_path}")
    print(f"Completed in {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
