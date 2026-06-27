#!/usr/bin/env python3
"""Pre-compute lightweight semantic capability vectors for all candidates.

This is an optional pre-computation step. If the vectors file already exists,
the ranker can load it for a small speedup; otherwise the ranker computes the
vectors on-the-fly within the 5-minute CPU budget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fitrank.embedder import precompute_candidate_vectors
from fitrank.jd_parser import parse_jd


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-compute candidate capability vectors")
    parser.add_argument("--candidates", default=str(ROOT / "data" / "candidates.jsonl"))
    parser.add_argument("--jd", default=str(ROOT / "data" / "job_description.txt"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "candidate_vectors.jsonl"))
    args = parser.parse_args()

    jd_text = Path(args.jd).read_text(encoding="utf-8")
    role_profile = parse_jd(jd_text)

    output = precompute_candidate_vectors(args.candidates, args.out, role_profile)
    print(f"Pre-computed vectors written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
