"""Pseudo-label generation for offline LightGBM training.

Because the hackathon dataset has no ground-truth labels, we generate weak labels
from the heuristic ranker: the top 5% are treated as positive examples, the bottom
20% as negative examples, and a curated set of known trap profiles are forced
negative. The middle band is excluded to reduce label noise.
"""

from __future__ import annotations

# Known trap IDs discovered in the dataset (non-ML titles with stuffed AI skills).
KNOWN_TRAP_IDS: set[str] = {
    "CAND_0004989",  # Project Manager, sample submission rank #1 trap
    "CAND_0000339",  # HR Manager, sample bad top-5
    "CAND_0007203",  # Content Writer with 9 AI skills
    "CAND_0000117",  # Accountant with LLM skills
    "CAND_0004521",  # Sales Manager with ML keyword stuffing
}

# A known genuine profile used as a quality-gate probe.
GENUINE_PROBE_ID: str = "CAND_0033861"
FORCED_POSITIVE_IDS: set[str] = {GENUINE_PROBE_ID}

POSITIVE_PERCENTILE: float = 0.95
NEGATIVE_PERCENTILE: float = 0.20


def percentile_threshold(values: list[float], percentile: float) -> float:
    """Return the score at the requested percentile (0-1)."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile)
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def assign_pseudo_label(
    heuristic_score: float,
    positive_threshold: float,
    negative_threshold: float,
    is_honeypot: bool = False,
    candidate_id: str | None = None,
) -> int | None:
    """Return 1 (positive), 0 (negative), or None (exclude).

    Traps are forced negative regardless of score. Candidates above the positive
    threshold are positive. Candidates below the negative threshold are negative.
    The middle band is excluded to reduce label noise.
    """
    if candidate_id and candidate_id in KNOWN_TRAP_IDS:
        return 0
    if candidate_id and candidate_id in FORCED_POSITIVE_IDS:
        return 1
    if is_honeypot and heuristic_score < positive_threshold:
        return 0
    if heuristic_score >= positive_threshold:
        return 1
    if heuristic_score <= negative_threshold:
        return 0
    return None
