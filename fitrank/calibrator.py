"""Multi-JD score calibration.

Linearly rescales raw scores so the 10th–90th percentile spans a fixed band
(default 0.40–0.90). This does NOT change the relative ordering, only the
absolute score values. It is intended for internal multi-JD analysis, NOT for
single-JD challenge submissions where raw scores are compared across teams.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitrank.ranker import ComponentScores


def _percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of a sorted list using linear interpolation."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    # Use nearest-rank variant: index = (p/100) * (n - 1)
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def calibrate_scores(
    ranked: list[tuple],
    target_p10: float = 0.40,
    target_p90: float = 0.90,
) -> list[tuple]:
    """Rescale final scores of a ranked list without changing ordering.

    Args:
        ranked: List of (Candidate, ComponentScores, CandidateScore) tuples as
            returned by rank_candidates.
        target_p10: Target score for the 10th percentile candidate.
        target_p90: Target score for the 90th percentile candidate.

    Returns:
        A new list with the same ordering but updated final_score values.
    """
    if not ranked:
        return ranked

    scores = [item[2].final_score for item in ranked]
    p10 = _percentile(scores, 10.0)
    p90 = _percentile(scores, 90.0)

    if p90 <= p10:
        # No spread to calibrate; return a shallow copy with clamped scores.
        return [
            (candidate, components, score_obj.__class__(**{**score_obj.__dict__, "final_score": max(0.0, min(1.0, score_obj.final_score))}))
            for candidate, components, score_obj in ranked
        ]

    target_span = target_p90 - target_p10
    raw_span = p90 - p10
    scale = target_span / raw_span

    calibrated: list[tuple] = []
    for candidate, components, score_obj in ranked:
        raw = score_obj.final_score
        new_score = target_p10 + (raw - p10) * scale
        new_score = max(0.0, min(1.0, new_score))
        # Mutate a copy of the score object so callers can still read components.
        new_score_obj = score_obj.__class__(**{**score_obj.__dict__, "final_score": round(new_score, 4)})
        new_components = components.__class__(**{**components.__dict__, "final_score": round(new_score, 4)})
        calibrated.append((candidate, new_components, new_score_obj))

    return calibrated
