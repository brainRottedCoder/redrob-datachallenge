"""Score calibration utilities for FitRank.

Ranking models (heuristic or learned) can produce top-100 scores that are tightly
clustered. Calibration spreads them to a wider, more interpretable range while
preserving the exact ordering required by the submission format.
"""

from __future__ import annotations

from fitrank.models import Candidate, CandidateScore
from fitrank.component_scores import ComponentScores


def _enforce_non_increasing(scores: list[float]) -> list[float]:
    """Clamp scores so each value is <= the previous (descending order)."""
    if not scores:
        return scores
    result = [scores[0]]
    for score in scores[1:]:
        result.append(min(score, result[-1]))
    return result


def _apply_rescaled_scores(
    ranked: list[tuple[Candidate, ComponentScores, CandidateScore]],
    new_scores: list[float],
) -> list[tuple[Candidate, ComponentScores, CandidateScore]]:
    result: list[tuple[Candidate, ComponentScores, CandidateScore]] = []
    for (candidate, components, score_obj), new_score in zip(ranked, new_scores, strict=True):
        clamped = round(max(0.0, min(1.0, new_score)), 4)
        score_obj.final_score = clamped
        components.final_score = clamped
        result.append((candidate, components, score_obj))
    return result


def calibrate_scores(
    ranked: list[tuple[Candidate, ComponentScores, CandidateScore]],
    low: float | None = None,
    high: float | None = None,
    target_p10: float | None = None,
    target_p90: float | None = None,
) -> list[tuple[Candidate, ComponentScores, CandidateScore]]:
    """Spread top-N scores to a wider range for better discrimination.

    Two modes are supported:

    * Min-max: pass ``low`` and ``high`` to linearly map the minimum score to
      ``low`` and the maximum score to ``high``.
    * Percentile: pass ``target_p10`` and ``target_p90`` to map the 10th
      percentile to ``target_p10`` and the 90th percentile to ``target_p90``.

    If percentile targets are provided they take precedence. Preserves ranking order.
    """
    if not ranked:
        return ranked

    scores = [item[2].final_score for item in ranked]

    if target_p10 is not None and target_p90 is not None:
        sorted_scores = sorted(scores)
        p10_index = max(0, int(0.1 * (len(sorted_scores) - 1)))
        p90_index = int(0.9 * (len(sorted_scores) - 1))
        src_low = sorted_scores[p10_index]
        src_high = sorted_scores[p90_index]
        dst_low = target_p10
        dst_high = target_p90
    else:
        src_low = min(scores)
        src_high = max(scores)
        dst_low = low if low is not None else 0.05
        dst_high = high if high is not None else 0.95

    span = src_high - src_low
    if span == 0:
        calibrated = [dst_low + (dst_high - dst_low) / 2.0] * len(scores)
    else:
        calibrated = [dst_low + (dst_high - dst_low) * (s - src_low) / span for s in scores]

    result: list[tuple[Candidate, ComponentScores, CandidateScore]] = []
    for (candidate, components, score_obj), new_score in zip(ranked, calibrated, strict=True):
        score_obj.final_score = round(max(0.0, min(1.0, new_score)), 4)
        components.final_score = score_obj.final_score
        result.append((candidate, components, score_obj))
    return result


def rescale_submission_scores(
    ranked: list[tuple[Candidate, ComponentScores, CandidateScore]],
    top_min: float = 0.55,
    top_max: float = 0.95,
) -> list[tuple[Candidate, ComponentScores, CandidateScore]]:
    """Rescale the top-100 shortlist to a recruiter-friendly 0.55-0.95 band.

    The minimum score maps to ``top_min`` and the maximum score maps to ``top_max``.
    Flat scores map to ``top_max`` so every candidate still appears attractive.
    Preserves ranking order.
    """
    if not ranked:
        return ranked

    scores = [item[2].final_score for item in ranked]
    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score

    if span == 0:
        rescaled = [top_max] * len(scores)
    else:
        rescaled = [top_min + (top_max - top_min) * (s - min_score) / span for s in scores]

    rescaled = _enforce_non_increasing([round(s, 4) for s in rescaled])
    return _apply_rescaled_scores(ranked, rescaled)
