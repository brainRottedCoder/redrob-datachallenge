"""Component score dataclass shared across the ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from fitrank.penalties import PenaltyScore


@dataclass
class ComponentScores:
    jd_fit: float
    career_evidence: float
    coherence: float
    platform_trust: float
    availability: float
    penalties: float
    final_score: float
    is_honeypot: bool
    penalties_detail: PenaltyScore
    ml_tenure_years: float = 0.0
