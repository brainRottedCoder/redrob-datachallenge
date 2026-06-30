"""Synthetic ground-truth labels derived from dataset signals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from fitrank.career_analyzer import analyze_career
from fitrank.coherence import _is_ml_skill
from fitrank.loader import load_candidates
from fitrank.models import Candidate
from fitrank.title_gate import TitleDomain, classify_title

LABEL_RELEVANCE = {
    "relevant_2": 2.0,
    "relevant_1": 1.0,
    "ai_adjacent": 0.0,
    "honeypot": 0.0,
    "irrelevant": 0.0,
}

GENUINE_LABELS = frozenset({"relevant_2", "relevant_1"})

HIGH_DEPTH_THRESHOLD = 4
MODERATE_DEPTH_THRESHOLD = 2
HONEYPOT_SKILL_THRESHOLD = 7


@dataclass(frozen=True)
class GroundTruth:
    candidate_id: str
    label: str
    tier: str
    is_honeypot: bool
    relevance: float
    title_domain: str
    career_ml_depth: int
    ml_skill_count: int


def label_candidate(candidate: Candidate) -> GroundTruth:
    """Assign a synthetic relevance label from profile signals only.

    Heuristics:
      - ML_AI title + high career depth -> relevant_2
      - ML_AI title + moderate career depth -> relevant_1
      - AI_ADJACENT title -> ai_adjacent
      - NON_TECH title with many ML skills -> honeypot
      - otherwise -> irrelevant
    """
    title_domain = classify_title(candidate.profile.current_title)
    career = analyze_career(candidate)
    career_ml_depth = career.all_career_ml_depth
    ml_skill_count = sum(1 for skill in candidate.skills if _is_ml_skill(skill.name))

    if title_domain == TitleDomain.NON_TECH and ml_skill_count >= HONEYPOT_SKILL_THRESHOLD:
        label = "honeypot"
    elif title_domain == TitleDomain.ML_AI and career_ml_depth >= HIGH_DEPTH_THRESHOLD:
        label = "relevant_2"
    elif title_domain == TitleDomain.ML_AI and career_ml_depth >= MODERATE_DEPTH_THRESHOLD:
        label = "relevant_1"
    elif title_domain == TitleDomain.AI_ADJACENT:
        label = "ai_adjacent"
    else:
        label = "irrelevant"

    return GroundTruth(
        candidate_id=candidate.candidate_id,
        label=label,
        tier=label,
        is_honeypot=label == "honeypot",
        relevance=LABEL_RELEVANCE[label],
        title_domain=title_domain.value,
        career_ml_depth=career_ml_depth,
        ml_skill_count=ml_skill_count,
    )


def build_ground_truth(
    candidates_path: str | Path,
    jd_path: str | Path,
    output_path: str | Path,
) -> dict[str, GroundTruth]:
    """Load candidates, label each one, and save the ground truth to disk."""
    candidates = load_candidates(candidates_path, validate=False)
    ground_truth = {candidate.candidate_id: label_candidate(candidate) for candidate in candidates}
    save_ground_truth(ground_truth, output_path)
    return ground_truth


def build_ground_truth_from_candidates(candidates: Iterator[Candidate]) -> dict[str, GroundTruth]:
    """Build a ground-truth dictionary from an iterable of candidates."""
    return {candidate.candidate_id: label_candidate(candidate) for candidate in candidates}


def save_ground_truth(ground_truth: dict[str, GroundTruth], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in ground_truth.values():
            handle.write(json.dumps(asdict(entry)) + "\n")
    return output


def load_ground_truth(path: str | Path) -> dict[str, GroundTruth]:
    ground_truth: dict[str, GroundTruth] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            entry = GroundTruth(**data)
            ground_truth[entry.candidate_id] = entry
    return ground_truth


def ground_truth_summary(ground_truth: dict[str, GroundTruth]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in ground_truth.values():
        counts[entry.label] = counts.get(entry.label, 0) + 1
    return counts
