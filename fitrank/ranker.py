"""Score aggregation, learned-model scoring, and ranking."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import heapq
import numpy as np

from fitrank.component_scores import ComponentScores
from fitrank.config_loader import load_weights as _load_weights
from fitrank.embedder import (
    batch_encode_candidates,
    encode_jd,
    load_candidate_embeddings,
)
from fitrank.features import extract_ranking_features
from fitrank.learned_ranker import DEFAULT_MODEL_PATH, load_model, predict_score, _model_feature_count_matches
from fitrank.models import Candidate, CandidateScore, RoleProfile
from fitrank.signals import load_normalization_constants

__all__ = [
    "ComponentScores",
    "load_weights",
    "score_candidate",
    "rank_candidates",
    "load_capability_vectors",
    "load_candidate_embeddings",
    "calibrate_scores",
]


def load_weights(path: str | Path = "config/weights.yaml") -> dict:
    return _load_weights(path)


def _resolve_final_score(
    features,
    components: ComponentScores,
    heuristic_score: float,
    ranking_mode: str,
    model_path: str | Path | None,
) -> float:
    if ranking_mode == "heuristic":
        return heuristic_score

    model = load_model(model_path)
    if ranking_mode == "learned" and model is None:
        raise FileNotFoundError(
            "Learned ranking mode requires models/fitrank_lgb.txt. "
            "Run: python scripts/train_learned_ranker.py"
        )
    if ranking_mode in ("learned", "auto") and model is not None:
        if _model_feature_count_matches(model, features):
            return predict_score(model, features, heuristic_score)
        if ranking_mode == "learned":
            raise ValueError(
                "Learned model feature count does not match current feature vector. "
                "Retrain with: python scripts/train_learned_ranker.py"
            )
    return heuristic_score


def _prepare_embedding_context(
    candidates,
    jd_text: str | None,
    jd_embedding: np.ndarray | None,
    candidate_embeddings: dict[str, np.ndarray] | None,
) -> tuple[Iterable[Candidate], np.ndarray | None, dict[str, np.ndarray] | None]:
    resolved_jd_embedding = jd_embedding
    if resolved_jd_embedding is None and jd_text:
        resolved_jd_embedding = encode_jd(jd_text)

    resolved_candidate_embeddings = candidate_embeddings
    candidate_list: list[Candidate] | Iterable[Candidate] = candidates
    if resolved_candidate_embeddings is None and resolved_jd_embedding is not None and jd_text:
        candidate_list = list(candidates)
        resolved_candidate_embeddings = batch_encode_candidates(candidate_list)

    return candidate_list, resolved_jd_embedding, resolved_candidate_embeddings


def score_candidate(
    candidate: Candidate,
    role_profile: RoleProfile,
    weights: dict | None = None,
    norms: dict[str, float] | None = None,
    jd_embedding: np.ndarray | None = None,
    candidate_embeddings: dict[str, np.ndarray] | None = None,
    ranking_mode: str = "auto",
    model_path: str | Path | None = None,
) -> tuple[ComponentScores, CandidateScore]:
    weights = weights or load_weights()
    norms = norms or load_normalization_constants()

    features, components, heuristic_score = extract_ranking_features(
        candidate,
        role_profile,
        weights=weights,
        norms=norms,
        jd_embedding=jd_embedding,
        candidate_embeddings=candidate_embeddings,
    )
    final_score = _resolve_final_score(
        features, components, heuristic_score, ranking_mode, model_path
    )

    components = ComponentScores(
        jd_fit=components.jd_fit,
        career_evidence=components.career_evidence,
        coherence=components.coherence,
        platform_trust=components.platform_trust,
        availability=components.availability,
        penalties=components.penalties,
        final_score=round(final_score, 4),
        is_honeypot=components.is_honeypot,
        penalties_detail=components.penalties_detail,
        ml_tenure_years=components.ml_tenure_years,
    )
    candidate_score = CandidateScore(
        candidate_id=candidate.candidate_id,
        jd_fit=components.jd_fit,
        career_evidence=components.career_evidence,
        coherence=components.coherence,
        platform_trust=components.platform_trust,
        availability=components.availability,
        penalties=components.penalties,
        final_score=components.final_score,
    )
    return components, candidate_score


def rank_candidates(
    candidates,
    role_profile: RoleProfile,
    weights: dict | None = None,
    norms: dict[str, float] | None = None,
    top_n: int = 100,
    jd_text: str | None = None,
    jd_embedding: np.ndarray | None = None,
    candidate_embeddings: dict[str, np.ndarray] | None = None,
    ranking_mode: str = "auto",
    model_path: str | Path | None = None,
) -> list[tuple[Candidate, ComponentScores, CandidateScore]]:
    weights = weights or load_weights()
    norms = norms or load_normalization_constants()
    candidates, jd_embedding, candidate_embeddings = _prepare_embedding_context(
        candidates,
        jd_text,
        jd_embedding,
        candidate_embeddings,
    )

    heap: list[tuple[tuple[float, str], Candidate, ComponentScores, CandidateScore]] = []
    for candidate in candidates:
        components, candidate_score = score_candidate(
            candidate,
            role_profile,
            weights,
            norms,
            jd_embedding=jd_embedding,
            candidate_embeddings=candidate_embeddings,
            ranking_mode=ranking_mode,
            model_path=model_path,
        )
        key = (candidate_score.final_score, candidate.candidate_id)
        if len(heap) < top_n:
            heapq.heappush(heap, (key, candidate, components, candidate_score))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, candidate, components, candidate_score))

    ranked = sorted(heap, key=lambda item: (-item[0][0], item[0][1]))
    return [(item[1], item[2], item[3]) for item in ranked]


DEFAULT_EMBEDDINGS_PATH = Path("outputs/candidate_embeddings.npz")
DEFAULT_IDS_PATH = Path("outputs/candidate_embedding_ids.json")


def load_candidate_embeddings_cache(
    embeddings_path: str | Path | None = None,
    ids_path: str | Path | None = None,
) -> dict[str, np.ndarray] | None:
    """Load pre-computed dense embeddings when cache files exist."""
    return load_candidate_embeddings(
        embeddings_path or DEFAULT_EMBEDDINGS_PATH,
        ids_path or DEFAULT_IDS_PATH,
    )


def load_capability_vectors(path: str | Path | None = None) -> dict[str, np.ndarray] | None:
    """Backward-compatible loader for pre-computed candidate embeddings."""
    if path is not None:
        ids_path = Path(path).with_name("candidate_embedding_ids.json")
        return load_candidate_embeddings(path, ids_path)
    return load_candidate_embeddings_cache()


def calibrate_scores(
    ranked: list[tuple[Candidate, ComponentScores, CandidateScore]],
    low: float = 0.05,
    high: float = 0.95,
) -> list[tuple[Candidate, ComponentScores, CandidateScore]]:
    """Spread top-N scores to a wider range for better discrimination.

    Uses a percentile-based linear mapping so the top candidate is near ``high``
    and the bottom of the shortlist is near ``low``. Preserves ranking order.
    """
    if not ranked:
        return ranked

    scores = [item[2].final_score for item in ranked]
    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score

    if span == 0:
        calibrated = [low + (high - low) / 2.0] * len(scores)
    else:
        calibrated = [low + (high - low) * (s - min_score) / span for s in scores]

    result: list[tuple[Candidate, ComponentScores, CandidateScore]] = []
    for (candidate, components, score_obj), new_score in zip(ranked, calibrated, strict=True):
        score_obj.final_score = round(new_score, 4)
        components.final_score = score_obj.final_score
        result.append((candidate, components, score_obj))
    return result
