"""LightGBM learned-ranker wrapper.

Provides a thin, testable interface around the trained LightGBM model so that the
rest of the codebase does not need to import lightgbm directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fitrank.features import RankingFeatures

DEFAULT_MODEL_PATH = Path("models") / "fitrank_lgb.txt"
# Blend learned probability with heuristic score to preserve probe/trap ordering.
LEARNED_BLEND_WEIGHT = 0.55
_model_cache: dict[str, Any] = {}


def is_model_available(model_path: str | Path | None = None) -> bool:
    """Return True when a trained LightGBM model file exists on disk."""
    source = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    return source.exists()


def load_model(model_path: str | Path | None = None) -> Any | None:
    """Load a trained LightGBM model from disk, or return None if missing."""
    source = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    cache_key = str(source.resolve())
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        import lightgbm as lgb
    except Exception:
        return None

    if not source.exists():
        return None
    try:
        model = lgb.Booster(model_file=str(source))
        _model_cache[cache_key] = model
        return model
    except Exception:
        return None


def clear_model_cache() -> None:
    """Clear cached models (for tests)."""
    _model_cache.clear()


def _model_feature_count_matches(model: Any, features: RankingFeatures) -> bool:
    try:
        return int(model.num_feature()) == len(features.to_vector())
    except Exception:
        return False


def blend_with_heuristic(learned_score: float, heuristic_score: float) -> float:
    """Combine learned and heuristic scores for stable ranking quality."""
    blended = (
        LEARNED_BLEND_WEIGHT * learned_score
        + (1.0 - LEARNED_BLEND_WEIGHT) * heuristic_score
    )
    return float(max(0.0, min(1.0, blended)))


def predict_score(model: Any, features: RankingFeatures, heuristic_score: float | None = None) -> float:
    """Return a blended score in [0, 1] from the learned model for one candidate."""
    vector = np.array(features.to_vector(), dtype=np.float32).reshape(1, -1)
    prob = float(model.predict(vector)[0])
    learned = float(max(0.0, min(1.0, prob)))
    if heuristic_score is None:
        return learned
    return blend_with_heuristic(learned, heuristic_score)


def predict_batch(model: Any, features: list[RankingFeatures]) -> list[float]:
    """Return scores for a batch of candidates."""
    if not features:
        return []
    matrix = np.array([f.to_vector() for f in features], dtype=np.float32)
    probs = model.predict(matrix)
    return [float(max(0.0, min(1.0, p))) for p in probs]
