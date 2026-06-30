"""Tests for dense semantic embedding module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fitrank.embedder import (
    EMBEDDING_DIM,
    batch_encode_candidates,
    cosine_similarity,
    encode_jd,
    load_candidate_embeddings,
    precompute_candidate_embeddings,
    reset_model_cache,
    semantic_similarity,
)


def test_cosine_similarity_identical_vectors():
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_semantic_similarity_maps_to_unit_interval():
    vec_a = np.array([1.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0], dtype=np.float32)
    assert semantic_similarity(vec_a, vec_b) == pytest.approx(0.5)


def test_similar_texts_score_higher_than_unrelated(jd_text):
    jd_embedding = encode_jd(jd_text)
    related = encode_jd(jd_text)
    unrelated = encode_jd("Corporate marketing manager focused on brand campaigns and social media ads.")
    assert semantic_similarity(jd_embedding, related) > semantic_similarity(jd_embedding, unrelated)


def test_precompute_and_load_round_trip(tmp_path, sample_path):
    output_dir = tmp_path / "outputs"
    embeddings_path, ids_path, manifest_path = precompute_candidate_embeddings(
        sample_path,
        output_dir=output_dir,
        batch_size=8,
    )

    assert embeddings_path.exists()
    assert ids_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dim"] == EMBEDDING_DIM
    assert manifest["count"] > 0

    loaded = load_candidate_embeddings(embeddings_path, ids_path)
    assert loaded is not None
    first_id = json.loads(ids_path.read_text(encoding="utf-8"))[0]
    assert first_id in loaded
    assert loaded[first_id].shape == (EMBEDDING_DIM,)


def test_batch_encode_candidates_from_loader(sample_path):
    from fitrank.loader import load_sample

    candidates = load_sample(sample_path)
    embeddings = batch_encode_candidates(candidates, batch_size=8)
    assert len(embeddings) == len(candidates)
    for vector in embeddings.values():
        assert vector.shape == (EMBEDDING_DIM,)
        assert np.linalg.norm(vector) == pytest.approx(1.0, rel=1e-4)


def test_load_missing_cache_returns_none(tmp_path):
    assert load_candidate_embeddings(
        tmp_path / "missing.npz",
        tmp_path / "missing.json",
    ) is None


def test_reset_model_cache_does_not_break_encoding(jd_text):
    from fitrank import embedder

    first = encode_jd(jd_text)
    reset_model_cache()
    second = encode_jd(jd_text)
    assert first.shape == second.shape == (EMBEDDING_DIM,)
