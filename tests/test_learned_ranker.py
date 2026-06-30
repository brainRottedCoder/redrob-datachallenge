"""Tests for learned LightGBM ranking."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fitrank.features import FEATURE_NAMES, RankingFeatures
from fitrank.jd_parser import parse_jd
from fitrank.learned_ranker import (
    DEFAULT_MODEL_PATH,
    _model_feature_count_matches,
    is_model_available,
    load_model,
    predict_score,
)
from fitrank.loader import load_sample
from fitrank.pseudo_labels import (
    KNOWN_TRAP_IDS,
    assign_pseudo_label,
    percentile_threshold,
)
from fitrank.ranker import rank_candidates, score_candidate

ROOT = Path(__file__).resolve().parents[1]


def _sample_features() -> RankingFeatures:
    return RankingFeatures(
        title_jd_match=0.5,
        capability_match=0.4,
        career_ml_depth=0.3,
        current_role_ml_depth=0.2,
        coherence_score=0.8,
        skill_trust_ratio=0.6,
        platform_trust=0.7,
        availability_score=0.5,
        penalties_total=0.1,
        assessment_jd_overlap=0.4,
        career_momentum=0.5,
        ml_tenure_years=3.0,
    )


def test_feature_names_stable():
    assert len(FEATURE_NAMES) == 12
    features = RankingFeatures(
        title_jd_match=0.5,
        capability_match=0.4,
        career_ml_depth=0.3,
        current_role_ml_depth=0.2,
        coherence_score=0.8,
        skill_trust_ratio=0.6,
        platform_trust=0.7,
        availability_score=0.5,
        penalties_total=0.1,
        assessment_jd_overlap=0.4,
        career_momentum=0.5,
        ml_tenure_years=3.0,
    )
    assert len(features.to_vector()) == 12


def test_pseudo_label_thresholds():
    scores = [float(i) for i in range(100)]
    pos = percentile_threshold(scores, 0.95)
    neg = percentile_threshold(scores, 0.20)
    assert assign_pseudo_label(95.0, pos, neg, is_honeypot=False, candidate_id="CAND_0000001") == 1
    assert assign_pseudo_label(neg, pos, neg, is_honeypot=False, candidate_id="CAND_0000002") == 0
    assert assign_pseudo_label(10.0, pos, neg, is_honeypot=False, candidate_id="CAND_0000003") == 0
    assert assign_pseudo_label(50.0, pos, neg, is_honeypot=False, candidate_id="CAND_0000004") is None


def test_honeypots_forced_negative():
    assert assign_pseudo_label(0.99, 0.5, 0.2, is_honeypot=False, candidate_id="CAND_0004989") == 0
    assert assign_pseudo_label(0.3, 0.5, 0.2, is_honeypot=True, candidate_id="CAND_0000999") == 0


def test_heuristic_fallback_no_model(tmp_path, monkeypatch, jd_text):
    missing = tmp_path / "missing.txt"
    monkeypatch.setattr("fitrank.learned_ranker.DEFAULT_MODEL_PATH", missing)

    role = parse_jd(jd_text)
    sample = load_sample()
    auto_ranked = rank_candidates(sample, role, top_n=5, ranking_mode="auto", model_path=missing, jd_text=jd_text)
    heuristic_ranked = rank_candidates(sample, role, top_n=5, ranking_mode="heuristic", jd_text=jd_text)
    auto_scores = [item[2].final_score for item in auto_ranked]
    heuristic_scores = [item[2].final_score for item in heuristic_ranked]
    assert auto_scores == heuristic_scores


def test_learned_mode_requires_model(tmp_path, jd_text):
    role = parse_jd(jd_text)
    sample = load_sample()
    missing = tmp_path / "no_model.txt"
    with pytest.raises(FileNotFoundError):
        rank_candidates(sample, role, top_n=5, ranking_mode="learned", model_path=missing, jd_text=jd_text)


@pytest.mark.skipif(not is_model_available(), reason="trained model not present")
def test_learned_mode_deterministic(jd_text):
    model = load_model()
    if model is None or not _model_feature_count_matches(model, _sample_features()):
        pytest.skip("trained model feature count mismatch; retrain with make train")

    role = parse_jd(jd_text)
    sample = load_sample()
    ranked_a = rank_candidates(sample, role, top_n=20, ranking_mode="learned", jd_text=jd_text)
    ranked_b = rank_candidates(sample, role, top_n=20, ranking_mode="learned", jd_text=jd_text)
    assert [item[0].candidate_id for item in ranked_a] == [item[0].candidate_id for item in ranked_b]


def test_train_and_rank_smoke(tmp_path):
    pytest.importorskip("lightgbm")
    import importlib.util

    train_path = ROOT / "scripts" / "train_learned_ranker.py"
    spec = importlib.util.spec_from_file_location("train_learned_ranker", train_path)
    train_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(train_mod)

    candidates = ROOT / "data" / "sample_candidates.json"
    jd = ROOT / "data" / "job_description.txt"
    features_path = tmp_path / "features.csv"
    model_path = tmp_path / "model.txt"

    rows = train_mod.extract_feature_rows(candidates, jd, features_path)
    matrix, labels, _ = train_mod.build_training_set(rows)
    assert len(matrix) >= 5
    metrics = train_mod.train_lightgbm(matrix, labels, model_path)
    temp_path = Path(metrics["temp_model_path"])
    train_mod.finalize_model(temp_path, model_path)
    assert model_path.exists()
    assert metrics["validation_auc"] >= 0.5

    role = parse_jd(jd.read_text(encoding="utf-8"))
    ranked = rank_candidates(
        load_sample(candidates),
        role,
        top_n=10,
        ranking_mode="learned",
        model_path=model_path,
        jd_text=jd.read_text(encoding="utf-8"),
    )
    assert len(ranked) == 10


@pytest.mark.skipif(not is_model_available(), reason="trained model not present")
def test_quality_gates_on_full_model(jd_text):
    from fitrank.pseudo_labels import GENUINE_PROBE_ID

    model = load_model()
    if model is None or not _model_feature_count_matches(model, _sample_features()):
        pytest.skip("trained model feature count mismatch; retrain with make train")

    role = parse_jd(jd_text)
    ranked = rank_candidates(
        load_sample(),
        role,
        top_n=100,
        ranking_mode="learned",
        jd_text=jd_text,
    )
    top_ids = {item[0].candidate_id for item in ranked}
    assert KNOWN_TRAP_IDS.isdisjoint(top_ids)
    assert GENUINE_PROBE_ID in top_ids or len(load_sample()) < 100


@pytest.mark.skipif(not is_model_available(), reason="trained model not present")
def test_model_predict_clamped():
    model = load_model()
    assert model is not None
    features = RankingFeatures(
        title_jd_match=0.9,
        capability_match=0.8,
        career_ml_depth=0.7,
        current_role_ml_depth=0.6,
        coherence_score=0.85,
        skill_trust_ratio=0.7,
        platform_trust=0.75,
        availability_score=0.6,
        penalties_total=0.05,
        assessment_jd_overlap=0.5,
        career_momentum=0.4,
        ml_tenure_years=5.0,
    )
    if not _model_feature_count_matches(model, features):
        pytest.skip("trained model feature count mismatch; retrain with make train")
    score = predict_score(model, features, heuristic_score=0.5)
    assert 0.0 <= score <= 1.0
