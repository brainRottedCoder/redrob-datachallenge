#!/usr/bin/env python3
"""Train a LightGBM ranker on heuristic features and pseudo-labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fitrank.embedder import batch_encode_candidates, encode_jd
from fitrank.features import FEATURE_NAMES, extract_ranking_features
from fitrank.jd_parser import parse_jd
from fitrank.loader import load_candidates
from fitrank.pseudo_labels import (
    GENUINE_PROBE_ID,
    KNOWN_TRAP_IDS,
    NEGATIVE_PERCENTILE,
    POSITIVE_PERCENTILE,
    assign_pseudo_label,
    percentile_threshold,
)
from fitrank.ranker import load_capability_vectors, load_weights, rank_candidates

DEFAULT_MODEL_PATH = ROOT / "models" / "fitrank_lgb.txt"
DEFAULT_FEATURES_PATH = ROOT / "outputs" / "training_features.csv"
DEFAULT_REPORT_PATH = ROOT / "outputs" / "training_report.json"


def extract_feature_rows(
    candidates_path: Path,
    jd_path: Path,
    features_path: Path,
) -> list[dict]:
    jd_text = jd_path.read_text(encoding="utf-8")
    role_profile = parse_jd(jd_text)
    weights = load_weights()
    candidate_embeddings = load_capability_vectors()
    jd_embedding = encode_jd(jd_text)
    candidates = list(load_candidates(candidates_path, validate=False))
    if candidate_embeddings is None:
        candidate_embeddings = batch_encode_candidates(candidates)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    fieldnames = [
        "candidate_id",
        *FEATURE_NAMES,
        "heuristic_score",
        "is_honeypot",
    ]

    with features_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            features, components, heuristic_score = extract_ranking_features(
                candidate,
                role_profile,
                weights=weights,
                jd_embedding=jd_embedding,
                candidate_embeddings=candidate_embeddings,
            )
            row = {
                "candidate_id": candidate.candidate_id,
                **features.to_dict(),
                "heuristic_score": round(heuristic_score, 6),
                "is_honeypot": int(components.is_honeypot),
            }
            writer.writerow(row)
            rows.append(row)

    return rows


def build_training_set(rows: list[dict]) -> tuple[list[list[float]], list[int], dict]:
    scores = [float(row["heuristic_score"]) for row in rows]
    pos_threshold = percentile_threshold(scores, POSITIVE_PERCENTILE)
    neg_threshold = percentile_threshold(scores, NEGATIVE_PERCENTILE)

    features_matrix: list[list[float]] = []
    labels: list[int] = []
    label_counts = {"positive": 0, "negative": 0, "excluded": 0}

    for row in rows:
        label = assign_pseudo_label(
            float(row["heuristic_score"]),
            pos_threshold,
            neg_threshold,
            is_honeypot=bool(int(row["is_honeypot"])),
            candidate_id=row["candidate_id"],
        )
        if label is None:
            label_counts["excluded"] += 1
            continue
        features_matrix.append([float(row[name]) for name in FEATURE_NAMES])
        labels.append(label)
        if label == 1:
            label_counts["positive"] += 1
        else:
            label_counts["negative"] += 1

    thresholds = {
        "positive_threshold": pos_threshold,
        "negative_threshold": neg_threshold,
        "positive_percentile": POSITIVE_PERCENTILE,
        "negative_percentile": NEGATIVE_PERCENTILE,
    }
    return features_matrix, labels, {**label_counts, **thresholds}


def train_lightgbm(
    features_matrix: list[list[float]],
    labels: list[int],
    model_path: Path,
) -> dict:
    import lightgbm as lgb
    import numpy as np

    x = np.array(features_matrix, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    rng = np.random.default_rng(42)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx = indices[:split]
    val_idx = indices[split:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    train_set = lgb.Dataset(x_train, label=y_train, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(x_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "min_data_in_leaf": 100,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "verbose": -1,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    temp_path = model_path.with_suffix(".tmp.txt")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(temp_path))

    val_probs = booster.predict(x_val)
    val_preds = (val_probs >= 0.5).astype(int)
    accuracy = float((val_preds == y_val).mean())
    best_auc = float(booster.best_score.get("valid_0", {}).get("auc", 0.0))

    importance = dict(zip(FEATURE_NAMES, booster.feature_importance().tolist(), strict=True))

    return {
        "best_iteration": booster.best_iteration,
        "validation_auc": round(best_auc, 4),
        "validation_accuracy": round(accuracy, 4),
        "feature_importance": {k: int(v) for k, v in sorted(importance.items(), key=lambda kv: -kv[1])},
        "temp_model_path": str(temp_path),
    }


def finalize_model(temp_path: Path, model_path: Path) -> None:
    if model_path.exists():
        model_path.unlink()
    temp_path.replace(model_path)


def discard_model(temp_path: Path) -> None:
    if temp_path.exists():
        temp_path.unlink()


def run_quality_gates(
    candidates_path: Path,
    jd_path: Path,
    model_path: Path,
) -> dict:
    jd_text = jd_path.read_text(encoding="utf-8")
    role_profile = parse_jd(jd_text)
    candidate_embeddings = load_capability_vectors()
    jd_embedding = encode_jd(jd_text)

    ranked = rank_candidates(
        load_candidates(candidates_path, validate=False),
        role_profile,
        jd_text=jd_text,
        jd_embedding=jd_embedding,
        candidate_embeddings=candidate_embeddings,
        ranking_mode="learned",
        model_path=model_path,
        top_n=100,
    )

    top_ids = {item[0].candidate_id for item in ranked}
    traps_present = sorted(KNOWN_TRAP_IDS & top_ids)
    probe_present = GENUINE_PROBE_ID in top_ids

    return {
        "top_100_count": len(ranked),
        "traps_in_top_100": traps_present,
        "probe_in_top_100": probe_present,
        "probe_rank": next(
            (idx + 1 for idx, item in enumerate(ranked) if item[0].candidate_id == GENUINE_PROBE_ID),
            None,
        ),
        "top_5_ids": [item[0].candidate_id for item in ranked[:5]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FitRank LightGBM model")
    parser.add_argument("--candidates", default=str(ROOT / "data" / "candidates.jsonl"))
    parser.add_argument("--jd", default=str(ROOT / "data" / "job_description.txt"))
    parser.add_argument("--features-out", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--skip-quality-gate", action="store_true")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing training features CSV instead of re-extracting.",
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    jd_path = Path(args.jd)
    features_path = Path(args.features_out)
    model_path = Path(args.model_out)
    report_path = Path(args.report_out)

    print("Extracting features...")
    if args.skip_extract and features_path.exists():
        with features_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        print(f"Loaded {len(rows)} feature rows from {features_path}")
    else:
        rows = extract_feature_rows(candidates_path, jd_path, features_path)
        print(f"Wrote {len(rows)} feature rows to {features_path}")

    print("Building pseudo-labels...")
    features_matrix, labels, label_info = build_training_set(rows)
    print(
        f"Training set: {label_info['positive']} positive, "
        f"{label_info['negative']} negative, {label_info['excluded']} excluded"
    )

    if label_info["positive"] < 10 or label_info["negative"] < 10:
        print("ERROR: insufficient labeled samples for training", file=sys.stderr)
        return 1

    print("Training LightGBM...")
    train_metrics = train_lightgbm(features_matrix, labels, model_path)
    temp_model = Path(train_metrics.pop("temp_model_path"))
    print(f"Model trained to temp path {temp_model}")
    print(f"Validation AUC: {train_metrics['validation_auc']}")

    report = {
        "label_counts": label_info,
        "train_metrics": train_metrics,
    }

    if not args.skip_quality_gate:
        print("Running quality gates...")
        gates = run_quality_gates(candidates_path, jd_path, temp_model)
        report["quality_gates"] = gates
        print(json.dumps(gates, indent=2))

        if gates["traps_in_top_100"]:
            discard_model(temp_model)
            print("ERROR: quality gate failed — traps in top 100", file=sys.stderr)
            return 1
        if not gates["probe_in_top_100"]:
            discard_model(temp_model)
            print("ERROR: quality gate failed — genuine probe missing from top 100", file=sys.stderr)
            return 1
        finalize_model(temp_model, model_path)
        print(f"Model saved to {model_path}")
    else:
        finalize_model(temp_model, model_path)
        print(f"Model saved to {model_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
