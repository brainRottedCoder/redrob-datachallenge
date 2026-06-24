"""Phase 1 scaffolding tests (PRD T1.1-T1.7)."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_t1_1_package_import():
    import fitrank  # noqa: F401


def test_t1_2_config_load():
    weights = yaml.safe_load((ROOT / "config" / "weights.yaml").read_text(encoding="utf-8"))
    for key in ("jd_fit", "career_evidence", "coherence", "platform_trust", "availability"):
        assert key in weights


def test_t1_4_weights_sum():
    weights = yaml.safe_load((ROOT / "config" / "weights.yaml").read_text(encoding="utf-8"))
    total = sum(weights[k] for k in ("jd_fit", "career_evidence", "coherence", "platform_trust", "availability"))
    assert abs(total - 1.0) < 0.001


def test_t1_5_directory_structure():
    for name in ("fitrank", "tests", "app", "scripts", "config", "outputs", "data", "docs", "deck"):
        assert (ROOT / name).exists()


def test_t1_7_metadata_stub():
    data = yaml.safe_load((ROOT / "submission_metadata.yaml").read_text(encoding="utf-8"))
    assert data["team_name"]
