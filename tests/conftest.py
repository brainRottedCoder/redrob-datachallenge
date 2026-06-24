from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def sample_path(project_root: Path) -> Path:
    return project_root / "data" / "sample_candidates.json"


@pytest.fixture
def candidates_path(project_root: Path) -> Path:
    return project_root / "data" / "candidates.jsonl"


@pytest.fixture
def schema_path(project_root: Path) -> Path:
    return project_root / "data" / "candidate_schema.json"
