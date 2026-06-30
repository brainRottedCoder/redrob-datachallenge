import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests that use the full 100K candidates fixture as slow."""
    for item in items:
        if "candidates_path" in getattr(item, "fixturenames", []):
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def jd_text(project_root: Path) -> str:
    return (project_root / "data" / "job_description.txt").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    """Use deterministic fake embeddings in tests without downloading the model.

    The fake model produces embeddings that correlate with keyword overlap so that
    semantic-similarity tests behave predictably.
    """
    from fitrank import embedder

    class FakeModel:
        def encode(self, texts, **kwargs):
            vectors = []
            for text in texts:
                text_lower = text.lower()
                vec = np.zeros(embedder.EMBEDDING_DIM, dtype=np.float32)
                # Seed a base direction from the text hash for stable noise.
                seed = abs(hash(text)) % (2**32)
                rng = np.random.default_rng(seed)
                # Each token contributes to a deterministic dimension.
                tokens = set(re.findall(r"[a-z0-9]+", text_lower))
                for token in tokens:
                    dim = abs(hash(token)) % embedder.EMBEDDING_DIM
                    vec[dim] += 1.0
                # Add small random noise so zero-token texts are not identical.
                vec += rng.standard_normal(embedder.EMBEDDING_DIM).astype(np.float32) * 0.05
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                vectors.append(vec)
            if len(vectors) == 1:
                return np.stack(vectors)
            return np.stack(vectors)

        def save(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    def fake_get_model(model_dir=None):
        return FakeModel()

    monkeypatch.setattr(embedder, "_get_model", fake_get_model)
    embedder.reset_model_cache()
    yield
    embedder.reset_model_cache()


@pytest.fixture
def sample_path(project_root: Path) -> Path:
    return project_root / "data" / "sample_candidates.json"


@pytest.fixture
def candidates_path(project_root: Path) -> Path:
    return project_root / "data" / "candidates.jsonl"


@pytest.fixture
def schema_path(project_root: Path) -> Path:
    return project_root / "data" / "candidate_schema.json"
