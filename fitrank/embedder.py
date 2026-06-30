"""Dense semantic embeddings for JD–candidate capability matching.

Uses sentence-transformers/all-MiniLM-L6-v2 to embed full JD text and aggregated
candidate profile text, then cosine similarity for the capability_match signal.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

from fitrank.models import Candidate, RoleProfile

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_EMBEDDINGS_PATH = DEFAULT_OUTPUT_DIR / "candidate_embeddings.npy"
DEFAULT_IDS_PATH = DEFAULT_OUTPUT_DIR / "candidate_ids.json"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "embedding_manifest.json"
DEFAULT_MODEL_DIR = Path("models/all-MiniLM-L6-v2")

_model: SentenceTransformer | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_model_path(model_dir: str | Path | None = None) -> str:
    local = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not local.is_absolute():
        local = _project_root() / local
    if local.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return str(local)
    return MODEL_NAME


def download_model(model_dir: str | Path | None = None) -> Path:
    """Download the embedding model to a local directory for offline ranking."""
    from sentence_transformers import SentenceTransformer

    target = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not target.is_absolute():
        target = _project_root() / target
    target.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(target))
    return target


def _get_model(model_dir: str | Path | None = None) -> SentenceTransformer:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_resolve_model_path(model_dir))
    return _model


def reset_model_cache() -> None:
    """Clear the lazy-loaded model singleton (for tests)."""
    global _model
    _model = None


def _candidate_text(candidate: Candidate) -> str:
    """Aggregate text fields from the candidate."""
    parts = [
        candidate.profile.headline,
        candidate.profile.summary,
        candidate.profile.current_title,
    ]
    parts.extend(skill.name for skill in candidate.skills)
    for entry in candidate.career_history:
        parts.append(entry.title)
        parts.append(entry.description)
    return " ".join(parts)


def encode_texts(
    texts: list[str],
    batch_size: int = 128,
    model_dir: str | Path | None = None,
) -> np.ndarray:
    """Encode texts to L2-normalized dense vectors."""
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model(model_dir)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_jd(jd_text: str, model_dir: str | Path | None = None) -> np.ndarray:
    """Encode full job description text to a single normalized vector."""
    return encode_texts([jd_text], batch_size=1, model_dir=model_dir)[0]


def encode_candidate(
    candidate: Candidate,
    model_dir: str | Path | None = None,
) -> np.ndarray:
    """Encode a single candidate profile to a normalized vector."""
    return encode_texts([_candidate_text(candidate)], batch_size=1, model_dir=model_dir)[0]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity for L2-normalized vectors."""
    return float(np.dot(vec_a, vec_b))


def semantic_similarity(jd_embedding: np.ndarray, candidate_embedding: np.ndarray) -> float:
    """Map cosine similarity from [-1, 1] to [0, 1]."""
    cosine = cosine_similarity(jd_embedding, candidate_embedding)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def dense_jd_candidate_similarity(
    jd_embedding: np.ndarray | None,
    candidate_embedding: np.ndarray | None,
) -> float:
    """Compatibility alias used by feature extraction."""
    if jd_embedding is None or candidate_embedding is None:
        return 0.0
    return semantic_similarity(jd_embedding, candidate_embedding)


def compute_semantic_capability_match(
    candidate: Candidate,
    jd_embedding: np.ndarray,
    model_dir: str | Path | None = None,
) -> float:
    """Compute semantic overlap between candidate text and JD embedding."""
    if jd_embedding.size == 0:
        return 0.0
    cand_embedding = encode_candidate(candidate, model_dir=model_dir)
    return semantic_similarity(jd_embedding, cand_embedding)


def semantic_match_from_precomputed(
    candidate_id: str,
    jd_embedding: np.ndarray,
    embeddings: dict[str, np.ndarray],
) -> float:
    """Compute semantic match using pre-computed candidate embeddings."""
    if jd_embedding.size == 0:
        return 0.0
    cand_embedding = embeddings.get(candidate_id)
    if cand_embedding is None:
        return 0.0
    return semantic_similarity(jd_embedding, cand_embedding)


def batch_encode_candidates(
    candidates: Iterable[Candidate],
    batch_size: int = 128,
    model_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Encode all candidates in batches and return an id -> vector mapping."""
    candidate_list = list(candidates)
    if not candidate_list:
        return {}

    embeddings: dict[str, np.ndarray] = {}
    batch_texts: list[str] = []
    batch_ids: list[str] = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        vectors = encode_texts(batch_texts, batch_size=batch_size, model_dir=model_dir)
        for index, candidate_id in enumerate(batch_ids):
            embeddings[candidate_id] = vectors[index]
        batch_texts.clear()
        batch_ids.clear()

    for candidate in candidate_list:
        batch_ids.append(candidate.candidate_id)
        batch_texts.append(_candidate_text(candidate))
        if len(batch_texts) >= batch_size:
            flush_batch()
    flush_batch()
    return embeddings


def precompute_candidate_embeddings(
    candidates_path: str | Path,
    output_dir: str | Path | None = None,
    batch_size: int = 128,
    model_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Pre-compute candidate embeddings and save NPZ + id index + manifest."""
    from fitrank.loader import load_candidates

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = out_dir / "candidate_embeddings.npy"
    ids_path = out_dir / "candidate_ids.json"
    manifest_path = out_dir / "embedding_manifest.json"

    candidate_ids: list[str] = []
    vector_batches: list[np.ndarray] = []
    batch_texts: list[str] = []
    batch_ids: list[str] = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        vector_batches.append(
            encode_texts(batch_texts, batch_size=batch_size, model_dir=model_dir)
        )
        candidate_ids.extend(batch_ids)
        batch_texts.clear()
        batch_ids.clear()

    for candidate in load_candidates(candidates_path, validate=False):
        batch_ids.append(candidate.candidate_id)
        batch_texts.append(_candidate_text(candidate))
        if len(batch_texts) >= batch_size:
            flush_batch()
            if len(candidate_ids) % 5000 == 0:
                print(f"Encoded {len(candidate_ids)} candidates...", flush=True)

    flush_batch()
    vectors = np.vstack(vector_batches) if vector_batches else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    np.save(embeddings_path, vectors)

    ids_path.write_text(json.dumps(candidate_ids), encoding="utf-8")
    manifest = {
        "model_name": MODEL_NAME,
        "dim": EMBEDDING_DIM,
        "count": len(candidate_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embeddings_path": str(embeddings_path),
        "ids_path": str(ids_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return embeddings_path, ids_path, manifest_path


def load_candidate_embeddings(
    embeddings_path: str | Path | None = None,
    ids_path: str | Path | None = None,
) -> dict[str, np.ndarray] | None:
    """Load pre-computed candidate embeddings when cache files exist."""
    emb_path = Path(embeddings_path) if embeddings_path else DEFAULT_EMBEDDINGS_PATH
    id_path = Path(ids_path) if ids_path else DEFAULT_IDS_PATH
    if not emb_path.exists() or not id_path.exists():
        return None

    if emb_path.suffix.lower() == ".npz":
        with np.load(emb_path) as data:
            matrix = np.asarray(data["embeddings"], dtype=np.float32)
    else:
        matrix = np.asarray(np.load(emb_path), dtype=np.float32)
    candidate_ids = json.loads(id_path.read_text(encoding="utf-8"))
    if len(candidate_ids) != matrix.shape[0]:
        raise ValueError("Embedding count does not match candidate id list length.")
    return {candidate_id: matrix[index] for index, candidate_id in enumerate(candidate_ids)}


# Backward-compatible aliases.
load_candidate_vectors = load_candidate_embeddings
precompute_candidate_vectors = precompute_candidate_embeddings

# ---------------------------------------------------------------------------
# Sparse capability vector fallback (no model download required)
# ---------------------------------------------------------------------------

CAPABILITY_VOCABULARY: dict[str, list[str]] = {
    "LLM fine-tuning": ["fine-tuning", "fine tuning", "finetuning", "llm fine-tuning", "instruction tuning"],
    "LoRA": ["lora"],
    "QLoRA": ["qlora"],
    "PEFT": ["peft"],
    "RAG": ["rag", "retrieval-augmented", "retrieval augmented", "retrieval augmented generation"],
    "embeddings": ["embedding", "vector representation", "dense vector"],
    "sentence-transformers": ["sentence-transformer", "sentence transformer"],
    "OpenAI embeddings": ["openai embedding", "ada embedding"],
    "BGE": ["bge"],
    "E5": ["e5"],
    "retrieval": ["retrieval", "information retrieval", "ir"],
    "ranking": ["ranking", "rerank", "re-ranking", "learning-to-rank", "ltr"],
    "hybrid retrieval": ["hybrid retrieval", "hybrid search", "sparse-dense"],
    "vector database": ["vector database", "vector db", "vectordb"],
    "Pinecone": ["pinecone"],
    "Weaviate": ["weaviate"],
    "Qdrant": ["qdrant"],
    "Milvus": ["milvus"],
    "FAISS": ["faiss"],
    "Elasticsearch": ["elasticsearch"],
    "OpenSearch": ["opensearch"],
    "LangChain": ["langchain"],
    "transformers": ["transformer", "hugging face", "huggingface"],
    "NLP": ["nlp", "natural language processing"],
    "computer vision": ["computer vision"],
    "XGBoost": ["xgboost", "gradient boosting"],
    "learning-to-rank": ["learning-to-rank", "learning to rank", "ltr"],
    "A/B testing": ["ab test", "a/b test", "ab testing", "online experiment"],
    "NDCG": ["ndcg"],
    "MRR": ["mrr"],
    "Python": ["python"],
    "distributed systems": ["distributed system", "distributed training", "scale"],
    "inference optimization": ["inference optimization", "model serving", "inference serving"],
}


def _sparse_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9/]+", text.lower())


def _sparse_build_ngram_sets(tokens: list[str]) -> tuple[set[str], set[str]]:
    if len(tokens) < 2:
        return set(), set()
    bigrams = {" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)}
    trigrams = (
        {" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)}
        if len(tokens) >= 3
        else set()
    )
    return bigrams, trigrams


def _sparse_candidate_text(candidate: Candidate) -> str:
    parts = [
        candidate.profile.headline,
        candidate.profile.summary,
        candidate.profile.current_title,
    ]
    parts.extend(skill.name for skill in candidate.skills)
    for entry in candidate.career_history:
        parts.append(entry.title)
        parts.append(entry.description)
    return " ".join(parts).lower()


def _sparse_capability_vector(text: str, capabilities: list[str]) -> dict[str, float]:
    vector: dict[str, float] = {}
    text_lower = text.lower()
    tokens = _sparse_tokenize(text_lower)
    token_set = set(tokens)
    bigrams, trigrams = _sparse_build_ngram_sets(tokens)
    ngrams = bigrams | trigrams

    for cap in capabilities:
        if cap not in CAPABILITY_VOCABULARY:
            cap_lower = cap.lower()
            if cap_lower in text_lower or cap_lower.replace(" ", "") in text_lower.replace(" ", ""):
                vector[cap] = 1.0
            continue

        score = 0.0
        for term in CAPABILITY_VOCABULARY[cap]:
            if " " in term:
                if term in ngrams:
                    score += 1.0
            else:
                if term in token_set:
                    score += 1.0
        if score > 0:
            vector[cap] = min(score, 3.0)
    return vector


def _sparse_cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    keys = set(vec_a.keys()) & set(vec_b.keys())
    if not keys:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_sparse_capability_match(
    candidate: Candidate,
    role_profile: RoleProfile,
) -> float:
    """Fast keyword/n-gram overlap between candidate text and JD capabilities.

    This is used as a zero-dependency fallback when dense embeddings are not
    pre-computed or the sentence-transformer model is unavailable.
    """
    if not role_profile.required_capabilities:
        return 0.0
    text = _sparse_candidate_text(candidate)
    cand_vector = _sparse_capability_vector(text, role_profile.required_capabilities)
    jd_vector = {cap: 1.0 for cap in role_profile.required_capabilities}
    return _sparse_cosine_similarity(cand_vector, jd_vector)


def precompute_sparse_candidate_vectors(
    candidates_path: str | Path,
    output_path: str | Path,
    role_profile: RoleProfile,
) -> Path:
    """Pre-compute sparse capability vectors for all candidates and save to JSONL."""
    from fitrank.loader import load_candidates

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for candidate in load_candidates(candidates_path, validate=False):
            text = _sparse_candidate_text(candidate)
            vector = _sparse_capability_vector(text, role_profile.required_capabilities)
            record = {
                "candidate_id": candidate.candidate_id,
                "vector": {k: round(v, 4) for k, v in vector.items()},
            }
            handle.write(json.dumps(record) + "\n")
    return output
