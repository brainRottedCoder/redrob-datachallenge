"""Lightweight, dependency-free semantic capability embedding.

This module builds a sparse TF-weighted vector for each candidate against the
JD capabilities and computes cosine similarity. It avoids heavy dependencies
like torch/sklearn so it can run in the constrained CPU-only sandbox.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from fitrank.models import Candidate, RoleProfile


# Capability phrase -> list of related terms/synonyms for soft matching.
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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9/]+", text.lower())


def _build_ngram_sets(tokens: list[str]) -> tuple[set[str], set[str]]:
    """Build 2-gram and 3-gram sets in a single pass over tokens."""
    if len(tokens) < 2:
        return set(), set()
    bigrams = {" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)}
    trigrams = (
        {" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)}
        if len(tokens) >= 3
        else set()
    )
    return bigrams, trigrams


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
    return " ".join(parts).lower()


def _capability_vector(text: str, capabilities: list[str]) -> dict[str, float]:
    """Return a sparse vector of capability relevance scores for the text."""
    vector: dict[str, float] = {}
    text_lower = text.lower()
    tokens = _tokenize(text_lower)
    token_set = set(tokens)
    bigrams, trigrams = _build_ngram_sets(tokens)
    ngrams = bigrams | trigrams

    for cap in capabilities:
        if cap not in CAPABILITY_VOCABULARY:
            # Fallback to exact/substring matching for unknown capabilities.
            cap_lower = cap.lower()
            if cap_lower in text_lower or cap_lower.replace(" ", "") in text_lower.replace(" ", ""):
                vector[cap] = 1.0
            continue

        score = 0.0
        for term in CAPABILITY_VOCABULARY[cap]:
            if " " in term:
                # Multi-word term: check n-grams.
                if term in ngrams:
                    score += 1.0
            else:
                if term in token_set:
                    score += 1.0
        if score > 0:
            vector[cap] = min(score, 3.0)  # Cap per-capability contribution.
    return vector


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
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


def compute_semantic_capability_match(
    candidate: Candidate,
    role_profile: RoleProfile,
) -> float:
    """Compute semantic overlap between candidate text and JD capabilities."""
    if not role_profile.required_capabilities:
        return 0.0

    text = _candidate_text(candidate)
    cand_vector = _capability_vector(text, role_profile.required_capabilities)
    # JD vector is a binary vector over its own capabilities.
    jd_vector = {cap: 1.0 for cap in role_profile.required_capabilities}
    return _cosine_similarity(cand_vector, jd_vector)


def precompute_candidate_vectors(
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
            text = _candidate_text(candidate)
            vector = _capability_vector(text, role_profile.required_capabilities)
            record = {
                "candidate_id": candidate.candidate_id,
                "vector": {k: round(v, 4) for k, v in vector.items()},
            }
            handle.write(json.dumps(record) + "\n")
    return output


def load_candidate_vectors(path: str | Path) -> dict[str, dict[str, float]]:
    """Load pre-computed candidate vectors from JSONL."""
    vectors: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            vectors[record["candidate_id"]] = record["vector"]
    return vectors


def semantic_match_from_precomputed(
    candidate_id: str,
    role_profile: RoleProfile,
    vectors: dict[str, dict[str, float]],
) -> float:
    """Compute semantic capability match using pre-computed vectors."""
    if not role_profile.required_capabilities:
        return 0.0
    cand_vector = vectors.get(candidate_id, {})
    if not cand_vector:
        return 0.0
    jd_vector = {cap: 1.0 for cap in role_profile.required_capabilities}
    return _cosine_similarity(cand_vector, jd_vector)
