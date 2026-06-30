"""Keyword-count baseline ranker for evaluation comparisons."""

from __future__ import annotations

import csv
import heapq
from dataclasses import dataclass
from pathlib import Path

from fitrank.loader import load_candidates
from fitrank.models import Candidate, RoleProfile

# Fixed AI/ML keyword vocabulary — naive teams typically match on these terms.
BASELINE_KEYWORDS: tuple[str, ...] = (
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "pytorch",
    "tensorflow",
    "keras",
    "jax",
    "scikit-learn",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "llm",
    "large language model",
    "transformers",
    "hugging face",
    "huggingface",
    "bert",
    "gpt",
    "openai",
    "langchain",
    "llamaindex",
    "rag",
    "retrieval augmented",
    "vector search",
    "embeddings",
    "faiss",
    "milvus",
    "weaviate",
    "pinecone",
    "chromadb",
    "computer vision",
    "opencv",
    "yolo",
    "object detection",
    "image classification",
    "speech recognition",
    "fine-tuning",
    "fine tuning",
    "lora",
    "qlora",
    "peft",
    "rlhf",
    "mlops",
    "mlflow",
    "wandb",
    "weights and biases",
    "model deployment",
    "inference",
    "feature engineering",
    "data science",
    "neural network",
    "reinforcement learning",
    "generative ai",
    "diffusion",
    "stable diffusion",
    "prompt engineering",
    "semantic search",
    "recommendation system",
    "ranking",
    "information retrieval",
)


@dataclass(frozen=True)
class BaselineResult:
    candidate_id: str
    keyword_score: int
    current_title: str


def _candidate_text(candidate: Candidate) -> str:
    parts = [
        candidate.profile.summary,
        candidate.profile.headline,
        *[skill.name for skill in candidate.skills],
    ]
    return " ".join(parts).lower()


def keyword_count(text: str, keywords: tuple[str, ...] = BASELINE_KEYWORDS) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def score_candidate_baseline(candidate: Candidate) -> BaselineResult:
    text = _candidate_text(candidate)
    return BaselineResult(
        candidate_id=candidate.candidate_id,
        keyword_score=keyword_count(text),
        current_title=candidate.profile.current_title,
    )


def baseline_rank(
    candidates,
    role_profile: RoleProfile | None = None,
    top_n: int = 100,
) -> list[BaselineResult]:
    del role_profile  # baseline ignores JD by design
    heap: list[tuple[tuple[int, str], BaselineResult]] = []
    for candidate in candidates:
        result = score_candidate_baseline(candidate)
        key = (result.keyword_score, result.candidate_id)
        if len(heap) < top_n:
            heapq.heappush(heap, (key, result))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, result))

    ranked = sorted(heap, key=lambda item: (-item[0][0], item[0][1]))
    return [item[1] for item in ranked]


def write_baseline_submission(
    ranked: list[BaselineResult],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not ranked:
        raise ValueError("No baseline results to write")

    max_score = ranked[0].keyword_score
    min_score = ranked[-1].keyword_score
    span = max(max_score - min_score, 1)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, result in enumerate(ranked, start=1):
            normalized = 1.0 - ((rank - 1) / max(len(ranked) - 1, 1)) * 0.8
            score = round(max(normalized, 0.2), 4)
            reasoning = (
                f"{result.current_title} | baseline_keywords={result.keyword_score} | "
                f"keyword-count ranker"
            )
            writer.writerow([result.candidate_id, rank, score, reasoning])
    return output


def run_baseline_ranker(
    candidates_path: str | Path,
    output_path: str | Path,
    top_n: int = 100,
) -> list[BaselineResult]:
    ranked = baseline_rank(load_candidates(candidates_path, validate=False), top_n=top_n)
    write_baseline_submission(ranked, output_path)
    return ranked
