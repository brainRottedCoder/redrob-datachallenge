"""Optional local LLM reasoning for FitRank top-100 candidates.

Provides a cache-aware reasoning engine that can use a local GGUF model (via
llama-cpp-python) when available. If the model is missing or the library is not
installed, the system falls back to the existing template-based reasoning.

This module is intentionally isolated so that ranking still works without any
LLM dependencies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fitrank.component_scores import ComponentScores
from fitrank.models import Candidate, RoleProfile
from fitrank.reasoning import ReasoningEvidence, _gather_evidence, build_reasoning, _build_score_line

ReasoningMode = Literal["template", "llm", "auto"]
DEFAULT_MODEL_PATH = Path("models") / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
DEFAULT_CACHE_PATH = Path("outputs") / "reasoning_cache.jsonl"


@dataclass
class ReasoningStats:
    cache_hits: int = 0
    llm_generated: int = 0
    template_fallbacks: int = 0


class ReasoningCache:
    """Simple append-only JSONL cache for LLM-generated reasoning strings."""

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self._cache: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._cache[record["key"]] = record["reasoning"]
                except (KeyError, json.JSONDecodeError):
                    continue

    def get(self, key: str) -> str | None:
        return self._cache.get(key)

    def set(self, key: str, reasoning: str) -> None:
        self.put(key, reasoning)

    def put(
        self,
        key: str,
        reasoning: str,
        candidate_id: str | None = None,
        jd_hash: str | None = None,
        model: str | None = None,
    ) -> None:
        if key in self._cache:
            return
        self._cache[key] = reasoning
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {"key": key, "reasoning": reasoning}
        if candidate_id is not None:
            record["candidate_id"] = candidate_id
        if jd_hash is not None:
            record["jd_hash"] = jd_hash
        if model is not None:
            record["model"] = model
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def jd_fingerprint(jd_text: str, role_profile: RoleProfile | None = None) -> str:
    """Stable hash of the JD text (and optionally role profile) for cache keys."""
    payload = jd_text
    if role_profile is not None:
        payload += str(role_profile.required_capabilities) + role_profile.domain
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_key(jd_hash: str, candidate_id: str, components: ComponentScores) -> str:
    """Cache key combining JD, candidate id, and a score signature."""
    sig = f"{components.jd_fit:.3f}:{components.career_evidence:.3f}:{components.coherence:.3f}"
    return f"{jd_hash}:{candidate_id}:{sig}"


def is_llama_available(model_path: Path | None = None) -> bool:
    """Return True if llama-cpp-python is installed and the model file exists."""
    try:
        from llama_cpp import Llama  # noqa: F401
    except Exception:
        return False
    path = model_path or DEFAULT_MODEL_PATH
    return Path(path).exists()


class LlamaReasoningEngine:
    """Thin wrapper around a local llama.cpp model."""

    def __init__(self, model_path: str | Path, n_ctx: int = 2048) -> None:
        from llama_cpp import Llama

        self.model = Llama(str(model_path), n_ctx=n_ctx, verbose=False)

    def generate(self, prompt: str, max_tokens: int = 120) -> str:
        output = self.model(prompt, max_tokens=max_tokens, stop=["\n\n"], temperature=0.2)
        return output["choices"][0]["text"].strip()


def build_reasoning_context(
    candidate: Candidate,
    components: ComponentScores,
    ml_tenure: float | None = None,
    role_profile: RoleProfile | None = None,
) -> ReasoningEvidence:
    """Gather evidence for LLM prompt generation or validation."""
    return _gather_evidence(candidate, components, ml_tenure, role_profile)


def build_llm_prompt(evidence: ReasoningEvidence) -> str:
    """Build a structured prompt for the local LLM."""
    return (
        "You are an experienced recruiter writing a one-line candidate summary. "
        "Use ONLY the facts below. Do not invent information.\n\n"
        f"Candidate ID: {evidence.candidate_id}\n"
        f"Title: {evidence.title}\n"
        f"Experience: {evidence.years_of_experience:.1f} years total, "
        f"{evidence.ml_tenure_years:.1f} years in ML\n"
        f"Score breakdown: {evidence.score_line}\n"
        f"Current role evidence: {', '.join(evidence.role_evidence) or 'none'}\n"
        f"Top assessments: {evidence.assessments}\n"
        f"GitHub score: {evidence.github_score}\n"
        f"Availability: {', '.join(evidence.availability_notes)}\n"
        f"Penalty reasons: {', '.join(evidence.penalty_reasons) or 'none'}\n\n"
        "Write a concise one-sentence summary of fit for this role."
    )


def ensure_valid_reasoning(
    text: str,
    candidate: Candidate,
    components: ComponentScores,
    evidence: ReasoningEvidence,
) -> str:
    """Ensure a reasoning string always contains the required score prefix and title."""
    prefix = f"{evidence.title} | {evidence.score_line}"
    if not text.startswith(prefix):
        text = f"{prefix} | {text}"
    if components.is_honeypot and "[HONEYPOT" not in text:
        text += " [HONEYPOT]"
    return text.strip()


def resolve_reasoning(
    candidate: Candidate,
    components: ComponentScores,
    *,
    mode: ReasoningMode = "template",
    role_profile: RoleProfile | None = None,
    jd_hash: str = "",
    cache: ReasoningCache | None = None,
    engine: LlamaReasoningEngine | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    ml_tenure: float | None = None,
    stats: ReasoningStats | None = None,
    max_new: int = 100,
) -> str:
    """Return a reasoning string, optionally using a local LLM with cache fallback."""
    evidence = build_reasoning_context(candidate, components, ml_tenure, role_profile)

    if mode == "template":
        if stats:
            stats.template_fallbacks += 1
        return build_reasoning(candidate, components, ml_tenure=evidence.ml_tenure_years, role_profile=role_profile)

    key = cache_key(jd_hash, candidate.candidate_id, components)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            if stats:
                stats.cache_hits += 1
            return ensure_valid_reasoning(cached, candidate, components, evidence)

    if engine is not None and max_new > 0:
        try:
            prompt = build_llm_prompt(evidence)
            llm_text = engine.generate(prompt)
            if llm_text:
                llm_text = ensure_valid_reasoning(llm_text, candidate, components, evidence)
                if cache is not None:
                    cache.set(key, llm_text)
                if stats:
                    stats.llm_generated += 1
                return llm_text
        except Exception:
            pass

    if stats:
        stats.template_fallbacks += 1
    return build_reasoning(candidate, components, ml_tenure=evidence.ml_tenure_years, role_profile=role_profile)
