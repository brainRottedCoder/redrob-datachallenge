"""Parse plain-text job descriptions into structured RoleProfile objects."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from fitrank.models import RoleProfile

logger = logging.getLogger(__name__)

DEFAULT_ROLE_PROFILE = RoleProfile(
    target_titles=[
        "ML Engineer",
        "Machine Learning Engineer",
        "AI Engineer",
        "Data Scientist",
    ],
    required_capabilities=[
        "PyTorch",
        "embeddings",
        "retrieval",
        "ranking",
        "LLM fine-tuning",
    ],
    nice_to_have=["RAG", "vector database", "MLOps"],
    seniority="mid",
    min_experience_years=3.0,
    preferred_work_mode="flexible",
    domain="general_ml",
)

TITLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsenior\s+ai\s+engineer\b", re.I), "Senior AI Engineer"),
    (re.compile(r"\bsenior\s+ml\s+engineer\b", re.I), "Senior ML Engineer"),
    (
        re.compile(r"\bsenior\s+machine\s+learning\s+engineer\b", re.I),
        "Senior Machine Learning Engineer",
    ),
    (re.compile(r"\bsenior\s+nlp\s+engineer\b", re.I), "Senior NLP Engineer"),
    (re.compile(r"\bsenior\s+data\s+scientist\b", re.I), "Senior Data Scientist"),
    (re.compile(r"\bmachine\s+learning\s+engineer\b", re.I), "Machine Learning Engineer"),
    (re.compile(r"\bml\s+engineer\b", re.I), "ML Engineer"),
    (re.compile(r"\bai\s+engineer\b", re.I), "AI Engineer"),
    (re.compile(r"\bdata\s+scientist\b", re.I), "Data Scientist"),
    (re.compile(r"\bnlp\s+engineer\b", re.I), "NLP Engineer"),
    (re.compile(r"\bcomputer\s+vision\s+engineer\b", re.I), "Computer Vision Engineer"),
    (re.compile(r"\bresearch\s+scientist\b", re.I), "Research Scientist"),
    (re.compile(r"\bapplied\s+ml\s+engineer\b", re.I), "Applied ML Engineer"),
]

CAPABILITY_KEYWORDS: list[tuple[str, str]] = [
    ("llm fine-tuning", "LLM fine-tuning"),
    ("fine-tuning", "LLM fine-tuning"),
    ("lora", "LoRA"),
    ("qlora", "QLoRA"),
    ("peft", "PEFT"),
    ("rag", "RAG"),
    ("retrieval-augmented", "RAG"),
    ("pytorch", "PyTorch"),
    ("tensorflow", "TensorFlow"),
    ("jax", "JAX"),
    ("embeddings", "embeddings"),
    ("sentence-transformers", "sentence-transformers"),
    ("sentence transformers", "sentence-transformers"),
    ("openai embeddings", "OpenAI embeddings"),
    ("bge", "BGE"),
    ("e5", "E5"),
    ("retrieval", "retrieval"),
    ("ranking", "ranking"),
    ("hybrid retrieval", "hybrid retrieval"),
    ("vector database", "vector database"),
    ("vector databases", "vector database"),
    ("pinecone", "Pinecone"),
    ("weaviate", "Weaviate"),
    ("qdrant", "Qdrant"),
    ("milvus", "Milvus"),
    ("faiss", "FAISS"),
    ("elasticsearch", "Elasticsearch"),
    ("opensearch", "OpenSearch"),
    ("langchain", "LangChain"),
    ("transformers", "transformers"),
    ("bert", "BERT"),
    ("nlp", "NLP"),
    ("computer vision", "computer vision"),
    ("mlops", "MLOps"),
    ("mlflow", "MLflow"),
    ("wandb", "Weights & Biases"),
    ("xgboost", "XGBoost"),
    ("learning-to-rank", "learning-to-rank"),
    ("a/b test", "A/B testing"),
    ("ab test", "A/B testing"),
    ("ndcg", "NDCG"),
    ("mrr", "MRR"),
    ("python", "Python"),
    ("distributed systems", "distributed systems"),
    ("inference optimization", "inference optimization"),
]

NICE_TO_HAVE_KEYWORDS: list[tuple[str, str]] = [
    ("open-source", "open-source contributions"),
    ("open source", "open-source contributions"),
    ("hr-tech", "HR-tech"),
    ("recruiting tech", "recruiting tech"),
    ("marketplace", "marketplace products"),
]

DOMAIN_RULES: list[tuple[str, list[str]]] = [
    ("NLP", ["nlp", "natural language", "transformers", "bert", "tokeniz"]),
    ("computer_vision", ["computer vision", "opencv", "object detection", "cnn"]),
    ("GenAI", ["generative ai", "llm", "fine-tuning", "rag", "prompt"]),
    ("MLOps", ["mlops", "kubeflow", "mlflow", "model deployment", "model serving"]),
    ("data_science", ["data scientist", "statistics", "experiment design"]),
    ("information_retrieval", ["retrieval", "ranking", "search", "hybrid retrieval"]),
]

SENIORITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(principal|staff|lead|senior)\b", re.I), "senior"),
    (re.compile(r"\b(junior|entry[- ]level|0\s*-\s*2\s*years)\b", re.I), "junior"),
]

EXPERIENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"experience\s+required:\s*(\d+)\s*[–-]\s*(\d+)\s*years", re.I),
    re.compile(r"(\d+)\s*\+\s*years", re.I),
    re.compile(r"at\s+least\s+(\d+)\s+years", re.I),
    re.compile(r"(\d+)\s*(?:or\s+more\s+)?years\s+of\s+experience", re.I),
    re.compile(r"(\d+)\s*[–-]\s*(\d+)\s*years", re.I),
]

WORK_MODE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bfully\s+remote\b|\bremote\s+role\b|\bremote[- ]first\b", re.I), "remote"),
    (re.compile(r"\bhybrid\b", re.I), "hybrid"),
    (re.compile(r"\bonsite\b|\bon-site\b|\bin[- ]office\b", re.I), "onsite"),
    (re.compile(r"\bflexible\b", re.I), "flexible"),
]


def parse_jd(text: str) -> RoleProfile:
    """Parse job description text into a RoleProfile."""
    normalized = (text or "").strip()
    if not normalized:
        logger.warning("Empty job description provided; using default ML engineer profile")
        return RoleProfile(
            target_titles=list(DEFAULT_ROLE_PROFILE.target_titles),
            required_capabilities=list(DEFAULT_ROLE_PROFILE.required_capabilities),
            nice_to_have=list(DEFAULT_ROLE_PROFILE.nice_to_have),
            seniority=DEFAULT_ROLE_PROFILE.seniority,
            min_experience_years=DEFAULT_ROLE_PROFILE.min_experience_years,
            preferred_work_mode=DEFAULT_ROLE_PROFILE.preferred_work_mode,
            domain=DEFAULT_ROLE_PROFILE.domain,
        )

    lowered = normalized.lower()
    return RoleProfile(
        target_titles=_extract_titles(normalized),
        required_capabilities=_extract_capabilities(lowered),
        nice_to_have=_extract_nice_to_have(lowered),
        seniority=_extract_seniority(lowered),
        min_experience_years=_extract_experience_years(normalized),
        preferred_work_mode=_extract_work_mode(lowered),
        domain=_extract_domain(lowered),
    )


def _extract_titles(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pattern, canonical in TITLE_PATTERNS:
        if pattern.search(text) and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)
    return found or ["ML Engineer"]


def _extract_capabilities(lowered: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for needle, canonical in CAPABILITY_KEYWORDS:
        if needle in lowered and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)
    return found


def _extract_nice_to_have(lowered: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for needle, canonical in NICE_TO_HAVE_KEYWORDS:
        if needle in lowered and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)
    return found


def _extract_seniority(lowered: str) -> str:
    for pattern, level in SENIORITY_PATTERNS:
        if pattern.search(lowered):
            return level
    return "mid"


def _extract_experience_years(text: str) -> float:
    values: list[float] = []
    for pattern in EXPERIENCE_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if len(groups) == 2 and groups[0] and groups[1]:
                values.append(float(groups[0]))
            elif groups[0]:
                values.append(float(groups[0]))
    return max(values) if values else 0.0


def _extract_work_mode(lowered: str) -> str:
    for pattern, mode in WORK_MODE_PATTERNS:
        if pattern.search(lowered):
            return mode
    return "flexible"


def _extract_domain(lowered: str) -> str:
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_RULES:
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score:
            scores[domain] = score
    if not scores:
        return "general_ml"
    return max(scores, key=scores.get)


def role_profile_to_dict(profile: RoleProfile) -> dict[str, object]:
    return asdict(profile)


def save_role_profile(
    profile: RoleProfile,
    path: str | Path = "outputs/role_profile.json",
) -> Path:
    """Serialize a RoleProfile to JSON for inspection and reproducibility."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(role_profile_to_dict(profile), handle, indent=2)
        handle.write("\n")
    return output_path


def parse_jd_file(
    jd_path: str | Path = "data/job_description.txt",
    output_path: str | Path = "outputs/role_profile.json",
) -> RoleProfile:
    """Load a JD text file, parse it, and persist the resulting profile."""
    source = Path(jd_path)
    text = source.read_text(encoding="utf-8")
    profile = parse_jd(text)
    save_role_profile(profile, output_path)
    return profile
