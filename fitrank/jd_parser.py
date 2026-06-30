"""Parse plain-text job descriptions into structured RoleProfile objects."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from fitrank.models import JDPrefs, RoleProfile

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

DOMAIN_EVIDENCE_KEYWORDS: dict[str, list[str]] = {
    "NLP": [
        "nlp", "natural language", "retrieval", "ranking", "transformer",
        "rag", "embedding", "semantic search", "bert", "llm",
    ],
    "information_retrieval": [
        "retrieval", "ranking", "search", "embedding", "semantic search",
        "learning-to-rank", "ndcg", "mrr",
    ],
    "computer_vision": [
        "computer vision", "opencv", "object detection", "cnn", "yolo",
        "image classification", "segmentation",
    ],
    "GenAI": [
        "llm", "fine-tuning", "rag", "prompt", "generative", "transformer",
    ],
    "MLOps": [
        "mlops", "deployment", "serving", "monitoring", "mlflow", "kubeflow",
    ],
    "data_science": [
        "statistics", "experiment", "model", "feature engineering", "analysis",
    ],
    "general_ml": [
        "pytorch", "model", "training", "deployment", "machine learning",
    ],
}

DOMAIN_RULES: list[tuple[str, list[str]]] = [
    ("NLP", ["nlp", "natural language", "transformers", "bert", "tokeniz"]),
    ("computer_vision", ["computer vision", "opencv", "object detection", "cnn"]),
    ("GenAI", ["generative ai", "llm", "fine-tuning", "rag", "prompt"]),
    ("MLOps", ["mlops", "kubeflow", "mlflow", "model deployment", "model serving"]),
    ("data_science", ["data scientist", "statistics", "experiment design"]),
    ("information_retrieval", ["retrieval", "ranking", "search", "hybrid retrieval"]),
]

CONSULTING_POSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"consulting\s+background", re.I),
    re.compile(
        r"consulting\s+experience\s+(?:is\s+)?(?:welcome|valued|preferred|a\s+plus)",
        re.I,
    ),
    re.compile(r"(?:welcome|value|valued|prefer|preferred).{0,40}consulting", re.I),
    re.compile(r"consulting.{0,40}(?:welcome|valued|preferred|a\s+plus)", re.I),
    re.compile(r"client[- ]facing", re.I),
    re.compile(r"services\s+background", re.I),
    re.compile(r"professional\s+services", re.I),
    re.compile(r"management\s+consulting", re.I),
    re.compile(r"\bbig\s+4\b", re.I),
    re.compile(r"\bbig\s+four\b", re.I),
]

RESEARCH_POSITIVE_KEYWORDS = [
    "phd", "postdoc", "post-doc", "research background", "publications",
    "published papers", "academic research", "research scientist",
]

LOCATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(bangalore|bengaluru)\b", re.I), "bangalore"),
    (re.compile(r"\b(pune)\b", re.I), "pune"),
    (re.compile(r"\b(noida)\b", re.I), "noida"),
    (re.compile(r"\b(mumbai)\b", re.I), "mumbai"),
    (re.compile(r"\b(delhi|new delhi)\b", re.I), "delhi"),
    (re.compile(r"\b(gurgaon|gurugram)\b", re.I), "gurgaon"),
    (re.compile(r"\b(hyderabad)\b", re.I), "hyderabad"),
    (re.compile(r"\b(chennai)\b", re.I), "chennai"),
    (re.compile(r"\b(kolkata)\b", re.I), "kolkata"),
    (re.compile(r"\b(ahmedabad)\b", re.I), "ahmedabad"),
    (re.compile(r"\b(india)\b", re.I), "india"),
    (re.compile(r"\b(remote)\b", re.I), "remote"),
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


def _build_jd_prefs(lowered: str, domain: str) -> JDPrefs:
    """Derive conditional penalty flags and evidence keywords from JD text."""
    evidence = list(DOMAIN_EVIDENCE_KEYWORDS.get(domain, DOMAIN_EVIDENCE_KEYWORDS["general_ml"]))

    penalize_consulting = not any(p.search(lowered) for p in CONSULTING_POSITIVE_PATTERNS)
    penalize_research = not any(kw in lowered for kw in RESEARCH_POSITIVE_KEYWORDS)
    penalize_domain_mismatch = domain != "computer_vision"
    values_production = not any(kw in lowered for kw in RESEARCH_POSITIVE_KEYWORDS)

    locations: list[str] = []
    seen_locs: set[str] = set()
    for pattern, canonical in LOCATION_PATTERNS:
        if pattern.search(lowered) and canonical not in seen_locs:
            locations.append(canonical)
            seen_locs.add(canonical)

    return JDPrefs(
        domain=domain,
        required_evidence_keywords=evidence,
        penalize_consulting_only=penalize_consulting,
        penalize_pure_research=penalize_research,
        penalize_domain_mismatch=penalize_domain_mismatch,
        preferred_locations=locations,
        values_production_experience=values_production,
    )


def _role_profile_from_parsed(lowered: str, normalized: str) -> RoleProfile:
    domain = _extract_domain(lowered)
    prefs = _build_jd_prefs(lowered, domain)
    return RoleProfile(
        target_titles=_extract_titles(normalized),
        required_capabilities=_extract_capabilities(lowered),
        nice_to_have=_extract_nice_to_have(lowered),
        seniority=_extract_seniority(lowered),
        min_experience_years=_extract_experience_years(normalized),
        preferred_work_mode=_extract_work_mode(lowered),
        domain=domain,
        prefs=prefs,
    )


def parse_jd(text: str) -> RoleProfile:
    """Parse job description text into a RoleProfile."""
    normalized = (text or "").strip()
    if not normalized:
        logger.warning("Empty job description provided; using default ML engineer profile")
        default_prefs = _build_jd_prefs("", DEFAULT_ROLE_PROFILE.domain)
        return RoleProfile(
            target_titles=list(DEFAULT_ROLE_PROFILE.target_titles),
            required_capabilities=list(DEFAULT_ROLE_PROFILE.required_capabilities),
            nice_to_have=list(DEFAULT_ROLE_PROFILE.nice_to_have),
            seniority=DEFAULT_ROLE_PROFILE.seniority,
            min_experience_years=DEFAULT_ROLE_PROFILE.min_experience_years,
            preferred_work_mode=DEFAULT_ROLE_PROFILE.preferred_work_mode,
            domain=DEFAULT_ROLE_PROFILE.domain,
            prefs=default_prefs,
        )

    lowered = normalized.lower()
    return _role_profile_from_parsed(lowered, normalized)


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
