"""Career history evidence extraction and template classification."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from fitrank.models import Candidate, CareerEntry

COMPANY_SIZE_ORDINAL = {
    "1-10": 1,
    "11-50": 2,
    "51-200": 3,
    "201-500": 4,
    "501-1000": 5,
    "1001-5000": 6,
    "5001-10000": 7,
    "10001+": 8,
}

DEEP_ML_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in [
        # Fine-tuning & GenAI
        r"fine[\s-]?tun",
        r"\bqlora\b",
        r"\blora(?!l)\b",
        r"\brlhf\b",
        r"\bdpo\b",
        r"\bsft\b",
        r"\bpeft\b",
        r"\bllm\b",
        r"\bgpt\b",
        r"\bllama\b",
        r"\bmistral\b",
        r"\bclaude\b",
        r"\bopenai\b",
        r"\bhf\b",
        r"\btransformers?\b",
        r"\battention\b",
        r"\bencoder\b",
        r"\bdecoder\b",
        r"\bpretrain\b",
        r"\binstruction tun",
        r"\bprompt engineer",
        # Retrieval & vector search
        r"\brag\b",
        r"sentence[\s-]?transformer",
        r"\bfaiss\b",
        r"\bmilvus\b",
        r"\bweaviate\b",
        r"\bpinecone\b",
        r"\bchromadb\b",
        r"\bqdrant\b",
        r"semantic[\s-]?search",
        r"vector[\s-]?search",
        r"\bembedding\b",
        r"\bretrieval\b",
        r"vector\s+index",
        r"dense\s+index",
        r"\bindex\s+search\b",
        r"\bnearest neighbor",
        # Frameworks & infra
        r"\bpytorch\b",
        r"\btensorflow\b",
        r"\bjax\b",
        r"\bkeras\b",
        r"\bmlflow\b",
        r"\bwandb\b",
        r"\bkubeflow\b",
        r"\bairflow\b",
        r"\bspark\b",
        r"\bray\s+(distributed|cluster|serve|tune|train|data)\b",
        r"\bray\.io\b",
        r"\bhorovod\b",
        r"\bdeep speed\b",
        r"\baccelerate\b",
        # Production & serving
        r"model[\s-]?serv",
        r"\btriton\b",
        r"\bbentoml\b",
        r"\bfastapi\b",
        r"\bflask\b",
        r"\bdocker\b",
        r"\bkubernetes\b",
        r"\bterraform\b",
        r"\bcicd\b",
        r"model\s+monitor",
        r"monitor\s+model",
        r"drift\s+monitor",
        r"production[\s-]?model",
        r"\bmodel deploy",
        r"\bonline model",
        r"\breal[- ]?time inference",
        r"\bbatch inference",
        r"\bedge deploy",
        # Evaluation & ranking
        r"\bndcg\b",
        r"\bmrr\b",
        r"\bmap@\b",
        r"mean average precision",
        r"\bprecision@\b",
        r"\brecall@\b",
        r"\bhit rate\b",
        r"\bab test\b",
        r"\ba/b test\b",
        r"\bonline eval",
        r"\boffline eval",
        r"\blearning[- ]?to[- ]?rank\b",
        r"\bltr\b",
        r"\branker\b",
        r"\brecommendation\b",
        r"\bcandidate[- ]?job\b",
        r"semantic\s+matching",
        r"candidate\s+matching",
        r"job\s+matching",
        # NLP & IR
        r"\bnlp\b",
        r"\bbert\b",
        r"\broberta\b",
        r"\bt5\b",
        r"\bnatural language\b",
        r"\btokeniz\b",
        r"\bner\b",
        r"\bsummariz\b",
        r"\bsentiment\b",
        r"text\s+classification",
        r"document\s+classification",
        r"image\s+classification",
        r"\binformation retrieval\b",
        # Training & systems
        r"training[\s-]?loop",
        r"hyperparameter",
        r"distributed[\s-]?train",
        r"\bdata parallel\b",
        r"\bmodel parallel\b",
        r"\bgradient accum",
        r"\bmixed precision",
        r"\bcheckpoint",
        r"\bexperiment tracking",
    ]
]

SHALLOW_AI_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in [
        r"curious about (how )?ai",
        r"experimenting with chatgpt",
        r"ai tools could augment",
        r"taking online courses (on|about)",
        r"played with the openai",
        r"excited about.*\bai\b",
        r"side project.*(rag|langchain)",
        r"(rag|langchain).*side project",
        r"exploring how llm",
        r"just starting to explore (ai|ml)",
        r"hobby project.*(ai|ml|gpt)",
        r"(ai|ml).*hobby project",
        r"interested in (ai|ml|llm|nlp)",
        r"learning (pytorch|tensorflow|machine learning|deep learning)",
        r"followed.*coursera.*(ml|ai|data science)",
        r"completed.*udemy.*(ml|ai|deep learning)",
        r"watched.*(andrew ng|fast\.ai|deeplearning\.ai)",
        r"personal project.*(chatgpt|openai api)",
    ]
]

TEMPLATE_PREFIXES: list[tuple[str, str]] = [
    ("enterprise sales of cloud software solutions", "sales"),
    ("customer support team lead at a saas product", "support"),
    ("marketing leadership role at a b2b saas company", "marketing"),
    ("business analyst at a consulting firm", "consulting"),
    ("brand design and creative direction", "brand_design"),
    ("mechanical engineering design role", "mechanical"),
    ("senior accounting role at a mid-sized company", "accounting"),
    ("fine-tuned llama-2-7b and mistral-7b", "ml_work"),
    ("developed a semantic search feature", "ml_work"),
    ("built and maintained data pipelines on apache airflow", "data_engineering"),
    ("operations management role at a logistics company", "operations"),
    ("content writing and seo strategy", "content"),
    ("cloud infrastructure and devops work", "devops"),
    ("android mobile development using java", "mobile"),
    ("frontend engineering at", "frontend"),
    ("java backend development at", "java_backend"),
    ("full-stack web application development", "fullstack"),
    ("test automation and qa engineering", "qa"),
    ("data analyst at", "data_analytics"),
    ("product management at", "product"),
    ("data science at", "data_science"),
    ("machine learning at", "ml_work"),
    ("ai engineering at", "ml_work"),
    ("nlp engineer at", "ml_work"),
    ("search engineer at", "ml_work"),
    ("recommendation systems at", "ml_work"),
    ("ranking and retrieval at", "ml_work"),
    ("built an embedding-based", "ml_work"),
    ("shipped a v2 ranking system", "ml_work"),
    ("designed an offline-to-online", "ml_work"),
    ("open-source contributions in", "ml_work"),
]

ML_DEPTH_MAX = 25.0


@dataclass
class CareerEvidence:
    all_career_ml_depth: int
    current_role_ml_depth: int
    career_momentum: float
    template_domain: str
    shallow_ai_count: int
    years_of_ml_experience: float = 0.0
    all_career_ml_depth_norm: float = 0.0
    current_role_ml_depth_norm: float = 0.0
    career_momentum_norm: float = 0.0


def analyze_career(candidate: Candidate) -> CareerEvidence:
    all_depth = score_career_ml_depth(candidate.career_history)
    current_depth = score_current_role_depth(candidate.career_history)
    momentum = score_career_momentum(candidate.career_history)
    template_domain = get_current_template_domain(candidate)
    shallow = count_shallow_ai_boilerplate(
        candidate.profile.summary,
        *[entry.description for entry in candidate.career_history],
    )
    ml_years = _years_of_ml_experience(candidate.career_history)

    return CareerEvidence(
        all_career_ml_depth=all_depth,
        current_role_ml_depth=current_depth,
        career_momentum=momentum,
        template_domain=template_domain,
        shallow_ai_count=shallow,
        years_of_ml_experience=ml_years,
        all_career_ml_depth_norm=min(all_depth / ML_DEPTH_MAX, 1.0),
        current_role_ml_depth_norm=min(current_depth / ML_DEPTH_MAX, 1.0),
        career_momentum_norm=_normalize_momentum(momentum),
    )


def score_career_ml_depth(career_history: list[CareerEntry]) -> int:
    # Cap per-entry depth so a single hyper-detailed role cannot dominate the career score.
    return sum(min(_count_deep_patterns(entry.description), 12) for entry in career_history)


def score_current_role_depth(career_history: list[CareerEntry]) -> int:
    current = [entry for entry in career_history if entry.is_current]
    if not current:
        return 0
    return _count_deep_patterns(current[-1].description)


def score_career_momentum(career_history: list[CareerEntry]) -> float:
    if len(career_history) < 2:
        return 0.0
    # Only include entries with a known company size; unknown sizes add noise.
    values = [
        COMPANY_SIZE_ORDINAL[entry.company_size]
        for entry in career_history
        if entry.company_size in COMPANY_SIZE_ORDINAL
    ]
    if len(values) < 2 or max(values) == min(values):
        return 0.0
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
    return numerator / denominator


def classify_template_domain(description: str) -> str:
    # Use a longer prefix to avoid false collisions between templates.
    return _classify_template_domain_cached(description.lower().strip()[:400])


@lru_cache(maxsize=4096)
def _classify_template_domain_cached(prefix: str) -> str:
    for template_prefix, domain in TEMPLATE_PREFIXES:
        if prefix.startswith(template_prefix):
            return domain
    if _count_deep_patterns_cached(prefix) >= 3:
        return "ml_work"
    return "unknown"


@lru_cache(maxsize=8192)
def _count_deep_patterns_cached(text: str) -> int:
    return sum(1 for pattern in DEEP_ML_PATTERNS if pattern.search(text or ""))


def get_current_template_domain(candidate: Candidate) -> str:
    current = [entry for entry in candidate.career_history if entry.is_current]
    if not current:
        return "unknown"
    return classify_template_domain(current[-1].description)


def count_shallow_ai_boilerplate(*texts: str) -> int:
    total = 0
    for text in texts:
        total += sum(1 for pattern in SHALLOW_AI_PATTERNS if pattern.search(text or ""))
    return total


def _count_deep_patterns(text: str) -> int:
    return _count_deep_patterns_cached(text or "")


def _years_of_ml_experience(career_history: list[CareerEntry]) -> float:
    """Sum duration of roles whose title or description contains ML patterns."""
    total_months = 0
    for entry in career_history:
        text = (entry.title + " " + entry.description).lower()
        if _count_deep_patterns_cached(text) >= 1:
            total_months += entry.duration_months
    return round(total_months / 12.0, 1)


def _normalize_momentum(momentum: float) -> float:
    # tanh maps (-inf, +inf) -> (-1, 1). Scale by /2 to keep typical slopes in range.
    return (math.tanh(momentum / 2.0) + 1.0) / 2.0
