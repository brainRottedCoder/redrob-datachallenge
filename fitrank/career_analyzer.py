"""Career history evidence extraction and template classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
        r"fine[\s-]?tun",
        r"\bqlora\b",
        r"\blora(?!l)\b",
        r"\brlhf\b",
        r"\bdpo\b",
        r"\bsft\b",
        r"sentence[\s-]?transformer",
        r"\bfaiss\b",
        r"\bmilvus\b",
        r"\bweaviate\b",
        r"\bpinecone\b",
        r"\bchromadb\b",
        r"\bqdrant\b",
        r"\bpytorch\b",
        r"\btensorflow\b",
        r"\bmlflow\b",
        r"\bwandb\b",
        r"model[\s-]?serv",
        r"\btriton\b",
        r"\bbentoml\b",
        r"\brag\b",
        r"semantic[\s-]?search",
        r"vector[\s-]?search",
        r"\bpeft\b",
        r"production[\s-]?model",
        r"\bllama\b",
        r"\bmistral\b",
        r"training[\s-]?loop",
        r"hyperparameter",
        r"distributed[\s-]?train",
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
]

ML_DEPTH_MAX = 15.0


@dataclass
class CareerEvidence:
    all_career_ml_depth: int
    current_role_ml_depth: int
    career_momentum: float
    template_domain: str
    shallow_ai_count: int
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

    return CareerEvidence(
        all_career_ml_depth=all_depth,
        current_role_ml_depth=current_depth,
        career_momentum=momentum,
        template_domain=template_domain,
        shallow_ai_count=shallow,
        all_career_ml_depth_norm=min(all_depth / ML_DEPTH_MAX, 1.0),
        current_role_ml_depth_norm=min(current_depth / ML_DEPTH_MAX, 1.0),
        career_momentum_norm=_normalize_momentum(momentum),
    )


def score_career_ml_depth(career_history: list[CareerEntry]) -> int:
    return sum(_count_deep_patterns(entry.description) for entry in career_history)


def score_current_role_depth(career_history: list[CareerEntry]) -> int:
    current = [entry for entry in career_history if entry.is_current]
    if not current:
        return 0
    return _count_deep_patterns(current[-1].description)


def score_career_momentum(career_history: list[CareerEntry]) -> float:
    if len(career_history) < 2:
        return 0.0
    values = [COMPANY_SIZE_ORDINAL.get(entry.company_size, 0) for entry in career_history]
    if not values or max(values) == min(values):
        return 0.0
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
    return numerator / denominator


def classify_template_domain(description: str) -> str:
    prefix = description.lower().strip()[:60]
    for template_prefix, domain in TEMPLATE_PREFIXES:
        if prefix.startswith(template_prefix):
            return domain
    if _count_deep_patterns(description) >= 3:
        return "ml_work"
    return "unknown"


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
    return sum(1 for pattern in DEEP_ML_PATTERNS if pattern.search(text or ""))


def _normalize_momentum(momentum: float) -> float:
    return max(0.0, min(1.0, (momentum + 2.0) / 4.0))
