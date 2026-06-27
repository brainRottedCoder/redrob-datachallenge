"""Profile coherence scoring and honeypot detection."""

from __future__ import annotations

from dataclasses import dataclass

from fitrank.career_analyzer import CareerEvidence
from fitrank.models import Candidate
from fitrank.title_gate import TitleDomain

ML_SKILL_KEYWORDS = {
    "pytorch", "tensorflow", "nlp", "machine learning", "deep learning", "llm",
    "lora", "rag", "transformers", "computer vision", "mlops", "scikit-learn",
    "xgboost", "keras", "jax", "huggingface", "fine-tuning", "embeddings",
    "bert", "gpt", "faiss", "milvus", "langchain", "mlflow", "wandb",
}

# Titles that are ML_AI in domain but should not receive full confirmation if
# the career lacks the NLP/IR focus required by the Senior AI Engineer JD.
_PENALIZED_ML_TITLES = {
    "computer vision", "cv engineer", "vision engineer", "speech recognition",
    "speech engineer", "robotics engineer", "roboticist", "control engineer",
}

DOMAIN_COMPATIBILITY: dict[tuple[str, TitleDomain], float] = {
    ("ml_work", TitleDomain.ML_AI): 1.0,
    ("ml_work", TitleDomain.AI_ADJACENT): 0.7,
    ("ml_work", TitleDomain.SOFTWARE): 0.5,
    ("ml_work", TitleDomain.NON_TECH): 0.0,
    ("data_engineering", TitleDomain.ML_AI): 0.8,
    ("data_engineering", TitleDomain.SOFTWARE): 0.7,
    ("data_science", TitleDomain.ML_AI): 0.9,
    ("data_science", TitleDomain.AI_ADJACENT): 0.6,
    ("data_science", TitleDomain.SOFTWARE): 0.3,
    ("data_science", TitleDomain.NON_TECH): 0.0,
    ("support", TitleDomain.NON_TECH): 1.0,
    ("marketing", TitleDomain.NON_TECH): 1.0,
    ("sales", TitleDomain.NON_TECH): 1.0,
    ("accounting", TitleDomain.NON_TECH): 1.0,
    ("consulting", TitleDomain.NON_TECH): 0.8,
    ("mechanical", TitleDomain.NON_TECH): 1.0,
    ("operations", TitleDomain.NON_TECH): 1.0,
    ("brand_design", TitleDomain.NON_TECH): 1.0,
    ("content", TitleDomain.NON_TECH): 1.0,
    ("product", TitleDomain.NON_TECH): 1.0,
    ("frontend", TitleDomain.SOFTWARE): 1.0,
    ("fullstack", TitleDomain.SOFTWARE): 1.0,
    ("java_backend", TitleDomain.SOFTWARE): 1.0,
    ("mobile", TitleDomain.SOFTWARE): 1.0,
    ("qa", TitleDomain.SOFTWARE): 0.8,
    ("data_analytics", TitleDomain.SOFTWARE): 0.7,
    ("data_analytics", TitleDomain.AI_ADJACENT): 0.4,
    ("devops", TitleDomain.SOFTWARE): 0.8,
}


@dataclass
class CoherenceScore:
    template_coherence: float
    title_domain_confirmation: float
    skill_career_alignment: float
    coherence_score: float
    is_honeypot: bool


def compute_coherence(
    candidate: Candidate,
    career_evidence: CareerEvidence,
    title_domain: TitleDomain,
) -> CoherenceScore:
    template = template_coherence_score(career_evidence.template_domain, title_domain)
    confirmation = title_domain_confirmation_score(
        candidate, title_domain, career_evidence
    )
    alignment = skill_career_alignment(candidate)

    score = 0.40 * template + 0.35 * confirmation + 0.25 * alignment

    if (
        title_domain == TitleDomain.NON_TECH
        and career_evidence.all_career_ml_depth < 2
        and career_evidence.current_role_ml_depth < 1
    ):
        score = min(score, 0.15)

    honeypot = (
        score < 0.2
        and title_domain == TitleDomain.NON_TECH
        and career_evidence.all_career_ml_depth >= 5
    )

    return CoherenceScore(
        template_coherence=template,
        title_domain_confirmation=confirmation,
        skill_career_alignment=alignment,
        coherence_score=round(score, 4),
        is_honeypot=honeypot,
    )


def template_coherence_score(template_domain: str, title_domain: TitleDomain) -> float:
    return DOMAIN_COMPATIBILITY.get((template_domain, title_domain), 0.0)


def title_domain_confirmation_score(
    candidate: Candidate,
    title_domain: TitleDomain,
    career_evidence: CareerEvidence,
) -> float:
    title_lower = candidate.profile.current_title.lower()
    if title_domain == TitleDomain.ML_AI:
        # Penalized ML titles (CV/speech/robotics) need NLP/IR evidence to confirm.
        if any(pt in title_lower for pt in _PENALIZED_ML_TITLES):
            career_text = " ".join(
                entry.description for entry in candidate.career_history
            ).lower()
            nlp_ir_terms = [
                "nlp", "natural language", "retrieval", "ranking", "search",
                "transformer", "bert", "llm", "rag", "semantic search", "embedding"
            ]
            if not any(term in career_text for term in nlp_ir_terms):
                return 0.3
        return 1.0
    if title_domain == TitleDomain.AI_ADJACENT:
        return 0.5
    if title_domain == TitleDomain.SOFTWARE and career_evidence.all_career_ml_depth >= 2:
        return 0.3
    return 0.0


def skill_career_alignment(candidate: Candidate) -> float:
    ml_skills = [skill for skill in candidate.skills if _is_ml_skill(skill.name)]
    if not ml_skills:
        return 0.0
    career_text = " ".join(entry.description for entry in candidate.career_history).lower()
    matched = 0
    for skill in ml_skills:
        name = skill.name.lower()
        if name in career_text:
            matched += 1
        else:
            tokens = [token for token in name.split() if len(token) >= 3]
            if tokens and all(token in career_text for token in tokens):
                matched += 1
    return matched / len(ml_skills)


def _is_ml_skill(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in ML_SKILL_KEYWORDS)
