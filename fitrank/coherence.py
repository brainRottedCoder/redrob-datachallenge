"""Profile coherence scoring and honeypot detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fitrank.career_analyzer import CareerEvidence
from fitrank.constants import CV_SPEECH_ROBOTICS_TITLES
from fitrank.models import Candidate, RoleProfile
from fitrank.title_gate import TitleDomain

ML_SKILL_KEYWORDS = {
    "pytorch", "tensorflow", "nlp", "machine learning", "deep learning", "llm",
    "lora", "rag", "transformers", "computer vision", "mlops", "scikit-learn",
    "xgboost", "keras", "jax", "huggingface", "fine-tuning", "embeddings",
    "bert", "gpt", "faiss", "milvus", "langchain", "mlflow", "wandb",
}

_ML_SKILL_PATTERN = re.compile(
    "|".join(re.escape(keyword) for keyword in sorted(ML_SKILL_KEYWORDS, key=len, reverse=True)),
    re.IGNORECASE,
)

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
    role_profile: RoleProfile | None = None,
) -> CoherenceScore:
    template = template_coherence_score(career_evidence.template_domain, title_domain)
    confirmation = title_domain_confirmation_score(
        candidate, title_domain, career_evidence, role_profile
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
        and (
            (
                title_domain == TitleDomain.NON_TECH
                and career_evidence.all_career_ml_depth >= 5
            )
            or (
                title_domain == TitleDomain.AI_ADJACENT
                and career_evidence.all_career_ml_depth_norm < 0.2
            )
        )
    )

    return CoherenceScore(
        template_coherence=template,
        title_domain_confirmation=confirmation,
        skill_career_alignment=alignment,
        coherence_score=round(score, 4),
        is_honeypot=honeypot,
    )


def template_coherence_score(template_domain: str, title_domain: TitleDomain) -> float:
    return DOMAIN_COMPATIBILITY.get((template_domain, title_domain), 0.3)


def title_domain_confirmation_score(
    candidate: Candidate,
    title_domain: TitleDomain,
    career_evidence: CareerEvidence,
    role_profile: RoleProfile | None = None,
) -> float:
    title_lower = candidate.profile.current_title.lower()
    if title_domain == TitleDomain.ML_AI:
        prefs = role_profile.prefs if role_profile else None
        domain = prefs.domain if prefs else "general_ml"
        penalize_mismatch = prefs.penalize_domain_mismatch if prefs else True
        evidence_keywords = list(prefs.required_evidence_keywords) if prefs else []

        if domain == "computer_vision":
            if any(pt in title_lower for pt in CV_SPEECH_ROBOTICS_TITLES):
                return 1.0

        if penalize_mismatch and any(pt in title_lower for pt in CV_SPEECH_ROBOTICS_TITLES):
            if evidence_keywords:
                career_text = " ".join(
                    entry.description for entry in candidate.career_history
                ).lower()
                if not any(kw in career_text for kw in evidence_keywords):
                    return 0.3
        return 1.0
    if title_domain == TitleDomain.AI_ADJACENT:
        if career_evidence.all_career_ml_depth >= 2:
            return 0.5
        return 0.2
    if title_domain == TitleDomain.SOFTWARE and career_evidence.all_career_ml_depth >= 2:
        return 0.3
    return 0.0


def skill_career_alignment(candidate: Candidate) -> float:
    ml_skills = [skill for skill in candidate.skills if _is_ml_skill(skill.name)]
    if not ml_skills:
        return 0.0
    current_text = " ".join(
        entry.description for entry in candidate.career_history if entry.is_current
    ).lower()
    past_text = " ".join(
        entry.description for entry in candidate.career_history if not entry.is_current
    ).lower()
    score = 0.0
    for skill in ml_skills:
        name = skill.name.lower()
        tokens = [token for token in name.split() if len(token) >= 3]
        in_current = name in current_text or (tokens and all(t in current_text for t in tokens))
        in_past = name in past_text or (tokens and all(t in past_text for t in tokens))
        if in_current:
            score += 2.0
        elif in_past:
            score += 1.0
    effective_total = max(len(ml_skills), 3) * 2.0
    return min(score / effective_total, 1.0)


def _is_ml_skill(name: str) -> bool:
    return bool(_ML_SKILL_PATTERN.search(name))
