"""Title domain classification and JD title matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from functools import lru_cache

from fitrank.models import RoleProfile


class TitleDomain(str, Enum):
    ML_AI = "ML_AI"
    SOFTWARE = "SOFTWARE"
    AI_ADJACENT = "AI_ADJACENT"
    NON_TECH = "NON_TECH"


DOMAIN_BONUS: dict[TitleDomain, float] = {
    TitleDomain.ML_AI: 1.0,
    TitleDomain.AI_ADJACENT: 0.5,
    TitleDomain.SOFTWARE: 0.3,
    TitleDomain.NON_TECH: 0.0,
}

# Checked before ML_AI patterns because some titles overlap semantically.
AI_ADJACENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bai\s+specialist\b", re.I),
    re.compile(r"\bsearch\s+engineer\b", re.I),
    re.compile(r"\brecommendation\s+systems?\s+engineer\b", re.I),
    re.compile(r"\bsenior\s+software\s+engineer\s*\(ml\)\b", re.I),
    re.compile(r"\bdata\s+engineer\b.*\bml\b", re.I),
    re.compile(r"\bml\b.*\bdata\s+engineer\b", re.I),
]

ML_AI_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*ml\s+engineer\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*machine\s+learning\s+engineer\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*data\s+scientist\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*ai\s+engineer\b", re.I),
    re.compile(r"\bai\s+research\s+engineer\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*nlp\s+engineer\b", re.I),
    re.compile(r"\bcomputer\s+vision\s+engineer\b", re.I),
    re.compile(r"\bapplied\s+ml\s+engineer\b", re.I),
    re.compile(r"\bresearch\s+scientist\b", re.I),
    re.compile(r"\bapplied\s+scientist\b", re.I),
]

SOFTWARE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsoftware\s+engineer\b", re.I),
    re.compile(r"\bfull[\s-]?stack\b", re.I),
    re.compile(r"\bbackend\s+engineer\b", re.I),
    re.compile(r"\bfrontend\s+engineer\b", re.I),
    re.compile(r"\bfront[\s-]?end\s+developer\b", re.I),
    re.compile(r"\bback[\s-]?end\s+developer\b", re.I),
    re.compile(r"\bfull\s+stack\s+developer\b", re.I),
    re.compile(r"\bcloud\s+engineer\b", re.I),
    re.compile(r"\bdevops\s+engineer\b", re.I),
    re.compile(r"\bplatform\s+engineer\b", re.I),
    re.compile(r"\bdeveloper\b", re.I),
    re.compile(r"\bengineer\b", re.I),
]

NON_TECH_KEYWORDS: list[str] = [
    "hr",
    "human resources",
    "accountant",
    "accounting",
    "project manager",
    "product manager",
    "marketing",
    "sales",
    "operations",
    "support",
    "recruiter",
    "consultant",
    "designer",
    "attorney",
    "lawyer",
    "nurse",
    "teacher",
]


@dataclass(frozen=True)
class TitleGateResult:
    domain: TitleDomain
    title_jd_match: float
    title_domain_bonus: float


def classify_title(title: str) -> TitleDomain:
    """Classify a candidate current title into a domain bucket."""
    return _classify_title_cached((title or "").strip())


@lru_cache(maxsize=256)
def _classify_title_cached(normalized: str) -> TitleDomain:
    if not normalized:
        return TitleDomain.NON_TECH

    for pattern in AI_ADJACENT_PATTERNS:
        if pattern.search(normalized):
            return TitleDomain.AI_ADJACENT

    for pattern in ML_AI_PATTERNS:
        if pattern.search(normalized):
            return TitleDomain.ML_AI

    for pattern in SOFTWARE_PATTERNS:
        if pattern.search(normalized):
            return TitleDomain.SOFTWARE

    lowered = normalized.lower()
    if any(keyword in lowered for keyword in NON_TECH_KEYWORDS):
        return TitleDomain.NON_TECH

    if re.search(r"\b(manager|director|lead|head|specialist|analyst|coordinator)\b", lowered):
        return TitleDomain.NON_TECH

    return TitleDomain.NON_TECH


def title_domain_bonus(domain: TitleDomain) -> float:
    return DOMAIN_BONUS[domain]


def title_jd_match(candidate_title: str, role_profile: RoleProfile) -> float:
    """Fuzzy-match a candidate title against parsed JD target titles."""
    return title_jd_match_targets(candidate_title, role_profile.target_titles)


def title_jd_match_targets(candidate_title: str, target_titles: list[str]) -> float:
    if not target_titles:
        return 0.0

    best = 0.0
    for target in target_titles:
        best = max(best, _title_similarity(candidate_title, target))
    return round(min(best, 1.0), 4)


def evaluate_title_gate(candidate_title: str, role_profile: RoleProfile) -> TitleGateResult:
    domain = classify_title(candidate_title)
    return TitleGateResult(
        domain=domain,
        title_jd_match=title_jd_match(candidate_title, role_profile),
        title_domain_bonus=title_domain_bonus(domain),
    )


ROLE_STOPWORDS = frozenset(
    {
        "senior",
        "junior",
        "lead",
        "staff",
        "principal",
        "engineer",
        "developer",
        "manager",
        "specialist",
        "analyst",
        "scientist",
        "the",
        "and",
        "of",
    }
)


def _title_similarity(left: str, right: str) -> float:
    left_norm = _normalize_title(left)
    right_norm = _normalize_title(right)

    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.92

    ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        ratio = max(ratio, overlap)

    left_core = _core_tokens(left_norm)
    right_core = _core_tokens(right_norm)
    if left_core and right_core and not (left_core & right_core):
        return min(ratio * 0.15, 0.08)

    return ratio


def _core_tokens(normalized_title: str) -> set[str]:
    return {token for token in normalized_title.split() if token not in ROLE_STOPWORDS}


def _normalize_title(title: str) -> str:
    normalized = title.lower().strip()
    normalized = re.sub(r"[^\w\s()+-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\bmachine learning\b", "ml", normalized)
    normalized = re.sub(r"\bartificial intelligence\b", "ai", normalized)
    return normalized.strip()
