"""Title domain classification and JD title matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*llm\s+engineer\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*gen(?:erative)?\s*ai\s+engineer\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*data\s+science\s+lead\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*ai\s+scientist\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*machine\s+learning\s+scientist\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*deep\s+learning\s+engineer\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*nlp\s+scientist\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*search\s+scientist\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*ranking\s+engineer\b", re.I),
    re.compile(r"\b(staff|senior|junior|lead|principal)?\s*recommendation\s+engineer\b", re.I),
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

NON_TECH_KEYWORDS: frozenset[str] = frozenset([
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
])


@dataclass(frozen=True)
class TitleGateResult:
    domain: TitleDomain
    title_jd_match: float
    title_domain_bonus: float


# Substring keywords used by the fast path below.  These are intentionally kept
# in sync with the regex patterns so that the public constants still document the
# classification rules, but the hot path avoids expensive regex matching on the
# full 100K-candidate dataset.
_AI_ADJACENT_KEYWORDS = {
    "ai specialist", "search engineer", "recommendation systems", "data engineer",
    "software engineer (ml)", "senior software engineer (ml)",
}

_ML_AI_KEYWORDS = {
    "ml engineer", "machine learning", "data scientist", "ai engineer", "nlp engineer",
    "llm engineer", "genai engineer", "generative ai engineer", "deep learning engineer",
    "computer vision engineer", "research scientist", "applied scientist",
    "machine learning scientist", "ai scientist", "search scientist", "ranking engineer",
    "recommendation engineer", "nlp scientist", "data science lead", "ml scientist",
}


def classify_title(title: str) -> TitleDomain:
    """Classify a candidate current title into a domain bucket."""
    return _classify_title_cached((title or "").strip())


@lru_cache(maxsize=65536)
def _classify_title_cached(normalized: str) -> TitleDomain:
    if not normalized:
        return TitleDomain.NON_TECH

    lowered = normalized.lower()

    # Fast non-tech rejection first (most profiles are non-ML).
    if any(keyword in lowered for keyword in NON_TECH_KEYWORDS):
        return TitleDomain.NON_TECH

    # AI-adjacent titles are checked before ML_AI because some overlap semantically.
    if "ai specialist" in lowered:
        return TitleDomain.AI_ADJACENT
    if "search engineer" in lowered:
        return TitleDomain.AI_ADJACENT
    if "recommendation systems" in lowered and "engineer" in lowered:
        return TitleDomain.AI_ADJACENT
    if "data engineer" in lowered and "ml" in lowered:
        return TitleDomain.AI_ADJACENT
    if "senior software engineer" in lowered and "ml" in lowered:
        return TitleDomain.AI_ADJACENT

    # Fast ML_AI path based on core substrings.
    if any(keyword in lowered for keyword in _ML_AI_KEYWORDS):
        return TitleDomain.ML_AI

    # Fast software path.
    sw_keywords = {
        "software engineer", "full stack", "fullstack", "backend", "frontend",
        "front end", "back end", "cloud engineer", "devops", "platform engineer",
        "developer",
    }
    if any(keyword in lowered for keyword in sw_keywords):
        return TitleDomain.SOFTWARE

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

    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    else:
        overlap = 0.0

    left_core = _core_tokens(left_norm)
    right_core = _core_tokens(right_norm)
    if left_core and right_core:
        core_overlap = len(left_core & right_core) / len(left_core | right_core)
    else:
        core_overlap = 0.0

    # If no core tokens overlap, the titles are fundamentally different.
    if left_core and right_core and not (left_core & right_core):
        return min(overlap * 0.15, 0.08)

    return max(overlap, core_overlap)


def _core_tokens(normalized_title: str) -> set[str]:
    return {token for token in normalized_title.split() if token not in ROLE_STOPWORDS}


def _normalize_title(title: str) -> str:
    normalized = title.lower().strip()
    normalized = re.sub(r"[^\w\s()+-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\bmachine learning\b", "ml", normalized)
    normalized = re.sub(r"\bartificial intelligence\b", "ai", normalized)
    return normalized.strip()
