"""FitRank — intelligent candidate discovery and ranking."""

from fitrank.models import (
    Candidate,
    CandidateScore,
    CareerEntry,
    Certification,
    Education,
    Language,
    Profile,
    RedrobSignals,
    RoleProfile,
    SalaryRange,
    Skill,
)
from fitrank.jd_parser import parse_jd, parse_jd_file, save_role_profile
from fitrank.title_gate import (
    TitleDomain,
    TitleGateResult,
    classify_title,
    evaluate_title_gate,
    title_domain_bonus,
    title_jd_match,
    title_jd_match_targets,
)

__all__ = [
    "Candidate",
    "CandidateScore",
    "CareerEntry",
    "Certification",
    "Education",
    "Language",
    "Profile",
    "RedrobSignals",
    "RoleProfile",
    "SalaryRange",
    "Skill",
    "TitleDomain",
    "TitleGateResult",
    "parse_jd",
    "parse_jd_file",
    "save_role_profile",
    "classify_title",
    "evaluate_title_gate",
    "title_domain_bonus",
    "title_jd_match",
    "title_jd_match_targets",
]
