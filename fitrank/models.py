"""Typed data models for FitRank candidate profiles and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Profile:
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        return cls(
            anonymized_name=str(data["anonymized_name"]),
            headline=str(data["headline"]),
            summary=str(data["summary"]),
            location=str(data["location"]),
            country=str(data["country"]),
            years_of_experience=float(data["years_of_experience"]),
            current_title=str(data["current_title"]),
            current_company=str(data["current_company"]),
            current_company_size=str(data["current_company_size"]),
            current_industry=str(data["current_industry"]),
        )


@dataclass(frozen=True)
class CareerEntry:
    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CareerEntry:
        return cls(
            company=str(data["company"]),
            title=str(data["title"]),
            start_date=str(data["start_date"]),
            end_date=data["end_date"],
            duration_months=int(data["duration_months"]),
            is_current=bool(data["is_current"]),
            industry=str(data["industry"]),
            company_size=str(data["company_size"]),
            description=str(data["description"]),
        )


@dataclass(frozen=True)
class Education:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: str | None = None
    tier: str = "unknown"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Education:
        return cls(
            institution=str(data["institution"]),
            degree=str(data["degree"]),
            field_of_study=str(data["field_of_study"]),
            start_year=int(data["start_year"]),
            end_year=int(data["end_year"]),
            grade=data.get("grade"),
            tier=str(data.get("tier", "unknown")),
        )


@dataclass(frozen=True)
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        return cls(
            name=str(data["name"]),
            proficiency=str(data["proficiency"]),
            endorsements=int(data["endorsements"]),
            duration_months=int(data.get("duration_months", 0)),
        )


@dataclass(frozen=True)
class Certification:
    name: str
    issuer: str
    year: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Certification:
        return cls(
            name=str(data["name"]),
            issuer=str(data["issuer"]),
            year=int(data["year"]),
        )


@dataclass(frozen=True)
class Language:
    language: str
    proficiency: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Language:
        return cls(
            language=str(data["language"]),
            proficiency=str(data["proficiency"]),
        )


@dataclass(frozen=True)
class SalaryRange:
    min: float
    max: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SalaryRange:
        return cls(min=float(data["min"]), max=float(data["max"]))


@dataclass(frozen=True)
class RedrobSignals:
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RedrobSignals:
        assessments = data.get("skill_assessment_scores") or {}
        return cls(
            profile_completeness_score=float(data["profile_completeness_score"]),
            signup_date=str(data["signup_date"]),
            last_active_date=str(data["last_active_date"]),
            open_to_work_flag=bool(data["open_to_work_flag"]),
            profile_views_received_30d=int(data["profile_views_received_30d"]),
            applications_submitted_30d=int(data["applications_submitted_30d"]),
            recruiter_response_rate=float(data["recruiter_response_rate"]),
            avg_response_time_hours=float(data["avg_response_time_hours"]),
            skill_assessment_scores={
                str(k): float(v) for k, v in assessments.items()
            },
            connection_count=int(data["connection_count"]),
            endorsements_received=int(data["endorsements_received"]),
            notice_period_days=int(data["notice_period_days"]),
            expected_salary_range_inr_lpa=SalaryRange.from_dict(
                data["expected_salary_range_inr_lpa"]
            ),
            preferred_work_mode=str(data["preferred_work_mode"]),
            willing_to_relocate=bool(data["willing_to_relocate"]),
            github_activity_score=float(data["github_activity_score"]),
            search_appearance_30d=int(data["search_appearance_30d"]),
            saved_by_recruiters_30d=int(data["saved_by_recruiters_30d"]),
            interview_completion_rate=float(data["interview_completion_rate"]),
            offer_acceptance_rate=float(data["offer_acceptance_rate"]),
            verified_email=bool(data["verified_email"]),
            verified_phone=bool(data["verified_phone"]),
            linkedin_connected=bool(data["linkedin_connected"]),
        )


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    profile: Profile
    career_history: list[CareerEntry]
    education: list[Education]
    skills: list[Skill]
    redrob_signals: RedrobSignals
    certifications: list[Certification] = field(default_factory=list)
    languages: list[Language] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        career = [CareerEntry.from_dict(entry) for entry in data["career_history"]]
        career.sort(key=lambda entry: entry.start_date)

        return cls(
            candidate_id=str(data["candidate_id"]),
            profile=Profile.from_dict(data["profile"]),
            career_history=career,
            education=[Education.from_dict(entry) for entry in data.get("education", [])],
            skills=[Skill.from_dict(entry) for entry in data.get("skills", [])],
            redrob_signals=RedrobSignals.from_dict(data["redrob_signals"]),
            certifications=[
                Certification.from_dict(entry)
                for entry in data.get("certifications", [])
            ],
            languages=[Language.from_dict(entry) for entry in data.get("languages", [])],
        )


@dataclass
class RoleProfile:
    target_titles: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    nice_to_have: list[str] = field(default_factory=list)
    seniority: str = "mid"
    min_experience_years: float = 0.0
    preferred_work_mode: str = "flexible"
    domain: str = "general_ml"


@dataclass
class CandidateScore:
    candidate_id: str
    jd_fit: float = 0.0
    career_evidence: float = 0.0
    coherence: float = 0.0
    platform_trust: float = 0.0
    availability: float = 0.0
    penalties: float = 0.0
    final_score: float = 0.0
    reasoning: str = ""
