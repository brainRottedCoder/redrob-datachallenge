"""Phase 3 job description parser tests (PRD T3.1-T3.10)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from fitrank.jd_parser import parse_jd, parse_jd_file, save_role_profile
from fitrank.models import RoleProfile


def test_t3_1_basic_capability_extraction():
    profile = parse_jd("We need LLM fine-tuning, RAG, and PyTorch experience.")
    capabilities = {item.lower() for item in profile.required_capabilities}
    assert "llm fine-tuning" in capabilities
    assert "rag" in capabilities
    assert "pytorch" in capabilities


def test_t3_2_seniority_detection_senior():
    profile = parse_jd("We're looking for a Senior ML Engineer")
    assert profile.seniority == "senior"


def test_t3_3_seniority_detection_junior():
    profile = parse_jd("Junior Data Scientist, 0-2 years")
    assert profile.seniority == "junior"


def test_t3_4_experience_years():
    profile = parse_jd("5+ years of experience in machine learning")
    assert profile.min_experience_years >= 5.0


def test_t3_5_domain_classification_nlp():
    profile = parse_jd(
        "Focus on NLP, transformers, and BERT for language understanding tasks."
    )
    assert "nlp" in profile.domain.lower()


def test_t3_6_work_mode_remote():
    profile = parse_jd("This is a fully remote role for distributed teams.")
    assert profile.preferred_work_mode == "remote"


def test_t3_7_empty_jd_fallback(caplog):
    caplog.set_level(logging.WARNING)
    profile = parse_jd("")
    assert isinstance(profile, RoleProfile)
    assert profile.target_titles
    assert any("Empty job description" in record.message for record in caplog.records)


def test_t3_8_title_extraction():
    profile = parse_jd(
        "We are hiring a Machine Learning Engineer or Data Scientist for the team."
    )
    titles = {title.lower() for title in profile.target_titles}
    assert "machine learning engineer" in titles
    assert "data scientist" in titles


def test_t3_9_role_profile_json_save(tmp_path):
    profile = parse_jd("Senior ML Engineer with PyTorch, RAG, and retrieval experience.")
    output_path = tmp_path / "role_profile.json"
    save_role_profile(profile, output_path)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["target_titles"]
    assert payload["required_capabilities"]


def test_t3_10_different_jds_produce_different_profiles():
    profile_a = parse_jd("Senior NLP Engineer, remote, transformers and BERT required.")
    profile_b = parse_jd("Junior Data Scientist, onsite, PyTorch and statistics.")
    assert profile_a != profile_b


def test_parse_official_job_description_file(project_root: Path):
    jd_path = project_root / "data" / "job_description.txt"
    assert jd_path.exists()
    profile = parse_jd(jd_path.read_text(encoding="utf-8"))
    assert len(profile.required_capabilities) >= 3
    assert profile.seniority == "senior"
    assert profile.min_experience_years >= 5.0


def test_parse_jd_file_writes_output(project_root: Path, tmp_path):
    jd_path = project_root / "data" / "job_description.txt"
    output_path = tmp_path / "role_profile.json"
    profile = parse_jd_file(jd_path, output_path)
    assert profile.required_capabilities
    assert output_path.exists()
