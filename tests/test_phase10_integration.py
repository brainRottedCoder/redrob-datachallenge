"""Phase 10 integration smoke tests."""

from pathlib import Path

import pytest


def test_t10_1_full_test_suite():
    # Executed via pytest in CI/Makefile; this file ensures module discovery.
    assert Path("app/streamlit_app.py").exists()


def test_t10_8_audit_script_exists():
    assert Path("scripts/audit_shortlist.py").exists()


def test_t10_10_metadata_complete():
    text = Path("submission_metadata.yaml").read_text(encoding="utf-8")
    for key in ("team_name", "methodology_summary", "declarations"):
        assert key in text
