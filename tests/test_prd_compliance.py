"""
PRD compliance and end-to-end integration tests.

Covers PRD Section 12 deliverables, Section 9 NFRs, and cross-phase exit criteria.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KNOWN_TRAPS = {"CAND_0004989", "CAND_0000339"}
GENUINE_PROBE = "CAND_0033861"
REQUIRED_MODULES = [
    "fitrank/models.py",
    "fitrank/loader.py",
    "fitrank/jd_parser.py",
    "fitrank/title_gate.py",
    "fitrank/career_analyzer.py",
    "fitrank/coherence.py",
    "fitrank/skill_trust.py",
    "fitrank/penalties.py",
    "fitrank/signals.py",
    "fitrank/ranker.py",
    "fitrank/reasoning.py",
    "rank.py",
]


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def submission_path(project_root: Path) -> Path:
    return project_root / "outputs" / "submission.csv"


class TestDeliverablesChecklist:
    def test_repo_has_fitrank_package(self, project_root: Path):
        assert (project_root / "fitrank" / "__init__.py").exists()

    @pytest.mark.parametrize("module_path", REQUIRED_MODULES)
    def test_required_modules_exist(self, project_root: Path, module_path: str):
        assert (project_root / module_path).exists()

    def test_submission_csv_valid(self, submission_path: Path):
        if not submission_path.exists():
            pytest.skip("Run rank.py first to generate submission.csv")
        result = subprocess.run(
            [sys.executable, "validate_submission.py", str(submission_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Submission is valid." in result.stdout

    def test_pdf_deck_exists(self, project_root: Path):
        pdf = project_root / "deck" / "Redrob_FitRank_Approach.pdf"
        if not pdf.exists():
            subprocess.run([sys.executable, "scripts/generate_deck_pdf.py"], cwd=ROOT, check=True)
        assert pdf.exists()
        assert pdf.stat().st_size > 1000

    def test_readme_has_reproduce_steps(self, project_root: Path):
        readme = (project_root / "README.md").read_text(encoding="utf-8")
        assert "pip install" in readme
        assert "rank.py" in readme
        assert "validate_submission.py" in readme

    def test_submission_metadata_complete(self, project_root: Path):
        text = (project_root / "submission_metadata.yaml").read_text(encoding="utf-8")
        for key in ("team_name", "methodology_summary", "declarations"):
            assert key in text

    def test_audit_report_exists(self, project_root: Path):
        path = project_root / "outputs" / "audit_report.json"
        if not path.exists():
            pytest.skip("audit report not generated yet")
        audit = json.loads(path.read_text(encoding="utf-8"))
        assert "runtime_seconds" in audit
        assert "score_histogram" in audit
        assert audit["honeypot_flags_in_shortlist"] == 0


class TestSubmissionQuality:
    def test_exactly_100_rows(self, submission_path: Path):
        if not submission_path.exists():
            pytest.skip("submission missing")
        rows = list(csv.DictReader(submission_path.open(encoding="utf-8")))
        assert len(rows) == 100

    def test_trap_profiles_excluded(self, submission_path: Path):
        if not submission_path.exists():
            pytest.skip("submission missing")
        ids = {row["candidate_id"] for row in csv.DictReader(submission_path.open(encoding="utf-8"))}
        assert KNOWN_TRAPS.isdisjoint(ids)

    def test_genuine_probe_in_top_100(self, submission_path: Path):
        if not submission_path.exists():
            pytest.skip("submission missing")
        ids = {row["candidate_id"] for row in csv.DictReader(submission_path.open(encoding="utf-8"))}
        assert GENUINE_PROBE in ids

    def test_reasoning_has_numeric_scores(self, submission_path: Path):
        if not submission_path.exists():
            pytest.skip("submission missing")
        pattern = re.compile(r"JD=\d+\.\d{2}")
        rows = list(csv.DictReader(submission_path.open(encoding="utf-8")))
        assert all(pattern.search(row["reasoning"]) for row in rows)

    def test_reasoning_unique(self, submission_path: Path):
        if not submission_path.exists():
            pytest.skip("submission missing")
        reasons = [row["reasoning"] for row in csv.DictReader(submission_path.open(encoding="utf-8"))]
        assert len(set(reasons)) == len(reasons)

    def test_scores_non_increasing(self, submission_path: Path):
        if not submission_path.exists():
            pytest.skip("submission missing")
        scores = [float(row["score"]) for row in csv.DictReader(submission_path.open(encoding="utf-8"))]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


class TestNonFunctionalRequirements:
    def test_no_network_imports_in_fitrank(self, project_root: Path):
        banned = {"requests", "urllib", "httpx", "aiohttp"}
        for path in (project_root / "fitrank").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            assert banned.isdisjoint(imported), f"{path} imports network library"

    def test_no_hardcoded_absolute_paths(self, project_root: Path):
        pattern = re.compile(r'[A-Z]:\\Users|[A-Z]:\\Projects|/home/|/Users/')
        scan_paths = [
            project_root / "fitrank",
            project_root / "rank.py",
            project_root / "app",
            project_root / "scripts",
        ]
        for base in scan_paths:
            paths = [base] if base.is_file() else base.rglob("*.py")
            for path in paths:
                text = path.read_text(encoding="utf-8", errors="ignore")
                assert not pattern.search(text), f"absolute path in {path}"

    def test_runtime_under_5_minutes(self, project_root: Path):
        audit_path = project_root / "outputs" / "audit_report.json"
        if not audit_path.exists():
            pytest.skip("audit report missing")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["runtime_seconds"] < 300

    def test_deterministic_sample_ranking(self):
        from fitrank.jd_parser import parse_jd
        from fitrank.loader import load_sample
        from fitrank.ranker import rank_candidates

        role = parse_jd((ROOT / "data" / "job_description.txt").read_text(encoding="utf-8"))
        ranked_a = rank_candidates(load_sample(), role, top_n=20)
        ranked_b = rank_candidates(load_sample(), role, top_n=20)
        ids_a = [item[0].candidate_id for item in ranked_a]
        ids_b = [item[0].candidate_id for item in ranked_b]
        assert ids_a == ids_b

    def test_all_phase_test_files_exist(self, project_root: Path):
        for phase in range(1, 11):
            matches = list((project_root / "tests").glob(f"test_phase{phase}*.py"))
            assert matches, f"missing tests for phase {phase}"


class TestCoherenceAndTraps:
    def test_trap_coherence_low(self, candidates_path):
        from fitrank.career_analyzer import analyze_career
        from fitrank.coherence import compute_coherence
        from fitrank.loader import load_candidates
        from fitrank.title_gate import classify_title

        target = None
        for candidate in load_candidates(candidates_path):
            if candidate.candidate_id == "CAND_0004989":
                target = candidate
                break
        assert target is not None
        career = analyze_career(target)
        domain = classify_title(target.profile.current_title)
        score = compute_coherence(target, career, domain)
        assert score.coherence_score <= 0.15

    def test_genuine_coherence_high(self, candidates_path):
        from fitrank.career_analyzer import analyze_career
        from fitrank.coherence import compute_coherence
        from fitrank.loader import load_candidates
        from fitrank.title_gate import classify_title

        target = None
        for candidate in load_candidates(candidates_path):
            if candidate.candidate_id == GENUINE_PROBE:
                target = candidate
                break
        assert target is not None
        career = analyze_career(target)
        domain = classify_title(target.profile.current_title)
        score = compute_coherence(target, career, domain)
        assert score.coherence_score >= 0.80
