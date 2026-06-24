"""Phase 9 ranker integration tests."""

import csv
import subprocess
import sys
from pathlib import Path

from fitrank.jd_parser import parse_jd
from fitrank.loader import load_sample
from fitrank.ranker import rank_candidates


def test_t9_1_csv_row_count_on_sample(tmp_path):
    role = parse_jd(Path("data/job_description.txt").read_text(encoding="utf-8"))
    ranked = rank_candidates(load_sample(), role, top_n=50)
    out = tmp_path / "submission.csv"
    from rank import _write_submission_csv

    subset = ranked[:50]
    _write_submission_csv(out, subset)
    rows = list(csv.reader(out.open(encoding="utf-8")))
    assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
    assert len(rows) == 51


def test_t9_3_scores_non_increasing(tmp_path):
    role = parse_jd(Path("data/job_description.txt").read_text(encoding="utf-8"))
    ranked = rank_candidates(load_sample(), role, top_n=50)
    out = tmp_path / "submission.csv"
    from rank import _write_submission_csv

    _write_submission_csv(out, ranked[:50])
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    scores = [float(row["score"]) for row in rows]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
