# FitRank Deliverables Checklist (PRD Section 12)

## Required for Submission

- [x] GitHub repo (public or accessible to organizers)
- [x] `rank.py` + full `fitrank/` package
- [x] `outputs/submission.csv` — 100 rows, validator-passing
- [x] PDF approach deck (`deck/Redrob_FitRank_Approach.pdf`)
- [x] `README.md` with install + reproduce steps

## Strongly Recommended

- [x] `submission_metadata.yaml` — fully completed
- [x] Streamlit sandbox (`streamlit run app/streamlit_app.py`)
- [x] `pytest -m "not slow" -q` — fast unit tests
- [x] `outputs/audit_report.json` — shortlist quality audit
- [x] Reasoning visible in CSV for all 100 rows
- [x] Trained LightGBM model (`models/fitrank_lgb.txt`)
- [x] Evaluation report (`outputs/eval_report.json`) with baseline comparison
- [x] Sensitivity report (`outputs/sensitivity_report.json`) with stability + ablation

## Pre-Submission QA Checklist

- [x] `python validate_submission.py outputs/submission.csv` → "Submission is valid."
- [x] End-to-end reproduce command runs in ≤ 5 minutes (with `make precompute`)
- [x] Zero network calls during `rank.py` execution (verified in tests)
- [x] Top 100 contains zero known honeypot profiles
- [x] Reasoning strings contain candidate-specific numeric values
- [x] `pytest -m "not slow" -q` → zero failures
- [x] Repo has no hardcoded absolute paths
- [x] `outputs/sensitivity_report.json` shows top-20 stability ≥ 19/20 under ±10% weight perturbation

## PRD Implementation Phases

- [x] Phase 1 — Project scaffolding & environment
- [x] Phase 2 — Data layer & candidate models
- [x] Phase 3 — Job description parser
- [x] Phase 4 — Title gate & domain classifier
- [x] Phase 5 — Career evidence analyzer
- [x] Phase 6 — Profile coherence engine
- [x] Phase 7 — Skill trust & penalty layer
- [x] Phase 8 — Redrob signals & platform trust
- [x] Phase 9 — Ranker, reasoning & output
- [x] Phase 10 — Demo, docs & final QA
- [x] Phase 11 — Dense semantic embeddings
- [x] Phase 12 — Learned ranker & evaluation framework
- [x] Phase 13 — Calibration, ablation & JD-adaptive penalties
