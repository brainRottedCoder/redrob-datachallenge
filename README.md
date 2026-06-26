# Redrob Datachallenge — FitRank

Hackathon project for the **Redrob Intelligent Candidate Discovery & Ranking Challenge** — FitRank candidate ranking system.

## Repository contents

| Path | Description |
|------|-------------|
| `PRD_Redrob_FitRank.md` | Full product requirements & implementation plan |
| `data/candidates.jsonl` | 100,000 candidate profiles (Git LFS) |
| `data/sample_candidates.json` | Sample profiles for development |
| `data/sample_submission.csv` | Submission format reference |
| `data/candidate_schema.json` | JSON schema for candidate profiles |
| `docs/job_description.docx` | Role/job description for ranking |
| `docs/submission_spec.docx` | Official submission specification |
| `docs/redrob_signals_doc.docx` | Redrob platform signals documentation |
| `validate_submission.py` | Validates ranked output CSV |
| `submission_metadata_template.yaml` | Metadata template for submission |

## Clone with Git LFS

The main dataset (`candidates.jsonl`, ~465 MB) is stored with **Git LFS** because it exceeds GitHub's 100 MB file limit.

```bash
git lfs install
git clone https://github.com/brainRottedCoder/redrob-datachallenge.git
cd redrob-datachallenge
git lfs pull
```

## Validate a submission

```bash
python validate_submission.py outputs/submission.csv
```

## Approach

See [`PRD_Redrob_FitRank.md`](PRD_Redrob_FitRank.md) for architecture, scoring methodology, and implementation plan.
