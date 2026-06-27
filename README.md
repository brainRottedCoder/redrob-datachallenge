# FitRank — Intelligent Candidate Discovery & Ranking

Offline, explainable candidate ranking for the Redrob Data & AI Challenge.

## Quickstart

```bash
pip install -r requirements.txt
python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt --out outputs/submission.csv
python validate_submission.py outputs/submission.csv
```

The ranker scores all 100,000 candidates, keeps the top 100 in a fixed-size heap, and runs end-to-end in under 5 minutes on CPU.

## Optional pre-computation

A lightweight semantic capability vector can be pre-computed for a small extra speedup:

```bash
python scripts/precompute_embeddings.py --candidates data/candidates.jsonl --jd data/job_description.txt
python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt --out outputs/submission.csv
```

If the pre-computed file is missing, the ranker computes the vectors on-the-fly within the 5-minute budget.

## Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/test_prd_compliance.py -v   # deliverables + NFR checks
```

See `DELIVERABLES_CHECKLIST.md` for full PRD Section 12 status.

## Demo

```bash
streamlit run app/streamlit_app.py
```

## Architecture

See `PRD_Redrob_FitRank.md` for the full 10-phase pipeline: JD parsing, title gate, career evidence, coherence, skill trust, Redrob signals, and ranked CSV output.

## Key improvements

- Profile-coherence scoring with honeypot detection
- Trusted skill chain (duration + endorsements)
- JD-conditioned semantic capability matching
- Penalties for consulting-only careers, pure research, CV/speech/robotics without NLP/IR, salary inversion, skill inflation, experience gaps, and job hopping
- Location, recency, recruiter-responsiveness, and tiered certification signals
- Candidate-specific, non-templated reasoning strings with ML tenure estimates and explicit penalty flags
- Actual model scores reported in the `score` column
- Shared constants module and refined component/penalty weights

## Reproduce submission

```bash
make install
make run
make validate
```
