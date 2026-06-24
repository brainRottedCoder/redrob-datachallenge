# FitRank — Intelligent Candidate Discovery & Ranking

Offline, explainable candidate ranking for the Redrob Data & AI Challenge.

## Quickstart

```bash
pip install -r requirements.txt
python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt --out outputs/submission.csv
python validate_submission.py outputs/submission.csv
```

## Tests

```bash
pytest tests/ -v
```

## Demo

```bash
streamlit run app/streamlit_app.py
```

## Architecture

See `PRD_Redrob_FitRank.md` for the full 10-phase pipeline: JD parsing, title gate, career evidence, coherence, skill trust, Redrob signals, and ranked CSV output.

## Reproduce submission

```bash
make install
make run
make validate
```
