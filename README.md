# FitRank — Intelligent Candidate Discovery & Ranking

Offline, explainable, learned candidate ranking for the Redrob Data & AI Challenge.

FitRank is a **two-stage ranking system**:

1. **Heuristic feature extraction** turns each candidate profile into 28 interpretable signals (JD fit, career evidence, profile coherence, skill trust, Redrob platform signals, penalties).
2. **A small LightGBM model** learns the optimal way to combine those signals from pseudo-labels derived from the dataset itself.

The result is a ranking that is both **data-driven** and **explainable**: every shortlisted candidate gets a human-readable reasoning string with component scores, key evidence, and penalty flags.

---

## Quickstart

```bash
pip install -r requirements.txt

# Optional but recommended: download the sentence-transformer model once
make download-model

# Optional: pre-compute dense semantic embeddings (one-time, ~2-3 min on CPU)
make precompute

# Train the learned LightGBM ranker (one-time, ~30-60 sec)
make train

# Rank and produce outputs/submission.csv
make run

# Validate the submission format
python validate_submission.py outputs/submission.csv
make validate
```

The full pipeline runs end-to-end in under 5 minutes on CPU and uses **zero network calls** during ranking.

---

## Ranking modes

| Mode | Behavior | Use case |
|------|----------|----------|
| `auto` (default) | Uses the LightGBM model if `models/fitrank_lgb.txt` exists; otherwise falls back to the heuristic scorer. | Normal operation |
| `heuristic` | Always uses the hand-tuned weighted formula. | Debugging, ablation, no model available |
| `learned` | Requires the trained model. Throws an error if it is missing. | Final submission when model is trained |

```bash
python rank.py --mode learned
python rank.py --mode heuristic
python rank.py --mode auto
```

---

## Semantic embeddings

FitRank uses dense sentence-transformer embeddings (`all-MiniLM-L6-v2`) for genuine semantic matching between the JD and candidate text. They are pre-computed once and cached:

```bash
python scripts/precompute_embeddings.py --candidates data/candidates.jsonl --output-dir outputs/
```

If the cache is missing, the system falls back to a lightweight sparse capability-vector matcher that still runs within the 5-minute budget.

---

## Evaluation framework

Because the challenge dataset has no public labels, FitRank builds a **synthetic evaluation set** from the heuristic ranker (top 5% positive, bottom 20% negative, known traps forced negative) and compares itself against a naive keyword-counting baseline.

```bash
python rank.py --eval
# or
python eval/compare.py --fitrank-out outputs/submission.csv --report outputs/eval_report.json
```

The report includes:

- `ndcg@10` and `ndcg@100`
- `precision@100` and `recall@100`
- Number of known traps in the top 100
- Rank of a known genuine probe candidate

---

## Ablation & sensitivity analysis

Prove design choices are justified, not arbitrary:

```bash
make sensitivity
# or
python scripts/run_sensitivity.py --rebuild-ground-truth
python rank.py --sensitivity   # after a normal ranking run
```

Produces `outputs/sensitivity_report.json` with:

- **Bootstrap stability** — perturb top-level weights by ±10% (30 runs) and measure top-20 overlap
- **Headline** — e.g. "Top 20 candidates remain stable under ±10% weight perturbation (mean 19.2/20, min 17/20)."
- **Ablation study** — remove coherence, skill trust, or Redrob signals; report NDCG@10/@100 drop (heuristic + learned modes)

Heuristic weight perturbation uses cached component scores (one full pass over the pool, then fast re-ranking). Learned ablation zeros feature groups before LightGBM inference.

For baseline comparison metrics used in the deck:

```bash
make eval
```

---

## Score calibration

Raw model scores can be tightly clustered. Use calibration to spread the top-100 scores to a more interpretable 0.05–0.95 range while preserving rank order:

```bash
python rank.py --calibrate
```

For the official challenge submission, leave `--calibrate` off to use the default 0.50–0.75 rescaling.

---

## Tests

Fast unit tests run in seconds. Slow integration tests (full 100K dataset, end-to-end pipeline) are marked separately.

```bash
# Fast unit tests only (seconds)
python -m pytest -m "not slow" -q

# Full suite including slow integration tests (minutes)
python -m pytest -q

# PRD compliance + deliverables
python -m pytest tests/test_prd_compliance.py -v
```

---

## Demo

```bash
streamlit run app/streamlit_app.py
```

Features:
- Paste any JD and re-rank the candidate pool
- Per-candidate radar chart of component scores
- Detailed reasoning and penalty breakdown
- Filter by title domain, country, and score
- Export the full shortlist as CSV
- Toggle **AI reasoning (local LLM)** for natural-language justifications

### Optional local LLM reasoning

FitRank can generate recruiter-friendly reasoning for the top-100 shortlist using a small local GGUF model (no network, CPU-only). Default `make run` keeps template reasoning for speed and reproducibility.

```bash
make install-llm
# Download Qwen2-1.5B-Instruct Q4_K_M (~1 GB) to models/qwen2-1.5b-instruct-q4_k_m.gguf

make precompute-reasoning
python rank.py --reasoning llm --reasoning-cache outputs/reasoning_cache.jsonl
python rank.py --reasoning llm --reasoning-max-new 0   # cache-only
```

If the model is missing or `llama-cpp-python` is not installed, reasoning falls back to template strings automatically.

---

## Architecture

```
Job Description
      │
      ▼
┌─────────────────────┐
│   JD Parser         │ → RoleProfile (target titles, capabilities, domain, seniority)
└─────────────────────┘
      │
      ▼
Candidates (100K) ──► Heuristic Feature Extractor
                         │
                         ├── Title gate & domain classification
                         ├── Career evidence analyzer
                         ├── Profile coherence engine
                         ├── Skill trust & penalties
                         └── Redrob platform signals
                         │
                         ▼
              28-d RankingFeatures
                         │
                         ▼
         ┌─────────────────────┐
         │   LightGBM ranker   │ (trained offline on pseudo-labels)
         └─────────────────────┘
                         │
                         ▼
              Top-100 submission.csv + audit_report.json
```

See `PRD_Redrob_FitRank.md` for the full v3.0 design.

---

## Key improvements

- **Two-stage learned ranker**: LightGBM model on 28 interpretable heuristic features
- **Dense semantic embeddings**: sentence-transformer `all-MiniLM-L6-v2` with CPU-friendly pre-computation
- **Evaluation framework**: synthetic labels, NDCG, baseline comparison, trap/probe quality gates
- **Score calibration**: optional 0.05–0.95 range spreading for better discrimination
- **JD-adaptive penalties**: CV/speech/robotics, consulting, and research penalties are disabled when the JD domain matches
- **Ablation & sensitivity analysis**: component importance and ranking stability under weight perturbation
- **Fast test suite**: unit tests run in seconds; slow tests are explicitly marked
- **Enhanced Streamlit demo**: radar charts, reasoning panels, filters, and CSV export
- **Explainable by default**: candidate-specific reasoning strings with numeric component scores

---

## Reproduce submission

```bash
make install
make download-model
make precompute
make train
make run
make validate
make eval          # baseline vs FitRank NDCG report
make sensitivity   # weight perturbation + ablation
make deck          # regenerate PDF approach deck
```

### Approach deck

The judge-ready PDF is built from [`deck/Redrob_FitRank_Approach.md`](deck/Redrob_FitRank_Approach.md). Metrics are injected from `outputs/audit_report.json`, `outputs/eval_report.json`, and `outputs/sensitivity_report.json`.

Optional demo screenshot for slide 9:

```bash
streamlit run app/streamlit_app.py
# Save a screenshot to deck/assets/demo_screenshot.png, then:
make deck
```

---

## Repository structure

| Path | Purpose |
|------|---------|
| `fitrank/` | Core ranking library |
| `fitrank/features.py` | 28-feature extractor for the learned model |
| `fitrank/learned_ranker.py` | LightGBM model wrapper |
| `fitrank/embedder.py` | Dense + sparse semantic matching |
| `fitrank/ranker.py` | Ranking orchestration (heuristic / learned / auto) |
| `fitrank/calibrator.py` | Score calibration and rescaling |
| `eval/compare.py` | Baseline comparison and NDCG metrics |
| `eval/sensitivity.py` | Weight perturbation bootstrap and component ablation |
| `scripts/run_sensitivity.py` | CLI for stability analysis |
| `scripts/` | Training, pre-computation, audit, ablation, deck generation |
| `app/streamlit_app.py` | Interactive demo |
| `tests/` | Unit and integration tests |
| `config/weights.yaml` | Heuristic scoring weights |
| `outputs/` | Submission CSV, audit report, eval report, embeddings |
| `deck/` | PDF approach deck |

---

## Submission metadata

See `submission_metadata.yaml` for team details, compute environment, and declarations.
