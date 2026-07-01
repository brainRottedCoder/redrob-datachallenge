# FitRank — Redrob Data & AI Challenge Submission

**Team:** FitRank  
**Challenge:** Intelligent candidate discovery & ranking (100K profiles → top-100 shortlist)  
**Approach:** Offline, explainable, two-stage ranking — heuristic feature extraction + learned LightGBM ranker

---

## What this submission delivers

| Deliverable | Location |
|-------------|----------|
| Ranking pipeline | `rank.py` + `fitrank/` |
| **Top-100 CSV** | `outputs/submission.csv` |
| Quality audit | `outputs/audit_report.json` |
| Approach deck (PDF) | `deck/Redrob_FitRank_Approach.pdf` |
| Team metadata | `submission_metadata.yaml` |
| Interactive demo | `app/streamlit_app.py` |
| Validator | `validate_submission.py` |

### Submission quality (verified)

- `python validate_submission.py outputs/submission.csv` → **Submission is valid.**
- **100/100** unique reasoning strings with `JD=X.XX` component scores
- Known traps **excluded** (`CAND_0004989`, `CAND_0000339`)
- Genuine probe **included** (`CAND_0033861`)
- End-to-end runtime **< 5 minutes** on CPU (see `outputs/audit_report.json`)
- **Zero network calls** during `rank.py` execution

---

## Reproduce from scratch

Requires **Python 3.11+** and a machine with **≥ 16 GB RAM**.

```bash
# 1. Install dependencies
pip install -r requirements.txt
# or: make install

# 2. One-time setup (recommended — keeps ranking under 5 min)
make download-model    # sentence-transformer for semantic matching
make precompute        # cache embeddings for 100K candidates (~2-3 min)
make train             # train LightGBM ranker on pseudo-labels (~1 min)

# 3. Rank and validate
make run
make validate

# 4. Fast test suite (seconds)
make test-fast
```

**Single command after setup:** `make run && make validate`

### Ranking modes

```bash
python rank.py --mode auto        # default: learned model when available
python rank.py --mode heuristic   # rule-based weights only
python rank.py --mode learned     # require LightGBM model
```

---

## How FitRank works

```
Job Description  →  JD Parser  →  RoleProfile
                                        │
100K Candidates  →  Feature Extractor  ├─ Title gate, career evidence, coherence
                      (28 signals)     ├─ Skill trust & penalties
                                        ├─ Redrob platform signals
                                        └─ Semantic capability match (MiniLM)
                                        │
                                        ▼
                              LightGBM ranker (offline-trained)
                                        │
                                        ▼
                         Top-100 CSV + per-candidate reasoning
```

**Scoring formula (heuristic fallback):**

`Final = 0.25×JD + 0.30×Career + 0.20×Coherence + 0.15×Platform + 0.10×Availability − Penalties`

Honeypot profiles with high penalties receive a **×0.25** score multiplier.

**Reasoning format (every row):**

```
{title} | JD=0.85 Career=0.58 Coh=0.79 Trust=0.85 | {candidate-specific evidence}
```

---

## Key design choices

1. **Explainable by default** — every rank has a structured reasoning string with numeric scores and evidence fragments (ML keywords, assessments, tenure, penalties).
2. **Honeypot-resistant** — coherence engine flags template-stuffed profiles; traps are pushed below the top 100.
3. **Semantic JD matching** — `all-MiniLM-L6-v2` embeddings (pre-computed offline) for genuine capability alignment.
4. **Learned re-ranking** — LightGBM trained on pseudo-labels (top 5% positive, bottom 20% negative, traps forced negative).
5. **JD-adaptive penalties** — CV/consulting/research penalties disabled when the JD domain matches.
6. **Stability validated** — weight perturbation bootstrap shows top-20 overlap ≥ 19/20 under ±10% jitter (`outputs/sensitivity_report.json`).

---

## Evaluation & evidence

```bash
make eval          # FitRank vs keyword baseline → outputs/eval_report.json
make sensitivity   # stability + ablation → outputs/sensitivity_report.json
make deck          # regenerate PDF deck from metrics
```

Because the challenge has no public labels, evaluation uses a **synthetic ground truth** built from heuristic scores plus known trap/probe IDs.

---

## Interactive demo

```bash
streamlit run app/streamlit_app.py
```

- Paste a JD and re-rank the pool
- Radar charts of component scores per candidate
- Penalty breakdown and reasoning panel
- Filter by title domain, country, score
- Export shortlist as CSV
- Optional **AI reasoning** toggle (local GGUF model — see below)

---

## Optional: local LLM reasoning

For demo purposes, FitRank can generate natural-language justifications for the top 100 using a small local model (no API calls):

```bash
make install-llm
# Place Qwen2-1.5B-Instruct Q4_K_M at models/qwen2-1.5b-instruct-q4_k_m.gguf
make precompute-reasoning
python rank.py --reasoning llm
```

Default `make run` uses fast template reasoning — no LLM required for submission.

---

## Tests

```bash
make test-fast                              # unit tests, ~60 s
python -m pytest tests/test_prd_compliance.py -v   # deliverables + NFR checks
make test                                   # full suite including 100K integration tests
```

---

## Repository layout

| Path | Purpose |
|------|---------|
| `fitrank/` | Core library (parser, features, ranker, reasoning, embeddings) |
| `rank.py` | CLI entry point |
| `validate_submission.py` | CSV format validator |
| `data/` | `candidates.jsonl`, `job_description.txt`, schema |
| `config/` | Scoring weights and normalization caps |
| `models/fitrank_lgb.txt` | Trained LightGBM model |
| `outputs/` | Submission CSV, audit, eval, sensitivity reports |
| `eval/` | Baseline comparison, metrics, sensitivity analysis |
| `scripts/` | Training, pre-compute, deck generation |
| `app/` | Streamlit demo |
| `deck/` | PDF approach deck |
| `tests/` | Unit and integration tests |
| `docs/` | Pointers to PRD and checklist |

---

## Constraints compliance

| Constraint | Status |
|------------|--------|
| CPU only | ✓ No GPU inference |
| No network during ranking | ✓ All models cached locally |
| ≤ 5 min on 100K candidates | ✓ **199.5 s** with pre-computed embeddings (see `outputs/audit_report.json`) |
| Deterministic output | ✓ Same inputs → same ranking |
| Explainable reasoning | ✓ 100 rows with component scores |

See `submission_metadata.yaml` for team declarations and `PRD_Redrob_FitRank.md` for the full design document.

---

## Contact & metadata

Fill in participant names in `submission_metadata.yaml` before final upload.

Repository: https://github.com/brainRottedCoder/redrob-datachallenge
