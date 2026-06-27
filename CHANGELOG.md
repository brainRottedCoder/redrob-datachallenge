# FitRank Changelog

This document records all changes made to the system during the testing and improvement phase.

## 1. Initial Dataset Analysis

Before making changes, the full 100,000-candidate dataset was analyzed to understand the challenge:

- **Dataset size:** 100,000 candidate profiles in `data/candidates.jsonl`
- **Countries:** 8 (75.1% India, 10.0% USA, others ~2.5% each)
- **ML/AI titles:** Only 1.2% of candidates have explicit ML/AI titles (~1,179)
- **Genuine strong candidates:** ~905 (0.9%) after filtering for title + trusted skills + career evidence
- **Key traps identified:**
  - 4,337 non-ML titles with 7+ AI skills (keyword stuffers)
  - 50,844 title-career description mismatches (~50.8%)
  - 18,865 salary inversion honeypots (`min > max`)
  - 3,279 shallow AI-curious boilerplate profiles
- **Job description parsed:** Senior AI Engineer — Founding Team at Redrob AI
  - Requires production retrieval/ranking/LLM experience
  - Disqualifies pure research, consulting-only careers, CV/speech without NLP/IR, title-chasers, framework-only experience

## 2. Critical Performance Fix

### Problem
The initial ranker ran in **528.29 seconds** (8.8 minutes), failing the 5-minute requirement.

### Root Cause
Profiling revealed that `fitrank/signals.py` was reloading `config/normalization.yaml` from disk and parsing it with PyYAML **for every candidate** (100,000 times). This consumed the majority of the runtime.

### Fix
- `fitrank/ranker.py` now loads normalization constants once and passes them to `compute_platform_trust()`.
- `fitrank/loader.py` now supports `validate=False` for fast production loading.
- `rank.py` uses `load_candidates(..., validate=False)` during ranking.

### Result
Runtime reduced from **528.29s to 114.53s** (clean environment) and stabilizes around **171.05s** under normal load — well under the 5-minute limit.

## 3. Critical Ranking Fixes

### 3.1 Removed Hardcoded Trap Exclusion
- **Before:** `rank.py` hardcoded `TRAP_IDS = {"CAND_0004989", "CAND_0000339"}` and manually filtered them out.
- **After:** The coherence/penalty layer naturally pushes these traps out of the top 100. Verified that both known traps are absent from the final submission.
- **Reason:** Hardcoding is fragile and would not generalize to hidden honeypots in the evaluation.

### 3.2 Fixed Score Reporting
- **Before:** The `score` column was artificially linearly mapped from rank 1 to rank 100, destroying the actual confidence signal.
- **After:** The CSV reports the candidate's **actual final model score**.
- **Validation:** Scores remain non-increasing, and the validator passes.

### 3.3 Improved Reasoning Strings
- **Before:** All reasoning strings used the same template, e.g., `"AI Engineer | JD=0.63 Career=0.32 Coh=0.88 Trust=0.42 | limited ML career evidence | low penalties (0.00)"`.
- **After:** Each reasoning string is candidate-specific and includes:
  - Current role ML/IR keywords (e.g., `LoRA, QLoRA, BentoML`)
  - Top skill assessment scores (e.g., `assessments LoRA=71`)
  - GitHub activity score (`GH 67` or `no GH`)
  - Years of experience and location
  - Availability signals (`OTW`, `≤30d notice`, `reloc`)
  - Penalty/trap notes when relevant
- **Result:** 100/100 reasoning strings are unique and contain specific profile facts, addressing the manual review criteria.

## 4. New Penalties Added

### 4.1 Consulting-Only Career Penalty
- Detects candidates whose entire career history is at consulting/services firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Deloitte, etc.).
- The JD explicitly disqualifies this profile.

### 4.2 Pure-Research Penalty
- Detects titles like `Research Scientist`, `Research Engineer`, `PhD`, `Postdoc`.
- Penalizes only if career text lacks production deployment/serving/monitoring/A-B testing keywords.
- The JD explicitly disqualifies pure research without production deployment.

### 4.3 CV/Speech/Robotics Without NLP/IR Penalty
- Detects titles like `Computer Vision Engineer`, `Speech Engineer`, `Robotics Engineer`.
- Penalizes if career text lacks NLP/IR evidence (NLP, retrieval, ranking, BERT, LLM, RAG, semantic search, embedding).
- The JD explicitly disqualifies this profile.

### 4.4 Salary Inversion Penalty
- Detects candidates with `expected_salary_range.min > max`.
- Dataset analysis found 18,865 such cases (~18.9%), indicating a honeypot signal.

### 4.5 Config Updates
- Added new weights in `config/weights.yaml` for the four new penalties:
  - `consulting_only: 0.10`
  - `pure_research: 0.10`
  - `cv_without_nlp: 0.10`
  - `salary_inverted: 0.10`

## 5. New Signals Added

### 5.1 Location / Recruitability Bonus
- Favors candidates in India/Tier-1 cities (Pune, Noida, Bangalore, Mumbai, Delhi, Gurgaon, Hyderabad, Chennai, etc.).
- Gives a bonus for `willing_to_relocate = true`.
- The JD strongly prefers Pune/Noida and relocation candidates.

### 5.2 Recency / Response Availability Multiplier
- Added `last_active_date` recency check (decay over 6 months).
- Added `recruiter_response_rate` and `avg_response_time_hours` (faster response = higher score).
- Added `applications_submitted_30d` as an activity signal.
- The JD and signals doc emphasize that inactive/low-response candidates are not actually available.

### 5.3 Certification Bonus
- Small bonus for relevant certifications (AWS, Azure, GCP, TensorFlow, PyTorch, ML, Deep Learning, Kubernetes, Docker, etc.).
- Added to `platform_trust` calculation.

### 5.4 Open-Source Contribution Bonus
- Detects language like "open source", "open-source", "GitHub contributions", "contributed to", "maintained", "published on GitHub", "PyPI", "HuggingFace Hub".
- Added as a small bonus to career evidence.
- The JD lists open-source contributions as nice-to-have.

## 6. Semantic Capability Matching

### 6.1 Implementation
- Created `fitrank/embedder.py` with a **dependency-free** semantic capability matcher.
- Builds a sparse TF-weighted vector per candidate over the JD capabilities.
- Computes cosine similarity between the candidate vector and the JD capability vector.
- Uses a vocabulary of ~40 capabilities with synonyms and related terms (e.g., `RAG` matches `retrieval-augmented`, `retrieval augmented generation`).
- Replaced the old literal-substring `_capability_match()` in `fitrank/ranker.py`.

### 6.2 Optional Pre-Computation
- Created `scripts/precompute_embeddings.py` to pre-compute candidate vectors.
- If pre-computed vectors exist, the ranker could load them for a small speedup.
- If missing, the ranker computes them on-the-fly within the 5-minute budget.
- No new heavy dependencies (no torch, no sklearn, no sentence-transformers) were added.

## 7. Career Evidence & Template Improvements

### 7.1 Expanded Deep ML Pattern Library
- Expanded from ~25 to ~75 regex patterns in `fitrank/career_analyzer.py`.
- New categories:
  - Fine-tuning & GenAI (LLM, GPT, BERT, transformers, instruction tuning)
  - Retrieval & vector search (RAG, FAISS, Milvus, Weaviate, embeddings, semantic search)
  - Frameworks & infra (PyTorch, TensorFlow, JAX, Keras, MLflow, W&B, Kubeflow, Airflow, Spark, Ray)
  - Production & serving (model serving, Triton, BentoML, FastAPI, Docker, Kubernetes, monitoring)
  - Evaluation & ranking (NDCG, MRR, MAP, A/B testing, learning-to-rank, recommendation, ranking)
  - NLP & IR (NLP, BERT, RoBERTa, T5, tokenization, information retrieval)
  - Training & systems (distributed training, data/model parallelism, mixed precision, checkpointing)

### 7.2 Improved Template Domain Classifier
- Added 20+ new template prefixes in `fitrank/career_analyzer.py`:
  - Content writing, DevOps, mobile, frontend, Java backend, full-stack, QA, data analyst, product, data science
  - ML work templates (ML at, AI engineering at, NLP engineer at, search/recommendation at, embedding-based ranking, offline-to-online evaluation, open-source contributions)
- This improves the template coherence score and honeypot detection for non-ML titles with mismatched career descriptions.

## 8. Title Matching Optimization

### 8.1 Removed difflib
- **Before:** `title_jd_match_targets()` used `difflib.SequenceMatcher`, which is O(n²) and expensive.
- **After:** Uses token/core-token Jaccard overlap only.
- **Result:** Faster and more interpretable title matching.

### 8.2 Title Classification
- Kept the existing domain classifier (ML_AI, AI_ADJACENT, SOFTWARE, NON_TECH).
- Kept AI Specialist as `AI_ADJACENT` (lower bonus) per the original trap design.

## 9. Testing & Verification

### 9.1 Full Test Suite
```bash
python -m pytest tests/ -v
```
Result: **93 passed in 1031.35s (17:11)**

### 9.2 PRD Compliance Tests
```bash
python -m pytest tests/test_prd_compliance.py -v
```
Result: **31 passed in 66.93s (1:06)**

### 9.3 Submission Validation
```bash
python validate_submission.py outputs/submission.csv
```
Result: `Submission is valid.`

### 9.4 End-to-End Ranker
```bash
python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt --out outputs/submission.csv
```
Result: `Ranked 100 candidates in 171.05s`

### 9.5 Key Checks Verified
- ✅ Runtime under 5 minutes (171.05s)
- ✅ No honeypot flags in top 100
- ✅ Known traps `CAND_0004989` and `CAND_0000339` excluded
- ✅ 100 unique reasoning strings
- ✅ Scores non-increasing
- ✅ No network imports in `fitrank/`
- ✅ No hardcoded absolute paths
- ✅ Submission CSV is valid

## 10. Documentation Updates

- Updated `README.md` with:
  - Optional pre-computation step
  - Key improvements list
  - Notes on the dependency-free semantic matcher

## 11. Files Changed

```
 M README.md
 M config/weights.yaml
 M fitrank/career_analyzer.py
 M fitrank/penalties.py
 M fitrank/ranker.py
 M fitrank/reasoning.py
 M fitrank/signals.py
 M fitrank/title_gate.py
 M outputs/audit_report.json
 M outputs/submission.csv
 M rank.py
?? fitrank/embedder.py
?? scripts/precompute_embeddings.py
```

## 12. Final Submission Metrics

| Metric | Value |
|--------|-------|
| Runtime | 171.05 seconds |
| Output rows | 100 |
| Validator | Pass |
| Honeypots in top 100 | 0 |
| Score range | 0.5351 – 0.7317 |
| Mean score | 0.5840 |
| Unique reasoning | 100/100 |
| Top candidate | CAND_0046064 (Senior NLP Engineer, score 0.7317) |

### Final Top-100 Title Distribution

| Title | Count |
|-------|-------|
| ML Engineer | 21 |
| AI Research Engineer | 17 |
| Computer Vision Engineer | 15 |
| Data Scientist | 10 |
| AI Specialist | 9 |
| Junior ML Engineer | 9 |
| AI Engineer | 5 |
| Machine Learning Engineer | 5 |
| Senior NLP Engineer | 3 |
| Applied ML Engineer | 2 |
| Senior Data Scientist | 2 |
| Search Engineer | 1 |
| Recommendation Systems Engineer | 1 |

## 13. Second Round of Correctness Improvements

A detailed review of the scoring logic identified a set of edge-case and correctness bugs. All of the following were fixed and the system was re-validated.

### 13.1 `fitrank/coherence.py`

- Added the missing entries to `DOMAIN_COMPATIBILITY` so penalized title domains (e.g., `CV_SPEECH_ROBOTICS`, `RESEARCH_ONLY`) are handled instead of silently falling back.
- For penalized title domains, the title is now confirmed only when the career description actually supports the penalized specialization, otherwise the title is treated as a mismatch.
- In `skill_career_alignment`, the skill token is now required to be at least 2 characters long, preventing trivial one-letter matches.

### 13.2 `fitrank/ranker.py`

- `career_score` is now hard-capped at `1.0` so the open-source bonus cannot inflate the component beyond its intended range.
- Education relevance is now based on exact field matches or field prefixes instead of substring matching, reducing false positives.
- The honeypot multiplier is now applied only when `coherence.is_honeypot` is true **and** `penalties.total_penalty > 0.30`, preventing over-penalization of borderline data-quality cases.

### 13.3 `fitrank/penalties.py`

- `pure_research` penalty now uses a dedicated title check (`Research Scientist`, `Research Engineer`, `Postdoc`, `PhD Researcher`, etc.) and is only applied when the career text lacks production deployment keywords.
- `AI_ADJACENT` keyword-stuffing is now detected only when the title domain is `AI_ADJACENT` or `NON_TECH`, not for genuine ML titles.
- Added `experience_gap` penalty (soft penalty for candidates well below the JD's required experience).
- Added `job_hopper` penalty (high role turnover with average tenure < 8 months).
- All consulting/research/CV constants were moved to a shared `fitrank/constants.py` module.

### 13.4 `fitrank/signals.py`

- `_availability_score` weights now sum to `1.0` so the score is a proper weighted average.
- `_assessment_jd_overlap` now uses word-boundary matching so unrelated skill names cannot accidentally match a JD capability substring.
- `_certification_bonus` is now tiered: high-tier certifications (e.g., AWS ML, Azure AI, Databricks, MLOps) give a larger bonus than general cloud/ML certs.
- The Tier-1 city list is now imported from `fitrank/constants.py` instead of being duplicated.

### 13.5 `fitrank/career_analyzer.py`

- The template-domain cache key was increased from 80 characters to 400 characters, removing false cache collisions between different career templates.
- Removed overly broad regex patterns that caused false positives:
  - `\bindex\b` → now requires `vector index`, `dense index`, or `index search`.
  - `\bray\b` → now requires `ray distributed/serve/tune/train` or `ray.io`.
  - `\bmap\b` → now requires `map@` or `mean average precision`.
  - `\bmonitor\b` → now requires `model monitor`, `monitor model`, or `drift monitor`.
  - `\bclassification\b` → now requires `text/document/image classification`.
  - `\bmatching\b` → now requires `semantic/candidate/job matching`.

### 13.6 `fitrank/title_gate.py`

- Added explicit ML/AI title patterns that were previously falling into generic buckets:
  - `LLM Engineer`, `GenAI Engineer`, `Generative AI Engineer`
  - `Data Science Lead`, `AI Scientist`, `Machine Learning Scientist`
  - `Deep Learning Engineer`, `NLP Scientist`, `Search Scientist`
  - `Ranking Engineer`, `Recommendation Engineer`
- Re-wrote the hot classification path to use substring gates instead of expensive regex on every title, keeping the public regex constants for documentation and compatibility.
- Increased the title cache size to `65536`.

### 13.7 `fitrank/constants.py`

- New shared constants module containing:
  - `CONSULTING_FIRMS`
  - `RESEARCH_ONLY_TITLES`, `PHD_TITLE_PATTERNS`
  - `CV_SPEECH_ROBOTICS_TITLES`
  - `INDIAN_TIER1`
  - `ML_KEYWORDS`

### 13.8 `fitrank/reasoning.py`

- Reasoning now reports an estimated **ML tenure** (years in roles whose descriptions contain deep ML patterns) alongside total years of experience.
- Penalty notes are now explicit: instead of a generic `high penalties (0.45)`, the reasoning lists the specific triggered flags (e.g., `consulting-only, salary inverted (0.45)`).
- Consulting firm names are now lowercased to avoid case-sensitivity issues.
- Shared constants are imported from `fitrank/constants.py`.

### 13.9 `config/weights.yaml`

- Top-level weights adjusted to put slightly more emphasis on JD fit:
  - `jd_fit: 0.25` (from `0.20`)
  - `coherence: 0.20` (from `0.25`)
- Penalty weights normalized to sum to `1.00` and the two new penalties added:
  - `experience_gap: 0.05`
  - `job_hopper: 0.04`

### 13.10 `rank.py`

- The validator subprocess now receives `cwd=str(Path(__file__).resolve().parent)` so `rank.py` works correctly regardless of the caller's working directory.

### 13.11 Testing & Validation After Improvements

**PRD compliance (fast):**
```bash
python -m pytest tests/test_prd_compliance.py -v
```
Result: **31 passed**

**Phase unit tests (fast — excludes two slow full-dataset tests):**
```bash
python -m pytest tests/ -k "not generator_memory and not all_candidates_classified" -v
```
Result: **~60 passed** in normal time

**Slow full-dataset tests (run individually; acceptable per project note):**

| Test | Result | Why Slow |
|------|--------|----------|
| `test_t2_7_generator_memory` | ✅ Pass | `tracemalloc` traces every allocation while streaming 100K candidates (~14 min) |
| `test_t4_12_all_candidates_classified` | ✅ Pass | Classifies all 100K titles with regex (~3 min) |

Both tests pass individually. They are excluded from the standard fast CI run because the `tracemalloc` overhead makes the combined suite impractical, not because of any defect.

**End-to-end ranker:**
```bash
python rank.py
```
Result: `Submission is valid.` / `Ranked 100 candidates in 189.6s`

### 13.12 Updated Submission Metrics

| Metric | Value |
|--------|-------|
| Runtime | 189.6 seconds |
| Output rows | 100 |
| Validator | Pass |
| Honeypots in top 100 | 0 |
| Score range | 0.5457 – 0.7889 |
| Mean score | 0.6024 |
| Unique reasoning | 100/100 |
| Top candidate | CAND_0071974 (Senior AI Engineer, score 0.7889) |

### Updated Top-100 Title Distribution

| Title | Count |
|-------|-------|
| AI Engineer | 14 |
| AI Specialist | 13 |
| Computer Vision Engineer | 11 |
| ML Engineer | 9 |
| Applied ML Engineer | 7 |
| Data Scientist | 7 |
| Senior NLP Engineer | 6 |
| Senior Machine Learning Engineer | 4 |
| Staff Machine Learning Engineer | 4 |
| Machine Learning Engineer | 4 |
| Junior ML Engineer | 4 |
| Senior Data Scientist | 3 |
| NLP Engineer | 3 |
| AI Research Engineer | 3 |
| Senior AI Engineer | 2 |
| Senior Applied Scientist | 2 |
| Recommendation Systems Engineer | 2 |
| Lead AI Engineer | 1 |
| Search Engineer | 1 |

### 13.13 Files Changed in This Round

```
 M config/weights.yaml
 M fitrank/career_analyzer.py
 M fitrank/coherence.py
 M fitrank/constants.py (new)
 M fitrank/penalties.py
 M fitrank/ranker.py
 M fitrank/reasoning.py
 M fitrank/signals.py
 M fitrank/title_gate.py
 M rank.py
 M CHANGELOG.md
 M outputs/audit_report.json
 M outputs/submission.csv
```

## 14. Third Round of Improvements

A full test run (83 fast tests) was executed, then additional improvements were identified and implemented.

### 14.1 `fitrank/career_analyzer.py`

- **Added `years_of_ml_experience`** field to `CareerEvidence` — sums `duration_months` of roles whose title or description contains ≥1 deep ML pattern, then converts to years. Previously this computation was duplicated in `reasoning.py` via `_estimate_ml_tenure()`.
- **Fixed `score_career_momentum()`** to exclude entries with unrecognized `company_size` values before computing the linear regression slope. Previously these mapped to ordinal `0`, introducing systematic downward slope bias for candidates at companies not in the dataset's size enumeration.

### 14.2 `fitrank/ranker.py`

- **Added `ml_tenure_years`** field to `ComponentScores` (sourced from `career.years_of_ml_experience`) so the reasoning module uses the already-computed value instead of re-walking career history.

### 14.3 `fitrank/reasoning.py`

- **Removed `_estimate_ml_tenure()`** — computation is now done once in `career_analyzer.py`.
- **Updated `build_reasoning()`** to accept an optional `ml_tenure: float` parameter.
- **Removed unused `DEEP_ML_PATTERNS` import.**
- **Added `seniority gap`** to the list of explicit penalty flag names.

### 14.4 `fitrank/penalties.py`

- **Added `seniority_mismatch` penalty** — fires when `role_profile.seniority == "senior"` and `career_evidence.years_of_ml_experience < 3.0`. This captures the case where a junior ML engineer ranks well on skills/depth but lacks the sustained ML tenure the Founding Team role demands.
- **Added `seniority_mismatch`** field to `PenaltyScore` dataclass.

### 14.5 `fitrank/coherence.py`

- **Unknown template domain fallback changed from `0.0` to `0.3`** — genuine candidates whose career descriptions don't match any known template prefix were being hard-zeroed in `template_coherence` (40% of coherence score). A `0.3` fallback avoids disproportionately penalizing non-templated descriptions.

### 14.6 `fitrank/title_gate.py`

- **`NON_TECH_KEYWORDS` converted from `list` to `frozenset`** — O(1) membership testing instead of O(n) linear scan for every title classification call.

### 14.7 `config/weights.yaml`

- Added `seniority_mismatch: 0.04` to `penalty_weights`.

### 14.8 `tests/test_phase11_improvements.py` (**NEW — 30 tests**)

New test file covering all improvements introduced in rounds 2 and 3:
- Title gate: 9 tests for new ML_AI title patterns (LLM Engineer, GenAI, Data Science Lead, etc.)
- Career analyzer: 3 tests for `years_of_ml_experience`
- Career momentum: 2 tests for unknown company size filtering
- Penalties: 2 tests for `experience_gap`, 3 tests for `job_hopper`
- Signals: 3 tests for tiered certification bonus, 3 tests for word-boundary assessment overlap
- Coherence: 2 tests for CV/NLP confirmation split, 2 tests for data_science domain
- Signals: 1 test for availability score bounds

### 14.9 Testing & Validation After Round 3

```bash
python -m pytest tests/test_prd_compliance.py tests/test_phase1_setup.py \
  tests/test_phase3_jd_parser.py tests/test_phase4_title_gate.py \
  tests/test_phase5_career.py tests/test_phase6_coherence.py \
  tests/test_phase7_penalties.py tests/test_phase7_skills.py \
  tests/test_phase8_signals.py tests/test_phase9_ranker.py \
  tests/test_phase10_integration.py tests/test_phase11_improvements.py -v
```

| Suite | Count | Result |
|-------|-------|--------|
| PRD compliance | 31 | ✅ Pass |
| Phase 1–10 (fast) | 83 | ✅ Pass |
| Phase 11 (new improvements) | 30 | ✅ Pass |
| **Total** | **144** | ✅ **All pass** |

### 14.10 Further Suggestions (Not Yet Implemented)

See `round3_report.md` for the full catalogue. Key items:

| Priority | Item | Description |
|----------|------|-------------|
| 🔴 High | `career_momentum` tanh normalization | Replace linear ÷4 with `tanh(m/2)` for proportional slope scaling |
| 🔴 High | `ML_DEPTH_MAX` increase | Raise from 15→25 to prevent single-role saturation |
| 🔴 High | AI_ADJACENT honeypot extension | Extend `is_honeypot` to also fire for `AI_ADJACENT` titles |
| 🟡 Medium | Skill-count damping in alignment | Avoid `1/1 = 1.0` for single-skill candidates |
| 🟡 Medium | Open-source deduplication | Avoid double-counting same phrase in summary + description |
| 🟡 Medium | Honeypot score label | Add `[HONEYPOT ×0.25]` to reasoning when multiplier applied |
| 🟡 Medium | Assessment overlap denominator cap | Cap at `min(len(caps), 5)` to avoid underscore for candidates with good but partial assessment coverage |
| 🟢 Low | Per-skill recency weighting | Weight current role skill matches higher than old ones |
| 🟢 Low | Multi-JD score calibration | Normalize scores to percentile band per JD |
| 🟢 Low | Streamlit app enhancements | Penalty breakdown table, score histogram, title domain filter |

### 14.11 Files Changed in Round 3

```
 M fitrank/career_analyzer.py  (years_of_ml_experience, momentum fix)
 M fitrank/ranker.py           (ml_tenure_years in ComponentScores)
 M fitrank/reasoning.py        (precomputed ML tenure, seniority flag)
 M fitrank/penalties.py        (seniority_mismatch penalty)
 M fitrank/coherence.py        (unknown template fallback = 0.3)
 M fitrank/title_gate.py       (NON_TECH_KEYWORDS → frozenset)
 M config/weights.yaml         (seniority_mismatch: 0.04)
?? tests/test_phase11_improvements.py  (30 new tests)
 M CHANGELOG.md
```

## 15. Fourth Round of Improvements

All suggestions from the Round 3 analysis report were implemented. This round focused on correctness, score discrimination, and tooling.

### 15.1 Tier 1 — Critical Correctness

#### `fitrank/career_analyzer.py`

- **Raised `ML_DEPTH_MAX` from 15 → 25** so a single hyper-detailed role cannot saturate the career-depth score and multi-role genuine ML candidates retain discrimination.
- **Added per-entry depth cap of 12** so one verbose role description does not dominate the entire career evidence.
- **Replaced linear momentum normalization with `tanh`**:
  - `momentum = 0` → `0.5` (neutral, not penalized)
  - Bounded and monotonic across all slope magnitudes.

#### `fitrank/coherence.py`

- **Extended honeypot detection to `AI_ADJACENT` titles**. The flag now fires when an AI Specialist / Search Engineer / Recommendation Systems Engineer has a low normalized career ML depth (`all_career_ml_depth_norm < 0.2`) and an overall coherence score below 0.2.
- **Made AI_ADJACENT title confirmation conditional on real career depth**. If `all_career_ml_depth < 2`, confirmation drops from 0.5 to 0.2, exposing title-chaser profiles.
- **Added skill-count damping in `skill_career_alignment`**. The effective denominator is `max(len(ml_skills), 3)`, so a candidate with one matching skill maxes at 0.33 instead of 1.0.
- **Added per-skill recency weighting** in `skill_career_alignment`. A skill found in the current role contributes `2×`; a skill found only in past roles contributes `1×`. This rewards up-to-date expertise.

### 15.2 Tier 2 — High Priority

#### `fitrank/ranker.py`

- **Deduplicated the open-source search corpus**. Sentences are fingerprinted (first 80 chars, lowercased) so a phrase repeated in summary and the first career description only counts once.

#### `fitrank/reasoning.py`

- **Added `[HONEYPOT ×0.25]` label** to the score line when `components.is_honeypot` is true, making the post-multiplier score transparent to reviewers.

#### `fitrank/signals.py`

- **Capped the assessment-overlap denominator at 5** (`min(len(capabilities), 5)`). A candidate with 2 verified matching assessments out of 20 JD capabilities now scores 0.40 instead of 0.10, better reflecting the signal of platform-verified skills.

### 15.3 Tier 3 — Medium Priority

#### `fitrank/penalties.py`

- **Made `template_mismatch` threshold-based**. It now fires only when `template_coherence < 0.1` **and** the template domain is known (not `unknown`), avoiding false positives for non-templated genuine profiles.

#### `rank.py`

- **Audit report path now follows the `--out` directory**. Running `rank.py --out tmp/sub.csv` writes `tmp/audit_report.json` instead of hardcoding `outputs/audit_report.json`.

#### `validate_submission.py`

- **Added reasoning quality checks**: empty reasoning, reasoning shorter than 20 characters, and missing `JD=X.XX` score field are now validation errors.

### 15.4 Tier 4 — Low Priority / Tooling

#### `fitrank/career_analyzer.py`

- **Expanded `SHALLOW_AI_PATTERNS`** with 9 new regexes covering hobby projects, "learning PyTorch/TensorFlow/ML/DL", course completions (Coursera/Udemy), Andrew Ng/Fast.ai/DL.ai, and personal ChatGPT/OpenAI API projects.

#### `fitrank/calibrator.py` (NEW)

- **Multi-JD score calibration module**. `calibrate_scores()` linearly rescales the 10th–90th percentile band to a fixed target (default 0.40–0.90). It does **not** change relative ordering. Activated via the `--calibrate` flag in `rank.py` for internal multi-JD analysis only; **not used** for the challenge submission.

#### `app/streamlit_app.py`

- **Full overhaul**:
  - Title-domain badges (🟢 ML_AI, 🟡 AI_ADJACENT, 🔵 SOFTWARE, 🔴 NON_TECH)
  - ML tenure / total experience progress bar per candidate
  - Expandable penalty breakdown table
  - Score distribution histogram using `plotly`
  - Sidebar filters: title domain, country, minimum score
  - Download button exports the full 100-candidate submission
- **`plotly` added to `requirements.txt`**.

### 15.5 New Tests

Created `tests/test_phase12_round4.py` with 34 focused tests covering every Round 4 change.

```bash
python -m pytest tests/test_phase12_round4.py -v
```
Result: **34 passed**.

### 15.6 Validation After Round 4

```bash
python -m pytest tests/ -k "not generator_memory and not all_candidates_classified" -q
```
Result: **155 passed, 2 deselected** (the 2 deselected are the full-dataset slow tests).

```bash
python -m pytest tests/test_prd_compliance.py -q
```
Result: **31 passed**.

```bash
python rank.py
```
Result: `Submission is valid.` / `Ranked 100 candidates in 233.21s`.

### 15.7 Round 4 Submission Metrics

| Metric | Value |
|--------|-------|
| Runtime | 233.21 seconds |
| Output rows | 100 |
| Validator | Pass |
| Honeypots in top 100 | 0 |
| Honeypots in top 20 | 0 |
| Score range | 0.5001 – 0.7419 |
| Mean score | 0.5541 |
| Unique reasoning | 100/100 |
| Top candidate | CAND_0071974 (Senior AI Engineer, 0.7419) |
| Computer Vision Engineers in top 100 | 7 (was 11) |
| Top 10 title domains | 10/10 ML_AI |

> **Note:** The mean score is slightly below the Round 3 target of 0.58 because raising `ML_DEPTH_MAX` to 25 (C1) intentionally reduces the normalized career-depth scores for the same raw evidence, improving discrimination. The ranking quality checks (top candidate preserved, CV count reduced, top-10 all ML_AI, no honeypots) are all met.

### 15.8 Files Changed in Round 4

```
 M fitrank/career_analyzer.py    (ML_DEPTH_MAX, per-entry cap, tanh momentum, shallow patterns)
 M fitrank/coherence.py          (AI_ADJACENT honeypot, recency-weighted skill alignment, damping)
 M fitrank/ranker.py             (deduplicated open-source corpus)
 M fitrank/reasoning.py          (honeypot label)
 M fitrank/signals.py            (assessment overlap denominator cap)
 M fitrank/penalties.py          (threshold-based template mismatch)
 M fitrank/calibrator.py         (NEW — multi-JD score calibration)
 M rank.py                       (audit path follows --out, --calibrate flag)
 M validate_submission.py        (reasoning quality checks)
 M app/streamlit_app.py          (full overhaul with plotly)
 M requirements.txt              (added plotly)
 M outputs/audit_report.json
 M outputs/submission.csv
?? tests/test_phase12_round4.py (34 new tests)
 M CHANGELOG.md
 M README.md
```
