# Product Requirements Document (PRD)
## Redrob FitRank — Intelligent Candidate Discovery & Ranking System

**Project:** Redrob India Runs Data & AI Hackathon  
**Product Name:** FitRank  
**Version:** 2.0  
**Date:** June 25, 2026  
**Author:** Hackathon Team  
**Status:** Active — Implementation Ready  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Product Vision & Core Idea](#3-product-vision--core-idea)
4. [Hackathon Context & Submission Requirements](#4-hackathon-context--submission-requirements)
5. [Dataset Overview & Key Insights](#5-dataset-overview--key-insights)
6. [System Architecture](#6-system-architecture)
7. [Scoring Methodology](#7-scoring-methodology)
8. [Differentiating Factors](#8-differentiating-factors)
9. [Non-Functional Requirements](#9-non-functional-requirements)
10. [Implementation Phases](#10-implementation-phases)
    - [Phase 1 — Project Scaffolding & Environment](#phase-1--project-scaffolding--environment)
    - [Phase 2 — Data Layer & Candidate Models](#phase-2--data-layer--candidate-models)
    - [Phase 3 — Job Description Parser](#phase-3--job-description-parser)
    - [Phase 4 — Title Gate & Domain Classifier](#phase-4--title-gate--domain-classifier)
    - [Phase 5 — Career Evidence Analyzer](#phase-5--career-evidence-analyzer)
    - [Phase 6 — Profile Coherence Engine](#phase-6--profile-coherence-engine)
    - [Phase 7 — Skill Trust & Penalty Layer](#phase-7--skill-trust--penalty-layer)
    - [Phase 8 — Redrob Signals & Platform Trust](#phase-8--redrob-signals--platform-trust)
    - [Phase 9 — Ranker, Reasoning & Output](#phase-9--ranker-reasoning--output)
    - [Phase 10 — Demo, Docs & Final QA](#phase-10--demo-docs--final-qa)
11. [Repository Structure](#11-repository-structure)
12. [Deliverables Checklist](#12-deliverables-checklist)
13. [Success Metrics](#13-success-metrics)
14. [Risks & Mitigations](#14-risks--mitigations)
15. [Future Roadmap](#15-future-roadmap)
16. [Appendix](#16-appendix)

---

## 1. Executive Summary

**FitRank** is an offline, explainable candidate ranking system for the Redrob Intelligent Candidate Discovery & Ranking Challenge. It ranks 100,000 synthetic candidate profiles against a job description (JD) the way an experienced recruiter would — by evaluating **cross-field profile coherence**, **career evidence depth**, **behavioral platform signals**, and **explicit honeypot detection** — rather than keyword overlap.

**Core thesis:** The right candidate is not the one who lists the most AI skills, but the one whose title, career narrative, trusted skills, and platform verification all tell the same story.

### Deliverables

| Deliverable | Description |
|-------------|-------------|
| GitHub Repository | Clean, modular, reproducible Python codebase |
| Ranked Output CSV | Top 100 candidates, scores, and reasoning |
| PDF Approach Deck | Problem, architecture, differentiation, examples |
| Streamlit Sandbox | Interactive recruiter-style shortlist demo |

---

## 2. Problem Statement

Recruiters reviewing large talent pools miss strong candidates because:

1. **Keyword filters surface the wrong people** — profiles optimized for search terms rank high without genuine fit.
2. **Skills lists are unreliable** — candidates claim advanced ML skills without corresponding work history.
3. **Titles lie** — current job titles frequently do not match actual role descriptions (~6,400 such profiles in this dataset).
4. **Boilerplate summaries mislead** — thousands of profiles share identical "curious about ChatGPT" templates.
5. **No explainability** — black-box rankings cannot be defended to hiring managers.
6. **Template noise** — ~300,000 career description entries are shared templates across candidates, making raw text analysis unreliable.

The dataset encodes all of these failure modes deliberately. A naive ranker counting AI-related skills promotes **honeypot profiles** while burying genuine practitioners.

**Product goal:** Deliver a top-100 shortlist that a recruiter can trust, with transparent reasoning for every rank.

---

## 3. Product Vision & Core Idea

### Vision Statement

> Build an AI-assisted ranking system that understands what a role needs, evaluates the full candidate picture, and produces an explainable shortlist — mirroring how Redrob would productize intelligent discovery for paying recruiter customers.

### Core Formula

```
Final Score = (0.20 × JD_Fit)
            + (0.30 × Career_Evidence)
            + (0.25 × Profile_Coherence)
            + (0.15 × Platform_Trust)
            + (0.10 × Availability)
            − Penalties
```

Each component is independently computable, testable, and explainable.

### Design Principles

| # | Principle | Implication |
|---|-----------|-------------|
| 1 | Recruiter-first | Every score component maps to a hiring decision a human would make |
| 2 | Explainable by default | No rank without a human-readable reasoning string |
| 3 | Offline & reproducible | Single command reproduces output from provided data only |
| 4 | JD-driven | System accepts any job description, not hardcoded for one role |
| 5 | Trap-aware | Explicit detection and penalization of adversarial profile patterns |
| 6 | Modular | Each scorer independently testable and replaceable |

---

## 4. Hackathon Context & Submission Requirements

### What Redrob Asks For

- Read a JD and understand what the role actually needs (not keyword extraction).
- Evaluate the full candidate picture: career history, skills, behavioral signals, platform activity.
- Deliver a shortlist recruiters can trust.
- Architecture is open: semantic search, LLM ranking, embeddings, hybrid scoring.

### Required Deliverables

| # | Item | Format |
|---|------|--------|
| 1 | GitHub repository | Clean, complete, working code |
| 2 | Approach deck | PPT → PDF |
| 3 | Ranked output | CSV per spec |

### Output CSV Specification

| Column | Rule |
|--------|------|
| `candidate_id` | Format `CAND_XXXXXXX` (7 digits) |
| `rank` | Integer 1–100, each used exactly once |
| `score` | Float, non-increasing by rank |
| `reasoning` | Short explanation per candidate |

**Tie-break rule:** Equal scores → lower `candidate_id` wins (ascending).

### Hard Technical Constraints

| Constraint | Value |
|------------|-------|
| Compute | CPU only — no GPU inference |
| Network | No API calls during ranking |
| Runtime | ≤ 5 minutes on 16 GB RAM |
| Input | Provided `candidates.jsonl` only |
| Reproducibility | One `reproduce_command` → deterministic output |

---

## 5. Dataset Overview & Key Insights

### Key Statistics

| Metric | Value |
|--------|-------|
| Total candidates | 100,000 |
| Countries | 8 (75.1% India) |
| Unique current titles | ~47 |
| Explicit AI/ML titled candidates | ~900 (0.9%) |
| Skills per candidate | 5–23 (mean 9.6) |
| Profiles with platform skill assessments | ~24.2% |
| No GitHub linked (score = -1) | ~64.6% |
| Open to work | ~35.3% |
| Both email + phone verified | ~44.5% |
| Career description entries shared by 50+ candidates | ~300,000 |

### Six Critical Data Insights

#### Insight 1 — Keyword stuffing (~3,865 profiles)
Non-ML titled candidates (HR Manager, Accountant, Content Writer) list 7+ AI/ML skills. The sample submission ranks these near the top — this is the primary **intentional anti-pattern**.

#### Insight 2 — Title ≠ career work (~6,400 profiles)
Current job titles frequently mismatch career descriptions. Example: "Project Manager" whose current role description is about brand design work in Adobe and Figma.

#### Insight 3 — Career descriptions are templated (~300K entries)
The top 5 description templates each appear 25,000+ times across candidates. These are synthetically assigned, not unique histories. Raw text depth scoring is unreliable without template classification.

#### Insight 4 — Genuine ML candidates are rare but distinctive (~900)
Real ML practitioners show: ML title + ML-heavy current role description + trusted skills (duration ≥ 12 months + endorsements > 0) + platform assessments on claimed skills. Example Rank #1 profile: Senior NLP Engineer who fine-tuned LLaMA-2 with LoRA/QLoRA, assessed on Weaviate (79) and RL (90).

#### Insight 5 — Shallow AI boilerplate (~6,000+ summaries)
Templates like "curious about AI, experimenting with ChatGPT, AI tools could augment my work" indicate AI-curious non-practitioners. These should trigger a penalty, not a bonus.

#### Insight 6 — Redrob signals are underused
`skill_assessment_scores`, `github_activity_score`, `saved_by_recruiters_30d`, `interview_completion_rate`, `offer_acceptance_rate` are verified behavioral signals. ~75% of teams will ignore most of these.

---

## 6. System Architecture

### Pipeline Overview

```
┌─────────────────────┐
│  job_description.txt│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐         ┌──────────────────────┐
│  Phase 3: JD Parser │────────▶│  RoleProfile (JSON)  │
└─────────────────────┘         └──────────┬───────────┘
                                            │
┌─────────────────────┐                    │
│  candidates.jsonl   │                    │
└──────────┬──────────┘                    │
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────┐
│  Phase 4: Title Gate & Domain Classification         │
│  Phase 5: Career Evidence Analyzer                   │
│  Phase 6: Profile Coherence Engine                   │
│  Phase 7: Skill Trust & Penalty Layer                │
│  Phase 8: Redrob Platform Signals                    │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  Phase 9: Score Aggregation → Rank → Reasoning       │
└──────────────────────────┬───────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
┌──────────────────┐             ┌─────────────────────┐
│  submission.csv  │             │  audit_report.json  │
└──────────────────┘             └─────────────────────┘
```

### Module Responsibilities

| Module | Phase | Responsibility |
|--------|-------|----------------|
| `fitrank/models.py` | 2 | Candidate + RoleProfile dataclasses |
| `fitrank/loader.py` | 2 | JSONL streaming loader |
| `fitrank/jd_parser.py` | 3 | JD → structured RoleProfile |
| `fitrank/title_gate.py` | 4 | Title domain classification |
| `fitrank/career_analyzer.py` | 5 | ML depth patterns, template classification, trajectory |
| `fitrank/coherence.py` | 6 | Cross-field consistency scoring |
| `fitrank/skill_trust.py` | 7 | Trusted skills, inflation detection |
| `fitrank/penalties.py` | 7 | Honeypot & boilerplate penalty rules |
| `fitrank/signals.py` | 8 | Redrob platform signal composite |
| `fitrank/ranker.py` | 9 | Weighted score aggregation |
| `fitrank/reasoning.py` | 9 | Human-readable explanation builder |
| `rank.py` | 9 | CLI entry point |
| `app/streamlit_app.py` | 10 | Demo UI |

---

## 7. Scoring Methodology

### Component A — JD Fit (Weight: 0.20)

Measures how well the candidate matches what the role explicitly requires.

```
jd_fit = 0.40 × title_jd_match
       + 0.30 × capability_match
       + 0.20 × assessment_jd_overlap
       + 0.10 × education_relevance
```

- `title_jd_match`: 1.0 if current title is in `RoleProfile.target_titles`, else partial match
- `capability_match`: fraction of JD required capabilities found in career text or skills
- `assessment_jd_overlap`: fraction of JD capabilities that have an assessment score ≥ 60
- `education_relevance`: CS/AI field score + institution tier weight

### Component B — Career Evidence (Weight: 0.30)

Measures depth of real ML work demonstrated in career history text.

```
career_evidence = 0.50 × normalize(all_career_ml_depth)
                + 0.35 × normalize(current_role_ml_depth)
                + 0.15 × career_momentum
```

- `all_career_ml_depth`: count of deep ML technical patterns across all role descriptions
- `current_role_ml_depth`: same, restricted to current (is_current=True) role only
- `career_momentum`: upward company-size progression slope across career history

Deep ML patterns scored: LoRA/QLoRA/PEFT, FAISS/Milvus/Weaviate, PyTorch/TensorFlow/JAX, MLflow/W&B, model deployment, RAG, inference serving, fine-tuning pipelines, eval harness, A/B test on model.

### Component C — Profile Coherence (Weight: 0.25)

Detects cross-field consistency — the core differentiator.

```
coherence = 0.40 × template_coherence
          + 0.35 × title_domain_confirmation
          + 0.25 × skill_career_alignment
```

- `template_coherence`: 1.0 if current role description template domain matches title domain, 0.0 if mismatch
- `title_domain_confirmation`: bonus when title is in the ML/AI category
- `skill_career_alignment`: fraction of claimed ML skills corroborated by career description text

**Hard cap:** If title_domain == non-tech AND career_ml_depth < 2 AND current_role_ml_depth < 1, coherence is capped at 0.15 regardless of other signals.

### Component D — Platform Trust (Weight: 0.15)

Uses Redrob behavioral signals to verify profile claims.

```
platform_trust = 0.35 × assessment_avg_norm
               + 0.20 × github_norm
               + 0.30 × engagement_composite
               + 0.15 × verification_score
```

- `assessment_avg_norm`: mean of all `skill_assessment_scores` / 100
- `github_norm`: `github_activity_score` / 100 (0 if score is -1)
- `engagement_composite`: weighted combination of recruiter saves, profile views, search appearances, interview completion, offer acceptance
- `verification_score`: verified_email + verified_phone flags

### Component E — Availability (Weight: 0.10)

Measures how accessible and ready the candidate is to engage.

```
availability = 0.40 × open_to_work_flag
             + 0.30 × (1 - notice_period_norm)
             + 0.30 × willing_to_relocate_or_remote_match
```

### Penalties (Subtracted from final score)

```
penalties = 0.25 × template_mismatch_flag
          + 0.20 × skill_inflation_risk
          + 0.15 × shallow_ai_boilerplate_norm
          + 0.10 × expert_zero_endorsement_flag
```

### Final Score

```
raw_score = 0.20×jd_fit + 0.30×career_evidence + 0.25×coherence
          + 0.15×platform_trust + 0.10×availability − penalties

final_score = clamp(raw_score, 0.0, 1.0)
```

### Tie-Breaking

1. Sort by `final_score` descending.
2. Equal scores → sort by `candidate_id` ascending (per submission rules).
3. Map top-100 scores to non-increasing floats (linear normalization within top 100).

---

## 8. Differentiating Factors

| # | Feature | Why Others Miss It |
|---|---------|-------------------|
| D1 | Profile Coherence Engine | Most teams score fields independently; we require cross-field agreement |
| D2 | Template-aware career analysis | ~300K entries are shared templates; treating text as unique gives noisy signal |
| D3 | Skill trust chain | Raw skill count is gamed; we require duration ≥ 12mo + endorsements > 0 together |
| D4 | JD-conditioned scoring via `--jd` flag | Most teams hardcode role assumptions |
| D5 | Platform engagement composite | 5 Redrob behavioral signals combined as readiness multiplier |
| D6 | Career momentum score | Company size progression rarely computed; maps directly to "full career history" requirement |
| D7 | Explicit honeypot reasoning | Reasoning shows *why a trap was excluded*, not just *why top candidates were included* |
| D8 | Shortlist diversity audit | Post-ranking quality audit demonstrates product thinking |
| D9 | Streamlit explainability demo | Score breakdown bars per component — "shortlist a recruiter can trust" made visible |

---

## 9. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | Performance | ≤ 5 min for 100K candidates, 16 GB RAM, CPU-only |
| NFR-2 | Reproducibility | Same inputs → same output every run (deterministic) |
| NFR-3 | Portability | Runs on Windows, macOS, Linux with single `pip install -r requirements.txt` |
| NFR-4 | Modularity | Each scorer module independently importable and testable |
| NFR-5 | Explainability | Every ranked row has a structured, non-generic reasoning string |
| NFR-6 | Compliance | Zero network calls during `rank.py` execution |
| NFR-7 | Documentation | README covers install, run, reproduce, architecture |
| NFR-8 | Testability | Each phase has unit tests covering normal cases, edge cases, and trap profiles |
| NFR-9 | Fail-safety | Invalid input → clear error with guidance, not silent wrong output |

---

## 10. Implementation Phases

Each phase below defines:
- **Goal** — what it builds
- **Steps** — granular implementation tasks
- **Inputs / Outputs** — what goes in and what comes out
- **Exit Criteria** — definition of "done"
- **Test Cases** — specific tests that must pass before moving to the next phase

---

### Phase 1 — Project Scaffolding & Environment

**Goal:** Establish a clean, reproducible Python project that any contributor can set up from scratch in one command.

**Estimated time:** 2 hours

#### Steps

| Step | Task | Detail |
|------|------|--------|
| 1.1 | Create repo directory structure | `fitrank/`, `tests/`, `app/`, `scripts/`, `config/`, `outputs/`, `data/`, `docs/`, `deck/` |
| 1.2 | Create `requirements.txt` | Pin versions: `python>=3.11`, `pytest`, `streamlit`, `pyyaml`; optional `sentence-transformers` |
| 1.3 | Create `README.md` stub | Install, quickstart, reproduce command |
| 1.4 | Create `config/weights.yaml` | Default scoring weights as defined in Section 7 |
| 1.5 | Create `fitrank/__init__.py` | Package init |
| 1.6 | Create `.gitignore` | Exclude `*.jsonl` (LFS), `__pycache__`, `outputs/`, `.env` |
| 1.7 | Create `Makefile` or `scripts/setup.sh` | One-command environment setup |
| 1.8 | Add `submission_metadata.yaml` stub | Pre-fill team info, compute specs |
| 1.9 | Verify Git LFS tracking for `data/candidates.jsonl` | Prevents repo bloat |

#### Inputs
- Empty repository

#### Outputs
- Importable `fitrank` package
- Installable environment via `pip install -r requirements.txt`
- `config/weights.yaml` with all scoring defaults

#### Exit Criteria
- `pip install -r requirements.txt` completes without errors on a clean Python 3.11 environment
- `python -c "import fitrank; print('OK')"` prints `OK`
- Directory structure matches Section 11 (Repository Structure)

#### Test Cases

| ID | Test | Input | Expected Output | Pass Condition |
|----|------|-------|-----------------|----------------|
| T1.1 | Package import | `import fitrank` | No ImportError | Module loads cleanly |
| T1.2 | Config load | `yaml.safe_load(open('config/weights.yaml'))` | Dict with keys: `jd_fit`, `career_evidence`, `coherence`, `platform_trust`, `availability` | All 5 weight keys present; values sum to ~1.0 |
| T1.3 | Environment completeness | `pip check` | No broken dependencies | Exit code 0 |
| T1.4 | Weights sum check | Load weights from config | Sum of component weights | `abs(sum - 1.0) < 0.001` |
| T1.5 | Directory structure | `os.path.exists()` for each required folder | All return True | All 9 directories present |
| T1.6 | Git LFS tracking | `git lfs ls-files` | `data/candidates.jsonl` appears | LFS pointer exists |
| T1.7 | Metadata stub validity | Load `submission_metadata.yaml` | Parseable YAML | No YAML parse errors |

---

### Phase 2 — Data Layer & Candidate Models

**Goal:** Build a typed, validated data loading layer that streams 100K candidates from JSONL without loading everything into memory simultaneously, and defines all data models used throughout the system.

**Estimated time:** 3 hours

#### Steps

| Step | Task | Detail |
|------|------|--------|
| 2.1 | Define `Candidate` dataclass in `fitrank/models.py` | Typed fields for all schema sections: profile, career_history, education, skills, certifications, languages, redrob_signals |
| 2.2 | Define `CareerEntry` dataclass | company, title, start_date, end_date, duration_months, is_current, industry, company_size, description |
| 2.3 | Define `Skill` dataclass | name, proficiency, endorsements, duration_months |
| 2.4 | Define `RedrobSignals` dataclass | All 25 signal fields from schema |
| 2.5 | Define `RoleProfile` dataclass | target_titles, required_capabilities, nice_to_have, seniority, min_experience_years, preferred_work_mode, domain |
| 2.6 | Define `CandidateScore` dataclass | candidate_id, jd_fit, career_evidence, coherence, platform_trust, availability, penalties, final_score, reasoning |
| 2.7 | Implement `fitrank/loader.py` | Generator-based JSONL streaming loader; parse + validate each line; skip malformed lines with warning |
| 2.8 | Add `load_sample()` helper | Loads `data/sample_candidates.json` for fast dev iteration |
| 2.9 | Add schema validation warnings | Log warnings for missing optional fields; raise errors on missing required fields |

#### Inputs
- `data/candidates.jsonl` (100K lines)
- `data/sample_candidates.json`
- `data/candidate_schema.json`

#### Outputs
- `Candidate`, `RoleProfile`, `CandidateScore` dataclasses
- `load_candidates(path)` generator function
- `load_sample()` function

#### Exit Criteria
- All 100,000 candidates load without errors
- Malformed lines are skipped with a warning log (not a crash)
- Memory usage during streaming stays below 2 GB (no full-list materialization required for scoring)

#### Test Cases

| ID | Test | Input | Expected Output | Pass Condition |
|----|------|-------|-----------------|----------------|
| T2.1 | Load sample | `list(load_candidates('data/sample_candidates.json'))` | List of `Candidate` objects | Length == number of entries in sample file; no exceptions |
| T2.2 | Required field presence | First loaded candidate | `candidate_id`, `profile`, `career_history`, `skills`, `redrob_signals` present | All required fields accessible as typed attributes |
| T2.3 | Type correctness | `c.profile.years_of_experience` | Float | `isinstance(val, float)` is True |
| T2.4 | Malformed line handling | JSONL with one corrupted line injected | Remaining valid candidates load | Warning logged; total count = expected − 1 |
| T2.5 | Career history ordering | Sort `career_history` by `start_date` | Chronological list | Latest entry last; `is_current=True` entry has null `end_date` |
| T2.6 | Signals completeness | `c.redrob_signals.github_activity_score` | Float or -1 | No AttributeError; -1 is valid |
| T2.7 | Generator memory | Stream all 100K with `for c in load_candidates(...)` | No MemoryError | Peak memory < 2 GB (monitored) |
| T2.8 | CandidateScore defaults | `CandidateScore(candidate_id='CAND_0000001')` | Object with all scores defaulting to 0.0 | No missing field errors |
| T2.9 | Candidate ID format | All loaded IDs | Match `CAND_[0-9]{7}` regex | 100% match rate |

---

### Phase 3 — Job Description Parser

**Goal:** Transform a plain-text job description into a structured `RoleProfile` that drives all subsequent scoring. The system must work for any JD, not just the one hidden in the challenge.

**Estimated time:** 4 hours

#### Steps

| Step | Task | Detail |
|------|------|--------|
| 3.1 | Implement `fitrank/jd_parser.py` | Main `parse_jd(text: str) → RoleProfile` function |
| 3.2 | Title extraction | Regex + keyword match for known ML/AI title patterns; fall back to noun phrases |
| 3.3 | Capability extraction | Extract required skills/capabilities using curated keyword lists + surrounding sentence context |
| 3.4 | Seniority detection | Patterns: "senior", "lead", "staff", "principal", "junior", "entry-level", years of experience requirements |
| 3.5 | Experience years parsing | Extract numeric experience requirements ("5+ years", "at least 4 years") |
| 3.6 | Domain classification | Classify JD domain: NLP, CV, GenAI, MLOps, Data Science, etc. |
| 3.7 | Work mode extraction | Remote / hybrid / onsite / flexible |
| 3.8 | Fallback default profile | If JD is empty or unparseable, use a sensible ML engineer default profile with a warning |
| 3.9 | Save parsed profile | Write `outputs/role_profile.json` for inspection and reproducibility |
| 3.10 | Create `data/job_description.txt` | Placeholder JD for development based on `docs/job_description.docx` content |

#### Inputs
- `job_description.txt` (plain text)

#### Outputs
- `RoleProfile` object
- `outputs/role_profile.json` (saved for inspection)

#### Exit Criteria
- Parser extracts ≥ 3 meaningful capabilities from any reasonable JD text
- Changing the JD text changes at least one field in `RoleProfile`
- Fallback profile activates gracefully when JD is empty (with logged warning)

#### Test Cases

| ID | Test | Input | Expected Output | Pass Condition |
|----|------|-------|-----------------|----------------|
| T3.1 | Basic capability extraction | JD text containing "LLM fine-tuning, RAG, PyTorch" | `required_capabilities` list | All 3 appear in output |
| T3.2 | Seniority detection — senior | JD: "We're looking for a Senior ML Engineer" | `seniority == "senior"` | Exact match |
| T3.3 | Seniority detection — junior | JD: "Junior Data Scientist, 0-2 years" | `seniority == "junior"` | Exact match |
| T3.4 | Experience years | JD: "5+ years of experience in machine learning" | `min_experience_years >= 5.0` | Value ≥ 5.0 |
| T3.5 | Domain classification | JD focused on NLP, transformers, BERT | `domain` contains "NLP" | Case-insensitive match |
| T3.6 | Work mode — remote | JD: "This is a fully remote role" | `preferred_work_mode == "remote"` | Exact match |
| T3.7 | Empty JD fallback | `parse_jd("")` | Returns default `RoleProfile`; warning logged | No exception; `target_titles` non-empty |
| T3.8 | Title extraction | JD: "hiring a Machine Learning Engineer or Data Scientist" | Both appear in `target_titles` | Both present in list |
| T3.9 | Role profile JSON save | Parse any JD | `outputs/role_profile.json` created | File exists; valid JSON |
| T3.10 | Different JDs → different profiles | Two distinct JD texts | Two distinct `RoleProfile` objects | At least one field differs between the two outputs |

---

### Phase 4 — Title Gate & Domain Classifier

**Goal:** Assign every candidate a title domain (ML/AI, software, ai-adjacent, non-tech) and compute a `title_jd_match` score against the parsed role profile. This is the first and most powerful filter.

**Estimated time:** 3 hours

#### Steps

| Step | Task | Detail |
|------|------|--------|
| 4.1 | Implement `fitrank/title_gate.py` | `classify_title(title: str) → TitleDomain` |
| 4.2 | Define `TitleDomain` enum | `ML_AI`, `SOFTWARE`, `AI_ADJACENT`, `NON_TECH` |
| 4.3 | Build ML_AI title pattern list | See Appendix A for complete list |
| 4.4 | Build SOFTWARE title patterns | software engineer, full stack, backend, frontend, cloud, devops, platform |
| 4.5 | Build AI_ADJACENT patterns | computer vision engineer, search engineer, recommendation systems, data engineer with ML, senior software engineer (ML) |
| 4.6 | Build NON_TECH catch-all | Everything else (hr, accountant, project manager, operations, etc.) |
| 4.7 | Implement `title_jd_match()` | Fuzzy match candidate title against `RoleProfile.target_titles`; return 0.0–1.0 |
| 4.8 | Implement domain bonus | `ML_AI` → 1.0 bonus; `AI_ADJACENT` → 0.5; `SOFTWARE` → 0.3; `NON_TECH` → 0.0 |
| 4.9 | Handle edge cases | "AI Specialist" title — classify as AI_ADJACENT (higher trap risk; apply coherence checks) |

#### Inputs
- Candidate `current_title` string
- `RoleProfile.target_titles`

#### Outputs
- `TitleDomain` enum per candidate
- `title_jd_match` float (0.0–1.0)
- `title_domain_bonus` float (0.0–1.0)

#### Exit Criteria
- All ~900 explicit ML titles classified as `ML_AI`
- All ~82,000+ non-tech titles classified as `NON_TECH`
- "AI Specialist" classified as `AI_ADJACENT` (not `ML_AI`)
- Title matching produces non-zero scores for near-matches (e.g., "Machine Learning Engineer" vs "ML Engineer")

#### Test Cases

| ID | Test | Input | Expected Output | Pass Condition |
|----|------|-------|-----------------|----------------|
| T4.1 | ML title classification | "Senior NLP Engineer" | `TitleDomain.ML_AI` | Exact |
| T4.2 | Data scientist | "Data Scientist" | `TitleDomain.ML_AI` | Exact |
| T4.3 | HR manager | "HR Manager" | `TitleDomain.NON_TECH` | Exact |
| T4.4 | Accountant | "Accountant" | `TitleDomain.NON_TECH` | Exact |
| T4.5 | Software engineer | "Full Stack Developer" | `TitleDomain.SOFTWARE` | Exact |
| T4.6 | AI Specialist trap | "AI Specialist" | `TitleDomain.AI_ADJACENT` | Exact (not ML_AI) |
| T4.7 | Computer Vision | "Computer Vision Engineer" | `TitleDomain.ML_AI` | Exact |
| T4.8 | JD match — exact | Candidate: "ML Engineer"; JD targets: ["ML Engineer"] | `title_jd_match ≥ 0.9` | Score high |
| T4.9 | JD match — near | Candidate: "Machine Learning Engineer"; JD: ["ML Engineer"] | `title_jd_match ≥ 0.6` | Score medium-high |
| T4.10 | JD match — miss | Candidate: "HR Manager"; JD: ["ML Engineer"] | `title_jd_match ≤ 0.1` | Score near-zero |
| T4.11 | Domain bonus correctness | ML_AI title | `title_domain_bonus == 1.0` | Exact |
| T4.12 | All 100K classified | Stream entire `candidates.jsonl` | No `None` domain | Zero null results |

---

### Phase 5 — Career Evidence Analyzer

**Goal:** Extract real ML work evidence from career history descriptions, classify description templates, compute career trajectory momentum, and produce a `career_evidence` score that is immune to template noise.

**Estimated time:** 5 hours

#### Steps

| Step | Task | Detail |
|------|------|--------|
| 5.1 | Implement `fitrank/career_analyzer.py` | Main `analyze_career(candidate) → CareerEvidence` |
| 5.2 | Build deep ML pattern library | ~20 regex patterns for frameworks, techniques, production work (see Appendix C) |
| 5.3 | Build shallow AI boilerplate patterns | ~8 regex patterns for "curious about AI" variants (see Appendix B) |
| 5.4 | Implement `score_career_ml_depth()` | Count deep ML pattern hits across all role descriptions |
| 5.5 | Implement `score_current_role_depth()` | Same, restricted to `is_current=True` role only |
| 5.6 | Implement template domain classifier | Hash/prefix-match description against known templates; assign domain label |
| 5.7 | Build template domain map | Map ~20 known template prefixes to domains: sales, support, marketing, accounting, brand_design, ml_work, data_engineering, mechanical, consulting |
| 5.8 | Implement `score_career_momentum()` | Convert company_size strings to ordinal integers; compute slope of size values over time |
| 5.9 | Implement `get_current_template_domain()` | Returns domain of current role's description template |
| 5.10 | Normalize all subscores to 0.0–1.0 | Use sigmoid or min-max normalization against dataset-derived max values |

#### Inputs
- Candidate `career_history` list
- Deep ML pattern library
- Template domain map

#### Outputs
- `CareerEvidence` dataclass: `all_career_ml_depth`, `current_role_ml_depth`, `career_momentum`, `template_domain`, `shallow_ai_count`, normalized scores

#### Exit Criteria
- Known genuine ML profiles score `all_career_ml_depth ≥ 5`
- Trap profiles with template-only ML mentions score `current_role_ml_depth == 0`
- Template domain correctly identified for ≥ 95% of top-5 common templates
- Career momentum score is positive for profiles with company size progression

#### Test Cases

| ID | Test | Input | Expected Output | Pass Condition |
|----|------|-------|-----------------|----------------|
| T5.1 | Deep ML depth — genuine | Career with "fine-tuned LLaMA-2 using LoRA and QLoRA; deployed via BentoML" | `all_career_ml_depth ≥ 5` | Score high |
| T5.2 | Deep ML depth — stuffed | Career with "curious about AI, ChatGPT tools" only | `all_career_ml_depth == 0` | Score zero |
| T5.3 | Current role focus | Candidate with ML in old roles only, non-ML current role | `current_role_ml_depth == 0`; `all_career_ml_depth > 0` | Scores differ correctly |
| T5.4 | Template classification — support | Description: "Customer support team lead at a SaaS product. Managed a team of 8 support agents..." | `template_domain == "support"` | Correct classification |
| T5.5 | Template classification — marketing | Description: "Marketing leadership role at a B2B SaaS company. Owned the demand-generation function..." | `template_domain == "marketing"` | Correct classification |
| T5.6 | Template classification — ML | Description: "Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA..." | `template_domain == "ml_work"` | Correct classification |
| T5.7 | Career momentum — growing | career: 11-50 → 201-500 → 5001-10000 | `career_momentum > 0` | Positive slope |
| T5.8 | Career momentum — flat | career: all roles at 10001+ | `career_momentum == 0` | Near-zero or exact zero |
| T5.9 | Career momentum — declining | career: 10001+ → 201-500 → 11-50 | `career_momentum < 0` | Negative slope |
| T5.10 | Shallow boilerplate detection | Summary: "I've been curious about AI and experimenting with ChatGPT" | `shallow_ai_count >= 1` | At least one pattern hit |
| T5.11 | Shallow boilerplate — genuine | Summary describing real model training pipeline | `shallow_ai_count == 0` | Zero hits |
| T5.12 | All career entries processed | Candidate with 10 career entries | All entries analyzed | No IndexError; no skipped entries |

---

### Phase 6 — Profile Coherence Engine

**Goal:** Compute how consistently a candidate's title, career description domain, and skills align with each other. This is the primary honeypot-detection mechanism and the most unique feature of FitRank.

**Estimated time:** 4 hours

#### Steps

| Step | Task | Detail |
|------|------|--------|
| 6.1 | Implement `fitrank/coherence.py` | Main `compute_coherence(candidate, career_evidence, title_domain) → CoherenceScore` |
| 6.2 | Implement `template_coherence_score()` | Compare `career_evidence.template_domain` to `title_domain`; return 1.0 for match, 0.0 for mismatch |
| 6.3 | Define domain compatibility matrix | Which title domains are compatible with which template domains (e.g., ML_AI + ml_work = 1.0; NON_TECH + ml_work = 0.0) |
| 6.4 | Implement `skill_career_alignment()` | Fraction of candidate's claimed ML skills that appear in career descriptions (by name or related keyword) |
| 6.5 | Implement hard cap rule | If NON_TECH title AND career_ml_depth < 2 AND current_role_ml_depth < 1 → cap coherence at 0.15 |
| 6.6 | Compute `title_domain_confirmation` | Bonus: 1.0 for ML_AI; 0.5 for AI_ADJACENT; 0.3 for SOFTWARE with ML career evidence; 0.0 for NON_TECH |
| 6.7 | Aggregate coherence score | Weighted sum per formula in Section 7 Component C |
| 6.8 | Add honeypot flag | Boolean `is_honeypot` = True when coherence < 0.2 AND title_domain == NON_TECH AND career_ml_depth ≥ 5 |

#### Inputs
- `TitleDomain` from Phase 4
- `CareerEvidence` from Phase 5
- Candidate `skills` list

#### Outputs
- `CoherenceScore` dataclass: `template_coherence`, `title_domain_confirmation`, `skill_career_alignment`, `coherence_score`, `is_honeypot`

#### Exit Criteria
- Known trap profiles (e.g., HR Manager with 9 AI skills) receive `coherence_score ≤ 0.15`
- Known genuine profiles (Senior NLP Engineer with matching career text) receive `coherence_score ≥ 0.80`
- `is_honeypot` flag correctly identifies profiles where skills are stuffed but career and title mismatch

#### Test Cases

| ID | Test | Input | Expected Output | Pass Condition |
|----|------|-------|-----------------|----------------|
| T6.1 | Perfect coherence | ML_AI title + ml_work template + ML skills with career mentions | `coherence_score ≥ 0.85` | High score |
| T6.2 | Hard cap trigger | NON_TECH title + career_ml_depth=0 + current_role_ml_depth=0 | `coherence_score ≤ 0.15` | Cap enforced |
| T6.3 | Honeypot detection | HR Manager + 10 AI skills + support template career | `is_honeypot == True` | Flag set |
| T6.4 | Template mismatch | Project Manager title + ml_work template career | `template_coherence == 0.0` | Zero (PM title ≠ ML work) |
| T6.5 | Template match | Data Scientist title + ml_work template career | `template_coherence == 1.0` | Perfect match |
| T6.6 | Skill-career alignment high | Claims PyTorch; career desc mentions "PyTorch training loop" | `skill_career_alignment ≥ 0.7` | Alignment confirmed |
| T6.7 | Skill-career alignment low | Claims 12 ML skills; career desc is customer support | `skill_career_alignment ≤ 0.1` | Near-zero |
| T6.8 | SOFTWARE with ML evidence | "Senior Software Engineer (ML)" + ML career descriptions | `coherence_score ≥ 0.50` | Not penalized |
| T6.9 | AI_ADJACENT score | "AI Specialist" + mixed ML/non-ML career | `0.20 ≤ coherence_score ≤ 0.70` | Reasonable range |
| T6.10 | Sample trap from submission | `CAND_0004989` (Project Manager, sample rank 1) | `coherence_score ≤ 0.15` | Trap correctly scored low |
| T6.11 | Sample genuine | `CAND_0033861` (Senior NLP Engineer, our probe rank 1) | `coherence_score ≥ 0.80` | Genuine scored high |

---

### Phase 7 — Skill Trust & Penalty Layer

**Goal:** Distinguish trusted ML skills from inflated or stuffed ones, and compute all penalty signals that will be subtracted from the final score.

**Estimated time:** 3 hours

#### Steps

| Step | Task | Detail |
|------|------|--------|
| 7.1 | Implement `fitrank/skill_trust.py` | `analyze_skills(skills, role_profile) → SkillTrustScore` |
| 7.2 | Build ML skill keyword set | ~60 ML-related skill name keywords (see Appendix C for reference set) |
| 7.3 | Implement `extract_ml_skills()` | Filter skills list to ML-relevant skills by keyword match on `name` |
| 7.4 | Implement trusted skill criterion | ML skill is "trusted" when: `duration_months ≥ 12` AND `endorsements > 0` |
| 7.5 | Compute `trusted_skill_ratio` | `len(trusted_ml_skills) / max(1, len(all_ml_skills))` |
| 7.6 | Implement `fitrank/penalties.py` | `compute_penalties(candidate, career_evidence, coherence) → PenaltyScore` |
| 7.7 | Template mismatch penalty | Binary flag × 0.25 weight |
| 7.8 | Skill inflation risk | Ratio of ML skills claimed as "advanced/expert" with < 6 months duration |
| 7.9 | Shallow AI boilerplate penalty | Normalize `shallow_ai_count` to 0.0–1.0 × 0.15 weight |
| 7.10 | Expert zero endorsement flag | Any skill with `proficiency == "expert"` and `endorsements == 0` → penalty |
| 7.11 | High skill count on non-ML title | NON_TECH title + total ML skills ≥ 7 → additional penalty weight |

#### Inputs
- Candidate `skills` list
- `CareerEvidence` from Phase 5
- `CoherenceScore` from Phase 6
- `RoleProfile` from Phase 3

#### Outputs
- `SkillTrustScore`: `trusted_ml_skills_count`, `all_ml_skills_count`, `trusted_skill_ratio`, `skill_inflation_risk`
- `PenaltyScore`: `template_mismatch`, `skill_inflation`, `shallow_boilerplate`, `expert_zero_endorse`, `non_ml_title_high_skill`, `total_penalty`

#### Exit Criteria
- Profiles with many ML skills listed but low duration and zero endorsements receive high `skill_inflation_risk`
- Profiles with `expert` skills and zero endorsements receive penalty
- Genuine ML profiles with long-duration, endorsed skills receive near-zero penalties

#### Test Cases

| ID | Test | Input | Expected Output | Pass Condition |
|----|------|-------|-----------------|----------------|
| T7.1 | Trusted skill — pass | PyTorch, duration=36, endorsements=12 | Included in trusted_ml_skills | Count increases |
| T7.2 | Trusted skill — fail duration | PyTorch, duration=4, endorsements=12 | NOT in trusted (duration < 12) | Not counted |
| T7.3 | Trusted skill — fail endorse | PyTorch, duration=36, endorsements=0 | NOT in trusted (zero endorsements) | Not counted |
| T7.4 | Trusted skill — fail both | PyTorch, duration=3, endorsements=0 | NOT in trusted | Not counted |
| T7.5 | Skill inflation high | 5 ML skills claimed "advanced", duration < 6 months each | `skill_inflation_risk ≥ 0.8` | High penalty |
| T7.6 | Skill inflation low | 5 ML skills, all duration ≥ 18 months | `skill_inflation_risk ≤ 0.1` | Low penalty |
| T7.7 | Expert zero endorsement | Skill with proficiency="expert", endorsements=0 | `expert_zero_endorse > 0` | Penalty triggered |
| T7.8 | Total penalty — clean profile | Genuine ML engineer with coherent profile | `total_penalty ≤ 0.05` | Near-zero |
| T7.9 | Total penalty — trap profile | HR Manager + 10 AI skills + support career | `total_penalty ≥ 0.40` | High penalty |
| T7.10 | Non-ML title high skill count | NON_TECH title + 9 claimed ML skills | `non_ml_title_high_skill` penalty triggered | Flag set |
| T7.11 | Penalty clamped | Any profile | `total_penalty ≤ 0.70` | Never exceeds 0.70 to avoid negative scores |

---

### Phase 8 — Redrob Signals & Platform Trust

**Goal:** Compute a behavioral trust score from Redrob platform signals that corroborates (or undermines) what the profile claims. This uses data that most competitors will ignore.

**Estimated time:** 3 hours

#### Steps

| Step | Task | Detail |
|------|------|--------|
| 8.1 | Implement `fitrank/signals.py` | `compute_platform_trust(candidate, role_profile) → PlatformTrustScore` |
| 8.2 | Assessment average normalization | `mean(skill_assessment_scores.values()) / 100`; -1 if no assessments |
| 8.3 | Assessment JD overlap | Count assessments whose skill name matches a JD required capability; normalize |
| 8.4 | GitHub score normalization | `github_activity_score / 100`; 0.0 if score is -1 |
| 8.5 | Platform engagement composite | Weighted: `saved_by_recruiters_30d` (0.35) + `profile_views_received_30d` (0.15) + `search_appearance_30d` (0.10) + `interview_completion_rate` (0.25) + `offer_acceptance_rate` (0.15); normalize each component to 0–1 |
| 8.6 | Verification score | `(verified_email + verified_phone + linkedin_connected) / 3` |
| 8.7 | Availability score | `open_to_work_flag × 0.4 + (1 - notice_days/150) × 0.3 + willing_to_relocate × 0.3` |
| 8.8 | Aggregate platform_trust | Weighted sum per formula in Section 7 Component D |
| 8.9 | Handle missing assessment data | Candidates with no assessments receive neutral (0.0) not penalized |
| 8.10 | Normalize engagement signals | Cap raw counts at 95th percentile of dataset distribution to avoid outlier dominance |

#### Inputs
- Candidate `redrob_signals` dict
- `RoleProfile.required_capabilities`
- Dataset-derived normalization constants (95th-percentile values per signal)

#### Outputs
- `PlatformTrustScore`: `assessment_avg`, `assessment_jd_overlap`, `github_norm`, `engagement_composite`, `verification_score`, `availability_score`, `platform_trust`

#### Exit Criteria
- Candidates with no assessments and no GitHub receive `platform_trust` in 0.1–0.3 range (neutral, not penalized)
- Candidates with high assessment scores on JD-relevant skills receive `platform_trust ≥ 0.6`
- Normalization constants calculated once at startup from the full dataset distribution

#### Test Cases

| ID | Test | Input | Expected Output | Pass Condition |
|----|------|-------|-----------------|----------------|
| T8.1 | Assessment average | `{"PyTorch": 85, "NLP": 70}` | `assessment_avg == 0.775` | Correct mean |
| T8.2 | No assessments | Empty `skill_assessment_scores` | `assessment_avg == 0.0` (not -1) | Zero, not error |
| T8.3 | GitHub present | `github_activity_score = 75` | `github_norm == 0.75` | Correct normalization |
| T8.4 | GitHub absent | `github_activity_score = -1` | `github_norm == 0.0` | Zero, not error |
| T8.5 | Verification full | email=True, phone=True, linkedin=True | `verification_score == 1.0` | Exact |
| T8.6 | Verification partial | email=True, phone=False, linkedin=False | `verification_score ≈ 0.33` | ±0.01 tolerance |
| T8.7 | Availability — open, short notice | open_to_work=True, notice=0, relocate=True | `availability_score ≥ 0.90` | High |
| T8.8 | Availability — not open | open_to_work=False, notice=90 | `availability_score ≤ 0.30` | Low |
| T8.9 | Assessment JD overlap | Assessments on "LoRA"; JD requires "LoRA fine-tuning" | `assessment_jd_overlap > 0` | Non-zero |
| T8.10 | Engagement composite — high saves | `saved_by_recruiters_30d = 15` (near 95th pct) | Engagement high | `engagement_composite ≥ 0.60` |
| T8.11 | Outlier capping | `profile_views_received_30d = 9999` | Capped to 95th-pct value | No score above 1.0 |
| T8.12 | Normalization constants | Computed on full 100K dataset | Constants stored in `config/` | File exists; all values > 0 |

---

### Phase 9 — Ranker, Reasoning & Output

**Goal:** Aggregate all component scores into a final ranking, generate human-readable per-candidate reasoning, produce the submission CSV, and run validation. This is the end-to-end integration phase.

**Estimated time:** 5 hours

#### Steps

| Step | Task | Detail |
|------|------|--------|
| 9.1 | Implement `fitrank/ranker.py` | `rank_candidates(candidates, role_profile, weights) → list[CandidateScore]` |
| 9.2 | Load weights from `config/weights.yaml` | Allow CLI override via `--weights` flag |
| 9.3 | Compute all component scores per candidate | Call Phases 4–8 modules; store in `CandidateScore` |
| 9.4 | Compute final score | Apply formula from Section 7; clamp to 0.0–1.0 |
| 9.5 | Sort and select top 100 | Sort descending; tie-break by ascending `candidate_id` |
| 9.6 | Map scores to submission format | Linear normalization within top 100 to ensure non-increasing constraint |
| 9.7 | Implement `fitrank/reasoning.py` | `build_reasoning(score: CandidateScore, candidate: Candidate) → str` |
| 9.8 | Reasoning template | `{title} | JD={jd:.2f} Career={ce:.2f} Coh={coh:.2f} Trust={pt:.2f} | {evidence_note} | {penalty_note}` |
| 9.9 | Implement `rank.py` CLI | `argparse`: `--candidates`, `--jd`, `--out`, `--weights`, `--top-n` |
| 9.10 | Write CSV output | Header + 100 data rows in correct column order |
| 9.11 | Run validator automatically | Call `validate_submission.py` on output; fail loudly if invalid |
| 9.12 | Write audit JSON | `outputs/audit_report.json`: title distribution, score histogram, trap exclusion summary |

#### Inputs
- All Phase 4–8 module outputs
- `config/weights.yaml`
- `candidates.jsonl`
- `RoleProfile` from Phase 3

#### Outputs
- `outputs/submission.csv` — 100 rows, valid format
- `outputs/audit_report.json` — quality audit
- Console: runtime, top-5 candidate summary, validation status

#### Exit Criteria
- `python validate_submission.py outputs/submission.csv` exits with code 0
- Runtime ≤ 5 minutes on full 100K dataset (CPU, 16 GB RAM)
- Top-100 contains 0 known trap profiles from the sample submission's bad top-10
- All 100 reasoning strings are non-generic (contain candidate-specific data values)

#### Test Cases

| ID | Test | Input | Expected Output | Pass Condition |
|----|------|-------|-----------------|----------------|
| T9.1 | CSV row count | Run `rank.py` end-to-end | Exactly 100 data rows + header | Row count == 101 |
| T9.2 | Rank uniqueness | Output CSV ranks | Each rank 1–100 used exactly once | `set(ranks) == set(range(1, 101))` |
| T9.3 | Score non-increasing | Output CSV scores | `score[i] >= score[i+1]` for all i | No violation |
| T9.4 | Tie-break correctness | Two candidates with equal scores | Lower candidate_id has lower rank number | Verified on synthetic tie |
| T9.5 | Validator pass | `validate_submission.py submission.csv` | "Submission is valid." | Exit code 0 |
| T9.6 | Trap exclusion | `CAND_0004989`, `CAND_0000339` (sample bad top-5) | None appear in top 100 | Not present in output |
| T9.7 | Runtime ≤ 5 min | Full 100K candidates | Elapsed time | `elapsed_seconds < 300` |
| T9.8 | Reasoning non-generic | All 100 reasoning strings | Each contains numeric scores | No string is identical to another |
| T9.9 | Deterministic output | Run twice with same inputs | Identical CSVs | `diff output1.csv output2.csv` is empty |
| T9.10 | JD change affects ranking | Run with two different JDs | Different top-100 sets | At least 10 candidates differ |
| T9.11 | Weight override | `--weights custom_weights.yaml` with altered values | Ranking changes | Output differs from default |
| T9.12 | Audit report written | Run end-to-end | `outputs/audit_report.json` exists | Valid JSON; contains title distribution |
| T9.13 | Candidate ID format | All 100 output IDs | Match `CAND_[0-9]{7}` | 100% match |
| T9.14 | No network calls | Run with network disabled (firewall/hosts block) | Completes successfully | No requests exception |

---

### Phase 10 — Demo, Docs & Final QA

**Goal:** Polish the project into a presentable, reproducible submission with an interactive demo, complete documentation, full test suite, and a judge-ready PDF deck.

**Estimated time:** 6 hours

#### Steps

| Step | Task | Detail |
|------|------|--------|
| 10.1 | Implement `app/streamlit_app.py` | JD input → ranked shortlist cards with score breakdown bars |
| 10.2 | Streamlit: score breakdown visualization | Per-candidate bar chart: JD Fit / Career / Coherence / Trust / Availability |
| 10.3 | Streamlit: trap examples toggle | "Show trap profiles" section contrasting bad vs good candidates |
| 10.4 | Streamlit: CSV export button | Download filtered shortlist |
| 10.5 | Write unit test suite `tests/` | Cover all phase exit criteria; `pytest tests/` must pass |
| 10.6 | Write `scripts/audit_shortlist.py` | CLI script producing diversity/quality report from submission CSV |
| 10.7 | Complete `README.md` | Installation, quickstart, reproduce command, architecture diagram, test command |
| 10.8 | Complete `submission_metadata.yaml` | All fields filled: team, compute, AI tools, methodology summary, declarations |
| 10.9 | Build PDF approach deck | 10 slides per outline in Section 16 |
| 10.10 | Fresh-environment reproduction test | Clone repo on clean machine/container; run reproduce command; validate output |
| 10.11 | Performance profiling | Identify slowest module; optimize if > 2 min |
| 10.12 | Final git push | Tag release `v1.0`; ensure all files committed |

#### Inputs
- All prior phase outputs
- `outputs/submission.csv`
- `outputs/audit_report.json`

#### Outputs
- Live Streamlit app (local + optionally hosted)
- Complete `tests/` suite passing
- Completed `README.md`
- Completed `submission_metadata.yaml`
- `deck/Redrob_FitRank_Approach.pdf`
- Tagged GitHub release `v1.0`

#### Exit Criteria
- `pytest tests/ -v` passes with zero failures
- `python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt --out outputs/submission.csv` completes in < 5 min on a clean environment
- `python validate_submission.py outputs/submission.csv` outputs "Submission is valid."
- Streamlit app runs locally with `streamlit run app/streamlit_app.py`
- All deliverable files present: repo, CSV, PDF deck, metadata YAML

#### Test Cases

| ID | Test | Input | Expected Output | Pass Condition |
|----|------|-------|-----------------|----------------|
| T10.1 | Full test suite | `pytest tests/ -v` | All tests pass | Zero failures, zero errors |
| T10.2 | End-to-end reproduce | Fresh clone + `pip install` + reproduce command | Valid `submission.csv` | Validator passes on fresh output |
| T10.3 | Streamlit starts | `streamlit run app/streamlit_app.py` | App runs on localhost | No import error; port opens |
| T10.4 | Streamlit JD input | Paste a JD + click Rank | Ranked cards appear | ≥ 10 candidate cards visible |
| T10.5 | Streamlit score bars | View any candidate card | Score breakdown bars visible | 5 components shown |
| T10.6 | Trap toggle | Click "Show trap examples" | Trap profiles displayed | At least 2 trap examples shown |
| T10.7 | CSV export | Click export in Streamlit | CSV downloads | Valid CSV with ≤ 20 rows |
| T10.8 | Audit script | `python scripts/audit_shortlist.py outputs/submission.csv` | Report printed | Title distribution visible |
| T10.9 | README reproduce command | Copy-paste reproduce command from README | Produces valid CSV | Validator passes |
| T10.10 | Metadata YAML completeness | Load `submission_metadata.yaml` | All required fields present | No missing keys |
| T10.11 | Performance benchmark | Time full pipeline | `elapsed_seconds` | < 300 s; log printed |
| T10.12 | Git tag | `git tag` | `v1.0` present | Tag visible in repo |
| T10.13 | No hardcoded paths | Grep for absolute paths in Python files | None found | Zero matches for `C:\` or `/home/` |
| T10.14 | Sample candidate demo | Load `data/sample_candidates.json` in Streamlit | Rankings computed | No error on sample input |

---

## 11. Repository Structure

```
redrob-fitrank/
├── rank.py                          # CLI entry point (reproduce command)
├── requirements.txt                 # Pinned dependencies
├── submission_metadata.yaml         # Hackathon metadata
├── README.md                        # Full documentation
├── Makefile                         # setup, test, run shortcuts
│
├── config/
│   └── weights.yaml                 # Scoring component weights
│
├── data/
│   ├── candidates.jsonl             # 100K profiles (Git LFS)
│   ├── sample_candidates.json       # Dev sample
│   ├── sample_submission.csv        # Format reference
│   ├── candidate_schema.json        # JSON schema
│   └── job_description.txt          # JD text input
│
├── docs/
│   ├── job_description.docx         # Official JD
│   ├── submission_spec.docx
│   ├── redrob_signals_doc.docx
│   └── README.docx
│
├── fitrank/
│   ├── __init__.py
│   ├── models.py                    # Dataclasses: Candidate, RoleProfile, CandidateScore
│   ├── loader.py                    # JSONL streaming loader
│   ├── jd_parser.py                 # Phase 3: JD → RoleProfile
│   ├── title_gate.py                # Phase 4: Title domain classification
│   ├── career_analyzer.py           # Phase 5: Career evidence + templates
│   ├── coherence.py                 # Phase 6: Cross-field consistency
│   ├── skill_trust.py               # Phase 7: Trusted skills
│   ├── penalties.py                 # Phase 7: Penalty computation
│   ├── signals.py                   # Phase 8: Redrob platform signals
│   ├── ranker.py                    # Phase 9: Score aggregation
│   └── reasoning.py                 # Phase 9: Reasoning builder
│
├── app/
│   └── streamlit_app.py             # Phase 10: Demo UI
│
├── scripts/
│   ├── audit_shortlist.py           # Diversity + trap audit
│   └── validate.sh                  # Validation wrapper
│
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_phase1_setup.py
│   ├── test_phase2_loader.py
│   ├── test_phase3_jd_parser.py
│   ├── test_phase4_title_gate.py
│   ├── test_phase5_career.py
│   ├── test_phase6_coherence.py
│   ├── test_phase7_skills.py
│   ├── test_phase8_signals.py
│   ├── test_phase9_ranker.py
│   └── test_phase10_integration.py
│
├── outputs/
│   ├── submission.csv               # Final ranked output
│   ├── role_profile.json            # Parsed JD profile
│   └── audit_report.json            # Post-ranking quality audit
│
├── deck/
│   └── Redrob_FitRank_Approach.pdf
│
└── validate_submission.py           # From challenge pack
```

---

## 12. Deliverables Checklist

### Required for Submission

- [ ] GitHub repo (public or accessible to organizers)
- [ ] `rank.py` + full `fitrank/` package
- [ ] `outputs/submission.csv` — 100 rows, validator-passing
- [ ] PDF approach deck (`deck/Redrob_FitRank_Approach.pdf`)
- [ ] `README.md` with install + reproduce steps

### Strongly Recommended

- [ ] `submission_metadata.yaml` — fully completed
- [ ] Streamlit or HuggingFace Spaces sandbox link
- [ ] `pytest tests/ -v` — all phases tested
- [ ] `outputs/audit_report.json` — shortlist quality audit
- [ ] Reasoning visible in CSV for all 100 rows

### Pre-Submission QA Checklist

- [ ] `python validate_submission.py outputs/submission.csv` → "Submission is valid."
- [ ] End-to-end reproduce command runs in ≤ 5 minutes
- [ ] Zero network calls during `rank.py` execution (verified)
- [ ] Top 100 contains zero honeypot profiles
- [ ] Reasoning strings contain candidate-specific numeric values
- [ ] `pytest tests/ -v` → zero failures
- [ ] Repo has no hardcoded absolute paths
- [ ] Git tag `v1.0` pushed

---

## 13. Success Metrics

### Submission Quality

| Metric | Target |
|--------|--------|
| Validator pass | 100% |
| Reproducibility | Deterministic, single command |
| Top-100 trap exclusion | All known honeypots outside top 100 |
| Reasoning coverage | 100 rows with meaningful reasoning |
| Runtime | ≤ 60 s preferred (well under 5 min limit) |

### Code Quality

| Metric | Target |
|--------|--------|
| Test coverage | All 10 phases have passing tests |
| Module independence | Each scorer independently importable |
| JD generalization | Different JDs → meaningfully different rankings |

### Differentiation

| Metric | Target |
|--------|--------|
| Coherence feature | Implemented, tested, explained in deck |
| Template-aware analysis | Template domain classified for ≥ 95% of profiles |
| Redrob signals used | ≥ 5 distinct signals in platform trust score |
| Honeypot flag | Explicitly surfaced in reasoning and audit report |

---

## 14. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Hidden JD differs from assumed ML role | Medium | High | JD-driven parser; never hardcode role |
| Template classifier misses new templates | Low | Medium | Fallback to raw ML depth if unclassified |
| Over-filtering to ML titles misses transitions | Medium | Medium | SOFTWARE + ML career evidence override |
| Runtime exceeds 5 min | Low | Critical | Regex patterns precompiled at startup; generator streaming |
| Network dependency introduced accidentally | Low | Critical | CI check: run with firewall; grep for `requests` and `urllib` calls |
| Equal scores break tie rules | Low | High | Deterministic sort by `candidate_id` enforced in ranker |
| Weights produce poor ranking | Medium | High | Configurable `weights.yaml`; ablation test 3 weight configs |
| Reasoning strings too generic | Medium | Medium | Assert uniqueness in T9.8; include numeric values in template |

---

## 15. Future Roadmap

If extended into a Redrob product:

1. **API endpoint** — `POST /rank { jd_text, candidate_ids } → shortlist`
2. **Recruiter feedback loop** — Online weight learning from save/reject signals
3. **Multi-role batch ranking** — Same talent pool ranked against multiple JDs simultaneously
4. **Confidence intervals** — Show rank stability under weight perturbation
5. **"Why not?" explainability** — Per-candidate gap analysis against JD requirements
6. **Real profile enrichment** — Replace synthetic template detection with live signal enrichment

---

## 16. Appendix

### A. ML/AI Title Patterns (Phase 4 — Title Gate)

```
ML Engineer, Senior ML Engineer, Junior ML Engineer, Staff ML Engineer
Machine Learning Engineer, Senior Machine Learning Engineer
Data Scientist, Senior Data Scientist
AI Engineer, Senior AI Engineer, Lead AI Engineer
AI Research Engineer
NLP Engineer, Senior NLP Engineer
Computer Vision Engineer
Search Engineer
Recommendation Systems Engineer
Applied ML Engineer
Senior Software Engineer (ML)
AI Specialist  ← classify as AI_ADJACENT, not ML_AI
```

### B. Shallow AI Boilerplate Patterns (Phase 5 — Penalties)

```
curious about (how )?ai
experimenting with chatgpt
ai tools could augment
taking online courses (on|about)
played with the openai
excited about.*\bai\b
side project.*(rag|langchain)
(rag|langchain).*side project
exploring how llm
```

### C. Deep ML Career Patterns (Phase 5 — Positive Evidence)

```
fine.tun|qlora|lora(?!l)|rlhf|dpo\b|sft\b|instruction.tun
sentence.transformer|all.minilm|bge.base|e5.base
faiss|milvus|weaviate|pinecone|chromadb|qdrant
pytorch|tensorflow|jax\b|keras
mlflow|wandb|weights.and.biases|kubeflow|vertex.ai
model.serv|triton.inference|bentoml|torchserve|onnx
gradient.descent|backprop|loss.function|epoch\b|batch.size
distributed.train|data.parallel|deepspeed|fsdp
semantic.search|vector.search|retrieval.augment|\brag\b
peft\b|accelerate\b|trl\b
production.model|shadow.deploy|ab.test.*model
llama|mistral|gemma|qwen|openai.api|anthropic.api
tokenizer|attention.mask|model.forward|training.loop
eval.harness|benchmark|ablation|hyperparameter.tun
```

### D. Career Description Template Domain Map

| Template Prefix (first 60 chars) | Domain |
|-----------------------------------|--------|
| "Enterprise sales of cloud software solutions..." | sales |
| "Customer support team lead at a SaaS product..." | support |
| "Marketing leadership role at a B2B SaaS company..." | marketing |
| "Business analyst at a consulting firm..." | consulting |
| "Brand design and creative direction..." | brand_design |
| "Mechanical engineering design role..." | mechanical |
| "Senior accounting role at a mid-sized company..." | accounting |
| "Fine-tuned LLaMA-2-7B and Mistral-7B..." | ml_work |
| "Developed a semantic search feature..." | ml_work |
| "Built and maintained data pipelines on Apache Airflow..." | data_engineering |
| "Operations management role at a logistics company..." | operations |

### E. Reproduce Command

```bash
# Install
pip install -r requirements.txt

# Run
python rank.py \
  --candidates data/candidates.jsonl \
  --jd data/job_description.txt \
  --out outputs/submission.csv

# Validate
python validate_submission.py outputs/submission.csv
```

### F. PDF Deck Structure (10 Slides)

| Slide | Title | Content |
|-------|-------|---------|
| 1 | Title | FitRank — Recruiter-Trustable AI Ranking |
| 2 | Problem | Keyword filters fail; why this dataset is adversarial |
| 3 | Data Insights | 6 key findings (traps, templates, 900 real ML profiles) |
| 4 | Core Idea | Profile Coherence > keyword matching; formula |
| 5 | Architecture | 10-phase pipeline diagram |
| 6 | Differentiators | 9 unique features vs typical approaches (table) |
| 7 | Scoring Model | Component weights, formulas, penalty rules |
| 8 | Examples | Rank #1 genuine vs sample rank #1 trap — side by side |
| 9 | Results | Validation pass, runtime, audit report, top-10 list |
| 10 | Redrob Vision | How FitRank becomes a Redrob product feature |

---

## Document Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-06-24 | Initial PRD |
| 2.0 | 2026-06-25 | Restructured into 10 implementation phases with detailed test cases per phase |

---

**End of PRD v2.0**
