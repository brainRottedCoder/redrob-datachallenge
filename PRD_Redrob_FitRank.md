# Product Requirements Document (PRD)
## Redrob FitRank — Intelligent Candidate Discovery & Ranking System

**Project:** Redrob India Runs Data & AI Hackathon  
**Product Name:** FitRank (working title)  
**Version:** 1.0  
**Date:** June 24, 2026  
**Author:** Hackathon Team  
**Status:** Draft for Implementation  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Product Vision & Core Idea](#3-product-vision--core-idea)
4. [Hackathon Context & Submission Requirements](#4-hackathon-context--submission-requirements)
5. [Dataset Overview & Key Insights](#5-dataset-overview--key-insights)
6. [Target Users & Use Cases](#6-target-users--use-cases)
7. [Product Scope](#7-product-scope)
8. [Differentiating Factors](#8-differentiating-factors)
9. [System Architecture](#9-system-architecture)
10. [Scoring Methodology (Deep Specification)](#10-scoring-methodology-deep-specification)
11. [Functional Requirements](#11-functional-requirements)
12. [Non-Functional Requirements](#12-non-functional-requirements)
13. [Implementation Plan & Steps](#13-implementation-plan--steps)
14. [Repository Structure](#14-repository-structure)
15. [Deliverables Checklist](#15-deliverables-checklist)
16. [Demo & PDF Deck Outline](#16-demo--pdf-deck-outline)
17. [Success Metrics](#17-success-metrics)
18. [Risks & Mitigations](#18-risks--mitigations)
19. [Future Roadmap (Post-Hackathon)](#19-future-roadmap-post-hackathon)
20. [Appendix](#20-appendix)

---

## 1. Executive Summary

**FitRank** is an offline, explainable candidate ranking system designed for the Redrob Intelligent Candidate Discovery & Ranking Challenge. It ranks 100,000 synthetic candidate profiles against a job description (JD) the way an experienced recruiter would — by evaluating **cross-field profile coherence**, **career evidence**, **behavioral platform signals**, and **explicit honeypot detection** — rather than keyword overlap.

The core thesis: **the right candidate is not the one who lists the most AI skills, but the one whose title, career narrative, trusted skills, and platform verification all tell the same story.**

### What We Will Deliver

| Deliverable | Description |
|-------------|-------------|
| GitHub Repository | Clean, modular, reproducible Python codebase |
| Ranked Output CSV | Top 100 candidates with scores and reasoning |
| PDF Approach Deck | Problem, architecture, differentiation, examples |
| Optional Sandbox | Streamlit/HuggingFace demo for recruiter-style shortlist review |

---

## 2. Problem Statement

Recruiters reviewing hundreds of profiles often miss strong candidates because:

1. **Keyword filters surface the wrong people** — profiles optimized for search terms rank high without genuine fit.
2. **Skills lists are unreliable** — candidates can claim advanced ML skills without corresponding work history.
3. **Titles lie** — current job titles frequently do not match actual role descriptions.
4. **Boilerplate summaries mislead** — thousands of profiles use identical “curious about ChatGPT” language.
5. **No explainability** — black-box rankings cannot be defended to hiring managers.

The Redrob hackathon dataset encodes all of these failure modes deliberately. A naive ranker that counts AI-related skills will promote **honeypot profiles** (non-ML titles with stuffed skill lists) while burying genuine practitioners.

**Product goal:** Deliver a top-100 shortlist that a recruiter can trust, with transparent reasoning for every rank.

---

## 3. Product Vision & Core Idea

### Vision Statement

> Build an AI-assisted ranking system that understands what a role needs, evaluates the full candidate picture, and produces an explainable shortlist — mirroring how Redrob would productize intelligent discovery for paying recruiter customers.

### Core Idea: Profile Coherence Ranking

FitRank treats candidate ranking as a **consistency verification problem**, not a keyword matching problem.

```
Final Score = JD Fit × Career Evidence × Profile Coherence × Platform Trust − Penalties
```

**Interpretation:**

- **JD Fit** — Does the candidate align with what the role requires (semantic + structured)?
- **Career Evidence** — Does their work history demonstrate real capability (especially current role)?
- **Profile Coherence** — Do title, career descriptions, and skills agree with each other?
- **Platform Trust** — Do Redrob behavioral signals (assessments, GitHub score, recruiter saves) corroborate the profile?
- **Penalties** — Honeypots, template mismatches, shallow AI-curious boilerplate, skill inflation

### Design Principles

1. **Recruiter-first** — Every score component maps to a hiring decision a human would make.
2. **Explainable by default** — No rank without a human-readable reasoning string.
3. **Offline & reproducible** — Single command reproduces output from provided data only.
4. **JD-driven** — System accepts any job description, not hardcoded for one role.
5. **Trap-aware** — Explicit detection and penalization of adversarial profile patterns.

---

## 4. Hackathon Context & Submission Requirements

### Official Hackathon Brief (Summary)

The hackathon asks participants to:

- Read a **job description** and understand what the role needs (not just extract keywords).
- Evaluate the **full candidate picture**: career history, skills, behavioral signals, platform activity.
- Deliver a **shortlist recruiters can trust**.
- Use any architecture: semantic search, LLM ranking, embeddings, hybrid scoring.

### Required Submissions

| # | Deliverable | Format |
|---|-------------|--------|
| 1 | GitHub repository | Clean, complete, working code |
| 2 | Approach deck | PPT converted to PDF |
| 3 | Ranked output | CSV per provided format |

### Output CSV Specification

Validated by `validate_submission.py`:

| Column | Rules |
|--------|-------|
| `candidate_id` | Format `CAND_XXXXXXX` (7 digits) |
| `rank` | Integer 1–100, each used exactly once |
| `score` | Float, non-increasing by rank |
| `reasoning` | Short explanation per candidate |

**Tie-break rule:** When scores are equal, lower `candidate_id` (ascending) wins.

### Technical Constraints (from challenge metadata)

| Constraint | Requirement |
|------------|-------------|
| Compute | CPU only (no GPU inference) |
| Network | No API calls during ranking execution |
| Runtime | ≤ 5 minutes on 16GB RAM |
| Reproducibility | Single `reproduce_command` generates submission from `candidates.jsonl` |
| Input | Provided dataset only at ranking time |

### Optional but Recommended

- `submission_metadata.yaml` — team info, methodology, compute declarations
- Hosted sandbox — Streamlit, HuggingFace Spaces, Colab, etc.

---

## 5. Dataset Overview & Key Insights

### Dataset Files

| File | Purpose |
|------|---------|
| `candidates.jsonl` | 100,000 candidate profiles (primary input) |
| `sample_candidates.json` | Small sample for development/demo |
| `sample_submission.csv` | Format reference (intentionally bad ranking logic) |
| `candidate_schema.json` | JSON schema for profile structure |
| `validate_submission.py` | Submission validator |
| `submission_metadata_template.yaml` | Metadata template |

**Note:** A dedicated `job_description.txt` may be provided separately by organizers. The system must accept JD via CLI flag regardless.

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total candidates | 100,000 |
| Countries | 8 (75.1% India) |
| Unique current titles | ~47 |
| Explicit AI/ML titles | ~900 (0.9%) |
| Skills per candidate | 5–23 (mean ~9.6) |
| Profiles with skill assessments | ~24.2% |
| No GitHub linked (`github_activity_score = -1`) | ~64.6% |
| Open to work | ~35.3% |

### Candidate Profile Schema (Required Fields)

Each candidate contains:

- **`profile`** — headline, summary, location, years_of_experience, current_title, current_company, industry
- **`career_history`** — up to 10 roles with descriptions, dates, company size, industry
- **`education`** — institution, degree, field, tier (tier_1–tier_4)
- **`skills`** — name, proficiency, endorsements, duration_months
- **`certifications`** / **`languages`** — optional
- **`redrob_signals`** — platform activity, assessments, salary, work mode, verification flags

### Critical Data Insights (Drive Product Design)

#### Insight 1: Keyword stuffing is pervasive (~3,865+ profiles)

Non-ML titles (HR Manager, Accountant, Content Writer) list 7+ AI/ML skills. The sample submission ranks these at the top — this is an **intentional anti-pattern**, not a target behavior.

#### Insight 2: Title ≠ career work (~6,400 suspicious profiles)

Current job descriptions often describe unrelated domains:

- Project Manager → brand design work
- Operations Manager → customer support or mechanical engineering
- HR Manager → marketing leadership

**Implication:** Title and career text must be evaluated together, not independently.

#### Insight 3: Career descriptions are templated (~300,000 shared entries)

Top description templates appear 25,000+ times each (sales, support, marketing, brand design). Descriptions are **synthetically assigned**, not unique histories.

**Implication:** Do not treat all career text as unique evidence. Use **template-domain classification** and **title–description domain coherence** instead of raw text depth alone.

#### Insight 4: Genuine ML candidates are rare but identifiable (~900 titles)

Strong candidates exhibit:

- ML/AI titles aligned with ML-heavy current role descriptions
- Trusted skills (duration ≥ 12 months AND endorsements > 0)
- Platform assessment scores on claimed skills
- Non-trivial GitHub activity (when present)

Example top genuine profile pattern:

- Title: Senior NLP Engineer
- Current role: “Fine-tuned LLaMA-2-7B and Mistral-7B using LoRA and QLoRA…”
- Assessments: LoRA (67), Weaviate (79), Reinforcement Learning (90)
- GitHub score: 32.6

#### Insight 5: Shallow AI boilerplate (~6,000+ profiles)

Repeated summary templates: “curious about AI,” “experimenting with ChatGPT,” “AI tools could augment my work.” These indicate **AI-curious non-practitioners**, not ML engineers.

#### Insight 6: Redrob signals are underused by most teams

Available behavioral signals:

- `skill_assessment_scores` — verified platform tests
- `github_activity_score` — engineering activity proxy
- `saved_by_recruiters_30d` — recruiter intent signal
- `profile_views_received_30d`, `search_appearance_30d`
- `interview_completion_rate`, `offer_acceptance_rate`
- `open_to_work_flag`, `notice_period_days`, `willing_to_relocate`
- `verified_email`, `verified_phone`

---

## 6. Target Users & Use Cases

### Primary User: Recruiter / Talent Acquisition Specialist

**Needs:**

- Quickly surface genuinely qualified candidates for a role
- Understand *why* each candidate was ranked
- Avoid false positives from keyword-optimized profiles
- Trust the shortlist enough to share with hiring managers

### Secondary User: Hackathon Evaluator / Redrob Product Team

**Needs:**

- Reproducible, well-documented approach
- Evidence of understanding adversarial data patterns
- Product thinking aligned with Redrob’s recruitment intelligence vision

### Core Use Cases

| ID | Use Case | Description |
|----|----------|-------------|
| UC-1 | Rank for role | Input JD + candidates.jsonl → output top-100 CSV |
| UC-2 | Explain ranking | Each row includes structured reasoning |
| UC-3 | Demo shortlist | Interactive UI to browse ranked candidates |
| UC-4 | Validate output | Run validator before submission |
| UC-5 | Audit shortlist | Review diversity, trap exclusion, score distribution |

---

## 7. Product Scope

### In Scope (MVP for Hackathon)

- JD parsing into structured role requirements
- Multi-signal hybrid scoring pipeline
- Profile coherence and honeypot detection
- Redrob behavioral signal integration
- Explainable reasoning generation
- CLI ranker (`rank.py`)
- Valid submission CSV output
- README + PDF approach deck
- Optional Streamlit sandbox demo

### Out of Scope (Hackathon MVP)

- Live GitHub/LinkedIn scraping
- Real-time API ranking with external LLMs
- User authentication / multi-tenant SaaS
- Production database or persistent storage
- Automated model training on hidden labels
- GPU-based inference pipeline

---

## 8. Differentiating Factors

These features separate FitRank from typical participant submissions.

### Differentiator 1: Profile Coherence Engine

Most teams score skills independently. FitRank requires **cross-field agreement**:

```
coherence = f(title_domain, career_description_domain, trusted_skills, assessments)
```

A profile with 12 AI skills but a non-ML title and mismatched career domain receives a near-zero coherence score regardless of skill count.

### Differentiator 2: Template-Aware Career Analysis

Because ~300K career description entries share templates, FitRank:

1. Classifies description templates into domains (support, marketing, mechanical, ML, sales, etc.)
2. Compares template domain to current title domain
3. Penalizes domain mismatches as honeypot indicators

**Unique insight:** Competitors treating career text as unique evidence will score inconsistently or be fooled by template noise.

### Differentiator 3: Skill Trust Chain (Not Skill Count)

ML skills count only when **both** conditions hold:

- `duration_months >= 12`
- `endorsements > 0`

Additionally penalize:

- `proficiency = expert` with `endorsements = 0`
- High endorsements with very short duration (inflation risk)

### Differentiator 4: JD-Conditioned Semantic Fit

FitRank accepts `--jd job_description.txt` and produces a structured role profile. Candidates are scored against **role intent**, not a fixed assumption. Demonstrates generalization beyond one hidden test role.

### Differentiator 5: Redrob Platform Engagement Composite

Combines multiple `redrob_signals` into a recruiter-realistic readiness score:

```
engagement = f(recruiter_saves, profile_views, search_appearances,
               interview_completion, offer_acceptance)
```

Mirrors what Redrob recruiters see on their dashboard — strong product alignment for judges.

### Differentiator 6: Career Trajectory / Momentum Score

Uses `company_size` progression across career history to detect upward momentum:

```
startup (51-200) → mid (1001-5000) → enterprise (10001+)
```

Rare signal; maps to “full career history” requirement in hackathon brief.

### Differentiator 7: Explicit Honeypot Reasoning

Reasoning strings include both **positive evidence** and **negative justification**:

> “Senior NLP Engineer | JD fit=0.88 | career_ml=4 | assess_avg=79 | NOT ranked for skill count alone; career-template coherence=1.0”

Shows evaluator-grade understanding of dataset traps.

### Differentiator 8: Shortlist Diversity Audit

Post-ranking audit report:

- Title concentration (expect ML-heavy top 100)
- Experience range distribution
- Geography spread
- Work mode alignment with JD

Demonstrates product thinking, not just model output.

### Differentiator 9: Interactive Explainability Demo

Streamlit cards with score breakdown bars per component — directly addresses “shortlist a recruiter can trust.”

---

## 9. System Architecture

### High-Level Pipeline

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│ Job Description │────▶│ Stage 1: JD Parser   │────▶│ Role Profile (JSON) │
└─────────────────┘     └──────────────────────┘     └──────────┬──────────┘
                                                                │
┌─────────────────┐     ┌──────────────────────┐                │
│ candidates.jsonl│────▶│ Stage 2: Feature     │◀───────────────┘
└─────────────────┘     │ Extraction per       │
                          │ Candidate            │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ Stage 3: Multi-Signal│
                          │ Scoring Engine       │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ Stage 4: Penalties & │
                          │ Trust Adjustment     │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ Stage 5: Rank, Tie-  │
                          │ break, Reasoning     │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ submission.csv       │
                          │ (+ optional audit)   │
                          └──────────────────────┘
```

### Component Breakdown

| Component | Responsibility |
|-----------|----------------|
| `jd_parser.py` | Parse JD → structured role requirements |
| `embedder.py` | Optional local semantic similarity (sentence-transformers, offline) |
| `title_gate.py` | Classify title domain; soft/hard filtering |
| `career_analyzer.py` | Template classification, ML depth patterns, trajectory |
| `coherence.py` | Cross-field consistency scoring |
| `skill_trust.py` | Trusted skill extraction, inflation detection |
| `signals.py` | Redrob platform signal composite |
| `penalties.py` | Shallow boilerplate, honeypot rules |
| `ranker.py` | Weighted score aggregation |
| `reasoning.py` | Human-readable explanation builder |
| `rank.py` | CLI entry point |
| `app/streamlit_app.py` | Demo UI |

### Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.11+ | Challenge ecosystem, fast iteration |
| Core scoring | Regex + heuristics + weighted math | Offline, fast, explainable |
| Optional semantic | `sentence-transformers` (pre-downloaded) | JD fit without runtime network |
| CLI | `argparse` | Simple reproducibility |
| Demo | Streamlit | Fast recruiter-style UI |
| Validation | Provided `validate_submission.py` | Compliance |

---

## 10. Scoring Methodology (Deep Specification)

### 10.1 Stage 1 — JD Understanding

**Input:** Plain text job description  
**Output:** `RoleProfile` object

```python
@dataclass
class RoleProfile:
    target_titles: list[str]           # e.g. ["ML Engineer", "Senior NLP Engineer"]
    required_capabilities: list[str]   # e.g. ["LLM fine-tuning", "RAG", "deployment"]
    nice_to_have: list[str]
    seniority: str                     # junior | mid | senior | staff
    min_experience_years: float
    preferred_work_mode: str | None    # remote | hybrid | onsite | flexible
    domain: str                        # e.g. "NLP/GenAI"
    embedding_vector: np.ndarray | None  # optional
```

**Implementation options (choose one primary):**

1. **Rule-based parser** — keyword/phrase extraction with seniority patterns
2. **Embedding similarity** — embed JD sentences; compare to candidate text (offline model)
3. **LLM parse (offline pre-step)** — run once before submission; save `role_profile.json`

**Recommendation for hackathon:** Hybrid rule-based + optional local embeddings.

---

### 10.2 Stage 2 — Per-Candidate Feature Extraction

For each candidate, compute:

| Feature | Description |
|---------|-------------|
| `title_domain` | ML / software / non-tech / ai-adjacent |
| `title_jd_match` | Overlap with RoleProfile.target_titles |
| `career_ml_depth` | Count of deep ML patterns in all career text |
| `current_role_ml_depth` | Deep ML patterns in current role description only |
| `career_template_domain` | Classified domain of current description template |
| `template_coherence` | title_domain vs template_domain alignment |
| `shallow_ai_count` | Count of boilerplate “AI curious” phrases |
| `trusted_ml_skills` | ML skills passing duration + endorsement checks |
| `skill_inflation_risk` | Ratio of suspicious skill entries |
| `assessment_avg` | Mean of skill_assessment_scores (or -1 if none) |
| `assessment_jd_overlap` | Assessments matching JD required capabilities |
| `github_norm` | Normalized github_activity_score (0 if -1) |
| `engagement_score` | Composite Redrob behavioral signal |
| `career_momentum` | Company size progression slope |
| `education_relevance` | CS/AI/DS field match + tier weight |
| `availability_score` | open_to_work, notice_period, relocate |
| `semantic_jd_fit` | Optional embedding cosine similarity |

---

### 10.3 Stage 3 — Component Scores (Normalized 0–1)

#### A. JD Fit Score (Weight: 0.20)

```
jd_fit = 0.4 * title_jd_match
       + 0.3 * semantic_jd_fit
       + 0.2 * assessment_jd_overlap
       + 0.1 * education_relevance
```

#### B. Career Evidence Score (Weight: 0.30)

```
career_evidence = 0.5 * normalize(career_ml_depth)
                + 0.35 * normalize(current_role_ml_depth)
                + 0.15 * career_momentum
```

**Deep ML patterns (examples):**

- Fine-tuning: LoRA, QLoRA, PEFT, RLHF, DPO, SFT
- Infrastructure: FAISS, Milvus, Weaviate, MLflow, W&B
- Frameworks: PyTorch, TensorFlow, JAX, Hugging Face
- Production: model deployment, inference serving, A/B testing models
- NLP/GenAI: RAG, sentence-transformers, LLaMA, Mistral

#### C. Profile Coherence Score (Weight: 0.25)

```
coherence = 0.4 * template_coherence
          + 0.35 * title_domain_ml_bonus
          + 0.25 * trusted_skill_alignment
```

**Hard rule:** If `title_domain == non-tech` AND `career_ml_depth < threshold` AND `current_role_ml_depth < threshold`, cap coherence at 0.15.

#### D. Platform Trust Score (Weight: 0.15)

```
platform_trust = 0.35 * assessment_avg_norm
               + 0.20 * github_norm
               + 0.30 * engagement_score
               + 0.15 * availability_score
```

#### E. Penalties (Subtracted)

```
penalties = 0.15 * shallow_ai_count_norm
          + 0.20 * skill_inflation_risk
          + 0.25 * template_mismatch_flag
          + 0.10 * non_ml_title_high_skill_flag
```

---

### 10.4 Final Score Formula

```
raw_score = 0.20 * jd_fit
          + 0.30 * career_evidence
          + 0.25 * coherence
          + 0.15 * platform_trust
          - penalties

final_score = clamp(raw_score, 0.0, 1.0)
```

### 10.5 Ranking & Tie-Breaking

1. Sort candidates by `final_score` descending
2. Take top 100
3. Assign ranks 1–100
4. Assign scores monotonically non-increasing (may transform final_score linearly for rank 1→100)
5. Tie-break equal scores by ascending `candidate_id`

### 10.6 Reasoning Template

```
{current_title} | JD={jd_fit:.2f} Career={career_evidence:.2f} Coherence={coherence:.2f} Trust={platform_trust:.2f} | {key_evidence} | {trap_note_if_any}
```

**Example:**

```
Senior NLP Engineer | JD=0.88 Career=0.91 Coherence=0.95 Trust=0.72 | LoRA/QLoRA in current role; assess LoRA=67, RL=90 | coherence-pass
```

---

## 11. Functional Requirements

### FR-1: CLI Ranking

| ID | Requirement |
|----|-------------|
| FR-1.1 | `rank.py` accepts `--candidates`, `--jd`, `--out` arguments |
| FR-1.2 | Produces CSV with exactly 100 data rows + header |
| FR-1.3 | Completes within 5 minutes on 100K candidates (CPU) |
| FR-1.4 | Runs without network access during execution |

### FR-2: JD Parsing

| ID | Requirement |
|----|-------------|
| FR-2.1 | Extract target titles, capabilities, seniority from JD |
| FR-2.2 | Output parseable role profile (JSON or internal object) |
| FR-2.3 | Support fallback default role profile if JD missing (with warning) |

### FR-3: Scoring

| ID | Requirement |
|----|-------------|
| FR-3.1 | Compute all component scores for every candidate |
| FR-3.2 | Apply honeypot penalties explicitly |
| FR-3.3 | Support configurable weights via config file (optional) |

### FR-4: Explainability

| ID | Requirement |
|----|-------------|
| FR-4.1 | Every ranked row includes reasoning string |
| FR-4.2 | Reasoning references at least 2 score components |
| FR-4.3 | Optionally flag honeypot avoidance in reasoning |

### FR-5: Validation

| ID | Requirement |
|----|-------------|
| FR-5.1 | Integrate `validate_submission.py` in workflow |
| FR-5.2 | Fail loudly on invalid output before submission |

### FR-6: Demo (Optional)

| ID | Requirement |
|----|-------------|
| FR-6.1 | Streamlit app accepts JD text input |
| FR-6.2 | Shows top-N ranked candidates with score breakdown |
| FR-6.3 | Displays honeypot/trap flags for sample bad profiles |

---

## 12. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | Performance | ≤ 5 min for 100K candidates on 16GB RAM CPU |
| NFR-2 | Reproducibility | Same inputs → same output (deterministic) |
| NFR-3 | Portability | Runs on Windows/macOS/Linux with documented setup |
| NFR-4 | Maintainability | Modular scorer packages, typed functions |
| NFR-5 | Explainability | No black-box-only ranking |
| NFR-6 | Compliance | No network/GPU during ranking |
| NFR-7 | Documentation | README with setup, run, architecture |
| NFR-8 | Testability | Unit tests for trap detection and scoring edge cases |

---

## 13. Implementation Plan & Steps

### Phase 0: Project Setup (Day 0 — 2 hours)

| Step | Task | Output |
|------|------|--------|
| 0.1 | Create GitHub repo structure | Folder skeleton |
| 0.2 | Add `requirements.txt`, README stub | Dev environment |
| 0.3 | Copy challenge files (validator, schema) | Local baseline |
| 0.4 | Create/sample `job_description.txt` | JD input for development |

---

### Phase 1: Core Pipeline MVP (Day 1 — 6 hours)

| Step | Task | Output |
|------|------|--------|
| 1.1 | Implement `rank.py` CLI skeleton | Runnable command |
| 1.2 | JSONL loader + candidate dataclass | Typed candidate objects |
| 1.3 | Basic title + skill scoring (baseline) | First valid CSV |
| 1.4 | Run `validate_submission.py` | Validation pass |
| 1.5 | Score monotonic mapping for ranks 1–100 | Compliant scores |

**Exit criteria:** Valid submission CSV exists (even if naive).

---

### Phase 2: JD Parser (Day 1–2 — 4 hours)

| Step | Task | Output |
|------|------|--------|
| 2.1 | Define `RoleProfile` schema | Structured JD output |
| 2.2 | Rule-based JD parser | title/capability extraction |
| 2.3 | Wire JD fit into scoring | Role-conditioned ranking |
| 2.4 | Optional: local embedding similarity | Semantic JD fit |

**Exit criteria:** Changing JD changes ranking meaningfully.

---

### Phase 3: Coherence & Trap Detection (Day 2 — 6 hours)

| Step | Task | Output |
|------|------|--------|
| 3.1 | Title domain classifier | ML vs non-tech vs software |
| 3.2 | Career template domain classifier | Template-aware analysis |
| 3.3 | Template coherence scoring | Mismatch penalties |
| 3.4 | Deep ML pattern library in career text | Career evidence score |
| 3.5 | Shallow AI boilerplate detector | Penalty signal |
| 3.6 | Skill trust chain + inflation detector | Trusted skills only |

**Exit criteria:** Known trap IDs (from sample submission) rank outside top 1000.

---

### Phase 4: Redrob Signals & Advanced Features (Day 3 — 4 hours)

| Step | Task | Output |
|------|------|--------|
| 4.1 | Platform engagement composite | Behavioral score |
| 4.2 | Assessment overlap with JD | Verification bonus |
| 4.3 | GitHub normalization | Technical credibility |
| 4.4 | Career momentum from company sizes | Trajectory score |
| 4.5 | Availability signals (OTW, notice, relocate) | Recruiter readiness |

**Exit criteria:** Top 100 predominantly ML/authentic profiles with diverse tie-breaking.

---

### Phase 5: Explainability & Reasoning (Day 3 — 3 hours)

| Step | Task | Output |
|------|------|--------|
| 5.1 | Reasoning generator module | Structured strings |
| 5.2 | Include component scores in reasoning | Transparent output |
| 5.3 | Optional honeypot notes in low-rank reasoning | Audit trail |

**Exit criteria:** Every top-20 row has defensible reasoning.

---

### Phase 6: Demo, Tests, Docs (Day 4 — 6 hours)

| Step | Task | Output |
|------|------|--------|
| 6.1 | Streamlit demo app | Interactive shortlist |
| 6.2 | Unit tests (traps, ties, validation) | Test suite |
| 6.3 | Shortlist diversity audit script | Audit report |
| 6.4 | Complete README | Reproduction guide |
| 6.5 | `submission_metadata.yaml` | Metadata file |

**Exit criteria:** Repo is submission-ready end-to-end.

---

### Phase 7: PDF Deck & Final QA (Day 4–5 — 4 hours)

| Step | Task | Output |
|------|------|--------|
| 7.1 | Write approach deck (8–10 slides) | PPT/PDF |
| 7.2 | Include architecture diagram + examples | Visual story |
| 7.3 | Fresh-machine reproduction test | Confidence check |
| 7.4 | Final validation + submission upload | Deliverables complete |

---

## 14. Repository Structure

```
redrob-fitrank/
├── rank.py                          # CLI entry: main reproduce command
├── job_description.txt              # Role input (or from organizers)
├── requirements.txt
├── submission_metadata.yaml
├── README.md
├── config/
│   └── weights.yaml                 # Optional scoring weights
├── data/
│   ├── candidates.jsonl             # Symlink or copy from challenge pack
│   └── sample_candidates.json
├── fitrank/
│   ├── __init__.py
│   ├── models.py                    # Candidate, RoleProfile dataclasses
│   ├── loader.py                    # JSONL loading
│   ├── jd_parser.py                 # Stage 1
│   ├── embedder.py                  # Optional semantic similarity
│   ├── title_gate.py                # Title domain classification
│   ├── career_analyzer.py           # ML depth, templates, trajectory
│   ├── coherence.py                 # Cross-field consistency
│   ├── skill_trust.py               # Trusted skills, inflation
│   ├── signals.py                   # Redrob platform composite
│   ├── penalties.py                 # Honeypot & boilerplate rules
│   ├── ranker.py                    # Score aggregation
│   └── reasoning.py                 # Explanation builder
├── app/
│   └── streamlit_app.py             # Demo UI
├── scripts/
│   ├── validate.sh                  # Run validator
│   └── audit_shortlist.py           # Diversity/trap audit
├── tests/
│   ├── test_coherence.py
│   ├── test_penalties.py
│   └── test_ranking.py
├── outputs/
│   └── submission.csv
├── deck/
│   └── Redrob_FitRank_Approach.pdf
└── validate_submission.py           # From challenge pack
```

---

## 15. Deliverables Checklist

### Required

- [ ] GitHub repo (public or accessible to organizers)
- [ ] `rank.py` + modular scorer code
- [ ] `submission.csv` — 100 rows, valid format
- [ ] PDF deck explaining approach
- [ ] README with install + reproduce steps

### Recommended

- [ ] `submission_metadata.yaml` completed
- [ ] Streamlit / HuggingFace sandbox link
- [ ] Unit tests for trap detection
- [ ] Shortlist audit report
- [ ] Example reasoning for top 10 candidates in deck

### Pre-Submission QA

- [ ] `python validate_submission.py submission.csv` passes
- [ ] Reproduce command runs in ≤ 5 minutes
- [ ] No network calls during ranking (verified)
- [ ] Top 100 contains no obvious honeypot profiles
- [ ] Reasoning is human-readable for all 100 rows

---

## 16. Demo & PDF Deck Outline

### Streamlit Demo Flow

1. Paste/upload job description
2. Click “Rank Candidates”
3. View top 20 cards with score breakdown bars
4. Toggle “Show trap examples” to contrast bad profiles
5. Export CSV button

### PDF Deck (8–10 Slides)

| Slide | Title | Content |
|-------|-------|---------|
| 1 | Title | FitRank — Recruiter-Trustable AI Ranking |
| 2 | Problem | Keyword search fails; recruiters miss real talent |
| 3 | Data Insight | Traps, templates, 100K adversarial profiles |
| 4 | Core Idea | Profile coherence > keyword matching |
| 5 | Architecture | 5-stage pipeline diagram |
| 6 | Differentiators | 7 unique features vs typical approaches |
| 7 | Scoring Model | Weights, formulas, penalty rules |
| 8 | Examples | Top candidate + trap candidate comparison |
| 9 | Results | Validation, runtime, shortlist audit |
| 10 | Future | Redrob product integration vision |

---

## 17. Success Metrics

### Hackathon Success (Primary)

| Metric | Target |
|--------|--------|
| Submission validity | 100% pass validator |
| Reproducibility | Single command, deterministic output |
| Top-100 quality | Predominantly genuine ML/role-aligned profiles |
| Trap exclusion | Sample honeypot IDs outside top 100 |
| Explainability | All 100 rows have meaningful reasoning |

### Product Success (Secondary)

| Metric | Target |
|--------|--------|
| Runtime | ≤ 60s preferred (well under 5 min limit) |
| Modularity | Each scorer independently testable |
| JD generalization | Different JDs produce different rankings |
| Demo usability | Non-technical reviewer understands output in < 2 min |

### Differentiation Success

| Metric | Target |
|--------|--------|
| Coherence feature | Documented + tested |
| Template-aware analysis | Implemented + explained in deck |
| Redrob signals usage | ≥ 4 signals in composite score |
| Honeypot awareness | Explicit in reasoning/audit |

---

## 18. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hidden JD differs from assumed ML role | Wrong ranking target | JD-driven parser; avoid hardcoding |
| Over-reliance on career text depth | fooled by templates | Template-domain coherence layer |
| Over-filtering to ML titles only | Miss strong software→ML transitions | Soft title scoring + career evidence override |
| Runtime exceeds 5 minutes | Disqualification risk | Precompute patterns; avoid heavy models |
| Network dependency | Repro failure | Offline-only ranking; pre-download embeddings |
| Equal scores break tie rules | Validation failure | Deterministic tie-break by candidate_id |
| Sample submission copied | Poor evaluator impression | Never use sample ranking logic |
| Reasoning too generic | Weak trust story | Template with component values + evidence |

---

## 19. Future Roadmap (Post-Hackathon)

If extended into a Redrob product feature:

1. **API endpoint** — `POST /rank { jd, candidate_ids } → shortlist`
2. **Recruiter feedback loop** — Learn weights from save/reject actions
3. **Multi-role batch ranking** — Rank same pool against multiple JDs
4. **Confidence intervals** — Show rank stability under weight perturbation
5. **Explainability UI** — Inline “why not this candidate?” comparisons
6. **Real profile integration** — Replace synthetic template logic with richer real-world signals

---

## 20. Appendix

### A. AI/ML Title Patterns (Title Gate)

```
ML Engineer, Senior/Junior/Staff ML Engineer, Machine Learning Engineer
Data Scientist, Senior Data Scientist
AI Engineer, Senior AI Engineer, Lead AI Engineer, AI Research Engineer
NLP Engineer, Senior NLP Engineer
Computer Vision Engineer
Search Engineer, Recommendation Systems Engineer
Applied ML Engineer
Senior Software Engineer (ML)
AI Specialist (apply coherence checks — higher trap risk)
```

### B. Shallow AI Boilerplate Patterns (Penalties)

```
"curious about ai"
"experimenting with chatgpt"
"ai tools could augment"
"taking online courses"
"played with the openai"
"excited about ai"
"side project" + "rag"/"langchain"
```

### C. Deep ML Career Patterns (Positive Evidence)

```
fine-tun, qlora, lora, rlhf, dpo, sft
sentence-transformer, faiss, milvus, weaviate
pytorch, tensorflow, jax, keras
model deploy, inference serv, production model
rag, embedding, semantic search, vector
mlflow, wandb, kubeflow, bentoml
hyperparameter, ablation, benchmark, eval harness
```

### D. Reproduce Command

```bash
python rank.py --jd ./job_description.txt --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
python validate_submission.py ./outputs/submission.csv
```

### E. Sample Job Description (Development Placeholder)

Until official JD is provided, use a senior ML/NLP role description including:

- LLM fine-tuning (LoRA/QLoRA)
- RAG and vector search
- Model deployment and evaluation
- 4+ years experience
- Python, PyTorch, Hugging Face

---

## Document Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-06-24 | Initial PRD — full implementation spec |

---

**End of PRD**
