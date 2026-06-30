# FitRank Approach Deck

## Slide 1: FitRank — Recruiter-Trustable AI Ranking

- Offline, explainable candidate discovery for the Redrob Data & AI Challenge
- Ranks 100,000 profiles against a JD in under 5 minutes on CPU
- Two-stage pipeline: interpretable heuristics + optional LightGBM learned ranker
- Every shortlist entry ships a candidate-specific reasoning string

## Slide 2: Problem — Keyword Filters Fail on Adversarial Profiles

**Concrete trap example**

- Profile: **HR Manager** (NON_TECH title) with **9 ML skills** listed (PyTorch, LLM, RAG, FAISS…)
- Salary range inverted: `min > max` (classic honeypot signal)
- Career history: generic HR responsibilities, zero production ML evidence
- **Keyword baseline** ranks this profile highly — it counts skill keywords and ignores the JD entirely
- Recruiter outcome: wasted screen time; genuine Senior AI Engineers buried below keyword stuffers

## Slide 3: Data Insights — An Adversarial 100K Pool

- **100,000** candidates; only **~905 (0.9%)** are genuine strong ML practitioners
- **4,337** non-ML titles with 7+ AI skills (keyword stuffers)
- **18,865** salary-inversion honeypots
- **50,844** title–career description mismatches (~51%)
- **3,279** shallow AI-curious boilerplate profiles
- JD: Senior AI Engineer at Redrob AI — production retrieval, ranking, and LLM systems required

## Slide 4: Innovation — Why Keyword Counting Fails (and How We Beat It)

| Keyword baseline | FitRank |
|----------------|---------|
| JD-blind: ignores role requirements | JD-conditioned title + semantic capability match |
| Counts skill keywords in summary | Career evidence: ML depth across roles + momentum |
| No coherence check | Profile coherence: title × skills × career alignment |
| No platform verification | Redrob signals: assessments, GitHub, engagement |
| No trap detection | 12 penalty types + honeypot multiplier |

**Core formula:** `final = 0.25·JD_fit + 0.30·career + 0.20·coherence + 0.15·trust + 0.10·availability − penalties`

## Slide 5: Architecture — 10-Phase Offline Pipeline

```
Job Description (JD)
        |
        v
  [1] JD Parser ---------> RoleProfile (skills, seniority, disqualifiers)
        |
        v
  [2] Title Gate --------> ML_AI | AI_ADJACENT | SOFTWARE | NON_TECH
        |
        v
  [3] Career Analyzer ---> ML depth, tenure, momentum, template domain
        |
        v
  [4] Coherence ---------> template + title-domain + skill-career alignment
        |
        v
  [5] Skill Trust -------> trusted chain (duration + endorsements)
        |
        v
  [6] Redrob Signals ----> assessments, GitHub, engagement, availability
        |
        v
  [7] Penalties ---------> 12 trap types (inflation, consulting-only, etc.)
        |
        v
  [8] Feature Extract ---> 28 interpretable features
        |
        v
  [9] LightGBM Ranker --> learned score (auto mode when model present)
        |
        v
 [10] CSV + Reasoning ---> top-100 shortlist with explainable strings
```

## Slide 6: Differentiators vs Typical Approaches

- **Profile coherence scoring** with honeypot detection (including AI_ADJACENT title-chasers)
- **Trusted skill chain** — duration ≥12 months + endorsements > 0
- **Semantic capability match** via `all-MiniLM-L6-v2` dense embeddings
- **Template-aware career analysis** — detects copy-paste ML boilerplate
- **JD-conditioned penalties** — consulting-only, pure research, CV-without-NLP, salary inversion
- **Redrob-native signals** — assessment overlap, GitHub activity, recruiter responsiveness
- **Candidate-specific reasoning** — ML tenure, penalty flags, honeypot labels
- **Stability analysis** — weight perturbation bootstrap + component ablation

## Slide 7: Scoring Model — Component Weights

| Component | Weight | What it measures |
|-----------|--------|------------------|
| JD fit | 0.25 | Title match, capability, assessments, education |
| Career evidence | 0.30 | ML depth across career + current role + momentum |
| Coherence | 0.20 | Template, title-domain, skill-career alignment |
| Platform trust | 0.15 | Assessments, GitHub, engagement, verification |
| Availability | 0.10 | OTW, notice period, location, recency |
| Penalties | subtract | Up to 0.70 cap; honeypot ×0.25 if penalty > 0.30 |

Configurable via `config/weights.yaml` — ablation tests prove each component earns its weight.

## Slide 8: Examples — Genuine ML Engineer vs Honeypot

**Rank #1 genuine — {{audit.top_5.0.candidate_id}}**
- Title: {{audit.top_5.0.title}}
- Score: {{audit.top_5.0.final_score}}
- Reasoning: {{audit.top_5.0.reasoning}}

**Known honeypot — CAND_0004989 (HR Manager + 9 ML skills)**
- FitRank: penalized for skill inflation + title mismatch; excluded from top-100
- Keyword baseline: surfaces in top-100 via keyword count alone
- Design goal: recruiters see *why* a profile is ranked, not just a number

## Slide 9: Results — Validation, Runtime, and Stability

**Production run**
- Runtime: **{{audit.runtime_seconds}}s** (< 5 min budget)
- Honeypots in shortlist: **{{audit.honeypot_flags_in_shortlist}}**
- Score range: {{audit.score_histogram.min}} – {{audit.score_histogram.max}} (mean {{audit.score_histogram.mean}})

**Baseline comparison (synthetic ground truth)**
- NDCG@100: baseline {{eval.baseline.ndcg.@100}} → FitRank {{eval.fitrank.ndcg.@100}} ({{eval.improvements.ndcg@100_pct}}% lift)
- Honeypot exclusion: {{eval.fitrank.honeypot_exclusion_rate}}

**Sensitivity analysis**
- {{sensitivity.headline}}
- Ablation NDCG@100 drops (heuristic): coherence {{sensitivity.ablation.heuristic.coherence.delta@100}}, skill_trust {{sensitivity.ablation.heuristic.skill_trust.delta@100}}, signals {{sensitivity.ablation.heuristic.signals.delta@100}}

![demo](deck/assets/demo_screenshot.png)

## Slide 10: Redrob Vision — Product Integration Roadmap

**Near-term (Q3–Q4)**
1. **API endpoint** — `POST /rank { jd_text, filters } → shortlist + reasoning`
2. **Recruiter feedback loop** — online weight learning from save/reject signals
3. **Multi-JD batch ranking** — same talent pool ranked against multiple roles simultaneously

**Delivered in this submission**
4. **Confidence intervals** — rank stability under ±10% weight perturbation (bootstrap analysis)
5. **"Why not?" explainability** — per-candidate reasoning with component breakdown

**Future**
6. **Real profile enrichment** — live Redrob signal ingestion replacing synthetic template detection
7. **Hiring manager dashboard** — shortlist exploration UI (Streamlit demo → production widget)

FitRank becomes Redrob's **intelligent discovery layer** — trustable ranking that survives adversarial profiles.
