"""Phase 7 penalty layer tests (PRD T7.7-T7.11)."""

from fitrank.career_analyzer import CareerEvidence, analyze_career
from fitrank.coherence import CoherenceScore, compute_coherence
from fitrank.loader import load_candidates
from fitrank.jd_parser import parse_jd
from fitrank.models import Candidate, CareerEntry, JDPrefs, Profile, RedrobSignals, RoleProfile, SalaryRange, Skill
from fitrank.penalties import compute_penalties
from fitrank.title_gate import TitleDomain, classify_title


def _signals():
    return RedrobSignals(
        80, "2024-01-01", "2024-06-01", True, 1, 1, 0.5, 1.0, {}, 1, 1, 30,
        SalaryRange(10, 20), "remote", True, -1, 1, 1, 0.5, -1, True, False, False,
    )


def test_t7_7_expert_zero_endorsement_penalty():
    candidate = Candidate(
        "CAND_0000099",
        Profile("A", "h", "s", "L", "IN", 5.0, "ML Engineer", "Co", "201-500", "Tech"),
        [],
        [],
        [Skill("PyTorch", "expert", 0, 36)],
        _signals(),
    )
    career = CareerEvidence(5, 3, 0.1, "ml_work", 0)
    coherence = CoherenceScore(1.0, 1.0, 0.8, 0.9, False)
    penalties = compute_penalties(candidate, career, coherence, RoleProfile())
    assert penalties.expert_zero_endorse > 0


def test_t7_10_non_ml_title_high_skill():
    from fitrank.models import Candidate, Profile, RedrobSignals, SalaryRange, Skill

    skills = [
        Skill("PyTorch", "advanced", 1, 12),
        Skill("NLP", "advanced", 1, 12),
        Skill("RAG", "advanced", 1, 12),
        Skill("LLM", "advanced", 1, 12),
        Skill("LoRA", "advanced", 1, 12),
        Skill("TensorFlow", "advanced", 1, 12),
        Skill("JAX", "advanced", 1, 12),
        Skill("MLOps", "advanced", 1, 12),
        Skill("BERT", "advanced", 1, 12),
    ]
    candidate = Candidate(
        "CAND_0000102",
        Profile("A", "h", "s", "L", "IN", 5.0, "HR Manager", "Co", "201-500", "Tech"),
        [],
        [],
        skills,
        RedrobSignals(
            80, "2024-01-01", "2024-06-01", True, 1, 1, 0.5, 1.0, {}, 1, 1, 30,
            SalaryRange(10, 20), "remote", True, -1, 1, 1, 0.5, -1, True, False, False,
        ),
    )
    career = CareerEvidence(0, 0, 0.0, "support", 1)
    coherence = CoherenceScore(0.0, 0.0, 0.0, 0.1, True)
    penalties = compute_penalties(candidate, career, coherence, RoleProfile())
    assert penalties.non_ml_title_high_skill == 1.0


def test_t7_11_penalty_clamped():
    candidate = Candidate(
        "CAND_0000100",
        Profile("A", "h", "curious about AI and ChatGPT tools", "L", "IN", 5.0, "HR Manager", "Co", "201-500", "Tech"),
        [CareerEntry("Co", "HR Manager", "2020-01-01", None, 24, True, "Tech", "201-500", "Customer support team lead at a SaaS product.")],
        [],
        [Skill("PyTorch", "expert", 0, 2), Skill("NLP", "expert", 0, 2), Skill("RAG", "expert", 0, 2),
         Skill("LLM", "expert", 0, 2), Skill("LoRA", "expert", 0, 2), Skill("BERT", "expert", 0, 2),
         Skill("TensorFlow", "expert", 0, 2), Skill("JAX", "expert", 0, 2), Skill("MLOps", "expert", 0, 2)],
        _signals(),
    )
    career = analyze_career(candidate)
    coherence = compute_coherence(candidate, career, TitleDomain.NON_TECH)
    penalties = compute_penalties(candidate, career, coherence, RoleProfile())
    assert penalties.total_penalty <= 0.70


def test_consulting_penalty_gated_by_jd_prefs():
    candidate = Candidate(
        "CAND_CONS",
        Profile("A", "h", "s", "L", "IN", 8.0, "ML Engineer", "Co", "201-500", "Tech"),
        [
            CareerEntry("TCS", "ML Engineer", "2018-01-01", "2020-01-01", 24, False, "Svc", "1000+", "ML"),
            CareerEntry("Infosys", "ML Engineer", "2020-01-01", None, 48, True, "Svc", "1000+", "ML"),
        ],
        [],
        [],
        _signals(),
    )
    career = CareerEvidence(5, 3, 0.2, "ml_work", 0)
    coherence = CoherenceScore(1.0, 1.0, 0.8, 0.9, False)
    strict_role = RoleProfile(prefs=JDPrefs(penalize_consulting_only=True))
    lenient_role = parse_jd("Consulting background welcome. Client-facing experience valued.")
    assert compute_penalties(candidate, career, coherence, strict_role).consulting_only == 1.0
    assert compute_penalties(candidate, career, coherence, lenient_role).consulting_only == 0.0


def test_domain_mismatch_penalty_gated_by_jd_prefs():
    candidate = Candidate(
        "CAND_CV",
        Profile("A", "h", "s", "L", "IN", 6.0, "Computer Vision Engineer", "Co", "201-500", "Tech"),
        [CareerEntry("Co", "CV Engineer", "2020-01-01", None, 48, True, "Tech", "201-500", "YOLO and OpenCV")],
        [],
        [],
        _signals(),
    )
    career = CareerEvidence(5, 3, 0.2, "ml_work", 0)
    coherence = CoherenceScore(1.0, 0.3, 0.8, 0.7, False)
    nlp_role = parse_jd("Senior NLP Engineer. Retrieval, ranking, transformers, RAG.")
    cv_role = parse_jd("Computer Vision Engineer. Object detection, OpenCV, CNN.")
    assert compute_penalties(candidate, career, coherence, nlp_role).domain_mismatch == 1.0
    assert compute_penalties(candidate, career, coherence, cv_role).domain_mismatch == 0.0
