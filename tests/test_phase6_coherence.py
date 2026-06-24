"""Phase 6 coherence tests."""

from fitrank.career_analyzer import CareerEvidence, analyze_career
from fitrank.coherence import compute_coherence
from fitrank.loader import load_candidates
from fitrank.models import Candidate, Profile, RedrobSignals, SalaryRange
from fitrank.title_gate import TitleDomain, classify_title


def test_t6_2_hard_cap():
    career = CareerEvidence(0, 0, 0.0, "support", 0)
    candidate = Candidate(
        "CAND_0000001",
        Profile("A", "h", "s", "L", "IN", 5.0, "HR Manager", "Co", "201-500", "Tech"),
        [],
        [],
        [],
        RedrobSignals(
            80, "2024-01-01", "2024-06-01", True, 1, 1, 0.5, 1.0, {}, 1, 1, 30,
            SalaryRange(10, 20), "remote", True, -1, 1, 1, 0.5, -1, True, False, False,
        ),
    )
    score = compute_coherence(candidate, career, TitleDomain.NON_TECH)
    assert score.coherence_score <= 0.15


def test_t6_10_sample_trap(candidates_path):
    target = None
    for candidate in load_candidates(candidates_path):
        if candidate.candidate_id == "CAND_0004989":
            target = candidate
            break
    assert target is not None
    career = analyze_career(target)
    domain = classify_title(target.profile.current_title)
    score = compute_coherence(target, career, domain)
    assert score.coherence_score <= 0.15
