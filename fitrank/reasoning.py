"""Human-readable per-candidate reasoning strings."""

from __future__ import annotations

from fitrank.models import Candidate
from fitrank.ranker import ComponentScores


def build_reasoning(
    candidate: Candidate,
    components: ComponentScores,
) -> str:
    title = candidate.profile.current_title
    evidence_note = "strong ML career evidence" if components.career_evidence >= 0.5 else "limited ML career evidence"
    if components.is_honeypot:
        penalty_note = "honeypot: title/career mismatch with stuffed AI skills"
    elif components.penalties >= 0.3:
        penalty_note = f"high penalties ({components.penalties:.2f})"
    else:
        penalty_note = f"low penalties ({components.penalties:.2f})"

    return (
        f"{title} | JD={components.jd_fit:.2f} Career={components.career_evidence:.2f} "
        f"Coh={components.coherence:.2f} Trust={components.platform_trust:.2f} | "
        f"{evidence_note} | {penalty_note}"
    )
