"""Descriptive governance posture for synthetic denial-management cases."""

from __future__ import annotations

from typing import Any


GOVERNANCE_POSTURES = {"SUPPORTED", "LOW_CONFIDENCE", "LOW_EVIDENCE", "AMBIGUOUS", "HIGH_RISK"}


def run(state: dict[str, Any]) -> dict[str, Any]:
    updated = dict(state)
    gaps = updated.get("documentation_gaps", {})
    recoverability = str(updated.get("recoverability", "unclear recoverability"))
    action = str(updated.get("routing_action", "ESCALATE"))
    category = str(updated.get("denial_category", "unclear denial"))
    metadata = updated.get("metadata", {})
    evidence = updated.get("evidence_profile", {})

    if metadata.get("specialist_review_candidate") or category == "unsupported service" or (
        evidence.get("has_specialist_notes") and gaps.get("ambiguity")
    ):
        posture = "HIGH_RISK"
    elif (
        gaps.get("ambiguity")
        or gaps.get("timeline_inconsistency")
        or action == "ESCALATE"
        or recoverability == "unclear recoverability"
    ):
        posture = "AMBIGUOUS"
    elif gaps.get("documentation_insufficiency") and recoverability == "low recoverability":
        posture = "LOW_EVIDENCE"
    elif gaps.get("missing_evidence") or gaps.get("documentation_insufficiency"):
        posture = "LOW_CONFIDENCE"
    else:
        posture = "SUPPORTED"

    updated["governance_posture"] = posture
    return updated
