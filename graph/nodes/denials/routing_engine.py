"""Deterministic denial routing actions kept separate from governance posture."""

from __future__ import annotations

from typing import Any


ROUTING_ACTIONS = {"APPEAL", "RESUBMIT", "WRITE_OFF", "ESCALATE"}


def run(state: dict[str, Any]) -> dict[str, Any]:
    updated = dict(state)
    gaps = updated.get("documentation_gaps", {})
    recoverability = str(updated.get("recoverability", "unclear recoverability"))
    category = str(updated.get("denial_category", "unclear denial"))
    evidence = updated.get("evidence_profile", {})
    high_risk = (
        bool(updated.get("metadata", {}).get("specialist_review_candidate"))
        or bool(evidence.get("has_specialist_notes") and gaps.get("ambiguity"))
        or "unsupported service" == category
    )
    ambiguous = (
        bool(gaps.get("ambiguity"))
        or bool(gaps.get("timeline_inconsistency"))
        or recoverability == "unclear recoverability"
        or category == "unclear denial"
    )

    if ambiguous or high_risk:
        action = "ESCALATE"
    elif recoverability == "partially recoverable" and (
        gaps.get("missing_evidence") or gaps.get("documentation_insufficiency")
    ):
        action = "RESUBMIT"
    elif recoverability == "likely recoverable" and not gaps.get("documentation_insufficiency"):
        action = "APPEAL"
    elif recoverability == "low recoverability" and (
        gaps.get("documentation_insufficiency") or gaps.get("missing_evidence")
    ):
        action = "WRITE_OFF"
    else:
        action = "ESCALATE"

    updated["routing_action"] = action
    return updated
