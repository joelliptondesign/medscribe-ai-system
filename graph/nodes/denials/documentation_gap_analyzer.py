"""Structured documentation gap analysis for synthetic denial cases."""

from __future__ import annotations

from typing import Any

from graph.config import get_execution_mode
from graph.llm_client import HybridLLMError, invoke_json


AMBIGUITY_LEVELS = {"low", "medium", "high"}
PROMPT = """You interpret synthetic, non-PHI denial documentation evidence.
Return strict JSON only. Use only the provided synthetic case content.
Do not invent PHI, patient facts, payer outcomes, reimbursement estimates, final routing decisions, or governance posture.
Interpret documentation gaps only with these fields:
missing_evidence, documentation_insufficiency, unsupported_specificity, conflicting_evidence,
partial_support, ambiguity_level, specialist_review_candidate, rationale.
Booleans must be true or false. ambiguity_level must be low, medium, or high.
"""

def _contains(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _deterministic_gaps(updated: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        str(updated.get(field, "")).lower()
        for field in ("denial_reason", "service_context", "documentation_context")
    )
    signals = updated.get("documentation_signals", {})
    evidence = updated.get("evidence_profile", {})
    strength = str(evidence.get("evidence_strength", "unspecified")).lower()
    gaps = {
        "missing_evidence": bool(
            signals.get("missing")
            or signals.get("missing_required_evidence")
            or evidence.get("has_missing_required_evidence")
            or _contains(text, ("missing", "omitted", "not included", "absent"))
        ),
        "unsupported_specificity": _contains(text, ("unsupported specificity", "specificity not supported", "unsupported code")),
        "documentation_insufficiency": bool(
            signals.get("missing")
            or evidence.get("has_missing_required_evidence")
            or strength in {"weak", "low"}
            or _contains(text, ("insufficient", "not documented", "no evidence", "low evidence"))
        ),
        "ambiguity": bool(
            signals.get("conflict")
            or signals.get("conflicting_documentation")
            or evidence.get("has_conflicting_documentation")
            or evidence.get("has_timeline_flags")
            or strength == "conflicting"
            or _contains(text, ("unclear", "ambiguous", "conflicting", "borderline", "inconsistent"))
        ),
        "partial_support": bool(evidence.get("has_partial_support") or strength == "partial"),
        "timeline_inconsistency": bool(evidence.get("has_timeline_flags")),
        "specialist_review_signal": bool(evidence.get("has_specialist_notes")),
    }
    gaps["gap_count"] = sum(1 for value in gaps.values() if value is True)
    return gaps


def _validate_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required_booleans = (
        "missing_evidence",
        "documentation_insufficiency",
        "unsupported_specificity",
        "conflicting_evidence",
        "partial_support",
        "specialist_review_candidate",
    )
    for field in required_booleans:
        if not isinstance(payload.get(field), bool):
            raise ValueError(f"{field} must be boolean")
    ambiguity_level = str(payload.get("ambiguity_level", "")).strip().lower()
    if ambiguity_level not in AMBIGUITY_LEVELS:
        raise ValueError("ambiguity_level must be low, medium, or high")
    return {
        "missing_evidence": payload["missing_evidence"],
        "unsupported_specificity": payload["unsupported_specificity"],
        "documentation_insufficiency": payload["documentation_insufficiency"],
        "ambiguity": payload["conflicting_evidence"] or ambiguity_level == "high",
        "partial_support": payload["partial_support"],
        "timeline_inconsistency": False,
        "specialist_review_signal": payload["specialist_review_candidate"],
        "conflicting_evidence": payload["conflicting_evidence"],
        "ambiguity_level": ambiguity_level,
        "llm_rationale": str(payload.get("rationale", "")).strip(),
    }


def _hybrid_gaps(updated: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: updated.get(key)
        for key in (
            "denial_reason",
            "service_context",
            "documentation_context",
            "documentation_signals",
            "evidence_profile",
            "metadata",
        )
    }
    response = invoke_json("denial_documentation_gap_analyzer", PROMPT, payload, diagnostic)
    gaps = _validate_llm_payload(response)
    gaps["gap_count"] = sum(
        1
        for key, value in gaps.items()
        if value is True
        and key
        in {
            "missing_evidence",
            "unsupported_specificity",
            "documentation_insufficiency",
            "ambiguity",
            "partial_support",
            "timeline_inconsistency",
            "specialist_review_signal",
            "conflicting_evidence",
        }
    )
    diagnostic["normalization_succeeded"] = True
    diagnostic["hybrid_interpretation_used"] = True
    return gaps


def run(state: dict[str, Any]) -> dict[str, Any]:
    updated = dict(state)
    diagnostic = {
        "node_name": "documentation_gap_analyzer",
        "execution_mode": get_execution_mode(),
        "live_call_attempted": False,
        "live_call_returned": False,
        "parse_succeeded": False,
        "normalization_succeeded": False,
        "fallback_triggered": False,
        "fallback_reason": "",
    }
    if diagnostic["execution_mode"] == "hybrid":
        try:
            gaps = _hybrid_gaps(updated, diagnostic)
        except (HybridLLMError, ValueError) as exc:
            diagnostic["fallback_triggered"] = True
            diagnostic["fallback_reason"] = diagnostic.get("fallback_reason") or exc.__class__.__name__
            diagnostic["hybrid_interpretation_used"] = False
            gaps = _deterministic_gaps(updated)
    else:
        gaps = _deterministic_gaps(updated)
        diagnostic["normalization_succeeded"] = True
    updated["documentation_gaps"] = gaps
    hybrid_diagnostics = dict(updated.get("_hybrid_node_diagnostics", {}))
    hybrid_diagnostics["documentation_gap_analyzer"] = diagnostic
    updated["_hybrid_node_diagnostics"] = hybrid_diagnostics
    return updated
