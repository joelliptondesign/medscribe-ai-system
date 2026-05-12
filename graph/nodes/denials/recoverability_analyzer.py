"""Deterministic recoverability posture for synthetic denial cases."""

from __future__ import annotations

from typing import Any

from graph.config import get_execution_mode
from graph.llm_client import HybridLLMError, invoke_json


RECOVERABLE_TERMS = ("available", "can provide", "attach", "resubmit", "included elsewhere", "supports", "consistent")
PARTIAL_TERMS = ("missing", "omitted", "not included", "gap", "clarify", "borderline")
LOW_TERMS = ("non-covered", "excluded", "experimental", "no evidence", "unsupported", "not documented")
UNCLEAR_TERMS = ("unclear", "conflicting", "ambiguous", "inconsistent", "contradict")
RECOVERABILITY_VALUES = {
    "likely recoverable",
    "partially recoverable",
    "low recoverability",
    "unclear recoverability",
}
PROMPT = """You interpret synthetic, non-PHI denial recoverability evidence.
Return strict JSON only. Use only the provided synthetic case content.
Do not invent PHI, patient facts, payer outcomes, reimbursement estimates, final routing decisions, or governance posture.
Interpret recoverability only with these fields:
recoverability, recoverability_factors, uncertainty, rationale.
recoverability must be one of: likely recoverable, partially recoverable, low recoverability, unclear recoverability.
recoverability_factors must be a JSON array of short strings. uncertainty must be low, medium, or high.
"""


def _text(state: dict[str, Any]) -> str:
    return " ".join(
        str(state.get(field, "")).lower()
        for field in ("denial_reason", "service_context", "documentation_context", "denial_type")
    )


def _deterministic_recoverability(updated: dict[str, Any]) -> str:
    text = _text(updated)
    signals = updated.get("documentation_signals", {})
    evidence = updated.get("evidence_profile", {})
    strength = str(evidence.get("evidence_strength", "unspecified")).lower()
    if (
        any(term in text for term in UNCLEAR_TERMS)
        or signals.get("conflict")
        or evidence.get("has_conflicting_documentation")
        or evidence.get("has_timeline_flags")
        or strength == "conflicting"
    ):
        recoverability = "unclear recoverability"
    elif (
        strength in {"weak", "low"}
        and evidence.get("has_missing_required_evidence")
        and not evidence.get("has_partial_support")
    ) or (any(term in text for term in LOW_TERMS) and not any(term in text for term in RECOVERABLE_TERMS)):
        recoverability = "low recoverability"
    elif (
        signals.get("missing")
        or evidence.get("has_missing_required_evidence")
        or evidence.get("has_partial_support")
        or any(term in text for term in PARTIAL_TERMS)
    ):
        recoverability = "partially recoverable"
    elif strength in {"strong", "moderate"} or any(term in text for term in RECOVERABLE_TERMS) or signals.get("clinical_evidence"):
        recoverability = "likely recoverable"
    else:
        recoverability = "unclear recoverability"
    return recoverability


def _validate_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    recoverability = str(payload.get("recoverability", "")).strip().lower()
    if recoverability not in RECOVERABILITY_VALUES:
        raise ValueError("invalid recoverability")
    factors = payload.get("recoverability_factors", [])
    if not isinstance(factors, list) or not all(isinstance(item, str) for item in factors):
        raise ValueError("recoverability_factors must be a string array")
    uncertainty = str(payload.get("uncertainty", "")).strip().lower()
    if uncertainty not in {"low", "medium", "high"}:
        raise ValueError("uncertainty must be low, medium, or high")
    return {
        "recoverability": recoverability,
        "recoverability_factors": factors,
        "recoverability_uncertainty": uncertainty,
        "recoverability_rationale": str(payload.get("rationale", "")).strip(),
    }


def _hybrid_recoverability(updated: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: updated.get(key)
        for key in (
            "denial_reason",
            "service_context",
            "documentation_context",
            "denial_type",
            "documentation_signals",
            "evidence_profile",
            "metadata",
        )
    }
    response = invoke_json("denial_recoverability_analyzer", PROMPT, payload, diagnostic)
    interpreted = _validate_llm_payload(response)
    diagnostic["normalization_succeeded"] = True
    diagnostic["hybrid_interpretation_used"] = True
    return interpreted


def run(state: dict[str, Any]) -> dict[str, Any]:
    updated = dict(state)
    diagnostic = {
        "node_name": "recoverability_analyzer",
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
            interpreted = _hybrid_recoverability(updated, diagnostic)
        except (HybridLLMError, ValueError) as exc:
            diagnostic["fallback_triggered"] = True
            diagnostic["fallback_reason"] = diagnostic.get("fallback_reason") or exc.__class__.__name__
            diagnostic["hybrid_interpretation_used"] = False
            interpreted = {"recoverability": _deterministic_recoverability(updated)}
    else:
        interpreted = {"recoverability": _deterministic_recoverability(updated)}
        diagnostic["normalization_succeeded"] = True
    updated.update(interpreted)
    hybrid_diagnostics = dict(updated.get("_hybrid_node_diagnostics", {}))
    hybrid_diagnostics["recoverability_analyzer"] = diagnostic
    updated["_hybrid_node_diagnostics"] = hybrid_diagnostics
    return updated
