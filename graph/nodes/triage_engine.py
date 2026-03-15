"""Deterministic triage assignment for the MED-SCRIBE demo."""

from __future__ import annotations

from typing import Any

from graph.config import get_execution_mode
from graph.diagnostics import append_node_diagnostic, init_node_diagnostic
from graph.llm_client import HybridLLMError, invoke_json
from graph.prompt_loader import load_prompt


VALID_TRIAGE_LEVELS = {"urgent", "routine", "home_care", "escalate"}


def _deterministic_triage(state: dict[str, Any]) -> dict[str, Any]:
    symptoms = set(state["intake_data"].get("symptoms", []))
    severity = set(state["intake_data"].get("severity_descriptors", []))

    if not symptoms:
        triage = "escalate"
        rationale = "No structured symptoms were extracted from the intake."
    elif "chest pain" in symptoms or "shortness of breath" in symptoms:
        triage = "urgent"
        rationale = "Symptoms include a higher-priority cardiorespiratory complaint."
    elif "severe" in severity or "sudden" in severity:
        triage = "urgent"
        rationale = "Severity language suggests prompt review."
    elif symptoms <= {"sore throat", "cough", "fever"}:
        triage = "home_care"
        rationale = "Symptoms fit a simple upper-respiratory demo pattern."
    else:
        triage = "routine"
        rationale = "Symptoms were recognized but do not meet urgent demo rules."

    return {
        "triage": {
            "level": triage,
            "rationale": rationale,
            "valid_levels": sorted(VALID_TRIAGE_LEVELS),
        }
    }


def _normalize_triage_output(parsed: dict[str, Any]) -> dict[str, Any]:
    level = str(parsed.get("level", "")).strip()
    if level not in VALID_TRIAGE_LEVELS:
        raise ValueError(f"Invalid hybrid triage level: {level}")

    rationale = str(parsed.get("rationale", "")).strip() or "Hybrid triage returned no rationale."
    return {
        "triage": {
            "level": level,
            "rationale": rationale,
            "valid_levels": sorted(VALID_TRIAGE_LEVELS),
        }
    }


def _hybrid_placeholder(state: dict[str, Any]) -> dict[str, Any]:
    prompt_text = load_prompt("triage_engine")
    diagnostic = init_node_diagnostic("triage_engine")
    try:
        parsed = invoke_json(
            "triage_engine",
            prompt_text,
            {
                "intake_data": state.get("intake_data", {}),
                "completeness": state.get("completeness", {}),
            },
            diagnostic,
        )
        normalized = _normalize_triage_output(parsed)
        diagnostic["normalization_succeeded"] = True
        normalized["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return normalized
    except HybridLLMError as exc:
        fallback = _deterministic_triage(state)
        diagnostic["fallback_triggered"] = True
        diagnostic["fallback_reason"] = diagnostic["fallback_reason"] or exc.failure_mode
        fallback["errors"] = state.get("errors", []) + [f"triage_engine_hybrid_fallback:{diagnostic['fallback_reason']}"]
        fallback["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return fallback
    except Exception as exc:
        fallback = _deterministic_triage(state)
        diagnostic["fallback_triggered"] = True
        diagnostic["fallback_reason"] = f"normalization_failure:{exc.__class__.__name__}"
        fallback["errors"] = state.get("errors", []) + [f"triage_engine_hybrid_fallback:{diagnostic['fallback_reason']}"]
        fallback["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return fallback


def run(state: dict[str, Any]) -> dict[str, Any]:
    mode = get_execution_mode()
    if mode == "hybrid":
        return _hybrid_placeholder(state)
    return _deterministic_triage(state)
