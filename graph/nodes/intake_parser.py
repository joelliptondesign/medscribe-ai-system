"""Deterministic intake parsing for the MED-SCRIBE demo."""

from __future__ import annotations

from typing import Any

from graph.config import get_execution_mode
from graph.diagnostics import append_node_diagnostic, init_node_diagnostic
from graph.llm_client import HybridLLMError, invoke_json
from graph.prompt_loader import load_prompt


SYMPTOM_RULES = {
    "chest pain": ["chest pain", "chest tightness"],
    "shortness of breath": ["shortness of breath", "breathing trouble", "can't breathe"],
    "fever": ["fever", "temperature", "chills"],
    "cough": ["cough", "coughing"],
    "sore throat": ["sore throat", "throat pain"],
    "headache": ["headache", "migraine"],
    "abdominal pain": ["abdominal pain", "stomach pain", "belly pain"],
    "nausea/vomiting": ["nausea", "vomiting", "throwing up"],
    "urinary burning": ["urinary burning", "burning urination", "burning when urinating"],
    "back pain": ["back pain", "lower back pain"],
}

SEVERITY_TERMS = ["severe", "worse", "worsening", "intense", "sudden"]
DURATION_HINTS = ["day", "days", "week", "weeks", "today", "yesterday"]
VALID_SYMPTOMS = set(SYMPTOM_RULES)


def _deterministic_intake(state: dict[str, Any]) -> dict[str, Any]:
    text = state["raw_input"].strip()
    lowered = text.lower()

    symptoms = [
        label
        for label, keywords in SYMPTOM_RULES.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    severity_descriptors = [term for term in SEVERITY_TERMS if term in lowered]
    duration = text if any(hint in lowered for hint in DURATION_HINTS) else None

    intake_data = {
        "symptoms": symptoms,
        "duration": duration,
        "medications": [],
        "allergies": [],
        "comorbidities": [],
        "severity_descriptors": severity_descriptors,
        "missing_fields": [],
        "missing_data_questions": [],
    }

    completeness = {
        "has_symptoms": bool(symptoms),
        "has_duration": duration is not None,
        "has_minimum_intake": bool(symptoms),
    }

    if not symptoms:
        intake_data["missing_fields"].append("symptoms")
        intake_data["missing_data_questions"].append("What symptoms are you experiencing?")

    if duration is None:
        intake_data["missing_fields"].append("duration")
        intake_data["missing_data_questions"].append("How long have the symptoms been present?")

    return {
        "intake_data": intake_data,
        "completeness": completeness,
    }


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_intake_output(parsed: dict[str, Any]) -> dict[str, Any]:
    symptoms = [item for item in _normalize_string_list(parsed.get("symptoms")) if item in VALID_SYMPTOMS]
    duration_raw = parsed.get("duration")
    duration = str(duration_raw).strip() if duration_raw not in (None, "") else None

    intake_data = {
        "symptoms": symptoms,
        "duration": duration,
        "medications": _normalize_string_list(parsed.get("medications")),
        "allergies": _normalize_string_list(parsed.get("allergies")),
        "comorbidities": _normalize_string_list(parsed.get("comorbidities")),
        "severity_descriptors": _normalize_string_list(parsed.get("severity_descriptors")),
        "missing_fields": _normalize_string_list(parsed.get("missing_fields")),
        "missing_data_questions": _normalize_string_list(parsed.get("missing_data_questions")),
    }

    if not symptoms and "symptoms" not in intake_data["missing_fields"]:
        intake_data["missing_fields"].append("symptoms")
        intake_data["missing_data_questions"].append("What symptoms are you experiencing?")

    if duration is None and "duration" not in intake_data["missing_fields"]:
        intake_data["missing_fields"].append("duration")
        intake_data["missing_data_questions"].append("How long have the symptoms been present?")

    completeness = {
        "has_symptoms": bool(symptoms),
        "has_duration": duration is not None,
        "has_minimum_intake": bool(symptoms),
    }

    return {
        "intake_data": intake_data,
        "completeness": completeness,
    }


def _hybrid_placeholder(state: dict[str, Any]) -> dict[str, Any]:
    prompt_text = load_prompt("intake_parser")
    diagnostic = init_node_diagnostic("intake_parser")
    try:
        parsed = invoke_json("intake_parser", prompt_text, {"raw_input": state.get("raw_input", "")}, diagnostic)
        normalized = _normalize_intake_output(parsed)
        diagnostic["normalization_succeeded"] = True
        normalized["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return normalized
    except HybridLLMError as exc:
        fallback = _deterministic_intake(state)
        diagnostic["fallback_triggered"] = True
        diagnostic["fallback_reason"] = diagnostic["fallback_reason"] or exc.failure_mode
        fallback["errors"] = state.get("errors", []) + [f"intake_parser_hybrid_fallback:{diagnostic['fallback_reason']}"]
        fallback["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return fallback
    except Exception as exc:
        fallback = _deterministic_intake(state)
        diagnostic["fallback_triggered"] = True
        diagnostic["fallback_reason"] = f"normalization_failure:{exc.__class__.__name__}"
        fallback["errors"] = state.get("errors", []) + [f"intake_parser_hybrid_fallback:{diagnostic['fallback_reason']}"]
        fallback["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return fallback


def run(state: dict[str, Any]) -> dict[str, Any]:
    mode = get_execution_mode()
    if mode == "hybrid":
        return _hybrid_placeholder(state)
    return _deterministic_intake(state)
