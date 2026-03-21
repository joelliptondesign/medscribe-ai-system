"""Deterministic intake parsing for the MED-SCRIBE demo."""

from __future__ import annotations

import re
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

STRUCTURED_SYMPTOM_ALIASES = {
    "abdominal discomfort": "abdominal pain",
    "burning urination": "urinary burning",
    "burning when urinating": "urinary burning",
    "mild headache": "headache",
    "nausea": "nausea/vomiting",
    "vomiting": "nausea/vomiting",
}

SEVERITY_TERMS = ["severe", "worse", "worsening", "intense", "sudden"]
DURATION_HINTS = ["day", "days", "week", "weeks", "today", "yesterday"]
VALID_SYMPTOMS = set(SYMPTOM_RULES)
SYMPTOM_CLUSTERS = {
    "respiratory": {"fever", "cough", "sore throat"},
    "cardio": {"chest pain", "shortness of breath"},
    "urinary": {"urinary burning", "frequent urination"},
    "musculoskeletal": {"joint pain", "morning stiffness", "back pain", "muscle aches"},
    "neuro": {"headache", "dizziness", "fatigue"},
    "gi": {"abdominal pain", "nausea/vomiting"},
}
AMBIGUOUS_PATTERNS = {
    frozenset({"fatigue", "headache"}),
    frozenset({"fever", "fatigue", "muscle aches"}),
    frozenset({"abdominal pain", "nausea/vomiting"}),
    frozenset({"cough", "fatigue"}),
    frozenset({"dizziness", "fatigue"}),
    frozenset({"chest pain", "leg swelling"}),
}
CONFLICT_PATTERNS = {
    frozenset({"chest pain", "leg swelling"}),
    frozenset({"fever", "cough", "urinary burning"}),
}


def _normalize_symptom(symptom: str) -> str:
    cleaned = symptom.strip().strip(".").lower()
    return STRUCTURED_SYMPTOM_ALIASES.get(cleaned, cleaned)


def _extract_structured_symptoms(text: str) -> list[str]:
    match = re.search(r"with symptoms:\s*(.+?)(?:\.\s*Duration:|$)", text, flags=re.IGNORECASE)
    if not match:
        return []

    symptoms: list[str] = []
    for item in match.group(1).split(","):
        normalized = _normalize_symptom(item)
        if normalized and normalized not in symptoms:
            symptoms.append(normalized)
    return symptoms


def _extract_duration(text: str, lowered: str) -> str | None:
    match = re.search(r"Duration:\s*([^\.]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if any(hint in lowered for hint in DURATION_HINTS):
        return text.strip()
    return None


def _ambiguity_details(symptoms: list[str]) -> tuple[bool, list[str]]:
    symptom_set = set(symptoms)
    reasons: list[str] = []
    active_clusters = [
        name for name, members in SYMPTOM_CLUSTERS.items()
        if symptom_set.intersection(members)
    ]

    if frozenset(symptom_set) in AMBIGUOUS_PATTERNS:
        reasons.append("ambiguous_pattern")
    if len(active_clusters) > 1:
        reasons.append("multiple_symptom_clusters")
    if len(symptom_set) <= 1:
        reasons.append("low_evidence_input")
    if frozenset(symptom_set) in CONFLICT_PATTERNS:
        reasons.append("conflicting_signals")

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return bool(deduped), deduped


def _deterministic_intake(state: dict[str, Any]) -> dict[str, Any]:
    text = state["raw_input"].strip()
    lowered = text.lower()

    symptoms = _extract_structured_symptoms(text)
    if not symptoms:
        symptoms = [
            label
            for label, keywords in SYMPTOM_RULES.items()
            if any(keyword in lowered for keyword in keywords)
        ]
    severity_descriptors = [term for term in SEVERITY_TERMS if term in lowered]
    duration = _extract_duration(text, lowered)

    intake_data = {
        "symptoms": symptoms,
        "duration": duration,
        "medications": [],
        "allergies": [],
        "comorbidities": [],
        "severity_descriptors": severity_descriptors,
        "missing_fields": [],
        "missing_data_questions": [],
        "ambiguity_flag": False,
        "ambiguity_reasons": [],
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

    ambiguity_flag, ambiguity_reasons = _ambiguity_details(symptoms)
    intake_data["ambiguity_flag"] = ambiguity_flag
    intake_data["ambiguity_reasons"] = ambiguity_reasons

    if ambiguity_flag and "What additional context would help disambiguate the presentation?" not in intake_data["missing_data_questions"]:
        intake_data["missing_data_questions"].append("What additional context would help disambiguate the presentation?")

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
        "ambiguity_flag": False,
        "ambiguity_reasons": [],
    }

    if not symptoms and "symptoms" not in intake_data["missing_fields"]:
        intake_data["missing_fields"].append("symptoms")
        intake_data["missing_data_questions"].append("What symptoms are you experiencing?")

    if duration is None and "duration" not in intake_data["missing_fields"]:
        intake_data["missing_fields"].append("duration")
        intake_data["missing_data_questions"].append("How long have the symptoms been present?")

    ambiguity_flag, ambiguity_reasons = _ambiguity_details(symptoms)
    intake_data["ambiguity_flag"] = ambiguity_flag
    intake_data["ambiguity_reasons"] = ambiguity_reasons

    if ambiguity_flag and "What additional context would help disambiguate the presentation?" not in intake_data["missing_data_questions"]:
        intake_data["missing_data_questions"].append("What additional context would help disambiguate the presentation?")

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
