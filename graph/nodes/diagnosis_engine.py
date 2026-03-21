"""Deterministic diagnosis suggestions for the MED-SCRIBE demo."""

from __future__ import annotations

from typing import Any

from graph.config import get_execution_mode
from graph.diagnostics import append_node_diagnostic, init_node_diagnostic
from graph.llm_client import HybridLLMError, invoke_json
from graph.prompt_loader import load_prompt


DIAGNOSIS_RULES = [
    ({"fever", "cough", "sore throat"}, "Upper respiratory infection"),
    ({"fever", "fatigue", "muscle aches"}, "Influenza-like illness"),
    ({"fever", "cough", "fatigue"}, "Viral syndrome"),
    ({"fever", "cough"}, "Viral syndrome"),
    ({"fever", "sore throat"}, "Pharyngitis"),
    ({"chest pain", "shortness of breath"}, "Chest pain syndrome"),
    ({"sore throat"}, "Pharyngitis"),
    ({"headache"}, "Tension headache"),
    ({"abdominal pain", "nausea/vomiting"}, "Gastroenteritis"),
    ({"urinary burning", "frequent urination"}, "Dysuria"),
    ({"urinary burning"}, "Dysuria"),
    ({"joint pain", "morning stiffness"}, "Arthralgia"),
    ({"back pain"}, "Musculoskeletal back pain"),
    ({"chest pain"}, "Chest pain syndrome"),
    ({"shortness of breath"}, "Dyspnea"),
]

AMBIGUITY_HYPOTHESES = {
    frozenset({"fatigue", "headache"}): ["Tension headache", "Fatigue syndrome"],
    frozenset({"fever", "fatigue", "muscle aches"}): ["Influenza-like illness", "Viral syndrome"],
    frozenset({"abdominal pain", "nausea/vomiting"}): ["Gastroenteritis", "Nausea/vomiting syndrome"],
    frozenset({"cough", "fatigue"}): ["Acute cough syndrome", "Viral syndrome"],
    frozenset({"dizziness", "fatigue"}): ["Dizziness syndrome", "Fatigue syndrome"],
    frozenset({"chest pain", "leg swelling"}): ["Chest pain syndrome", "Peripheral edema"],
}

VAGUE_DIAGNOSIS_TERMS = ("unspecified", "unknown", "general", "nos")


def _refine_diagnosis(diagnosis: str, symptoms: set[str]) -> str:
    lowered = diagnosis.lower()
    if any(term in lowered for term in VAGUE_DIAGNOSIS_TERMS):
        if {"fever", "fatigue", "muscle aches"}.issubset(symptoms):
            return "Influenza-like illness"
        if {"fever", "cough", "sore throat"}.issubset(symptoms):
            return "Upper respiratory infection"
        if {"urinary burning", "frequent urination"}.issubset(symptoms):
            return "Dysuria"
    return diagnosis


def _deterministic_diagnoses(state: dict[str, Any]) -> dict[str, Any]:
    symptoms = set(state["intake_data"].get("symptoms", []))
    intake_data = state.get("intake_data", {})
    diagnoses: list[str] = []

    if intake_data.get("ambiguity_flag"):
        ranked = AMBIGUITY_HYPOTHESES.get(frozenset(symptoms), [])
        if ranked:
            return {"diagnoses": ranked[:2]}

    for required_symptoms, diagnosis in DIAGNOSIS_RULES:
        if required_symptoms.issubset(symptoms) and diagnosis not in diagnoses:
            diagnoses.append(_refine_diagnosis(diagnosis, symptoms))
        if len(diagnoses) == 3:
            break

    if not diagnoses and symptoms:
        diagnoses.append("Symptom-based follow-up needed")

    return {"diagnoses": diagnoses[:3]}


def _normalize_diagnosis_output(parsed: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    raw = parsed.get("diagnoses")
    if not isinstance(raw, list):
        raise ValueError("Hybrid diagnosis output must include a diagnoses list")

    diagnoses = [str(item).strip() for item in raw if str(item).strip()][:3]
    if not diagnoses and state.get("intake_data", {}).get("symptoms"):
        diagnoses = ["Symptom-based follow-up needed"]
    return {"diagnoses": diagnoses}


def _hybrid_placeholder(state: dict[str, Any]) -> dict[str, Any]:
    prompt_text = load_prompt("diagnosis_engine")
    diagnostic = init_node_diagnostic("diagnosis_engine")
    try:
        parsed = invoke_json(
            "diagnosis_engine",
            prompt_text,
            {
                "intake_data": state.get("intake_data", {}),
                "triage": state.get("triage", {}),
            },
            diagnostic,
        )
        normalized = _normalize_diagnosis_output(parsed, state)
        diagnostic["normalization_succeeded"] = True
        normalized["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return normalized
    except HybridLLMError as exc:
        fallback = _deterministic_diagnoses(state)
        diagnostic["fallback_triggered"] = True
        diagnostic["fallback_reason"] = diagnostic["fallback_reason"] or exc.failure_mode
        fallback["errors"] = state.get("errors", []) + [f"diagnosis_engine_hybrid_fallback:{diagnostic['fallback_reason']}"]
        fallback["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return fallback
    except Exception as exc:
        fallback = _deterministic_diagnoses(state)
        diagnostic["fallback_triggered"] = True
        diagnostic["fallback_reason"] = f"normalization_failure:{exc.__class__.__name__}"
        fallback["errors"] = state.get("errors", []) + [f"diagnosis_engine_hybrid_fallback:{diagnostic['fallback_reason']}"]
        fallback["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return fallback


def run(state: dict[str, Any]) -> dict[str, Any]:
    mode = get_execution_mode()
    if mode == "hybrid":
        return _hybrid_placeholder(state)
    return _deterministic_diagnoses(state)
