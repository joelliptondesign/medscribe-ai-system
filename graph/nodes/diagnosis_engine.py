"""Deterministic diagnosis suggestions for the MED-SCRIBE demo."""

from __future__ import annotations

from typing import Any

from graph.config import get_execution_mode
from graph.diagnostics import append_node_diagnostic, init_node_diagnostic
from graph.llm_client import HybridLLMError, invoke_json
from graph.prompt_loader import load_prompt


DIAGNOSIS_RULES = [
    ({"fever", "cough", "sore throat"}, "Upper respiratory infection"),
    ({"fever", "cough"}, "Viral syndrome"),
    ({"sore throat"}, "Pharyngitis"),
    ({"headache"}, "Tension headache"),
    ({"abdominal pain", "nausea/vomiting"}, "Gastroenteritis"),
    ({"urinary burning"}, "Dysuria"),
    ({"back pain"}, "Musculoskeletal back pain"),
    ({"chest pain"}, "Chest pain syndrome"),
    ({"shortness of breath"}, "Dyspnea"),
]


def _deterministic_diagnoses(state: dict[str, Any]) -> dict[str, Any]:
    symptoms = set(state["intake_data"].get("symptoms", []))
    diagnoses: list[str] = []

    for required_symptoms, diagnosis in DIAGNOSIS_RULES:
        if required_symptoms.issubset(symptoms) and diagnosis not in diagnoses:
            diagnoses.append(diagnosis)
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
