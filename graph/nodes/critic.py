"""Structured critic evidence generation for the MED-SCRIBE demo."""

from __future__ import annotations

import re
from typing import Any

from graph.config import get_execution_mode
from graph.diagnostics import append_node_diagnostic, init_node_diagnostic
from graph.llm_client import HybridLLMError, invoke_json
from graph.prompt_loader import load_prompt


VALID_TRIAGE_LEVELS = {"urgent", "routine", "home_care", "escalate"}
VALID_ICD_STATUSES = {"OK", "PARTIAL_MATCH", "NO_MATCH_FOUND"}
VALID_CRITIC_STATUSES = {"pass", "revise", "fail"}
REASON_CODE_PATTERN = re.compile(r"^[A-Z0-9]+(?:_[A-Z0-9]+)*$")
REQUIRED_CRITIC_KEYS = {
    "diagnosis_consistency_score",
    "symptom_alignment_score",
    "icd_specificity_score",
    "recommended_status",
    "confidence",
    "reason_codes",
    "summary",
}
CRITIC_CONTRACT_SUFFIX = """

Return exactly one JSON object with exactly these keys:
- diagnosis_consistency_score
- symptom_alignment_score
- icd_specificity_score
- recommended_status
- confidence
- reason_codes
- summary

Contract requirements:
- scores and confidence must be floats between 0.0 and 1.0
- recommended_status must be one of: pass, revise, fail
- reason_codes must be an array of uppercase underscore-delimited strings
- summary must be short
- no markdown
- no prose outside JSON
""".strip()
PARSE_FAILURE_REVIEW = {
    "diagnosis_consistency_score": 0.0,
    "symptom_alignment_score": 0.0,
    "icd_specificity_score": 0.0,
    "recommended_status": "fail",
    "confidence": 0.0,
    "reason_codes": [
        "CRITIC_OUTPUT_PARSE_FAILURE",
    ],
    "summary": "Critic output could not be parsed.",
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 2)))


def _dedupe_reason_codes(codes: list[str]) -> list[str]:
    deduped: list[str] = []
    for code in codes:
        if code not in deduped:
            deduped.append(code)
    return deduped


def _validate_score(value: Any, field_name: str) -> float:
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return round(score, 2)


def _normalize_reason_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("reason_codes must be a list")

    normalized: list[str] = []
    for item in value:
        code = str(item).strip()
        if not code or not REASON_CODE_PATTERN.fullmatch(code):
            raise ValueError(f"Invalid reason code: {item}")
        normalized.append(code)
    return _dedupe_reason_codes(normalized)


def _normalize_summary(value: Any) -> str:
    summary = str(value).strip()
    if not summary:
        raise ValueError("summary is required")
    if len(summary) > 160:
        raise ValueError("summary must be short")
    return summary


def _build_review(
    diagnosis_consistency_score: float,
    symptom_alignment_score: float,
    icd_specificity_score: float,
    recommended_status: str,
    confidence: float,
    reason_codes: list[str],
    summary: str,
    extra_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    review = {
        "diagnosis_consistency_score": _clamp(diagnosis_consistency_score),
        "symptom_alignment_score": _clamp(symptom_alignment_score),
        "icd_specificity_score": _clamp(icd_specificity_score),
        "recommended_status": recommended_status,
        "confidence": _clamp(confidence),
        "reason_codes": _dedupe_reason_codes(reason_codes),
        "summary": summary,
    }
    if extra_scores:
        for key, value in extra_scores.items():
            review[key] = _clamp(value)
    return {
        "critic_review": review
    }


def _parse_failure_review() -> dict[str, Any]:
    return {"critic_review": dict(PARSE_FAILURE_REVIEW)}


def _deterministic_review(state: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    severity = "pass"

    diagnoses = state.get("diagnoses", [])
    triage = state.get("triage", {})
    intake_data = state.get("intake_data", {})
    icd_mappings = state.get("icd_mappings", [])
    ambiguity_flag = bool(intake_data.get("ambiguity_flag"))
    ambiguity_reasons = set(intake_data.get("ambiguity_reasons", []))
    contradiction_flag = "conflicting_signals" in ambiguity_reasons
    diagnosis_score = 1.0
    symptom_score = 1.0
    icd_score = 1.0
    completeness_score = 0.0
    specificity_score = 0.0
    coherence_score = 0.0

    if len(diagnoses) > 3:
        reason_codes.append("DIAGNOSIS_COUNT_EXCEEDS_LIMIT")
        diagnosis_score = 0.0
        severity = "fail"
    elif not diagnoses and intake_data.get("symptoms"):
        reason_codes.append("NO_DIAGNOSES_GENERATED")
        diagnosis_score = 0.45
        if severity != "fail":
            severity = "revise"

    triage_level = triage.get("level")
    if triage_level not in VALID_TRIAGE_LEVELS:
        reason_codes.append("INVALID_TRIAGE_LEVEL")
        symptom_score = 0.0
        severity = "fail"

    for mapping in icd_mappings:
        if mapping.get("status") not in VALID_ICD_STATUSES:
            reason_codes.append("INVALID_ICD_MAPPING_STATUS")
            icd_score = 0.0
            severity = "fail"
            break

    if not intake_data or not triage or not isinstance(icd_mappings, list):
        reason_codes.append("REQUIRED_OUTPUTS_MISSING")
        diagnosis_score = 0.0
        symptom_score = 0.0
        icd_score = 0.0
        severity = "fail"

    symptoms = set(intake_data.get("symptoms", []))
    strong_patterns = {
        frozenset({"fever", "cough", "fatigue"}),
        frozenset({"chest pain", "shortness of breath"}),
        frozenset({"fever", "sore throat"}),
        frozenset({"urinary burning", "frequent urination"}),
        frozenset({"joint pain", "morning stiffness"}),
    }
    ambiguous_patterns = {
        frozenset({"fatigue", "headache"}),
        frozenset({"fever", "fatigue", "muscle aches"}),
        frozenset({"abdominal pain", "nausea/vomiting"}),
        frozenset({"cough", "fatigue"}),
        frozenset({"dizziness", "fatigue"}),
    }
    cluster_map = {
        "respiratory": {"fever", "cough", "sore throat", "runny nose", "sneezing"},
        "cardio": {"chest pain", "shortness of breath", "leg swelling"},
        "urinary": {"urinary burning", "frequent urination"},
        "musculoskeletal": {"joint pain", "morning stiffness", "back pain", "muscle aches"},
        "neuro": {"headache", "dizziness", "fatigue"},
        "gi": {"abdominal pain", "nausea/vomiting"},
    }
    active_clusters = {
        name for name, members in cluster_map.items()
        if symptoms.intersection(members)
    }
    structured_ambiguity = (
        ambiguity_flag
        and bool(diagnoses)
        and len(diagnoses) <= 2
        and bool(icd_mappings)
        and all(mapping.get("status") in {"OK", "PARTIAL_MATCH"} for mapping in icd_mappings)
    )
    if triage_level == "home_care" and ("chest pain" in symptoms or "shortness of breath" in symptoms):
        reason_codes.append("HOME_CARE_HIGH_PRIORITY_CONFLICT")
        symptom_score = 0.0
        severity = "fail"

    if (
        len(active_clusters) > 1
        and frozenset(symptoms) not in strong_patterns
        and frozenset(symptoms) not in ambiguous_patterns
    ):
        if structured_ambiguity:
            reason_codes.append("MULTI_CLUSTER_CONFLICT_HANDLED")
            diagnosis_score = min(diagnosis_score, 0.7)
            symptom_score = min(symptom_score, 0.7)
            severity = "revise"
        else:
            reason_codes.append("MULTI_CLUSTER_CONFLICT")
            diagnosis_score = min(diagnosis_score, 0.2)
            symptom_score = 0.0
            severity = "fail"

    if len(symptoms) <= 1 and triage_level != "urgent":
        reason_codes.append("LOW_EVIDENCE_PRESENTATION")
        diagnosis_score = min(diagnosis_score, 0.45)
        symptom_score = min(symptom_score, 0.45)
        severity = "fail"

    if frozenset(symptoms) in ambiguous_patterns and severity == "pass":
        reason_codes.append("AMBIGUOUS_SYMPTOM_PATTERN")
        diagnosis_score = min(diagnosis_score, 0.7)
        symptom_score = min(symptom_score, 0.7)
        severity = "revise"

    if any(diagnosis == "Symptom-based follow-up needed" for diagnosis in diagnoses):
        reason_codes.append("GENERIC_DIAGNOSIS")
        diagnosis_score = min(diagnosis_score, 0.45)
        if severity != "fail":
            severity = "revise"

    if triage_level == "escalate" and not reason_codes:
        reason_codes.append("INCOMPLETE_INTAKE_ESCALATION")
        symptom_score = min(symptom_score, 0.6)
        severity = "revise"

    if isinstance(icd_mappings, list) and icd_mappings:
        score_total = 0.0
        has_partial_match = False
        has_no_match = False
        for mapping in icd_mappings:
            status = mapping.get("status")
            if status == "OK":
                score_total += 1.0
            elif status == "PARTIAL_MATCH":
                score_total += 0.5
                has_partial_match = True
            elif status == "NO_MATCH_FOUND":
                has_no_match = True
        icd_score = min(icd_score, score_total / len(icd_mappings))
        if has_partial_match:
            reason_codes.append("PARTIAL_ICD_MAPPING")
            if severity == "pass":
                severity = "revise"
        if has_no_match:
            reason_codes.append("NO_ICD_MATCH_FOUND")
            severity = "fail"
            icd_score = 0.0
    elif diagnoses or symptoms:
        reason_codes.append("ICD_MAPPING_MISSING")
        severity = "fail"
        icd_score = 0.0

    has_complete_intake = bool(symptoms) and bool(intake_data.get("duration")) and not intake_data.get("missing_fields")
    is_strong_pattern = frozenset(symptoms) in strong_patterns
    all_ok_mappings = bool(icd_mappings) and all(mapping.get("status") == "OK" for mapping in icd_mappings)
    coherent_diagnoses = bool(diagnoses) and len(set(diagnoses)) == len(diagnoses)
    single_cluster = len(active_clusters) <= 1 or is_strong_pattern

    if has_complete_intake:
        completeness_score = 0.6 if "LOW_EVIDENCE_PRESENTATION" in reason_codes else 1.0
    if all_ok_mappings and diagnoses:
        specificity_score = 1.0 if (is_strong_pattern or triage_level == "urgent") else 0.45
    if coherent_diagnoses and all_ok_mappings and single_cluster:
        coherence_score = 1.0 if severity == "pass" else 0.4

    if structured_ambiguity:
        diagnosis_score = max(diagnosis_score, 0.7)
        symptom_score = max(symptom_score, 0.7)
        completeness_score = 1.0
        specificity_score = max(specificity_score, 0.45 if not all_ok_mappings else 1.0)
        coherence_score = max(coherence_score, 1.0 if all_ok_mappings else 0.4)
        if "LOW_EVIDENCE_PRESENTATION" not in reason_codes:
            severity = "revise" if severity != "fail" or frozenset(symptoms) in ambiguous_patterns else severity
        if "AMBIGUITY_STRUCTURED_HANDLING" not in reason_codes:
            reason_codes.append("AMBIGUITY_STRUCTURED_HANDLING")

    if contradiction_flag:
        diagnosis_score = min(diagnosis_score, 1.0)
        symptom_score = min(symptom_score, 1.0)
        completeness_score = min(completeness_score, 1.0)
        specificity_score = min(specificity_score, 0.45 if all_ok_mappings else specificity_score)
        coherence_score = min(coherence_score, 0.4)
        severity = "revise" if severity == "pass" else severity
        if "STRUCTURAL_CONTRADICTION_CAP" not in reason_codes:
            reason_codes.append("STRUCTURAL_CONTRADICTION_CAP")

    if (
        diagnosis_score >= 0.9
        and symptom_score >= 0.9
        and icd_score >= 0.9
        and completeness_score >= 0.9
        and specificity_score >= 0.9
        and coherence_score >= 0.9
        and not contradiction_flag
    ):
        severity = "pass"
        if "CRITIC_REVIEW_CLEAR" not in reason_codes:
            reason_codes = ["CRITIC_REVIEW_CLEAR"]

    if severity == "pass" and not reason_codes:
        reason_codes.append("CRITIC_REVIEW_CLEAR")

    confidence = sum((diagnosis_score, symptom_score, icd_score)) / 3
    if severity == "fail":
        confidence = min(confidence, 0.45)
    elif severity == "revise":
        confidence = min(confidence, 0.62 if ambiguity_flag else 0.69)
    if contradiction_flag:
        confidence = min(confidence, 0.62)

    summary_map = {
        "pass": "Critic evidence supports pass.",
        "revise": "Critic evidence suggests revision.",
        "fail": "Critic evidence indicates failure.",
    }
    status = severity if severity in VALID_CRITIC_STATUSES else "fail"
    return _build_review(
        diagnosis_consistency_score=diagnosis_score,
        symptom_alignment_score=symptom_score,
        icd_specificity_score=icd_score,
        recommended_status=status,
        confidence=confidence,
        reason_codes=reason_codes,
        summary=summary_map[status],
        extra_scores={
            "completeness_score": completeness_score,
            "specificity_score": specificity_score,
            "coherence_score": coherence_score,
        },
    )


def _normalize_critic_output(parsed: dict[str, Any]) -> dict[str, Any]:
    if set(parsed.keys()) != REQUIRED_CRITIC_KEYS:
        raise ValueError("critic output keys do not match required contract")

    status = str(parsed.get("recommended_status", "")).strip().lower()
    if status not in VALID_CRITIC_STATUSES:
        raise ValueError(f"Invalid hybrid critic status: {status}")

    return _build_review(
        diagnosis_consistency_score=_validate_score(parsed.get("diagnosis_consistency_score"), "diagnosis_consistency_score"),
        symptom_alignment_score=_validate_score(parsed.get("symptom_alignment_score"), "symptom_alignment_score"),
        icd_specificity_score=_validate_score(parsed.get("icd_specificity_score"), "icd_specificity_score"),
        recommended_status=status,
        confidence=_validate_score(parsed.get("confidence"), "confidence"),
        reason_codes=_normalize_reason_codes(parsed.get("reason_codes")),
        summary=_normalize_summary(parsed.get("summary")),
    )


def _hybrid_placeholder(state: dict[str, Any]) -> dict[str, Any]:
    prompt_text = f"{load_prompt('critic')}\n\n{CRITIC_CONTRACT_SUFFIX}"
    diagnostic = init_node_diagnostic("critic")
    try:
        parsed = invoke_json(
            "critic",
            prompt_text,
            {
                "intake_data": state.get("intake_data", {}),
                "triage": state.get("triage", {}),
                "diagnoses": state.get("diagnoses", []),
                "icd_mappings": state.get("icd_mappings", []),
            },
            diagnostic,
        )
        normalized = _normalize_critic_output(parsed)
        diagnostic["normalization_succeeded"] = True
        normalized["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return normalized
    except HybridLLMError as exc:
        if exc.failure_mode in {"json_parse_failure", "non_json_response", "contract_rejection"}:
            fallback = _parse_failure_review()
        else:
            fallback = _deterministic_review(state)
        diagnostic["fallback_triggered"] = True
        diagnostic["fallback_reason"] = diagnostic["fallback_reason"] or exc.failure_mode
        fallback["errors"] = state.get("errors", []) + [f"critic_hybrid_fallback:{diagnostic['fallback_reason']}"]
        fallback["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return fallback
    except Exception as exc:
        fallback = _parse_failure_review()
        diagnostic["fallback_triggered"] = True
        diagnostic["fallback_reason"] = f"normalization_failure:{exc.__class__.__name__}"
        fallback["errors"] = state.get("errors", []) + [f"critic_hybrid_fallback:{diagnostic['fallback_reason']}"]
        fallback["node_diagnostics"] = append_node_diagnostic(state, diagnostic)
        return fallback


def run(state: dict[str, Any]) -> dict[str, Any]:
    mode = get_execution_mode()
    if mode == "hybrid":
        return _hybrid_placeholder(state)
    return _deterministic_review(state)
