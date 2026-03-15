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
) -> dict[str, Any]:
    return {
        "critic_review": {
            "diagnosis_consistency_score": _clamp(diagnosis_consistency_score),
            "symptom_alignment_score": _clamp(symptom_alignment_score),
            "icd_specificity_score": _clamp(icd_specificity_score),
            "recommended_status": recommended_status,
            "confidence": _clamp(confidence),
            "reason_codes": _dedupe_reason_codes(reason_codes),
            "summary": summary,
        }
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
    diagnosis_score = 1.0
    symptom_score = 1.0
    icd_score = 1.0

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
    if triage_level == "home_care" and ("chest pain" in symptoms or "shortness of breath" in symptoms):
        reason_codes.append("HOME_CARE_HIGH_PRIORITY_CONFLICT")
        symptom_score = 0.0
        severity = "fail"

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

    if severity == "pass" and not reason_codes:
        reason_codes.append("CRITIC_REVIEW_CLEAR")

    confidence = sum((diagnosis_score, symptom_score, icd_score)) / 3
    if severity == "fail":
        confidence = min(confidence, 0.45)
    elif severity == "revise":
        confidence = min(confidence, 0.69)

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
