"""Deterministic governance arbitration for MED-SCRIBE critic evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


POLICY_PATH = Path(__file__).resolve().parents[2] / "governance" / "policy_rules.json"
EXPERIMENTAL_ESCALATED_REVIEW_STATUS = "REVISE_ESCALATED"
EXPERIMENTAL_ESCALATION_RULE = "EXPERIMENTAL_RULE_URGENT_TRIAGE_ICD_SPECIFICITY_ESCALATED_REVIEW"
EXPERIMENTAL_ESCALATION_REASON = "EXPERIMENTAL_URGENT_TRIAGE_CODING_REVIEW"
STATUS_RANK = {"PASS": 0, "REVISE": 1, EXPERIMENTAL_ESCALATED_REVIEW_STATUS: 1, "FAIL": 2}
FAIL_RULE_IDS = {
    "diagnosis_consistency_score": "RULE_DIAGNOSIS_CONSISTENCY_FAIL",
    "symptom_alignment_score": "RULE_SYMPTOM_ALIGNMENT_FAIL",
    "icd_specificity_score": "RULE_ICD_SPECIFICITY_FAIL",
    "confidence": "RULE_CONFIDENCE_FAIL",
}
GOVERNANCE_INPUTS_USED = [
    "critic_review.diagnosis_consistency_score",
    "critic_review.symptom_alignment_score",
    "critic_review.icd_specificity_score",
    "critic_review.confidence",
    "critic_review.recommended_status",
    "critic_review.reason_codes",
]
GOVERNANCE_INPUTS_IGNORED = [
    "intake_data.symptoms",
    "intake_data.severity_descriptors",
    "intake_data.duration",
    "triage.level",
    "triage.rationale",
    "diagnoses",
    "icd_mappings",
]


def _load_policy() -> dict[str, Any]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _escalate_status(current: str, candidate: str) -> str:
    return candidate if STATUS_RANK[candidate] > STATUS_RANK[current] else current


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _score_band(score: float, pass_min: float, revise_min: float) -> str:
    if score >= pass_min:
        return "pass"
    if score >= revise_min:
        return "revise"
    return "fail"


def _evaluate_metric(
    score: float,
    pass_min: float,
    revise_min: float,
    pass_rule: str,
    revise_rule: str,
    fail_rule: str,
    low_reason: str,
) -> tuple[str, str, str | None]:
    band = _score_band(score, pass_min, revise_min)
    if band == "pass":
        return "PASS", pass_rule, None
    if band == "revise":
        return "REVISE", revise_rule, low_reason
    return "FAIL", fail_rule, low_reason


def _upstream_context_summary(state: dict[str, Any]) -> dict[str, Any]:
    intake_data = state.get("intake_data", {})
    triage = state.get("triage", {})
    diagnoses = state.get("diagnoses", [])
    icd_mappings = state.get("icd_mappings", [])
    return {
        "triage_level": triage.get("level"),
        "triage_rationale": triage.get("rationale"),
        "symptoms": intake_data.get("symptoms", []),
        "severity_descriptors": intake_data.get("severity_descriptors", []),
        "duration": intake_data.get("duration"),
        "diagnosis_count": len(diagnoses) if isinstance(diagnoses, list) else 0,
        "icd_mapping_statuses": [
            mapping.get("status")
            for mapping in icd_mappings
            if isinstance(mapping, dict)
        ],
    }


def _experimental_escalated_review_applies(
    state: dict[str, Any],
    governance_status: str,
    governance_fail_drivers: list[dict[str, Any]],
    recommended_status: str,
) -> bool:
    triage_level = str(state.get("triage", {}).get("level", "")).strip().lower()
    return (
        governance_status == "FAIL"
        and triage_level in {"urgent", "escalate"}
        and recommended_status == "revise"
        and len(governance_fail_drivers) == 1
        and governance_fail_drivers[0].get("input") == "critic_review.icd_specificity_score"
    )


def run(state: dict[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    thresholds = policy["thresholds"]
    rule_ids = policy["rule_ids"]
    critic_review = state.get("critic_review", {})

    governance_status = "PASS"
    applied_rules: list[str] = []
    reason_codes = _dedupe(list(critic_review.get("reason_codes", [])))
    governance_rule_evaluations: list[dict[str, Any]] = []
    governance_fail_drivers: list[dict[str, Any]] = []

    metric_specs = [
        (
            "diagnosis_consistency_score",
            thresholds["diagnosis_consistency_min_for_pass"],
            thresholds["diagnosis_consistency_min_for_revise"],
            rule_ids["diagnosis_consistency_pass"],
            rule_ids["diagnosis_consistency_revise"],
            "LOW_DIAGNOSIS_CONSISTENCY",
        ),
        (
            "symptom_alignment_score",
            thresholds["symptom_alignment_min_for_pass"],
            thresholds["symptom_alignment_min_for_revise"],
            rule_ids["symptom_alignment_pass"],
            rule_ids["symptom_alignment_revise"],
            "LOW_SYMPTOM_ALIGNMENT",
        ),
        (
            "icd_specificity_score",
            thresholds["icd_specificity_min_for_pass"],
            thresholds["icd_specificity_min_for_revise"],
            rule_ids["icd_specificity_pass"],
            rule_ids["icd_specificity_revise"],
            "LOW_ICD_SPECIFICITY",
        ),
        (
            "confidence",
            thresholds["confidence_min_for_pass"],
            thresholds["confidence_min_for_revise"],
            rule_ids["confidence_pass"],
            rule_ids["confidence_revise"],
            "LOW_CRITIC_CONFIDENCE",
        ),
    ]

    for field_name, pass_min, revise_min, pass_rule, revise_rule, low_reason in metric_specs:
        score = float(critic_review.get(field_name, 0.0))
        severity, applied_rule, reason_code = _evaluate_metric(
            score=score,
            pass_min=pass_min,
            revise_min=revise_min,
            pass_rule=pass_rule,
            revise_rule=revise_rule,
            fail_rule=FAIL_RULE_IDS[field_name],
            low_reason=low_reason,
        )
        applied_rules.append(applied_rule)
        if reason_code is not None:
            reason_codes.append(reason_code)
        governance_status = _escalate_status(governance_status, severity)
        governance_rule_evaluations.append(
            {
                "input": f"critic_review.{field_name}",
                "score": score,
                "severity": severity,
                "applied_rule": applied_rule,
                "reason_code": reason_code,
            }
        )
        if severity == "FAIL":
            governance_fail_drivers.append(
                {
                    "input": f"critic_review.{field_name}",
                    "score": score,
                    "applied_rule": applied_rule,
                    "reason_code": reason_code,
                }
            )

    recommended_status = str(critic_review.get("recommended_status", "")).strip().lower()
    if recommended_status == "fail":
        applied_rules.append(rule_ids["critic_recommendation_fail"])
        governance_status = _escalate_status(governance_status, "FAIL")
        governance_rule_evaluations.append(
            {
                "input": "critic_review.recommended_status",
                "value": recommended_status,
                "severity": "FAIL",
                "applied_rule": rule_ids["critic_recommendation_fail"],
                "reason_code": None,
            }
        )
        governance_fail_drivers.append(
            {
                "input": "critic_review.recommended_status",
                "value": recommended_status,
                "applied_rule": rule_ids["critic_recommendation_fail"],
                "reason_code": None,
            }
        )
    elif recommended_status == "revise":
        applied_rules.append(rule_ids["critic_recommendation_revise"])
        governance_status = _escalate_status(governance_status, "REVISE")
        governance_rule_evaluations.append(
            {
                "input": "critic_review.recommended_status",
                "value": recommended_status,
                "severity": "REVISE",
                "applied_rule": rule_ids["critic_recommendation_revise"],
                "reason_code": None,
            }
        )

    policy_simulation_applied = _experimental_escalated_review_applies(
        state,
        governance_status,
        governance_fail_drivers,
        recommended_status,
    )
    if policy_simulation_applied:
        governance_status = EXPERIMENTAL_ESCALATED_REVIEW_STATUS
        applied_rules.append(EXPERIMENTAL_ESCALATION_RULE)
        reason_codes.append(EXPERIMENTAL_ESCALATION_REASON)
        governance_rule_evaluations.append(
            {
                "input": "triage.level",
                "value": state.get("triage", {}).get("level"),
                "severity": EXPERIMENTAL_ESCALATED_REVIEW_STATUS,
                "applied_rule": EXPERIMENTAL_ESCALATION_RULE,
                "reason_code": EXPERIMENTAL_ESCALATION_REASON,
                "experimental_policy_simulation": True,
            }
        )

    governance_result = {
        "final_status": governance_status,
        "escalation_required": governance_status in {"REVISE", "FAIL", EXPERIMENTAL_ESCALATED_REVIEW_STATUS},
        "policy_version": policy["policy_version"],
        "governance_version": policy["governance_version"],
        "applied_rules": _dedupe(applied_rules),
        "reason_codes": _dedupe(reason_codes),
        "policy_simulation": {
            "enabled": True,
            "applied": policy_simulation_applied,
            "experimental_status": EXPERIMENTAL_ESCALATED_REVIEW_STATUS,
            "condition": "urgent_or_escalate_triage_with_single_icd_specificity_fail_driver_and_critic_revise",
            "preserves_escalation_required": True,
        },
        "governance_attribution": {
            "governance_inputs_used": list(GOVERNANCE_INPUTS_USED),
            "governance_inputs_ignored": list(GOVERNANCE_INPUTS_IGNORED),
            "experimental_policy_inputs_used": ["triage.level"] if policy_simulation_applied else [],
            "governance_fail_drivers": governance_fail_drivers,
            "governance_rule_evaluations": governance_rule_evaluations,
            "upstream_context_summary": _upstream_context_summary(state),
        },
    }
    return {
        "governance_result": governance_result,
        "escalation_required": governance_result["escalation_required"],
    }
