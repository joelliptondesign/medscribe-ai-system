"""Deterministic governance arbitration for MED-SCRIBE critic evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


POLICY_PATH = Path(__file__).resolve().parents[2] / "governance" / "policy_rules.json"
STATUS_RANK = {"PASS": 0, "REVISE": 1, "FAIL": 2}
FAIL_RULE_IDS = {
    "diagnosis_consistency_score": "RULE_DIAGNOSIS_CONSISTENCY_FAIL",
    "symptom_alignment_score": "RULE_SYMPTOM_ALIGNMENT_FAIL",
    "icd_specificity_score": "RULE_ICD_SPECIFICITY_FAIL",
    "confidence": "RULE_CONFIDENCE_FAIL",
}


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


def run(state: dict[str, Any]) -> dict[str, Any]:
    policy = _load_policy()
    thresholds = policy["thresholds"]
    rule_ids = policy["rule_ids"]
    critic_review = state.get("critic_review", {})

    governance_status = "PASS"
    applied_rules: list[str] = []
    reason_codes = _dedupe(list(critic_review.get("reason_codes", [])))

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

    recommended_status = str(critic_review.get("recommended_status", "")).strip().lower()
    if recommended_status == "fail":
        applied_rules.append(rule_ids["critic_recommendation_fail"])
        governance_status = _escalate_status(governance_status, "FAIL")
    elif recommended_status == "revise":
        applied_rules.append(rule_ids["critic_recommendation_revise"])
        governance_status = _escalate_status(governance_status, "REVISE")

    governance_result = {
        "final_status": governance_status,
        "escalation_required": governance_status in {"REVISE", "FAIL"},
        "policy_version": policy["policy_version"],
        "governance_version": policy["governance_version"],
        "applied_rules": _dedupe(applied_rules),
        "reason_codes": _dedupe(reason_codes),
    }
    return {
        "governance_result": governance_result,
        "escalation_required": governance_result["escalation_required"],
    }
