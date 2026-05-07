"""Run the bounded synthetic incident pack through the existing MedScribe runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "evaluation" / "synthetic_incidents" / "incidents.json"
SUMMARY_DIR = REPO_ROOT / "evaluation" / "synthetic_incidents"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_case_tags(case: dict[str, Any]) -> list[str]:
    tags = ["medscribe", "synthetic-incident", "phase2"]
    for tag in case.get("tags", []):
        text = str(tag).strip()
        if text and text not in tags:
            tags.append(text)
    tags.append(f"incident_id:{case['incident_id']}")
    tags.append(f"incident_class:{case['incident_class']}")
    return tags


def _failure_localization(record: dict[str, Any]) -> str:
    if record.get("failed_stage"):
        return f"failed_stage:{record['failed_stage']}"
    fallback_nodes = record.get("fallback_nodes") or []
    if fallback_nodes:
        return "fallback_nodes:" + ",".join(str(node) for node in fallback_nodes)
    reason_codes = record.get("summary", {}).get("reason_codes") or []
    if reason_codes:
        return "reason_codes:" + ",".join(str(code) for code in reason_codes[:5])
    decision = record.get("decision", "")
    return f"decision:{decision or 'unknown'}"


def _high_level_match(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> bool:
    if error or not record:
        return False
    if record.get("status") not in {"completed", "degraded", "failed"}:
        return False
    incident_class = case["incident_class"]
    if incident_class == "malformed_downstream_payload":
        return record.get("status") in {"completed", "degraded", "failed"}
    if incident_class == "incorrect_icd_specificity":
        mappings = record.get("icd_mapping", {}).get("mappings")
        if not isinstance(mappings, list):
            return False
        return any(
            isinstance(mapping, dict)
            and str(mapping.get("status", "")).strip().upper() == "OK"
            and bool(str(mapping.get("icd_code", "")).strip())
            and bool(str(mapping.get("icd_label", "")).strip())
            for mapping in mappings
        )
    if incident_class in {"governance_override", "policy_change_divergence"}:
        return bool(record.get("summary", {}).get("governance_version") or record.get("summary", {}).get("policy_version"))
    if incident_class == "critic_false_positive":
        return bool(record.get("scores") is not None)
    if incident_class == "ambiguous_case_overconfidence":
        parsed = record.get("parsed_input", {})
        return "ambiguity_flag" in parsed or bool(record.get("diagnosis", {}).get("diagnoses"))
    return bool(record.get("diagnosis") and record.get("summary"))


def _governance_override_reporting(record: dict[str, Any]) -> dict[str, Any]:
    triage = record.get("diagnosis", {}).get("triage", {})
    triage_level = str(triage.get("level", "")).strip().lower()
    reason_codes = [
        str(code)
        for code in record.get("summary", {}).get("reason_codes", [])
    ]
    governed_decision = str(record.get("decision", "")).strip().upper()
    coding_reason_markers = (
        "ICD",
        "CODING",
        "DOCUMENTATION",
        "MISSING_DURATION",
        "LOW_ICD_SPECIFICITY",
    )
    documentation_or_coding_failure = any(
        any(marker in reason_code for marker in coding_reason_markers)
        for reason_code in reason_codes
    )
    urgent_triage_detected = triage_level in {"urgent", "escalate"}
    experimental_escalated_review = governed_decision == "REVISE_ESCALATED"
    governance_vs_triage_divergence = (
        urgent_triage_detected
        and governed_decision in {"REVISE", "FAIL", "REVISE_ESCALATED"}
        and documentation_or_coding_failure
    )
    if documentation_or_coding_failure:
        governance_failure_reason = "coding_or_documentation_specificity"
    elif governed_decision in {"REVISE", "FAIL", "REVISE_ESCALATED"}:
        governance_failure_reason = "critic_or_policy_threshold"
    else:
        governance_failure_reason = "none"

    if governance_vs_triage_divergence:
        operational_interpretation = (
            "Urgent triage was detected, while final governance failure is explained "
            "by coding/documentation specificity signals rather than direct triage urgency."
        )
    elif urgent_triage_detected:
        operational_interpretation = "Urgent triage was detected and governance did not report a coding/documentation failure."
    else:
        operational_interpretation = "No urgent triage signal was detected in the runtime summary."

    return {
        "triage_level": triage_level,
        "urgent_triage_detected": urgent_triage_detected,
        "governance_ignored_triage": not experimental_escalated_review,
        "experimental_escalated_review": experimental_escalated_review,
        "documentation_or_coding_failure": documentation_or_coding_failure,
        "governance_failure_reason": governance_failure_reason,
        "governance_vs_triage_divergence": governance_vs_triage_divergence,
        "operational_interpretation": operational_interpretation,
    }


def _shape_name(value: Any) -> str:
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return type(value).__name__


def _malformed_payload_reporting(case: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    input_text = str(case.get("input_text", "")).lower()
    mappings = record.get("icd_mapping", {}).get("mappings")
    diagnoses = record.get("diagnosis", {}).get("diagnoses")
    fallback_nodes = record.get("fallback_nodes") or []
    malformed_payload_observed = not isinstance(mappings, list)
    malformed_instruction_detected = (
        "mappings should be a string not an array" in input_text
        or "should be a string" in input_text
    )
    observed_output_shape = {
        "diagnosis.diagnoses": _shape_name(diagnoses),
        "icd_mapping.mappings": _shape_name(mappings),
    }
    shape_validation_result = (
        "valid_structured_outputs"
        if not malformed_payload_observed
        else "malformed_downstream_payload_observed"
    )
    if malformed_payload_observed:
        operational_interpretation = (
            "Malformed downstream payload structure was observed in the runtime summary."
        )
    else:
        operational_interpretation = (
            "Malformed instruction language was present in the synthetic input, but downstream "
            "structured outputs remained valid and no malformed payload propagated."
        )

    return {
        "malformed_instruction_detected": malformed_instruction_detected,
        "malformed_payload_observed": malformed_payload_observed,
        "incident_behavior_classification": (
            "malformed_instruction_resilience"
            if malformed_instruction_detected and not malformed_payload_observed
            else "malformed_payload_propagation"
        ),
        "expected_shape": {
            "diagnosis.diagnoses": "list",
            "icd_mapping.mappings": "list",
        },
        "observed_output_shape": observed_output_shape,
        "shape_validation_result": shape_validation_result,
        "malformed_payload_stage": "none" if not malformed_payload_observed else "icd_mapping",
        "structured_outputs_valid": not malformed_payload_observed,
        "fallback_triggered_by_malformed_payload": bool(fallback_nodes) and malformed_payload_observed,
        "operational_interpretation": operational_interpretation,
    }


def _ambiguous_overconfidence_reporting(record: dict[str, Any]) -> dict[str, Any]:
    parsed = record.get("parsed_input", {})
    diagnoses = record.get("diagnosis", {}).get("diagnoses")
    scores = record.get("scores", {})
    summary = record.get("summary", {})
    reason_codes = [
        str(code)
        for code in summary.get("reason_codes", [])
    ]
    critic_reason_codes = [
        str(code)
        for code in scores.get("reason_codes", [])
    ]
    all_reason_codes = reason_codes + [
        code for code in critic_reason_codes if code not in reason_codes
    ]
    diagnosis_count = len(diagnoses) if isinstance(diagnoses, list) else 0
    critic_confidence = scores.get("confidence")
    try:
        confidence_value = float(critic_confidence)
    except (TypeError, ValueError):
        confidence_value = None

    ambiguity_detected = bool(parsed.get("ambiguity_flag"))
    insufficient_evidence_detected = (
        bool(parsed.get("missing_fields"))
        or diagnosis_count == 0
        or any(
            marker in code
            for code in all_reason_codes
            for marker in (
                "INSUFFICIENT",
                "LOW_EVIDENCE",
                "NO_DIAGNOSES",
                "NO_DIAGNOSES_PROVIDED",
                "NO_DIAGNOSES_FOUND",
            )
        )
    )
    low_confidence = confidence_value is not None and confidence_value <= 0.4
    governed_decision = str(record.get("decision", "")).strip().upper()
    single_diagnosis_on_ambiguous_input = ambiguity_detected and diagnosis_count == 1
    overconfidence_observed = bool(
        ambiguity_detected
        and (
            governed_decision == "PASS"
            or (
                single_diagnosis_on_ambiguous_input
                and confidence_value is not None
                and confidence_value >= 0.7
            )
        )
    )
    ambiguity_preserved = bool(
        ambiguity_detected
        and insufficient_evidence_detected
        and not overconfidence_observed
        and governed_decision in {"FAIL", "REVISE", "REVISE_ESCALATED"}
    )

    if overconfidence_observed:
        incident_behavior_classification = "true_overconfidence_behavior"
        operational_interpretation = (
            "Ambiguity was detected, but the summary indicates a confident pass or a "
            "single high-confidence diagnosis on ambiguous evidence."
        )
    elif ambiguity_preserved:
        incident_behavior_classification = "ambiguity_preservation"
        operational_interpretation = (
            "Ambiguity was detected and preserved through low-confidence critic scoring "
            "and a non-pass governance outcome."
        )
    else:
        incident_behavior_classification = "ambiguous_inconclusive"
        operational_interpretation = (
            "The summary does not clearly establish either true overconfidence or clean "
            "ambiguity preservation."
        )

    return {
        "ambiguity_detected": ambiguity_detected,
        "ambiguity_reasons": parsed.get("ambiguity_reasons", []),
        "diagnosis_count": diagnosis_count,
        "critic_confidence": critic_confidence,
        "critic_recommended_status": scores.get("recommended_status"),
        "single_diagnosis_on_ambiguous_input": single_diagnosis_on_ambiguous_input,
        "insufficient_evidence_detected": insufficient_evidence_detected,
        "low_confidence_detected": low_confidence,
        "overconfidence_observed": overconfidence_observed,
        "ambiguity_preserved": ambiguity_preserved,
        "incident_behavior_classification": incident_behavior_classification,
        "operational_interpretation": operational_interpretation,
    }


def _critic_false_positive_reporting(case: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    input_text = str(case.get("input_text", "")).lower()
    parsed = record.get("parsed_input", {})
    diagnoses = record.get("diagnosis", {}).get("diagnoses")
    mappings = record.get("icd_mapping", {}).get("mappings")
    scores = record.get("scores", {})
    summary = record.get("summary", {})

    diagnosis_count = len(diagnoses) if isinstance(diagnoses, list) else 0
    partial_icd_mapping_count = sum(
        1
        for mapping in mappings or []
        if isinstance(mapping, dict)
        and str(mapping.get("status", "")).strip().upper() == "PARTIAL_MATCH"
    )
    critic_reason_codes = [
        str(code)
        for code in scores.get("reason_codes", [])
    ]
    summary_reason_codes = [
        str(code)
        for code in summary.get("reason_codes", [])
    ]
    critic_status = str(scores.get("recommended_status", "")).strip().lower()
    governed_decision = str(record.get("decision", "")).strip().upper()
    critic_confidence = scores.get("confidence")
    low_risk_markers = (
        "mild",
        "no fever",
        "no neurologic",
        "no trauma",
        "improved",
        "after rest",
        "skipped lunch",
    )
    low_risk_context_present = any(marker in input_text for marker in low_risk_markers)
    reassuring_context_missing_from_structured_output = bool(
        low_risk_context_present
        and not parsed.get("comorbidities")
        and not parsed.get("allergies")
        and not parsed.get("medications")
        and "no fever" in input_text
        and "fever" not in parsed.get("symptoms", [])
    )
    representation_loss_detected = bool(
        low_risk_context_present
        and (
            parsed.get("duration") is None
            or bool(parsed.get("missing_fields"))
            or bool(parsed.get("ambiguity_flag"))
            or reassuring_context_missing_from_structured_output
        )
    )
    diagnosis_broadening_detected = diagnosis_count > 1
    icd_broadening_detected = partial_icd_mapping_count > 0
    caution_amplification_detected = bool(
        representation_loss_detected
        and (diagnosis_broadening_detected or icd_broadening_detected)
        and (critic_status in {"revise", "fail"} or governed_decision in {"REVISE", "FAIL", "REVISE_ESCALATED"})
    )
    governance_amplified_critic = bool(
        (critic_status == "pass" and governed_decision in {"REVISE", "FAIL", "REVISE_ESCALATED"})
        or (critic_status == "revise" and governed_decision in {"FAIL", "REVISE_ESCALATED"})
    )
    false_positive_risk_observed = bool(
        low_risk_context_present
        and governed_decision in {"REVISE", "FAIL", "REVISE_ESCALATED"}
    )

    if caution_amplification_detected:
        incident_behavior_classification = "representation_loss_caution_amplification"
        false_positive_risk_classification = "representation_loss_driven_false_positive_risk"
        operational_interpretation = (
            "Low-risk narrative context was present, but structured representation loss, "
            "diagnosis/ICD broadening, and critic caution combined into a non-pass outcome."
        )
    elif false_positive_risk_observed:
        incident_behavior_classification = "standalone_critic_false_positive_risk"
        false_positive_risk_classification = "simple_false_positive_risk"
        operational_interpretation = (
            "Low-risk narrative context was present and the critic/governance outcome was "
            "non-pass, without enough summary evidence to attribute the caution to "
            "representation loss."
        )
    else:
        incident_behavior_classification = "critic_caution_not_observed"
        false_positive_risk_classification = "not_observed"
        operational_interpretation = (
            "The summary does not show a non-pass critic/governance outcome for the "
            "low-risk case."
        )

    return {
        "low_risk_context_present": low_risk_context_present,
        "diagnosis_count": diagnosis_count,
        "partial_icd_mapping_count": partial_icd_mapping_count,
        "critic_reason_codes": critic_reason_codes,
        "critic_confidence": critic_confidence,
        "critic_recommended_status": scores.get("recommended_status"),
        "governance_amplified_critic": governance_amplified_critic,
        "representation_loss_detected": representation_loss_detected,
        "reassuring_context_missing_from_structured_output": reassuring_context_missing_from_structured_output,
        "diagnosis_broadening_detected": diagnosis_broadening_detected,
        "icd_broadening_detected": icd_broadening_detected,
        "caution_amplification_detected": caution_amplification_detected,
        "false_positive_risk_observed": false_positive_risk_observed,
        "false_positive_risk_classification": false_positive_risk_classification,
        "incident_behavior_classification": incident_behavior_classification,
        "governance_reason_codes": summary_reason_codes,
        "operational_interpretation": operational_interpretation,
    }


def _summarize_record(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    if record is None:
        return {
            "incident_id": case["incident_id"],
            "incident_class": case["incident_class"],
            "runtime_status": "error",
            "governed_decision": "",
            "fallback_used": False,
            "failure_localization_clue": error or "runner_error",
            "high_level_expected_match": False,
        }

    summary = {
        "incident_id": case["incident_id"],
        "incident_class": case["incident_class"],
        "runtime_status": record.get("status", ""),
        "governed_decision": record.get("decision", ""),
        "fallback_used": bool(record.get("fallback_used")),
        "fallback_nodes": record.get("fallback_nodes", []),
        "failure_localization_clue": _failure_localization(record),
        "high_level_expected_match": _high_level_match(case, record, error),
        "trace_stage_count": len(record.get("trace", [])),
    }
    if case["incident_class"] == "governance_override":
        summary.update(_governance_override_reporting(record))
    if case["incident_class"] == "malformed_downstream_payload":
        summary.update(_malformed_payload_reporting(case, record))
    if case["incident_class"] == "critic_false_positive":
        summary.update(_critic_false_positive_reporting(case, record))
    if case["incident_class"] == "ambiguous_case_overconfidence":
        summary.update(_ambiguous_overconfidence_reporting(record))
    return summary


def _langsmith_visibility(project: str, expected_count: int) -> dict[str, Any]:
    if not os.getenv("LANGCHAIN_API_KEY", "").strip():
        return {"queried": False, "reason": "LANGCHAIN_API_KEY not configured"}

    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        from langsmith import Client

        wait_for_all_tracers()
        client = Client()
        runs = list(client.list_runs(project_name=project, run_type="chain", limit=max(50, expected_count * 10)))
        incident_runs = [run for run in runs if run.name == "medscribe.synthetic_incident"]
        stage_names = {
            "medscribe.intake_parser",
            "medscribe.triage_engine",
            "medscribe.diagnosis_engine",
            "medscribe.icd_mapper",
            "medscribe.critic",
            "medscribe.governance_policy",
        }
        stage_output_visible = False
        if incident_runs:
            trace_id = getattr(incident_runs[0], "trace_id", None) or getattr(incident_runs[0], "id", None)
            trace_runs = list(client.list_runs(project_name=project, trace_id=trace_id, limit=100))
            stage_output_visible = all(
                any(run.name == stage_name and bool(getattr(run, "outputs", None)) for run in trace_runs)
                for stage_name in stage_names
            )
        return {
            "queried": True,
            "project": project,
            "recent_synthetic_incident_root_count": len(incident_runs),
            "traces_appeared": len(incident_runs) > 0,
            "incident_tags_metadata_expected": bool(incident_runs),
            "stage_outputs_visible_in_latest_trace": stage_output_visible,
        }
    except Exception as exc:
        return {"queried": False, "reason": exc.__class__.__name__}


def run_incidents(dataset_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    load_dotenv(REPO_ROOT / ".env")
    if os.getenv("LANGCHAIN_API_KEY", "").strip():
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    from graph.tracing import trace_span
    from service.run_manager import execute

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    incidents = payload["incidents"]
    project = os.getenv("LANGCHAIN_PROJECT", "medscribe-phase1-runtime")
    started_at = _utc_stamp()
    summaries: list[dict[str, Any]] = []

    for case in incidents:
        run_id = f"synthetic-{case['incident_id'].lower()}-{started_at.replace(':', '').replace('-', '')}"
        metadata = {
            "incident_id": case["incident_id"],
            "incident_class": case["incident_class"],
            "dataset_name": payload["dataset_name"],
            "dataset_version": payload["dataset_version"],
            "contains_phi": False,
            "expected_failure_mode": case["expected_failure_mode"],
        }
        record = None
        error = None
        with trace_span(
            "medscribe.synthetic_incident",
            inputs={"incident_id": case["incident_id"], "input_text": case["input_text"]},
            metadata=metadata,
            tags=_safe_case_tags(case),
        ) as span:
            try:
                record = execute(case["input_text"], run_id=run_id, persist=False)
            except Exception as exc:
                error = exc.__class__.__name__
                partial = getattr(exc, "partial_record", None)
                if isinstance(partial, dict):
                    record = partial
            summary = _summarize_record(case, record, error)
            span.set_outputs(summary)
        summaries.append(summary)
        print(
            f"{case['incident_id']} {case['incident_class']} "
            f"status={summary['runtime_status']} decision={summary['governed_decision']} "
            f"match={summary['high_level_expected_match']}"
        )

    passed = sum(1 for item in summaries if item["high_level_expected_match"])
    failed = len(summaries) - passed
    langsmith = _langsmith_visibility(project, len(summaries))
    result = {
        "dataset_name": payload["dataset_name"],
        "dataset_version": payload["dataset_version"],
        "started_at": started_at,
        "completed_at": _utc_stamp(),
        "langsmith_project": project,
        "incident_count": len(summaries),
        "passed_high_level": passed,
        "failed_high_level": failed,
        "summaries": summaries,
        "langsmith_visibility": langsmith,
    }

    if output_path is None:
        output_path = SUMMARY_DIR / "last_run_summary.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    timestamped = SUMMARY_DIR / f"run_summary_{started_at.replace(':', '').replace('-', '')}.json"
    timestamped.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("summary_path=" + str(output_path))
    print("timestamped_summary_path=" + str(timestamped))
    print("incident_count=" + str(result["incident_count"]))
    print("passed_high_level=" + str(passed))
    print("failed_high_level=" + str(failed))
    print("langsmith_project=" + project)
    print("langsmith_visibility=" + json.dumps(langsmith, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic MedScribe incident pack.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--output", default=str(SUMMARY_DIR / "last_run_summary.json"))
    args = parser.parse_args()
    run_incidents(Path(args.dataset), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
