"""Minimal regenerated LangSmith experiment runner for operational benchmarks."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "evaluation" / "operational_benchmark_cases.json"
DENIAL_DATASET_PATH = REPO_ROOT / "evaluation" / "denial_benchmark_cases.json"
DEFAULT_DATASET_NAME = "medscribe-operational-benchmark-poc"
DEFAULT_DENIAL_DATASET_NAME = "medscribe-denial-operational-benchmark-poc"
EXPERIMENT_LABELS = {"baseline", "threshold_variant", "routing_sensitivity_variant"}
WORKFLOWS = {"cdi", "denial"}
DENIAL_ROUTING_ACTIONS = ("APPEAL", "RESUBMIT", "WRITE_OFF", "ESCALATE")
DENIAL_GOVERNANCE_POSTURES = ("SUPPORTED", "LOW_CONFIDENCE", "LOW_EVIDENCE", "AMBIGUOUS", "HIGH_RISK")
DENIAL_VARIANT_DESCRIPTIONS = {
    "baseline": "baseline denial routing and governance outputs",
    "threshold_variant": (
        "stricter denial sensitivity for ambiguity, low evidence, specialist review, "
        "unsupported certainty, and recoverability boundaries"
    ),
    "routing_sensitivity_variant": (
        "routing-sensitive denial perturbation that reduces avoidable escalation for "
        "recoverable partial-support and documentation-gap cases while preserving high-risk escalation"
    ),
}
THRESHOLD_VARIANT_OVERRIDES = {
    "confidence_min_for_revise": 0.7,
}
ROUTING_SENSITIVITY_VARIANT_OVERRIDES = {
    "diagnosis_consistency_min_for_pass": 0.0,
    "diagnosis_consistency_min_for_revise": 0.0,
    "symptom_alignment_min_for_pass": 0.0,
    "symptom_alignment_min_for_revise": 0.0,
    "icd_specificity_min_for_pass": 0.0,
    "icd_specificity_min_for_revise": 0.0,
    "confidence_min_for_pass": 0.0,
    "confidence_min_for_revise": 0.0,
}
VARIANT_THRESHOLD_OVERRIDES = {
    "threshold_variant": THRESHOLD_VARIANT_OVERRIDES,
    "routing_sensitivity_variant": ROUTING_SENSITIVITY_VARIANT_OVERRIDES,
}
FINAL_STATUS_BUCKETS = ("PASS", "REVISE", "FAIL", "ESCALATE", "OTHER", "MISSING")
STATUS_RANK = {
    "PASS": 0,
    "REVISE": 1,
    "REVISE_ESCALATED": 2,
    "ESCALATE": 2,
    "FAIL": 3,
}
LOW_EVIDENCE_MARKERS = ("LOW_EVIDENCE", "INSUFFICIENT", "MISSING", "NO_ICD", "NO_DIAGNOS")
SPECIALIST_MARKERS = ("SPECIALIST", "STROKE", "SEPSIS", "RESPIRATORY_FAILURE")
UNSUPPORTED_SPECIFICITY_MARKERS = ("UNSUPPORTED", "SPECIFICITY", "LOW_ICD", "PARTIAL_ICD", "NO_ICD")
SECRET_VALUE_PATTERNS = (
    re.compile(r"lsv2_[A-Za-z0-9_\-]+"),
    re.compile(r"lsv2_[^\s'\"),]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]+"),
    re.compile(r"API Key:\s*[^\n]+"),
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class EvaluationResult:
    key: str
    score: bool
    comment: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_cases(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("operational benchmark dataset must be a JSON list")
    required = {
        "case_id",
        "title",
        "input_text",
        "operational_theme",
        "expected_high_level_behavior",
        "metadata",
    }
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case at index {index} must be an object")
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"{case.get('case_id', index)} missing required fields: {missing}")
        if not str(case["input_text"]).strip():
            raise ValueError(f"{case['case_id']} input_text must be non-empty")
    return cases


def load_denial_cases(path: Path = DENIAL_DATASET_PATH) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("denial benchmark dataset must be a JSON list")
    required = {
        "case_id",
        "title",
        "denial_type",
        "payer_reason",
        "clinical_summary",
        "documentation_summary",
        "operational_theme",
        "expected_routing_action",
        "expected_governance_posture",
        "metadata",
    }
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case at index {index} must be an object")
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"{case.get('case_id', index)} missing required fields: {missing}")
        if case.get("expected_routing_action") not in DENIAL_ROUTING_ACTIONS:
            raise ValueError(f"{case['case_id']} has invalid expected_routing_action")
        if case.get("expected_governance_posture") not in DENIAL_GOVERNANCE_POSTURES:
            raise ValueError(f"{case['case_id']} has invalid expected_governance_posture")
        metadata = case.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("phi") is not False:
            raise ValueError(f"{case['case_id']} must be synthetic non-PHI")
    return cases


def load_workflow_cases(workflow: str) -> list[dict[str, Any]]:
    if workflow == "denial":
        return load_denial_cases()
    return load_cases()


def langsmith_credentials_available() -> bool:
    return bool(os.getenv("LANGCHAIN_API_KEY", "").strip() or os.getenv("LANGSMITH_API_KEY", "").strip())


def _sanitize_error_message(exc: Exception) -> str:
    message = str(exc)
    for pattern in SECRET_VALUE_PATTERNS:
        message = pattern.sub("[REDACTED]", message)
    return message


def _get_langsmith_client() -> Any | None:
    if not langsmith_credentials_available():
        return None
    try:
        from langsmith import Client

        return Client()
    except Exception as exc:
        print(f"langsmith_client_status=unavailable error_class={exc.__class__.__name__} reason={_sanitize_error_message(exc)}")
        return None


def langsmith_preflight(
    hosted_requested: bool,
    *,
    hosted_tracing_enabled_by_runner: bool = False,
) -> tuple[dict[str, Any], Any | None]:
    preflight: dict[str, Any] = {
        "hosted_requested": hosted_requested,
        "hosted_tracing_enabled_by_runner": hosted_tracing_enabled_by_runner,
        "langchain_api_key_present": bool(os.getenv("LANGCHAIN_API_KEY", "").strip()),
        "langsmith_api_key_present": bool(os.getenv("LANGSMITH_API_KEY", "").strip()),
        "langchain_tracing_v2_set": "LANGCHAIN_TRACING_V2" in os.environ,
        "langchain_tracing_v2_enabled": os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower()
        in {"1", "true", "yes"},
        "langchain_endpoint_set": bool(os.getenv("LANGCHAIN_ENDPOINT", "").strip()),
        "langchain_project_set": bool(os.getenv("LANGCHAIN_PROJECT", "").strip()),
        "client_construction_succeeded": False,
        "readiness_check_attempted": False,
        "readiness_check_succeeded": False,
    }
    client = _get_langsmith_client()
    if client is None:
        preflight["client_status"] = "unavailable"
        return preflight, None

    preflight["client_construction_succeeded"] = True
    preflight["client_status"] = "ready"
    if not hosted_requested:
        preflight["readiness_check_skipped_reason"] = "hosted_not_requested"
        return preflight, client

    preflight["readiness_check_attempted"] = True
    try:
        info = getattr(client, "info")
        if callable(info):
            info()
        list(client.list_datasets(limit=1))
        preflight["readiness_check_succeeded"] = True
    except Exception as exc:
        preflight["client_status"] = "readiness_check_failed"
        preflight["readiness_error_class"] = exc.__class__.__name__
        preflight["readiness_error"] = _sanitize_error_message(exc)
    return preflight, client


def _print_langsmith_preflight(preflight: dict[str, Any]) -> None:
    fields = [
        f"hosted_requested={preflight.get('hosted_requested')}",
        f"hosted_tracing_enabled_by_runner={preflight.get('hosted_tracing_enabled_by_runner')}",
        f"langchain_api_key_present={preflight.get('langchain_api_key_present')}",
        f"langsmith_api_key_present={preflight.get('langsmith_api_key_present')}",
        f"langchain_tracing_v2_set={preflight.get('langchain_tracing_v2_set')}",
        f"langchain_tracing_v2_enabled={preflight.get('langchain_tracing_v2_enabled')}",
        f"langchain_endpoint_set={preflight.get('langchain_endpoint_set')}",
        f"client_construction_succeeded={preflight.get('client_construction_succeeded')}",
        f"readiness_check_attempted={preflight.get('readiness_check_attempted')}",
        f"readiness_check_succeeded={preflight.get('readiness_check_succeeded')}",
        f"client_status={preflight.get('client_status')}",
    ]
    if preflight.get("readiness_error_class"):
        fields.append(f"readiness_error_class={preflight.get('readiness_error_class')}")
    print("langsmith_preflight_status " + " ".join(fields))


def _tracing_enabled_value() -> bool:
    return os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() in {"1", "true", "yes"}


def _enable_hosted_tracing_if_needed(hosted_requested: bool) -> bool:
    if not hosted_requested or not langsmith_credentials_available() or _tracing_enabled_value():
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    return True


def _dataset_exists(client: Any, dataset_name: str) -> bool:
    try:
        client.read_dataset(dataset_name=dataset_name)
        return True
    except Exception:
        return False


def create_or_reuse_dataset(
    client: Any | None,
    cases: list[dict[str, Any]],
    dataset_name: str,
    *,
    workflow: str = "cdi",
) -> str:
    if client is None:
        print("langsmith_dataset_status=skipped credentials_or_client_missing")
        return dataset_name

    try:
        if not _dataset_exists(client, dataset_name):
            client.create_dataset(
                dataset_name=dataset_name,
                description=(
                    "Synthetic non-PHI MedScribe denial-management operational benchmark cases."
                    if workflow == "denial"
                    else "Synthetic non-PHI MedScribe CDI coding-support operational benchmark cases."
                ),
            )
        existing_ids: set[str] = set()
        try:
            for example in client.list_examples(dataset_name=dataset_name):
                metadata = getattr(example, "metadata", None) or {}
                case_id = metadata.get("case_id")
                if case_id:
                    existing_ids.add(str(case_id))
        except Exception:
            existing_ids = set()

        for case in cases:
            if case["case_id"] in existing_ids:
                continue
            if workflow == "denial":
                inputs = {"case_payload": case}
                outputs = {
                    "expected_routing_action": case.get("expected_routing_action"),
                    "expected_governance_posture": case.get("expected_governance_posture"),
                }
            else:
                inputs = {"input_text": case["input_text"]}
                outputs = {
                    "expected_high_level_behavior": case["expected_high_level_behavior"],
                    "expected_governance_status": case.get("expected_governance_status"),
                    "expected_routing_category": case.get("expected_routing_category"),
                }
            client.create_example(
                inputs=inputs,
                outputs=outputs,
                metadata={
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "workflow": workflow,
                    "operational_theme": case["operational_theme"],
                    **case.get("metadata", {}),
                },
                dataset_name=dataset_name,
            )
        print(f"langsmith_dataset_status=ready dataset_name={dataset_name}")
    except Exception as exc:
        print(
            "langsmith_dataset_status=degraded "
            f"dataset_name={dataset_name} error_class={exc.__class__.__name__} "
            f"reason={_sanitize_error_message(exc)}"
        )
    return dataset_name


def _case_by_id(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): case for case in cases}


def _case_by_input(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(case["input_text"]): case for case in cases}


def _case_by_denial_payload(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {json.dumps(case, sort_keys=True): case for case in cases}


def _example_case_id(example: Any) -> str:
    metadata = getattr(example, "metadata", None) or {}
    return str(metadata.get("case_id", "")).strip()


def _list_selected_examples(client: Any, dataset_name: str, cases: list[dict[str, Any]]) -> list[Any]:
    selected_ids = {str(case["case_id"]) for case in cases}
    by_id: dict[str, Any] = {}
    for example in client.list_examples(dataset_name=dataset_name):
        case_id = _example_case_id(example)
        if case_id in selected_ids:
            by_id[case_id] = example
    missing = [case["case_id"] for case in cases if case["case_id"] not in by_id]
    if missing:
        raise ValueError(f"LangSmith dataset is missing selected cases: {missing}")
    return [by_id[str(case["case_id"])] for case in cases]


def _case_from_example(example: Any, cases_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_id = _example_case_id(example)
    if case_id in cases_by_id:
        return cases_by_id[case_id]

    metadata = getattr(example, "metadata", None) or {}
    outputs = getattr(example, "outputs", None) or {}
    inputs = getattr(example, "inputs", None) or {}
    if "case_payload" in inputs and isinstance(inputs["case_payload"], dict):
        payload = dict(inputs["case_payload"])
        payload.setdefault("metadata", metadata)
        return payload
    return {
        "case_id": case_id or "unknown",
        "title": metadata.get("title", case_id or "unknown"),
        "input_text": inputs.get("input_text", ""),
        "operational_theme": metadata.get("operational_theme", ""),
        "expected_high_level_behavior": outputs.get("expected_high_level_behavior", ""),
        "expected_governance_status": outputs.get("expected_governance_status"),
        "expected_routing_category": outputs.get("expected_routing_category"),
        "metadata": metadata,
    }


def _record_status(record: dict[str, Any]) -> str:
    return str(record.get("decision") or record.get("summary", {}).get("status") or "").strip().upper()


def _status_bucket(status: str) -> str:
    normalized = str(status).strip().upper()
    if not normalized:
        return "MISSING"
    if normalized in {"PASS", "REVISE", "FAIL"}:
        return normalized
    if "ESCALATE" in normalized:
        return "ESCALATE"
    return "OTHER"


def _record_summary(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    summary = record.get("summary")
    return summary if isinstance(summary, dict) else {}


def _record_reason_codes(record: dict[str, Any] | None) -> list[str]:
    summary = _record_summary(record)
    reason_codes = summary.get("reason_codes")
    if not isinstance(reason_codes, list):
        scores = record.get("scores") if isinstance(record, dict) else {}
        reason_codes = scores.get("reason_codes") if isinstance(scores, dict) else []
    return [str(code).strip().upper() for code in reason_codes if str(code).strip()]


def _record_operational_snapshot(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    snapshot = record.get("operational_observability")
    return snapshot if isinstance(snapshot, dict) else {}


def _record_critic_snapshot(record: dict[str, Any] | None) -> dict[str, Any]:
    snapshot = _record_operational_snapshot(record)
    critic = snapshot.get("critic_metric_snapshot")
    if isinstance(critic, dict):
        return critic
    scores = record.get("scores") if isinstance(record, dict) else {}
    return scores if isinstance(scores, dict) else {}


def _record_governance_snapshot(record: dict[str, Any] | None) -> dict[str, Any]:
    snapshot = _record_operational_snapshot(record)
    governance = snapshot.get("governance_snapshot")
    return governance if isinstance(governance, dict) else {}


def _record_timing(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    timing = record.get("timing")
    if isinstance(timing, dict):
        return timing
    snapshot = _record_operational_snapshot(record)
    stage_timing = snapshot.get("stage_timing_ms")
    return stage_timing if isinstance(stage_timing, dict) else {}


def _llm_used(record: dict[str, Any] | None, workflow: str) -> bool:
    if not isinstance(record, dict):
        return False
    diagnostics = record.get("node_diagnostics")
    if not isinstance(diagnostics, list):
        return False
    if workflow == "denial":
        return any(
            isinstance(diagnostic, dict)
            and (
                bool(diagnostic.get("hybrid_interpretation_used"))
                or bool(diagnostic.get("live_call_returned"))
            )
            for diagnostic in diagnostics
        )
    return any(
        isinstance(diagnostic, dict) and bool(diagnostic.get("live_call_returned"))
        for diagnostic in diagnostics
    )


def _provider_call_attempted(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return False
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("provider_call_attempted"):
        return True
    diagnostics = record.get("node_diagnostics")
    if not isinstance(diagnostics, list):
        return False
    return any(isinstance(diagnostic, dict) and bool(diagnostic.get("live_call_attempted")) for diagnostic in diagnostics)


def _token_cost_status(record: dict[str, Any] | None, workflow: str) -> tuple[bool, str, dict[str, Any]]:
    token_metadata: dict[str, Any] = {}
    if isinstance(record, dict):
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            raw_token_metadata = metadata.get("token_usage")
            if isinstance(raw_token_metadata, dict):
                token_metadata = raw_token_metadata
    if token_metadata:
        return True, "", token_metadata
    if _llm_used(record, workflow) or _provider_call_attempted(record):
        return False, "provider_metadata_unavailable", {}
    return False, "no_provider_call", {}


def _case_contains_phi(case: dict[str, Any]) -> bool | None:
    metadata = case.get("metadata")
    if isinstance(metadata, dict) and "phi" in metadata:
        return bool(metadata.get("phi"))
    return None


def _primary_output(workflow: str, result: dict[str, Any]) -> str:
    record = result.get("record") if isinstance(result.get("record"), dict) else {}
    if workflow == "denial":
        route = str(result.get("routing_action") or record.get("routing_action") or "").strip()
        posture = str(result.get("governance_posture") or record.get("governance_posture") or "").strip()
        status = str(
            result.get("status")
            or result.get("runtime_status")
            or record.get("status")
            or record.get("runtime_status")
            or ""
        ).strip()
        if route:
            return f"route={route} posture={posture or 'missing'} status={status or 'missing'}"
        if status:
            return f"status={status}"
        for key in ("governance_posture", "denial_category", "final_status"):
            value = result.get(key) or record.get(key)
            text = str(value or "").strip()
            if text:
                return text
        return "MISSING"
    for value in (
        result.get("final_status"),
        record.get("decision"),
        record.get("summary", {}).get("status") if isinstance(record.get("summary"), dict) else "",
        record.get("status"),
    ):
        text = str(value or "").strip().upper()
        if text:
            return text
    return "MISSING"


def _safe_tag(prefix: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_:\-]+", "_", text)
    return f"{prefix}:{text}"


def _dedupe_tags(tags: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        text = str(tag or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _operational_metadata(
    case: dict[str, Any],
    result: dict[str, Any],
    *,
    workflow: str,
    variant: str,
    latency_ms: int,
) -> dict[str, Any]:
    record = result.get("record") if isinstance(result.get("record"), dict) else {}
    gaps = record.get("documentation_gaps") if isinstance(record, dict) else {}
    if not isinstance(gaps, dict):
        gaps = {}
    evidence = record.get("evidence_profile") if isinstance(record, dict) else {}
    if not isinstance(evidence, dict):
        evidence = {}
    token_available, token_unavailable_reason, _ = _token_cost_status(record, workflow)
    execution_metadata = record.get("metadata") if isinstance(record, dict) else {}
    if not isinstance(execution_metadata, dict):
        execution_metadata = {}
    timing = _record_timing(record)
    metadata: dict[str, Any] = {
        "workflow": workflow,
        "trace_type": "operational",
        "case_id": case.get("case_id"),
        "variant": variant,
        "routing_action": result.get("routing_action") or record.get("routing_action"),
        "governance_posture": result.get("governance_posture") or record.get("governance_posture"),
        "denial_category": result.get("denial_category") or record.get("denial_category"),
        "recoverability": result.get("recoverability") or record.get("recoverability"),
        "evidence_strength": evidence.get("evidence_strength"),
        "ambiguity_level": evidence.get("ambiguity_level"),
        "ambiguity_flag": bool(gaps.get("ambiguity")) if workflow == "denial" else None,
        "conflicting_evidence": bool(evidence.get("has_conflicting_documentation")) if workflow == "denial" else None,
        "specialist_review_candidate": bool(case.get("metadata", {}).get("specialist_review_candidate"))
        if isinstance(case.get("metadata"), dict)
        else None,
        "degraded_mode": bool(record.get("degraded_mode")) if isinstance(record, dict) else False,
        "fallback_used": bool(record.get("fallback_used")) if isinstance(record, dict) else False,
        "contains_phi": _case_contains_phi(case),
        "execution_mode": execution_metadata.get("execution_mode"),
        "llm_used": _llm_used(record, workflow),
        "token_cost_available": token_available,
        "token_cost_unavailable_reason": token_unavailable_reason or None,
        "latency_ms": latency_ms,
        "node_latency_ms": timing or None,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _operational_tags(metadata: dict[str, Any]) -> list[str]:
    alerts = metadata.get("alerts") if isinstance(metadata.get("alerts"), list) else []
    return _dedupe_tags(
        [
            _safe_tag("workflow", metadata.get("workflow")),
            _safe_tag("trace_type", metadata.get("trace_type")),
            _safe_tag("variant", metadata.get("variant")),
            _safe_tag("mode", metadata.get("execution_mode")),
            _safe_tag("route", metadata.get("routing_action")),
            _safe_tag("posture", metadata.get("governance_posture")),
            _safe_tag("routing_action", metadata.get("routing_action")),
            _safe_tag("governance_posture", metadata.get("governance_posture")),
            "contains_phi:false" if metadata.get("contains_phi") is False else None,
            _safe_tag("degraded_mode", str(bool(metadata.get("degraded_mode"))).lower()),
            _safe_tag("fallback_used", str(bool(metadata.get("fallback_used"))).lower()),
            _safe_tag("status", metadata.get("status")),
            _safe_tag("severity", metadata.get("max_alert_severity")),
            *[_safe_tag("alert", alert.get("class")) for alert in alerts if isinstance(alert, dict)],
        ]
    )


def _contains_marker(values: Iterable[Any], markers: tuple[str, ...]) -> bool:
    text = " ".join(str(value).upper() for value in values)
    return any(marker in text for marker in markers)


def _expected_case_signals(case: dict[str, Any]) -> dict[str, bool]:
    metadata = case.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    expected_route = str(case.get("expected_routing_category", "")).lower()
    theme = str(case.get("operational_theme", "")).lower()
    return {
        "unsafe_pass_risk": bool(metadata.get("unsafe_pass_risk"))
        or str(case.get("expected_governance_status", "")).strip().upper() in {"FAIL", "REVISE"},
        "expects_low_evidence": bool(metadata.get("low_evidence_boundary"))
        or str(metadata.get("documentation_confidence", "")).lower() == "low",
        "expects_specialist_review": bool(metadata.get("specialist_review_expected"))
        or "specialist" in expected_route
        or "specialist" in theme,
        "expects_coder_review": bool(metadata.get("coder_review_expected")) or "coder" in expected_route,
        "expects_unsupported_specificity": metadata.get("documentation_supports_specificity") is False
        or "unsupported specificity" in theme,
        "expects_spillover_watch": bool(metadata.get("adjacent_case_spillover_watch")),
    }


def _status_distance(actual: str, expected: str) -> int:
    actual_rank = STATUS_RANK.get(str(actual).strip().upper(), 4)
    expected_rank = STATUS_RANK.get(str(expected).strip().upper(), 4)
    return abs(actual_rank - expected_rank)


def _variant_overrides(experiment_label: str) -> dict[str, float]:
    return VARIANT_THRESHOLD_OVERRIDES.get(experiment_label, {})


def _denial_strength(record: dict[str, Any], case: dict[str, Any] | None = None) -> str:
    evidence = record.get("evidence_profile")
    if isinstance(evidence, dict) and evidence.get("evidence_strength"):
        return str(evidence.get("evidence_strength")).strip().lower()
    if isinstance(case, dict):
        return str(case.get("evidence_strength") or "").strip().lower()
    return ""


def _denial_metadata(case: dict[str, Any] | None) -> dict[str, Any]:
    metadata = case.get("metadata") if isinstance(case, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _denial_variant_signals(record: dict[str, Any], case: dict[str, Any] | None = None) -> dict[str, Any]:
    gaps = record.get("documentation_gaps") if isinstance(record, dict) else {}
    if not isinstance(gaps, dict):
        gaps = {}
    evidence = record.get("evidence_profile") if isinstance(record, dict) else {}
    if not isinstance(evidence, dict):
        evidence = {}
    metadata = _denial_metadata(case)
    strength = _denial_strength(record, case)
    ambiguity_level = str(gaps.get("ambiguity_level") or evidence.get("ambiguity_level") or "").strip().lower()
    recoverability = str(record.get("recoverability") or "").strip().lower()
    category = str(record.get("denial_category") or (case or {}).get("denial_type") or "").strip().lower()
    conflicting = bool(
        gaps.get("conflicting_evidence")
        or evidence.get("has_conflicting_documentation")
        or strength == "conflicting"
    )
    timeline = bool(gaps.get("timeline_inconsistency") or evidence.get("has_timeline_flags"))
    specialist = bool(
        gaps.get("specialist_review_signal")
        or evidence.get("has_specialist_notes")
        or metadata.get("specialist_review_candidate")
    )
    missing = bool(gaps.get("missing_evidence") or evidence.get("has_missing_required_evidence"))
    insufficiency = bool(gaps.get("documentation_insufficiency") or strength in {"weak", "low"})
    partial = bool(gaps.get("partial_support") or evidence.get("has_partial_support") or strength == "partial")
    high_ambiguity = bool(gaps.get("ambiguity") or ambiguity_level == "high" or conflicting or timeline)
    boundary = bool(metadata.get("boundary_case") or partial or recoverability in {"partially recoverable", "unclear recoverability"})
    return {
        "ambiguity_level": ambiguity_level,
        "boundary": boundary,
        "category": category,
        "conflicting": conflicting,
        "high_ambiguity": high_ambiguity,
        "insufficiency": insufficiency,
        "low_evidence": bool(strength in {"weak", "low"} or insufficiency or missing),
        "missing": missing,
        "partial": partial,
        "recoverability": recoverability,
        "specialist": specialist,
        "strength": strength,
        "timeline": timeline,
        "unsupported_service": category == "unsupported service",
    }


def _set_denial_variant_route(record: dict[str, Any], route: str, posture: str, reason: str) -> None:
    if route in DENIAL_ROUTING_ACTIONS:
        record["routing_action"] = route
    if posture in DENIAL_GOVERNANCE_POSTURES:
        record["governance_posture"] = posture
    adjustment = record.setdefault("variant_adjustment", {})
    if isinstance(adjustment, dict):
        adjustment["applied_reason"] = reason


def apply_denial_variant(
    record: dict[str, Any],
    experiment_label: str,
    case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply deterministic runner-scoped denial variant behavior without changing node rules."""
    if experiment_label not in {"threshold_variant", "routing_sensitivity_variant"}:
        return record

    updated = deepcopy(record)
    signals = _denial_variant_signals(updated, case)
    original_route = str(updated.get("routing_action") or "").strip().upper()
    original_posture = str(updated.get("governance_posture") or "").strip().upper()
    updated["variant_adjustment"] = {
        "variant": experiment_label,
        "description": DENIAL_VARIANT_DESCRIPTIONS.get(experiment_label),
        "baseline_routing_action": original_route,
        "baseline_governance_posture": original_posture,
        "signals": signals,
        "applied_reason": "no_adjustment",
    }

    if experiment_label == "threshold_variant":
        if signals["specialist"] or signals["unsupported_service"]:
            _set_denial_variant_route(updated, "ESCALATE", "HIGH_RISK", "strict_specialist_or_unsupported_service")
        elif signals["high_ambiguity"]:
            posture = "HIGH_RISK" if signals["conflicting"] and signals["low_evidence"] else "AMBIGUOUS"
            _set_denial_variant_route(updated, "ESCALATE", posture, "strict_high_ambiguity")
        elif signals["low_evidence"] and signals["recoverability"] == "low recoverability":
            _set_denial_variant_route(updated, "WRITE_OFF", "LOW_EVIDENCE", "strict_low_evidence_low_recoverability")
        elif signals["low_evidence"] and signals["boundary"]:
            _set_denial_variant_route(updated, "ESCALATE", "AMBIGUOUS", "strict_low_evidence_boundary")
        elif signals["partial"] and signals["recoverability"] in {"partially recoverable", "unclear recoverability"}:
            _set_denial_variant_route(updated, "ESCALATE", "AMBIGUOUS", "strict_partial_support_boundary")
        elif signals["recoverability"] == "likely recoverable" and signals["strength"] in {"strong", "moderate"}:
            _set_denial_variant_route(updated, "APPEAL", "SUPPORTED", "strict_supported_recoverable")

    if experiment_label == "routing_sensitivity_variant":
        if signals["specialist"] or signals["unsupported_service"]:
            _set_denial_variant_route(updated, "ESCALATE", "HIGH_RISK", "sensitive_preserve_high_risk_escalation")
        elif signals["conflicting"] or signals["timeline"]:
            _set_denial_variant_route(updated, "ESCALATE", "AMBIGUOUS", "sensitive_preserve_conflict_escalation")
        elif signals["low_evidence"] and signals["recoverability"] == "low recoverability":
            _set_denial_variant_route(updated, "WRITE_OFF", "LOW_EVIDENCE", "sensitive_low_recoverability_write_off")
        elif signals["partial"] and signals["recoverability"] in {"likely recoverable", "partially recoverable", "unclear recoverability"}:
            _set_denial_variant_route(updated, "RESUBMIT", "LOW_CONFIDENCE", "sensitive_partial_support_resubmit")
        elif signals["missing"] or signals["insufficiency"]:
            if signals["recoverability"] in {"likely recoverable", "partially recoverable"}:
                _set_denial_variant_route(updated, "RESUBMIT", "LOW_CONFIDENCE", "sensitive_recoverable_documentation_gap")
        elif signals["recoverability"] == "likely recoverable" and signals["strength"] in {"strong", "moderate"}:
            _set_denial_variant_route(updated, "APPEAL", "SUPPORTED", "sensitive_supported_appeal")

    adjustment = updated.get("variant_adjustment")
    if isinstance(adjustment, dict):
        adjustment["routing_changed"] = original_route != str(updated.get("routing_action") or "").strip().upper()
        adjustment["governance_changed"] = original_posture != str(updated.get("governance_posture") or "").strip().upper()
        adjustment["variant_routing_action"] = updated.get("routing_action")
        adjustment["variant_governance_posture"] = updated.get("governance_posture")
    return updated


@contextmanager
def _temporary_threshold_variant(experiment_label: str) -> Any:
    overrides = _variant_overrides(experiment_label)
    if not overrides:
        yield
        return

    from graph.nodes import governance_policy

    original_load_policy = governance_policy._load_policy

    def load_policy_with_variant() -> dict[str, Any]:
        policy = deepcopy(original_load_policy())
        policy["thresholds"].update(overrides)
        policy["governance_version"] = str(policy.get("governance_version", "")) + f"_{experiment_label}"
        return policy

    governance_policy._load_policy = load_policy_with_variant
    try:
        yield
    finally:
        governance_policy._load_policy = original_load_policy


def high_level_match(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    if error or not record:
        return EvaluationResult("high_level_match", False, error or "missing record")
    status = str(record.get("status", "")).strip().lower()
    has_core_outputs = bool(record.get("summary")) and bool(record.get("trace"))
    return EvaluationResult(
        "high_level_match",
        status in {"completed", "degraded"} and has_core_outputs,
        f"status={status} theme={case['operational_theme']}",
    )


def governance_status_match(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    expected = str(case.get("expected_governance_status", "")).strip().upper()
    if not expected:
        return EvaluationResult("governance_status_match", True, "no expected governance status")
    if error or not record:
        return EvaluationResult("governance_status_match", False, error or "missing record")
    actual = _record_status(record)
    return EvaluationResult("governance_status_match", actual == expected, f"expected={expected} actual={actual}")


def completed_without_fallback(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    if error or not record:
        return EvaluationResult("completed_without_fallback", False, error or "missing record")
    fallback_used = bool(record.get("fallback_used") or record.get("fallback_nodes"))
    status = str(record.get("status", "")).strip().lower()
    return EvaluationResult(
        "completed_without_fallback",
        status == "completed" and not fallback_used,
        f"status={status} fallback_used={fallback_used}",
    )


def final_status_present(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    if error or not record:
        return EvaluationResult("final_status_present", False, error or "missing record")
    final_status = _record_status(record)
    return EvaluationResult("final_status_present", bool(final_status), f"final_status={final_status or 'missing'}")


def evaluate_case(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> list[EvaluationResult]:
    return [
        high_level_match(case, record, error),
        governance_status_match(case, record, error),
        completed_without_fallback(case, record, error),
        final_status_present(case, record, error),
    ]


def denial_high_level_match(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    if error or not record:
        return EvaluationResult("denial_high_level_match", False, error or "missing record")
    status = str(record.get("status", "")).strip().lower()
    has_outputs = bool(record.get("routing_action")) and bool(record.get("governance_posture"))
    has_diagnostics = isinstance(record.get("node_diagnostics"), list) and bool(record.get("node_diagnostics"))
    return EvaluationResult(
        "denial_high_level_match",
        status in {"completed", "degraded"} and has_outputs and has_diagnostics,
        f"status={status} route={record.get('routing_action')} posture={record.get('governance_posture')}",
    )


def denial_routing_action_match(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    expected = str(case.get("expected_routing_action", "")).strip().upper()
    if error or not record:
        return EvaluationResult("denial_routing_action_match", False, error or "missing record")
    actual = str(record.get("routing_action", "")).strip().upper()
    return EvaluationResult("denial_routing_action_match", actual == expected, f"expected={expected} actual={actual}")


def denial_governance_posture_match(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    expected = str(case.get("expected_governance_posture", "")).strip().upper()
    if error or not record:
        return EvaluationResult("denial_governance_posture_match", False, error or "missing record")
    actual = str(record.get("governance_posture", "")).strip().upper()
    return EvaluationResult("denial_governance_posture_match", actual == expected, f"expected={expected} actual={actual}")


def denial_completed_without_fallback(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    if error or not record:
        return EvaluationResult("denial_completed_without_fallback", False, error or "missing record")
    fallback_used = bool(record.get("fallback_used"))
    status = str(record.get("status", "")).strip().lower()
    return EvaluationResult(
        "denial_completed_without_fallback",
        status == "completed" and not fallback_used,
        f"status={status} fallback_used={fallback_used}",
    )


def denial_final_route_present(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    if error or not record:
        return EvaluationResult("denial_final_route_present", False, error or "missing record")
    route = str(record.get("routing_action", "")).strip().upper()
    posture = str(record.get("governance_posture", "")).strip().upper()
    return EvaluationResult(
        "denial_final_route_present",
        route in DENIAL_ROUTING_ACTIONS and posture in DENIAL_GOVERNANCE_POSTURES,
        f"route={route or 'missing'} posture={posture or 'missing'}",
    )


def evaluate_denial_case(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> list[EvaluationResult]:
    return [
        denial_high_level_match(case, record, error),
        denial_routing_action_match(case, record, error),
        denial_governance_posture_match(case, record, error),
        denial_completed_without_fallback(case, record, error),
        denial_final_route_present(case, record, error),
    ]


def _run_pipeline(case: dict[str, Any], experiment_label: str) -> tuple[dict[str, Any] | None, str | None, int]:
    from service.run_manager import execute

    previous_mode = os.environ.get("MEDSCRIBE_EXECUTION_MODE")
    os.environ["MEDSCRIBE_EXECUTION_MODE"] = "hybrid"
    started = time.perf_counter()
    try:
        with _temporary_threshold_variant(experiment_label):
            record = execute(
                case["input_text"],
                run_id=f"{experiment_label}-{case['case_id']}",
                timestamp=_utc_stamp(),
                persist=False,
            )
        return record, None, max(1, int((time.perf_counter() - started) * 1000))
    except Exception as exc:
        partial = getattr(exc, "partial_record", None)
        record = partial if isinstance(partial, dict) else None
        return record, str(exc), max(1, int((time.perf_counter() - started) * 1000))
    finally:
        if previous_mode is None:
            os.environ.pop("MEDSCRIBE_EXECUTION_MODE", None)
        else:
            os.environ["MEDSCRIBE_EXECUTION_MODE"] = previous_mode


def _denial_operational_observability(record: dict[str, Any], latency_ms: int) -> dict[str, Any]:
    gaps = record.get("documentation_gaps", {})
    if not isinstance(gaps, dict):
        gaps = {}
    return {
        "observability_version": "denial_regenerated_v1",
        "comparison_profile": "denial_local_regenerated",
        "stage_timing_ms": {"total_ms": latency_ms},
        "denial_snapshot": {
            "routing_action": record.get("routing_action"),
            "governance_posture": record.get("governance_posture"),
            "denial_category": record.get("denial_category"),
            "recoverability": record.get("recoverability"),
            "evidence_strength": record.get("evidence_profile", {}).get("evidence_strength"),
            "documentation_gap_count": gaps.get("gap_count"),
            "ambiguity": bool(gaps.get("ambiguity")),
            "missing_evidence": bool(gaps.get("missing_evidence")),
            "partial_support": bool(gaps.get("partial_support")),
            "timeline_inconsistency": bool(gaps.get("timeline_inconsistency")),
            "specialist_review_signal": bool(gaps.get("specialist_review_signal")),
        },
        "node_diagnostic_count": len(record.get("node_diagnostics", [])),
        "fallback_used": bool(record.get("fallback_used")),
        "degraded_mode": bool(record.get("degraded_mode")),
    }


def _run_denial_pipeline(
    case: dict[str, Any],
    experiment_label: str,
    *,
    execution_mode: str = "",
) -> tuple[dict[str, Any] | None, str | None, int]:
    from graph.denial_graph import run_denial_graph

    previous_mode = os.environ.get("MEDSCRIBE_EXECUTION_MODE")
    if execution_mode:
        os.environ["MEDSCRIBE_EXECUTION_MODE"] = execution_mode
    started = time.perf_counter()
    try:
        record = run_denial_graph(case)
        record = apply_denial_variant(record, experiment_label, case)
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        record["run_id"] = f"denial-{experiment_label}-{case['case_id']}"
        record["experiment_label"] = experiment_label
        record["workflow"] = "denial"
        record["trace"] = [diagnostic.get("node_name") for diagnostic in record.get("node_diagnostics", [])]
        record["runtime_status"] = record.get("status", "completed")
        record["operational_observability"] = _denial_operational_observability(record, latency_ms)
        return record, None, latency_ms
    except Exception as exc:
        return None, str(exc), max(1, int((time.perf_counter() - started) * 1000))
    finally:
        if execution_mode:
            if previous_mode is None:
                os.environ.pop("MEDSCRIBE_EXECUTION_MODE", None)
            else:
                os.environ["MEDSCRIBE_EXECUTION_MODE"] = previous_mode


def _run_case_result(
    case: dict[str, Any],
    experiment_label: str,
    dataset_name: str,
    *,
    workflow: str = "cdi",
    execution_mode: str = "",
) -> dict[str, Any]:
    from graph.operational_alerts import build_layer1_payload

    if workflow == "denial":
        record, error, latency_ms = _run_denial_pipeline(case, experiment_label, execution_mode=execution_mode)
        evaluations = evaluate_denial_case(case, record, error)
    else:
        record, error, latency_ms = _run_pipeline(case, experiment_label)
        evaluations = evaluate_case(case, record, error)
    token_metadata = {}
    if record:
        token_metadata = record.get("metadata", {}).get("token_usage", {}) or {}
    final_status = (
        str(record.get("routing_action") or record.get("governance_posture", "")).strip().upper()
        if workflow == "denial" and record
        else _record_status(record or {})
    )
    result = {
        "case_id": case["case_id"],
        "title": case["title"],
        "workflow": workflow,
        "experiment_label": experiment_label,
        "dataset_name": dataset_name,
        "threshold_variant_overrides": THRESHOLD_VARIANT_OVERRIDES if experiment_label == "threshold_variant" else {},
        "variant_threshold_overrides": _variant_overrides(experiment_label),
        "runtime_status": record.get("status") if record else "error",
        "final_status": final_status,
        "routing_action": record.get("routing_action") if record else "",
        "governance_posture": record.get("governance_posture") if record else "",
        "denial_category": record.get("denial_category") if record else "",
        "recoverability": record.get("recoverability") if record else "",
        "latency_ms": latency_ms,
        "token_metadata": token_metadata,
        "error": error,
        "evaluations": [
            {"key": item.key, "score": item.score, "comment": item.comment}
            for item in evaluations
        ],
        "record": record,
    }
    result["output"] = _primary_output(workflow, result)
    result["metadata"] = _operational_metadata(
        case,
        result,
        workflow=workflow,
        variant=experiment_label,
        latency_ms=latency_ms,
    )
    token_available, token_unavailable_reason, token_details = _token_cost_status(record, workflow)
    layer1 = build_layer1_payload(
        workflow=workflow,
        status=str(result.get("runtime_status") or "failed"),
        output=result["output"],
        metadata=result["metadata"],
        record=record,
        existing_error=error,
        token_metadata=token_details,
        expected_trace_count=6,
    )
    result["status"] = layer1["status"]
    result["error"] = layer1["error"]
    result["alerts"] = layer1["alerts"]
    result["alert_count"] = layer1["alert_count"]
    result["max_alert_severity"] = layer1["max_alert_severity"]
    result["operational_metrics"] = layer1["operational_metrics"]
    result["operational_thresholds"] = layer1["operational_thresholds"]
    result["output"] = _primary_output(workflow, result)
    result["metadata"].update(
        {
            "status": result["status"],
            "error": result["error"],
            "alerts": result["alerts"],
            "alert_count": result["alert_count"],
            "max_alert_severity": result["max_alert_severity"],
            "operational_metrics": result["operational_metrics"],
            "token_cost_available": token_available,
            "token_cost_unavailable_reason": token_unavailable_reason or None,
        }
    )
    result["tags"] = _operational_tags(result["metadata"])
    return result


def _console_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "record"}


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _distribution(counter: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {key: round(value / total, 4) for key, value in sorted(counter.items())}


def _top_counts(counter: dict[str, int], limit: int = 5) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _routing_distribution_report(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    raw_status_counts: dict[str, int] = {}
    bucket_counts = {bucket: 0 for bucket in FINAL_STATUS_BUCKETS}
    expected_status_counts: dict[str, int] = {}
    expected_routing_counts: dict[str, int] = {}
    reason_code_counts: dict[str, int] = {}
    critic_recommended_status_counts: dict[str, int] = {}
    runtime_status_counts: dict[str, int] = {}
    zero_metric_counts = {
        "diagnosis_consistency_score": 0,
        "symptom_alignment_score": 0,
        "icd_specificity_score": 0,
        "confidence": 0,
    }
    low_evidence_volume = 0
    specialist_review_volume = 0
    coder_review_volume = 0
    unsupported_specificity_volume = 0
    degraded_count = 0
    fallback_count = 0
    adjacent_spillover_indicators = {
        "watched_cases": 0,
        "pass_on_spillover_watch": 0,
        "expected_status_mismatch": 0,
        "unsafe_pass_count": 0,
    }
    latency_values: list[int] = []
    token_metadata_available = False
    token_metadata_keys: set[str] = set()

    cases_by_id = _case_by_id(cases)
    for result in results:
        case = cases_by_id.get(str(result.get("case_id")), {})
        record = result.get("record") if isinstance(result.get("record"), dict) else None
        final_status = str(result.get("final_status") or _record_status(record or {})).strip().upper()
        bucket = _status_bucket(final_status)
        _increment(raw_status_counts, final_status or "MISSING")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        runtime_status = str(result.get("runtime_status") or (record or {}).get("status") or "missing").strip().lower()
        _increment(runtime_status_counts, runtime_status)
        if runtime_status == "degraded":
            degraded_count += 1
        if isinstance(record, dict) and bool(record.get("fallback_used") or record.get("fallback_nodes")):
            fallback_count += 1

        expected_status = str(case.get("expected_governance_status", "")).strip().upper()
        if expected_status:
            _increment(expected_status_counts, expected_status)
            if final_status != expected_status:
                adjacent_spillover_indicators["expected_status_mismatch"] += 1
        expected_route = str(case.get("expected_routing_category", "")).strip()
        if expected_route:
            _increment(expected_routing_counts, expected_route)

        reason_codes = _record_reason_codes(record)
        for code in reason_codes:
            _increment(reason_code_counts, code)
        critic = _record_critic_snapshot(record)
        recommended_status = str(critic.get("recommended_status", "")).strip().lower()
        if recommended_status:
            _increment(critic_recommended_status_counts, recommended_status)
        for metric_name in zero_metric_counts:
            value = critic.get(metric_name)
            if isinstance(value, (int, float)) and value == 0:
                zero_metric_counts[metric_name] += 1

        signals = _expected_case_signals(case)
        if signals["expects_low_evidence"] or _contains_marker(reason_codes, LOW_EVIDENCE_MARKERS):
            low_evidence_volume += 1
        if signals["expects_specialist_review"] or _contains_marker(reason_codes, SPECIALIST_MARKERS):
            specialist_review_volume += 1
        if signals["expects_coder_review"]:
            coder_review_volume += 1
        if signals["expects_unsupported_specificity"] or _contains_marker(reason_codes, UNSUPPORTED_SPECIFICITY_MARKERS):
            unsupported_specificity_volume += 1
        if signals["expects_spillover_watch"]:
            adjacent_spillover_indicators["watched_cases"] += 1
            if bucket == "PASS":
                adjacent_spillover_indicators["pass_on_spillover_watch"] += 1
        if signals["unsafe_pass_risk"] and bucket == "PASS":
            adjacent_spillover_indicators["unsafe_pass_count"] += 1

        latency = result.get("latency_ms")
        if isinstance(latency, int):
            latency_values.append(latency)
        token_metadata = result.get("token_metadata")
        if isinstance(token_metadata, dict) and token_metadata:
            token_metadata_available = True
            token_metadata_keys.update(str(key) for key in token_metadata)

    latency_report: dict[str, Any] = {"available": bool(latency_values)}
    if latency_values:
        latency_report.update(
            {
                "count": len(latency_values),
                "min_ms": min(latency_values),
                "max_ms": max(latency_values),
                "mean_ms": round(sum(latency_values) / len(latency_values), 2),
            }
        )

    return {
        "layer": "layer_3_routing_distribution",
        "status_counts": {key: raw_status_counts.get(key, 0) for key in sorted(raw_status_counts)},
        "status_bucket_counts": bucket_counts,
        "status_bucket_distribution": _distribution(bucket_counts, total),
        "expected_status_counts": {key: expected_status_counts.get(key, 0) for key in sorted(expected_status_counts)},
        "expected_routing_category_counts": {
            key: expected_routing_counts.get(key, 0) for key in sorted(expected_routing_counts)
        },
        "low_evidence_routing_volume": low_evidence_volume,
        "specialist_review_oriented_volume": specialist_review_volume,
        "coder_review_oriented_volume": coder_review_volume,
        "unsupported_specificity_volume": unsupported_specificity_volume,
        "governance_reason_code_counts": {
            key: reason_code_counts.get(key, 0) for key in sorted(reason_code_counts)
        },
        "dominant_reason_codes": _top_counts(reason_code_counts),
        "runtime_status_counts": {
            key: runtime_status_counts.get(key, 0) for key in sorted(runtime_status_counts)
        },
        "degraded_count": degraded_count,
        "fallback_count": fallback_count,
        "critic_recommended_status_counts": {
            key: critic_recommended_status_counts.get(key, 0)
            for key in sorted(critic_recommended_status_counts)
        },
        "zero_metric_counts": zero_metric_counts,
        "dominance_diagnostics": {
            "all_results_degraded": total > 0 and degraded_count == total,
            "all_results_used_fallback": total > 0 and fallback_count == total,
            "all_critic_recommendations_fail": total > 0
            and critic_recommended_status_counts.get("fail", 0) == total,
            "all_core_metrics_zero": total > 0
            and all(
                zero_metric_counts[key] == total
                for key in (
                    "diagnosis_consistency_score",
                    "symptom_alignment_score",
                    "icd_specificity_score",
                )
            ),
        },
        "adjacent_case_spillover_indicators": adjacent_spillover_indicators,
        "latency": latency_report,
        "token_and_cost": {
            "available": token_metadata_available,
            "metadata_keys": sorted(token_metadata_keys),
            "note": "unavailable" if not token_metadata_available else "reported from runtime metadata",
        },
    }


def _denial_routing_distribution_report(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    routing_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    expected_routing_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    posture_counts = {posture: 0 for posture in DENIAL_GOVERNANCE_POSTURES}
    expected_posture_counts = {posture: 0 for posture in DENIAL_GOVERNANCE_POSTURES}
    category_counts: dict[str, int] = {}
    recoverability_counts: dict[str, int] = {}
    evidence_strength_counts: dict[str, int] = {}
    recoverability_movement_observations: dict[str, int] = {}
    runtime_status_counts: dict[str, int] = {}
    low_evidence_routing_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    ambiguity_routing_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    specialist_review_candidate_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    conflicting_evidence_routing_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    partial_support_routing_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    timeline_inconsistency_routing_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    recoverability_boundary_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    threshold_boundary_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    ambiguity_spillover_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    specialist_review_movement_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    low_evidence_movement_counts = {action: 0 for action in DENIAL_ROUTING_ACTIONS}
    variant_adjustment_counts: dict[str, int] = {}
    variant_route_movement_counts: dict[str, int] = {}
    variant_governance_movement_counts: dict[str, int] = {}
    degraded_count = 0
    fallback_count = 0
    latency_values: list[int] = []
    token_metadata_available = False
    token_metadata_keys: set[str] = set()

    cases_by_id = _case_by_id(cases)
    for result in results:
        case = cases_by_id.get(str(result.get("case_id")), {})
        record = result.get("record") if isinstance(result.get("record"), dict) else {}
        route = str(result.get("routing_action") or record.get("routing_action") or "").strip().upper()
        posture = str(result.get("governance_posture") or record.get("governance_posture") or "").strip().upper()
        if route in routing_counts:
            routing_counts[route] += 1
        if posture in posture_counts:
            posture_counts[posture] += 1
        expected_route = str(case.get("expected_routing_action", "")).strip().upper()
        expected_posture = str(case.get("expected_governance_posture", "")).strip().upper()
        if expected_route in expected_routing_counts:
            expected_routing_counts[expected_route] += 1
        if expected_posture in expected_posture_counts:
            expected_posture_counts[expected_posture] += 1
        _increment(category_counts, str(record.get("denial_category") or result.get("denial_category") or "missing"))
        recoverability = str(record.get("recoverability") or result.get("recoverability") or "missing")
        _increment(recoverability_counts, recoverability)
        evidence = record.get("evidence_profile") if isinstance(record, dict) else {}
        if not isinstance(evidence, dict):
            evidence = {}
        strength = str(evidence.get("evidence_strength") or "unspecified")
        _increment(evidence_strength_counts, strength)
        if expected_route and route and expected_route != route:
            _increment(recoverability_movement_observations, f"{recoverability}:{expected_route}->{route}")
        runtime_status = str(result.get("runtime_status") or record.get("status") or "missing").strip().lower()
        _increment(runtime_status_counts, runtime_status)
        if runtime_status == "degraded" or record.get("degraded_mode"):
            degraded_count += 1
        if record.get("fallback_used"):
            fallback_count += 1
        gaps = record.get("documentation_gaps") if isinstance(record, dict) else {}
        if not isinstance(gaps, dict):
            gaps = {}
        if posture == "LOW_EVIDENCE" or gaps.get("documentation_insufficiency"):
            if route in low_evidence_routing_counts:
                low_evidence_routing_counts[route] += 1
        if posture == "AMBIGUOUS" or gaps.get("ambiguity"):
            if route in ambiguity_routing_counts:
                ambiguity_routing_counts[route] += 1
        if gaps.get("ambiguity") and evidence.get("has_conflicting_documentation"):
            if route in conflicting_evidence_routing_counts:
                conflicting_evidence_routing_counts[route] += 1
        if gaps.get("partial_support"):
            if route in partial_support_routing_counts:
                partial_support_routing_counts[route] += 1
        if gaps.get("timeline_inconsistency"):
            if route in timeline_inconsistency_routing_counts:
                timeline_inconsistency_routing_counts[route] += 1
        if recoverability in {"partially recoverable", "unclear recoverability"}:
            if route in recoverability_boundary_counts:
                recoverability_boundary_counts[route] += 1
        metadata = case.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("specialist_review_candidate"):
            if route in specialist_review_candidate_counts:
                specialist_review_candidate_counts[route] += 1
            if route in specialist_review_movement_counts:
                specialist_review_movement_counts[route] += 1
        boundary_case = bool(isinstance(metadata, dict) and metadata.get("boundary_case"))
        if boundary_case:
            if route in threshold_boundary_counts:
                threshold_boundary_counts[route] += 1
        if (gaps.get("ambiguity") or posture in {"AMBIGUOUS", "HIGH_RISK"}) and route != "ESCALATE":
            if route in ambiguity_spillover_counts:
                ambiguity_spillover_counts[route] += 1
        if (posture == "LOW_EVIDENCE" or gaps.get("documentation_insufficiency")) and route in low_evidence_movement_counts:
            low_evidence_movement_counts[route] += 1
        adjustment = record.get("variant_adjustment") if isinstance(record, dict) else {}
        if isinstance(adjustment, dict) and adjustment:
            _increment(variant_adjustment_counts, str(adjustment.get("applied_reason") or "no_adjustment"))
            baseline_route = str(adjustment.get("baseline_routing_action") or "").strip().upper()
            variant_route = str(adjustment.get("variant_routing_action") or route).strip().upper()
            if baseline_route and variant_route and baseline_route != variant_route:
                _increment(variant_route_movement_counts, f"{baseline_route}->{variant_route}")
            baseline_posture = str(adjustment.get("baseline_governance_posture") or "").strip().upper()
            variant_posture = str(adjustment.get("variant_governance_posture") or posture).strip().upper()
            if baseline_posture and variant_posture and baseline_posture != variant_posture:
                _increment(variant_governance_movement_counts, f"{baseline_posture}->{variant_posture}")
        latency = result.get("latency_ms")
        if isinstance(latency, int):
            latency_values.append(latency)
        token_metadata = result.get("token_metadata")
        if isinstance(token_metadata, dict) and token_metadata:
            token_metadata_available = True
            token_metadata_keys.update(str(key) for key in token_metadata)

    latency_report: dict[str, Any] = {"available": bool(latency_values)}
    if latency_values:
        latency_report.update(
            {
                "count": len(latency_values),
                "min_ms": min(latency_values),
                "max_ms": max(latency_values),
                "mean_ms": round(sum(latency_values) / len(latency_values), 2),
            }
        )

    return {
        "layer": "layer_3_denial_routing_distribution",
        "routing_action_counts": routing_counts,
        "routing_action_distribution": _distribution(routing_counts, total),
        "expected_routing_action_counts": expected_routing_counts,
        "governance_posture_counts": posture_counts,
        "governance_posture_distribution": _distribution(posture_counts, total),
        "expected_governance_posture_counts": expected_posture_counts,
        "denial_category_counts": {key: category_counts.get(key, 0) for key in sorted(category_counts)},
        "recoverability_counts": {key: recoverability_counts.get(key, 0) for key in sorted(recoverability_counts)},
        "evidence_strength_counts": {key: evidence_strength_counts.get(key, 0) for key in sorted(evidence_strength_counts)},
        "low_evidence_routing_counts": low_evidence_routing_counts,
        "ambiguity_routing_counts": ambiguity_routing_counts,
        "specialist_review_candidate_routing_counts": specialist_review_candidate_counts,
        "conflicting_evidence_routing_counts": conflicting_evidence_routing_counts,
        "partial_support_routing_counts": partial_support_routing_counts,
        "timeline_inconsistency_routing_counts": timeline_inconsistency_routing_counts,
        "recoverability_boundary_routing_counts": recoverability_boundary_counts,
        "threshold_boundary_routing_counts": threshold_boundary_counts,
        "ambiguity_spillover_routing_counts": ambiguity_spillover_counts,
        "specialist_review_movement_counts": specialist_review_movement_counts,
        "low_evidence_movement_counts": low_evidence_movement_counts,
        "variant_adjustment_counts": {
            key: variant_adjustment_counts.get(key, 0) for key in sorted(variant_adjustment_counts)
        },
        "variant_route_movement_counts": {
            key: variant_route_movement_counts.get(key, 0) for key in sorted(variant_route_movement_counts)
        },
        "variant_governance_movement_counts": {
            key: variant_governance_movement_counts.get(key, 0) for key in sorted(variant_governance_movement_counts)
        },
        "recoverability_movement_observations": {
            key: recoverability_movement_observations.get(key, 0)
            for key in sorted(recoverability_movement_observations)
        },
        "runtime_status_counts": {key: runtime_status_counts.get(key, 0) for key in sorted(runtime_status_counts)},
        "degraded_count": degraded_count,
        "fallback_count": fallback_count,
        "latency": latency_report,
        "token_and_cost": {
            "available": token_metadata_available,
            "metadata_keys": sorted(token_metadata_keys),
            "note": "unavailable" if not token_metadata_available else "reported from runtime metadata",
        },
    }


def _layer1_alert_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    alert_class_counts: dict[str, int] = {}
    total_alerts = 0
    error_count = 0
    for result in results:
        status = str(result.get("status") or result.get("runtime_status") or "missing").strip().lower()
        _increment(status_counts, status or "missing")
        if result.get("error"):
            error_count += 1
        alerts = result.get("alerts") if isinstance(result.get("alerts"), list) else []
        total_alerts += len(alerts)
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            _increment(alert_class_counts, str(alert.get("class") or "missing"))
            _increment(severity_counts, str(alert.get("severity") or "info"))
    return {
        "layer": "layer_1_operational_alerting",
        "status_counts": {key: status_counts.get(key, 0) for key in sorted(status_counts)},
        "alert_class_counts": {key: alert_class_counts.get(key, 0) for key in sorted(alert_class_counts)},
        "severity_counts": {key: severity_counts.get(key, 0) for key in sorted(severity_counts)},
        "total_alerts": total_alerts,
        "error_count": error_count,
    }


def _summary(
    *,
    cases: list[dict[str, Any]],
    experiment_label: str,
    dataset_name: str,
    results: list[dict[str, Any]],
    workflow: str = "cdi",
    hosted_experiment_name: str | None = None,
    hosted_experiment_url: str | None = None,
    formal_hosted: bool = False,
) -> dict[str, Any]:
    routing_distribution = (
        _denial_routing_distribution_report(cases, results)
        if workflow == "denial"
        else _routing_distribution_report(cases, results)
    )
    summary: dict[str, Any] = {
        "dataset_name": dataset_name,
        "workflow": workflow,
        "experiment_label": experiment_label,
        "case_count": len(cases),
        "generated_at": _utc_stamp(),
        "formal_hosted_experiment": formal_hosted,
        "threshold_variant_overrides": THRESHOLD_VARIANT_OVERRIDES if experiment_label == "threshold_variant" else {},
        "variant_threshold_overrides": _variant_overrides(experiment_label),
        "routing_distribution": routing_distribution,
        "layer1_alert_summary": _layer1_alert_summary(results),
        "results": [_console_result(result) for result in results],
    }
    if hosted_experiment_name:
        summary["hosted_experiment_name"] = hosted_experiment_name
    if hosted_experiment_url:
        summary["hosted_experiment_url"] = hosted_experiment_url
    return summary


def run_experiment(
    cases: list[dict[str, Any]],
    experiment_label: str,
    dataset_name: str,
    *,
    workflow: str = "cdi",
    execution_mode: str = "",
) -> dict[str, Any]:
    if experiment_label not in EXPERIMENT_LABELS:
        raise ValueError(f"experiment_label must be one of {sorted(EXPERIMENT_LABELS)}")

    results = [
        _run_case_result(case, experiment_label, dataset_name, workflow=workflow, execution_mode=execution_mode)
        for case in cases
    ]
    return _summary(
        cases=cases,
        experiment_label=experiment_label,
        dataset_name=dataset_name,
        results=results,
        workflow=workflow,
    )


def _result_from_run_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    if "output" in outputs and isinstance(outputs["output"], dict):
        outputs = outputs["output"]
    return dict(outputs)


def _runtime_error_for_evaluator(result: dict[str, Any]) -> str | None:
    error = result.get("error")
    if not error:
        return None
    status = str(result.get("status") or result.get("runtime_status") or "").strip().lower()
    if status == "failed":
        return str(error)
    alerts = result.get("alerts") if isinstance(result.get("alerts"), list) else []
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        if str(alert.get("class")) == str(error) and str(alert.get("severity")) != "critical":
            return None
    return str(error)


def _rows_to_results(rows: Iterable[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        run = row.get("run") if isinstance(row, dict) else getattr(row, "run", None)
        outputs = getattr(run, "outputs", None) if run is not None else None
        if isinstance(outputs, dict):
            results.append(_result_from_run_outputs(outputs))
    return results


def _hosted_evaluator(
    key: str,
    cases_by_id: dict[str, dict[str, Any]],
    evaluator_fn: Any,
) -> Any:
    def evaluator(run: Any, example: Any) -> dict[str, Any]:
        outputs = getattr(run, "outputs", None) or {}
        result = _result_from_run_outputs(outputs) if isinstance(outputs, dict) else {}
        record = result.get("record")
        if not isinstance(record, dict):
            record = None
        error = _runtime_error_for_evaluator(result)
        case = _case_from_example(example, cases_by_id)
        evaluation = evaluator_fn(case, record, error)
        return {
            "key": key,
            "score": evaluation.score,
            "comment": evaluation.comment,
        }

    evaluator.__name__ = key
    return evaluator


def _hosted_denial_evaluator(
    key: str,
    cases_by_id: dict[str, dict[str, Any]],
    evaluator_fn: Any,
) -> Any:
    def evaluator(run: Any, example: Any) -> dict[str, Any]:
        outputs = getattr(run, "outputs", None) or {}
        result = _result_from_run_outputs(outputs) if isinstance(outputs, dict) else {}
        record = result.get("record")
        if not isinstance(record, dict):
            record = None
        error = _runtime_error_for_evaluator(result)
        case = _case_from_example(example, cases_by_id)
        evaluation = evaluator_fn(case, record, error)
        return {
            "key": key,
            "score": evaluation.score,
            "comment": evaluation.comment,
        }

    evaluator.__name__ = key
    return evaluator


def run_hosted_experiment(
    client: Any,
    cases: list[dict[str, Any]],
    experiment_label: str,
    dataset_name: str,
    *,
    workflow: str = "cdi",
    execution_mode: str = "",
) -> dict[str, Any]:
    if experiment_label not in EXPERIMENT_LABELS:
        raise ValueError(f"experiment_label must be one of {sorted(EXPERIMENT_LABELS)}")

    from langsmith import evaluate

    cases_by_id = _case_by_id(cases)
    cases_by_input = {} if workflow == "denial" else _case_by_input(cases)
    cases_by_payload = _case_by_denial_payload(cases) if workflow == "denial" else {}
    examples = _list_selected_examples(client, dataset_name, cases)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        if workflow == "denial":
            payload = inputs.get("case_payload")
            case = cases_by_payload[json.dumps(payload, sort_keys=True)]
        else:
            input_text = str(inputs.get("input_text", ""))
            case = cases_by_input[input_text]
        return _run_case_result(case, experiment_label, dataset_name, workflow=workflow, execution_mode=execution_mode)

    target.__name__ = f"medscribe_{workflow}_{experiment_label}_operational_benchmark"

    if workflow == "denial":
        evaluators = [
            _hosted_denial_evaluator("denial_high_level_match", cases_by_id, denial_high_level_match),
            _hosted_denial_evaluator("denial_routing_action_match", cases_by_id, denial_routing_action_match),
            _hosted_denial_evaluator("denial_governance_posture_match", cases_by_id, denial_governance_posture_match),
            _hosted_denial_evaluator("denial_completed_without_fallback", cases_by_id, denial_completed_without_fallback),
            _hosted_denial_evaluator("denial_final_route_present", cases_by_id, denial_final_route_present),
        ]
    else:
        evaluators = [
            _hosted_evaluator("high_level_match", cases_by_id, high_level_match),
            _hosted_evaluator("governance_status_match", cases_by_id, governance_status_match),
            _hosted_evaluator("completed_without_fallback", cases_by_id, completed_without_fallback),
            _hosted_evaluator("final_status_present", cases_by_id, final_status_present),
        ]
    experiment_prefix = f"medscribe-{workflow}-operational-{experiment_label}"
    hosted_results = evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        description="Regenerated MedScribe operational benchmark experiment.",
        metadata={
            "dataset_name": dataset_name,
            "workflow": workflow,
            "experiment_label": experiment_label,
            "case_limit": len(cases),
            "runner": "evaluation/langsmith_experiment_runner.py",
        },
        max_concurrency=0,
        client=client,
        blocking=True,
        upload_results=True,
        error_handling="log",
    )
    rows = list(hosted_results)
    results = _rows_to_results(rows)
    hosted_experiment_name = getattr(hosted_results, "experiment_name", None)
    hosted_experiment_url = None
    if hosted_experiment_name:
        try:
            project = client.read_project(project_name=hosted_experiment_name)
            hosted_experiment_url = getattr(project, "url", None)
        except Exception:
            hosted_experiment_url = None
    return _summary(
        cases=cases,
        experiment_label=experiment_label,
        dataset_name=dataset_name,
        results=results,
        workflow=workflow,
        hosted_experiment_name=hosted_experiment_name,
        hosted_experiment_url=hosted_experiment_url,
        formal_hosted=True,
    )


def _pairwise_score(case: dict[str, Any], result: dict[str, Any]) -> tuple[int, list[str]]:
    record = result.get("record") if isinstance(result.get("record"), dict) else None
    final_status = str(result.get("final_status") or _record_status(record or {})).strip().upper()
    expected_status = str(case.get("expected_governance_status", "")).strip().upper()
    bucket = _status_bucket(final_status)
    reason_codes = _record_reason_codes(record)
    signals = _expected_case_signals(case)
    score = 0
    signals_seen: list[str] = []

    if expected_status:
        distance = _status_distance(final_status, expected_status)
        score += max(0, 4 - distance)
        signals_seen.append(f"expected_status_distance={distance}")
        if final_status == expected_status:
            score += 3
            signals_seen.append("matched_expected_governance_status")

    if signals["unsafe_pass_risk"] and bucket == "PASS":
        score -= 6
        signals_seen.append("unsafe_pass_penalty")
    elif signals["unsafe_pass_risk"] and bucket in {"REVISE", "FAIL", "ESCALATE"}:
        score += 2
        signals_seen.append("avoided_unsafe_pass")

    if signals["expects_unsupported_specificity"]:
        if _contains_marker(reason_codes, UNSUPPORTED_SPECIFICITY_MARKERS) or bucket in {"REVISE", "FAIL", "ESCALATE"}:
            score += 2
            signals_seen.append("unsupported_specificity_preserved")
        else:
            score -= 2
            signals_seen.append("unsupported_specificity_not_visible")

    if signals["expects_low_evidence"]:
        if _contains_marker(reason_codes, LOW_EVIDENCE_MARKERS) or bucket in {"REVISE", "FAIL", "ESCALATE"}:
            score += 2
            signals_seen.append("low_evidence_handled_conservatively")
        else:
            score -= 2
            signals_seen.append("low_evidence_not_visible")

    if signals["expects_specialist_review"] and bucket in {"REVISE", "FAIL", "ESCALATE"}:
        score += 1
        signals_seen.append("specialist_review_route_preserved")
    if signals["expects_coder_review"] and bucket in {"REVISE", "FAIL", "ESCALATE"}:
        score += 1
        signals_seen.append("coder_review_route_preserved")
    if signals["expects_spillover_watch"] and bucket != "PASS":
        score += 1
        signals_seen.append("adjacent_spillover_guarded")

    critic = _record_critic_snapshot(record)
    if any(critic.get(key) is not None for key in ("confidence", "recommended_status", "icd_specificity_score")):
        score += 1
        signals_seen.append("inspectable_critic_snapshot")
    if reason_codes:
        score += 1
        signals_seen.append("inspectable_reason_codes")
    if result.get("error"):
        score -= 4
        signals_seen.append("runtime_error_penalty")

    return score, signals_seen


def _denial_pairwise_score(case: dict[str, Any], result: dict[str, Any]) -> tuple[int, list[str]]:
    record = result.get("record") if isinstance(result.get("record"), dict) else {}
    route = str(result.get("routing_action") or record.get("routing_action") or "").strip().upper()
    posture = str(result.get("governance_posture") or record.get("governance_posture") or "").strip().upper()
    expected_route = str(case.get("expected_routing_action", "")).strip().upper()
    expected_posture = str(case.get("expected_governance_posture", "")).strip().upper()
    recoverability = str(record.get("recoverability") or result.get("recoverability") or "").strip().lower()
    gaps = record.get("documentation_gaps") if isinstance(record, dict) else {}
    if not isinstance(gaps, dict):
        gaps = {}
    evidence = record.get("evidence_profile") if isinstance(record, dict) else {}
    if not isinstance(evidence, dict):
        evidence = {}
    metadata = case.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    ambiguous = bool(gaps.get("ambiguity")) or expected_posture in {"AMBIGUOUS", "HIGH_RISK"}
    low_evidence = bool(gaps.get("documentation_insufficiency")) or expected_posture == "LOW_EVIDENCE"
    recoverable = recoverability in {"likely recoverable", "partially recoverable"}
    partial_support = bool(gaps.get("partial_support") or evidence.get("has_partial_support"))
    conflicting = bool(evidence.get("has_conflicting_documentation") or gaps.get("ambiguity") and evidence.get("evidence_strength") == "conflicting")
    specialist_signal = bool(metadata.get("specialist_review_candidate") or evidence.get("has_specialist_notes"))
    score = 0
    signals_seen: list[str] = []

    if route == expected_route:
        score += 4
        signals_seen.append("matched_expected_routing_action")
    if posture == expected_posture:
        score += 3
        signals_seen.append("matched_expected_governance_posture")
    if ambiguous:
        if route == "ESCALATE":
            score += 3
            signals_seen.append("ambiguity_escalated")
        elif route == "RESUBMIT" and partial_support and recoverable and not conflicting and not specialist_signal:
            score += 1
            signals_seen.append("recoverable_ambiguity_resubmit")
        elif route in {"APPEAL", "WRITE_OFF"}:
            score -= 4
            signals_seen.append("unsafe_certainty_on_ambiguity")
    if low_evidence:
        if route in {"ESCALATE", "WRITE_OFF", "RESUBMIT"}:
            score += 1
            signals_seen.append("low_evidence_not_appealed")
        if route == "APPEAL":
            score -= 3
            signals_seen.append("low_evidence_appeal_penalty")
    if partial_support and route == "WRITE_OFF":
        score -= 4
        signals_seen.append("partial_support_write_off_penalty")
    elif partial_support and route in {"RESUBMIT", "ESCALATE"}:
        score += 2
        signals_seen.append("partial_support_preserved")
    if conflicting and route == "ESCALATE":
        score += 2
        signals_seen.append("conflicting_evidence_escalated")
    elif conflicting and route in {"APPEAL", "WRITE_OFF"}:
        score -= 3
        signals_seen.append("conflicting_evidence_certainty_penalty")
    if recoverable:
        if route in {"APPEAL", "RESUBMIT", "ESCALATE"}:
            score += 2
            signals_seen.append("recoverable_route_preserved")
        if route == "WRITE_OFF":
            score -= 5
            signals_seen.append("premature_write_off_penalty")
    if specialist_signal:
        if route == "ESCALATE" and posture == "HIGH_RISK":
            score += 3
            signals_seen.append("specialist_review_escalated")
        elif route != "ESCALATE":
            score -= 3
            signals_seen.append("specialist_review_not_escalated")
    adjustment = record.get("variant_adjustment") if isinstance(record, dict) else {}
    if isinstance(adjustment, dict) and adjustment:
        reason = str(adjustment.get("applied_reason") or "")
        if reason != "no_adjustment":
            signals_seen.append(f"variant_adjustment={reason}")
        if adjustment.get("routing_changed"):
            score += 1
            signals_seen.append("variant_routing_movement")
        if route == "WRITE_OFF" and (ambiguous or partial_support or recoverable):
            score -= 4
            signals_seen.append("over_aggressive_write_off_penalty")
        if route == "ESCALATE" and recoverable and partial_support and not conflicting and not specialist_signal:
            score -= 1
            signals_seen.append("avoidable_recoverable_escalation_penalty")
    if record.get("denial_category"):
        score += 1
        signals_seen.append("inspectable_denial_category")
    if record.get("node_diagnostics"):
        score += 1
        signals_seen.append("inspectable_node_diagnostics")
    if result.get("error"):
        score -= 4
        signals_seen.append("runtime_error_penalty")
    return score, signals_seen


def _pairwise_preference(case: dict[str, Any], baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    baseline_score, baseline_signals = _pairwise_score(case, baseline)
    variant_score, variant_signals = _pairwise_score(case, variant)
    if baseline_score > variant_score:
        preferred = "baseline"
    elif variant_score > baseline_score:
        preferred = "threshold_variant"
    else:
        preferred = "tie"
    baseline_record = baseline.get("record") if isinstance(baseline.get("record"), dict) else None
    variant_record = variant.get("record") if isinstance(variant.get("record"), dict) else None
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "operational_theme": case["operational_theme"],
        "question": "Which output provides the safer, clearer, and more operationally useful CDI review-routing outcome before human review?",
        "expected_governance_status": case.get("expected_governance_status"),
        "expected_routing_category": case.get("expected_routing_category"),
        "preferred_output": preferred,
        "scores": {
            "baseline": baseline_score,
            "threshold_variant": variant_score,
        },
        "status_movement": {
            "baseline": baseline.get("final_status"),
            "threshold_variant": variant.get("final_status"),
        },
        "reason_codes": {
            "baseline": _record_reason_codes(baseline_record),
            "threshold_variant": _record_reason_codes(variant_record),
        },
        "governance_snapshots": {
            "baseline": _record_governance_snapshot(baseline_record),
            "threshold_variant": _record_governance_snapshot(variant_record),
        },
        "critic_snapshots": {
            "baseline": _record_critic_snapshot(baseline_record),
            "threshold_variant": _record_critic_snapshot(variant_record),
        },
        "preference_signals": {
            "baseline": baseline_signals,
            "threshold_variant": variant_signals,
        },
    }


def _denial_pairwise_preference(
    case: dict[str, Any],
    baseline: dict[str, Any],
    variant: dict[str, Any],
    variant_label: str,
) -> dict[str, Any]:
    baseline_score, baseline_signals = _denial_pairwise_score(case, baseline)
    variant_score, variant_signals = _denial_pairwise_score(case, variant)
    if baseline_score > variant_score:
        preferred = "baseline"
    elif variant_score > baseline_score:
        preferred = variant_label
    else:
        preferred = "tie"
    baseline_record = baseline.get("record") if isinstance(baseline.get("record"), dict) else {}
    variant_record = variant.get("record") if isinstance(variant.get("record"), dict) else {}
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "operational_theme": case["operational_theme"],
        "question": "Which routing outcome is safer and more operationally useful before human review?",
        "expected_routing_action": case.get("expected_routing_action"),
        "expected_governance_posture": case.get("expected_governance_posture"),
        "preferred_output": preferred,
        "scores": {
            "baseline": baseline_score,
            variant_label: variant_score,
        },
        "routing_action_movement": {
            "baseline": baseline.get("routing_action"),
            variant_label: variant.get("routing_action"),
        },
        "governance_posture_movement": {
            "baseline": baseline.get("governance_posture"),
            variant_label: variant.get("governance_posture"),
        },
        "recoverability": {
            "baseline": baseline_record.get("recoverability"),
            variant_label: variant_record.get("recoverability"),
        },
        "documentation_gaps": {
            "baseline": baseline_record.get("documentation_gaps", {}),
            variant_label: variant_record.get("documentation_gaps", {}),
        },
        "evidence_profile": {
            "baseline": baseline_record.get("evidence_profile", {}),
            variant_label: variant_record.get("evidence_profile", {}),
        },
        "preference_signals": {
            "baseline": baseline_signals,
            variant_label: variant_signals,
        },
    }


def _routing_distribution_movement(
    baseline_report: dict[str, Any],
    variant_report: dict[str, Any],
    *,
    variant_label: str = "threshold_variant",
) -> dict[str, Any]:
    movement: dict[str, Any] = {}
    for key in (
        "status_bucket_counts",
        "governance_reason_code_counts",
        "adjacent_case_spillover_indicators",
        "routing_action_counts",
        "governance_posture_counts",
        "low_evidence_routing_counts",
        "ambiguity_routing_counts",
        "specialist_review_candidate_routing_counts",
        "recoverability_counts",
        "evidence_strength_counts",
        "conflicting_evidence_routing_counts",
        "partial_support_routing_counts",
        "timeline_inconsistency_routing_counts",
        "recoverability_boundary_routing_counts",
        "threshold_boundary_routing_counts",
        "ambiguity_spillover_routing_counts",
        "specialist_review_movement_counts",
        "low_evidence_movement_counts",
        "variant_adjustment_counts",
        "variant_route_movement_counts",
        "variant_governance_movement_counts",
    ):
        baseline_values = baseline_report.get(key, {})
        variant_values = variant_report.get(key, {})
        if isinstance(baseline_values, dict) and isinstance(variant_values, dict):
            all_keys = sorted(set(baseline_values) | set(variant_values))
            movement[key] = {
                item: {
                    "baseline": baseline_values.get(item, 0),
                    variant_label: variant_values.get(item, 0),
                    "delta": variant_values.get(item, 0) - baseline_values.get(item, 0),
                }
                for item in all_keys
            }
    for key in ("low_evidence_routing_volume", "specialist_review_oriented_volume", "unsupported_specificity_volume"):
        movement[key] = {
            "baseline": baseline_report.get(key),
            variant_label: variant_report.get(key),
            "delta": (variant_report.get(key) or 0) - (baseline_report.get(key) or 0),
        }
    movement["latency"] = {
        "baseline": baseline_report.get("latency", {}),
        variant_label: variant_report.get("latency", {}),
    }
    movement["token_and_cost"] = {
        "baseline": baseline_report.get("token_and_cost", {}),
        variant_label: variant_report.get("token_and_cost", {}),
    }
    status_counts_identical = baseline_report.get("status_bucket_counts", {}) == variant_report.get("status_bucket_counts", {})
    reason_codes_identical = baseline_report.get("governance_reason_code_counts", {}) == variant_report.get(
        "governance_reason_code_counts", {}
    )
    movement["movement_blocked_by_reason_codes"] = {
        "inferable": bool(status_counts_identical and reason_codes_identical),
        "status_bucket_counts_identical": bool(status_counts_identical),
        "reason_code_counts_identical": bool(reason_codes_identical),
        "dominant_baseline_reason_codes": baseline_report.get("dominant_reason_codes", {}),
        "dominant_variant_reason_codes": variant_report.get("dominant_reason_codes", {}),
        "baseline_dominance_diagnostics": baseline_report.get("dominance_diagnostics", {}),
        "variant_dominance_diagnostics": variant_report.get("dominance_diagnostics", {}),
    }
    return movement


def run_pairwise_comparison(cases: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    return run_pairwise_comparison_for_workflow(cases, dataset_name, workflow="cdi", variant_label="threshold_variant")


def run_pairwise_comparison_for_workflow(
    cases: list[dict[str, Any]],
    dataset_name: str,
    *,
    workflow: str,
    variant_label: str,
    execution_mode: str = "",
) -> dict[str, Any]:
    baseline_results = [
        _run_case_result(case, "baseline", dataset_name, workflow=workflow, execution_mode=execution_mode)
        for case in cases
    ]
    variant_results = [
        _run_case_result(case, variant_label, dataset_name, workflow=workflow, execution_mode=execution_mode)
        for case in cases
    ]
    baseline = _summary(
        cases=cases,
        experiment_label="baseline",
        dataset_name=dataset_name,
        results=baseline_results,
        workflow=workflow,
    )
    variant_summary = _summary(
        cases=cases,
        experiment_label=variant_label,
        dataset_name=dataset_name,
        results=variant_results,
        workflow=workflow,
    )
    baseline_by_id = {str(result["case_id"]): result for result in baseline_results}
    variant_by_id = {str(result["case_id"]): result for result in variant_results}
    if workflow == "denial":
        preferences = [
            _denial_pairwise_preference(
                case,
                baseline_by_id[str(case["case_id"])],
                variant_by_id[str(case["case_id"])],
                variant_label,
            )
            for case in cases
        ]
    else:
        preferences = [
            _pairwise_preference(case, baseline_by_id[str(case["case_id"])], variant_by_id[str(case["case_id"])])
            for case in cases
        ]
    preference_counts: dict[str, int] = {}
    status_changed_count = 0
    for item in preferences:
        _increment(preference_counts, str(item["preferred_output"]))
        movement = item.get("status_movement") or item.get("routing_action_movement") or {}
        if movement.get("baseline") != movement.get(variant_label):
            status_changed_count += 1
    unchanged_case_count = len(cases) - status_changed_count
    routing_distribution_movement = _routing_distribution_movement(
        baseline["routing_distribution"],
        variant_summary["routing_distribution"],
        variant_label=variant_label,
    )
    routing_distribution_movement["unchanged_case_count"] = unchanged_case_count

    return {
        "dataset_name": dataset_name,
        "workflow": workflow,
        "experiment_label": f"pairwise_baseline_vs_{variant_label}",
        "generated_at": _utc_stamp(),
        "case_count": len(cases),
        "layer": "layer_2_pairwise_denial_routing_usefulness" if workflow == "denial" else "layer_2_pairwise_governance_usefulness",
        "judge": {
            "type": "deterministic_local_heuristic",
            "hosted_langsmith_feedback_attached": False,
            "limitation": "Pairwise preferences are emitted locally because adding a hosted LLM-as-judge path would overbuild the existing regenerated workflow.",
            "criteria": (
                [
                    "ambiguity escalation quality",
                    "low-evidence handling",
                    "recoverability alignment",
                    "unsafe write-off avoidance",
                    "unsupported certainty avoidance",
                    "operational inspectability",
                ]
                if workflow == "denial"
                else [
                    "preserves reviewability",
                    "avoids unsupported specificity",
                    "handles ambiguity conservatively",
                    "identifies documentation insufficiency appropriately",
                    "avoids unsafe PASS behavior",
                    "provides operationally useful rationale",
                    "routes borderline cases appropriately",
                ]
            ),
        },
        "threshold_variant_overrides": THRESHOLD_VARIANT_OVERRIDES,
        "preference_counts": {key: preference_counts.get(key, 0) for key in sorted(preference_counts)},
        "status_changed_count": status_changed_count,
        "unchanged_case_count": unchanged_case_count,
        "routing_distribution_movement": routing_distribution_movement,
        "pairwise_preferences": preferences,
        "baseline_summary": baseline,
        f"{variant_label}_summary": variant_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default="cdi", choices=sorted(WORKFLOWS))
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--experiment-label", default="baseline", choices=sorted(EXPERIMENT_LABELS))
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit for smoke runs.")
    parser.add_argument("--skip-runtime", action="store_true", help="Only validate and create/reuse dataset.")
    parser.add_argument("--local-only", action="store_true", help="Run local summary mode without LangSmith evaluate().")
    parser.add_argument(
        "--execution-mode",
        default="",
        choices=["", "deterministic", "hybrid"],
        help="Optional denial workflow execution mode override for smoke and experiment runs.",
    )
    parser.add_argument(
        "--pairwise-variant",
        default="threshold_variant",
        choices=["threshold_variant", "routing_sensitivity_variant"],
        help="Variant label for local pairwise comparison.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Run local baseline vs variant pairwise governance or denial-routing usefulness comparison.",
    )
    args = parser.parse_args()

    load_dotenv()
    cases = load_workflow_cases(args.workflow)
    if args.limit > 0:
        cases = cases[: args.limit]
    default_dataset_name = DEFAULT_DENIAL_DATASET_NAME if args.workflow == "denial" else DEFAULT_DATASET_NAME
    requested_dataset_name = args.dataset_name or default_dataset_name

    hosted_requested = not args.local_only and not args.pairwise
    preflight, client = langsmith_preflight(hosted_requested)
    hosted_tracing_enabled_by_runner = False
    if hosted_requested and client is not None and preflight.get("readiness_check_succeeded"):
        hosted_tracing_enabled_by_runner = _enable_hosted_tracing_if_needed(hosted_requested)
        preflight["hosted_tracing_enabled_by_runner"] = hosted_tracing_enabled_by_runner
        preflight["langchain_tracing_v2_set"] = "LANGCHAIN_TRACING_V2" in os.environ
        preflight["langchain_tracing_v2_enabled"] = _tracing_enabled_value()
    _print_langsmith_preflight(preflight)
    if args.local_only:
        print("langsmith_hosted_mode_status=local_only_requested")
    elif args.pairwise:
        print("langsmith_hosted_mode_status=pairwise_local_only")
    elif preflight.get("readiness_check_succeeded"):
        print("langsmith_hosted_mode_status=hosted_requested_preflight_ready")
    elif client is None:
        print("langsmith_hosted_mode_status=hosted_unavailable_client_missing")
    else:
        print("langsmith_hosted_mode_status=hosted_requested_preflight_degraded")

    dataset_name = requested_dataset_name
    if hosted_requested and client is not None and preflight.get("readiness_check_succeeded"):
        dataset_name = create_or_reuse_dataset(client, cases, requested_dataset_name, workflow=args.workflow)
    elif hosted_requested:
        print("langsmith_dataset_status=skipped hosted_preflight_unavailable")

    if not langsmith_credentials_available():
        print("langsmith_run_status=skipped credentials_absent")
        if args.skip_runtime:
            return 0

    if args.skip_runtime:
        print("runtime_status=skipped")
        return 0

    if args.pairwise:
        summary = run_pairwise_comparison_for_workflow(
            cases,
            dataset_name,
            workflow=args.workflow,
            variant_label=args.pairwise_variant,
            execution_mode=args.execution_mode if args.workflow == "denial" else "",
        )
        summary["langsmith_preflight"] = preflight
        summary["hosted_mode_status"] = "pairwise_local_only"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if hosted_requested and client is not None and preflight.get("readiness_check_succeeded"):
        try:
            summary = run_hosted_experiment(
                client,
                cases,
                args.experiment_label,
                dataset_name,
                workflow=args.workflow,
                execution_mode=args.execution_mode if args.workflow == "denial" else "",
            )
            summary["hosted_mode_status"] = "hosted_succeeded"
        except Exception as exc:
            print(
                "langsmith_formal_experiment_status=degraded "
                f"error_class={exc.__class__.__name__} reason={_sanitize_error_message(exc)}"
            )
            summary = run_experiment(
                cases,
                args.experiment_label,
                dataset_name,
                workflow=args.workflow,
                execution_mode=args.execution_mode if args.workflow == "denial" else "",
            )
            summary["hosted_mode_status"] = "hosted_requested_local_summary_produced"
    else:
        summary = run_experiment(
            cases,
            args.experiment_label,
            dataset_name,
            workflow=args.workflow,
            execution_mode=args.execution_mode if args.workflow == "denial" else "",
        )
        summary["hosted_mode_status"] = "local_only_requested" if args.local_only else "hosted_unavailable_local_summary_produced"
    summary["langsmith_preflight"] = preflight
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
