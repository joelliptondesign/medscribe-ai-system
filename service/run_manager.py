"""Execution wrapper for the existing MED-SCRIBE pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from threading import Thread
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from graph.config import get_execution_mode, get_langsmith_metadata
from graph.nodes import governance_policy, triage_engine
from graph.operational_alerts import build_layer1_payload
from graph.state import initial_state, load_schema
from graph.tracing import trace_span
from service import storage
from service.tools import call_tool


TRACE = [
    "intake_parser",
    "triage_engine",
    "diagnosis_engine",
    "icd_mapper",
    "critic",
    "governance_policy",
]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _merge_state(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    state.update(update)
    return state


def _elapsed_ms(start: float, end: float) -> int:
    return max(1, int((end - start) * 1000))


def _timing_snapshot(
    parse_ms: int | None,
    diagnosis_ms: int | None,
    mapping_ms: int | None,
    scoring_ms: int | None,
    total_ms: int | None,
) -> dict[str, int]:
    timing: dict[str, int] = {}
    if parse_ms is not None:
        timing["parse_ms"] = parse_ms
    if diagnosis_ms is not None:
        timing["diagnosis_ms"] = diagnosis_ms
    if mapping_ms is not None:
        timing["mapping_ms"] = mapping_ms
    if scoring_ms is not None:
        timing["scoring_ms"] = scoring_ms
    if total_ms is not None:
        timing["total_ms"] = total_ms
    return timing


def _build_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": state.get("governance_result", {}).get("final_status", "FAIL"),
        "escalation_required": state.get("governance_result", {}).get("escalation_required", False),
        "policy_version": state.get("governance_result", {}).get("policy_version", ""),
        "governance_version": state.get("governance_result", {}).get("governance_version", ""),
        "reason_codes": state.get("governance_result", {}).get("reason_codes", []),
        "summary": (
            state.get("critic_review", {}).get("summary", "")
            if state.get("governance_result", {}).get("final_status", "FAIL") == "PASS"
            else "Escalation required due to policy thresholds."
        ),
    }


def _text_word_count(value: Any) -> int:
    if not isinstance(value, str):
        return 0
    return len(value.split())


def _operational_observability_snapshot(
    state: dict[str, Any],
    timing: dict[str, int],
) -> dict[str, Any]:
    intake_data = state.get("intake_data", {})
    triage = state.get("triage", {})
    diagnoses = state.get("diagnoses", [])
    icd_mappings = state.get("icd_mappings", [])
    critic_review = state.get("critic_review", {})
    governance_result = state.get("governance_result", {})
    return {
        "observability_version": "regen_stabilization_v1",
        "comparison_profile": "regenerated_live_hybrid",
        "stage_timing_ms": timing,
        "reasoning_verbosity": {
            "triage_rationale_words": _text_word_count(triage.get("rationale")),
            "critic_summary_words": _text_word_count(critic_review.get("summary")),
            "critic_reason_code_count": len(critic_review.get("reason_codes", [])),
            "governance_reason_code_count": len(governance_result.get("reason_codes", [])),
        },
        "critic_metric_snapshot": {
            "diagnosis_consistency_score": critic_review.get("diagnosis_consistency_score"),
            "symptom_alignment_score": critic_review.get("symptom_alignment_score"),
            "icd_specificity_score": critic_review.get("icd_specificity_score"),
            "confidence": critic_review.get("confidence"),
            "recommended_status": critic_review.get("recommended_status"),
        },
        "governance_snapshot": {
            "final_status": governance_result.get("final_status"),
            "policy_version": governance_result.get("policy_version"),
            "governance_version": governance_result.get("governance_version"),
            "applied_rule_count": len(governance_result.get("applied_rules", [])),
            "reason_code_count": len(governance_result.get("reason_codes", [])),
        },
        "output_shape_snapshot": {
            "symptom_count": len(intake_data.get("symptoms", [])),
            "diagnosis_count": len(diagnoses),
            "icd_mapping_count": len(icd_mappings),
            "critic_reason_code_count": len(critic_review.get("reason_codes", [])),
        },
    }


def _fallback_used(state: dict[str, Any]) -> bool:
    node_diagnostics = state.get("node_diagnostics")
    if not isinstance(node_diagnostics, list):
        return False

    return any(
        isinstance(diagnostic, dict)
        and (
            diagnostic.get("fallback") is True
            or diagnostic.get("fallback_triggered") is True
        )
        for diagnostic in node_diagnostics
    )


def _diagnostic_snapshot(state: dict[str, Any]) -> list[dict[str, Any]]:
    node_diagnostics = state.get("node_diagnostics")
    if not isinstance(node_diagnostics, list):
        return []

    snapshot: list[dict[str, Any]] = []
    for diagnostic in node_diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        snapshot.append(
            {
                "node_name": diagnostic.get("node_name"),
                "live_call_attempted": bool(diagnostic.get("live_call_attempted")),
                "live_call_returned": bool(diagnostic.get("live_call_returned")),
                "parse_succeeded": bool(diagnostic.get("parse_succeeded")),
                "normalization_succeeded": bool(diagnostic.get("normalization_succeeded")),
                "fallback_triggered": bool(
                    diagnostic.get("fallback") is True
                    or diagnostic.get("fallback_triggered") is True
                ),
                "fallback_reason": str(diagnostic.get("fallback_reason", "")).strip(),
            }
        )
    return snapshot


def _fallback_nodes(node_diagnostics: list[dict[str, Any]]) -> list[str]:
    return [
        str(diagnostic.get("node_name"))
        for diagnostic in node_diagnostics
        if diagnostic.get("fallback_triggered") and diagnostic.get("node_name")
    ]


def _llm_used(node_diagnostics: list[dict[str, Any]]) -> bool:
    return any(bool(diagnostic.get("live_call_returned")) for diagnostic in node_diagnostics)


def _fallback_reasons(node_diagnostics: list[dict[str, Any]]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for diagnostic in node_diagnostics:
        node_name = diagnostic.get("node_name")
        reason = str(diagnostic.get("fallback_reason", "")).strip()
        if diagnostic.get("fallback_triggered") and node_name and reason:
            reasons[str(node_name)] = reason
    return reasons


def _build_record(
    *,
    run_id: str,
    timestamp: str,
    input_text: str,
    state: dict[str, Any],
    execution_mode: str,
    status: str,
    retry_count: int,
    failed_stage: str | None,
    fallback_used: bool,
    degraded_mode: bool,
    error: str | None,
    parse_ms: int | None,
    diagnosis_ms: int | None,
    mapping_ms: int | None,
    scoring_ms: int | None,
    total_ms: int | None,
    ) -> dict[str, Any]:
    node_diagnostics = _diagnostic_snapshot(state)
    timing = _timing_snapshot(parse_ms, diagnosis_ms, mapping_ms, scoring_ms, total_ms)
    final_decision = state.get("governance_result", {}).get("final_status", "FAIL")
    llm_used = _llm_used(node_diagnostics)
    metadata = {
        "pipeline_version": "v1",
        "model_version": "unknown",
        "execution_mode": execution_mode,
        "hybrid_attempted": execution_mode == "hybrid",
        "langsmith": get_langsmith_metadata(),
        "workflow": "cdi",
        "contains_phi": False,
        "llm_used": llm_used,
        "token_cost_available": False,
        "token_cost_unavailable_reason": "provider_metadata_unavailable" if llm_used else "no_provider_call",
        "latency_ms": total_ms,
        "node_latency_ms": timing,
        "degraded_mode": degraded_mode,
        "fallback_used": fallback_used,
    }
    layer1 = build_layer1_payload(
        workflow="cdi",
        status=status,
        output=final_decision,
        metadata=metadata,
        record={
            "status": status,
            "failed_stage": failed_stage,
            "fallback_used": fallback_used,
            "degraded_mode": degraded_mode,
            "trace": list(TRACE),
            "latency_ms": total_ms,
        },
        existing_error=error,
        expected_trace_count=len(TRACE),
    )
    metadata.update(
        {
            "status": layer1["status"],
            "error": layer1["error"],
            "alerts": layer1["alerts"],
            "alert_count": layer1["alert_count"],
            "max_alert_severity": layer1["max_alert_severity"],
            "operational_metrics": layer1["operational_metrics"],
        }
    )
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "input": input_text,
        "output": final_decision,
        "parsed_input": state.get("intake_data", {}),
        "diagnosis": {
            "diagnoses": state.get("diagnoses", []),
            "triage": state.get("triage", {}),
        },
        "icd_mapping": {
            "mappings": state.get("icd_mappings", []),
        },
        "scores": state.get("critic_review", {}),
        "decision": final_decision,
        "summary": _build_summary(state),
        "timing": timing,
        "latency_ms": total_ms,
        "operational_observability": _operational_observability_snapshot(state, timing),
        "trace": list(TRACE),
        "node_diagnostics": node_diagnostics,
        "fallback_nodes": _fallback_nodes(node_diagnostics),
        "fallback_reasons": _fallback_reasons(node_diagnostics),
        "metadata": metadata,
        "status": layer1["status"],
        "retry_count": retry_count,
        "failed_stage": failed_stage,
        "fallback_used": fallback_used,
        "degraded_mode": degraded_mode,
        "error": error,
        "alerts": layer1["alerts"],
        "alert_count": layer1["alert_count"],
        "max_alert_severity": layer1["max_alert_severity"],
        "operational_metrics": layer1["operational_metrics"],
        "operational_thresholds": layer1["operational_thresholds"],
    }


def _persist_failure(
    *,
    persist: bool,
    run_id: str,
    timestamp: str,
    input_text: str,
    state: dict[str, Any],
    execution_mode: str,
    retry_count: int,
    failed_stage: str,
    error: str,
    parse_ms: int | None,
    diagnosis_ms: int | None,
    mapping_ms: int | None,
    scoring_ms: int | None,
    total_ms: int,
) -> dict[str, Any]:
    partial_record = _build_record(
        run_id=run_id,
        timestamp=timestamp,
        input_text=input_text,
        state=state,
        execution_mode=execution_mode,
        status="failed",
        retry_count=retry_count,
        failed_stage=failed_stage,
        fallback_used=_fallback_used(state),
        degraded_mode=False,
        error=error,
        parse_ms=parse_ms,
        diagnosis_ms=diagnosis_ms,
        mapping_ms=mapping_ms,
        scoring_ms=scoring_ms,
        total_ms=total_ms,
    )
    if persist:
        storage.append_run(partial_record)
    return partial_record


def execute(
    input_text: str,
    *,
    run_id: str | None = None,
    timestamp: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    load_dotenv()
    with trace_span(
        "medscribe.governed_run",
        inputs={"input_text": input_text},
        metadata={
            "run_id": run_id or "generated_on_execute",
            "pipeline_version": "v1",
            "persist": persist,
            "trace_stages": TRACE,
            "operational_observability_version": "regen_stabilization_v1",
            "comparison_profile": "regenerated_live_hybrid",
        },
        tags=["medscribe", "governed-runtime", "workflow:cdi"],
    ) as span:
        result = _execute_pipeline(
            input_text,
            run_id=run_id,
            timestamp=timestamp,
            persist=persist,
        )
        span.set_outputs(result)
        return result


def _execute_pipeline(
    input_text: str,
    *,
    run_id: str | None = None,
    timestamp: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("input_text must be a non-empty string")

    load_dotenv()
    load_schema()

    normalized_input = input_text.strip()
    run_id = run_id or str(uuid4())
    timestamp = timestamp or _utc_timestamp()
    state = initial_state(normalized_input)
    execution_mode = get_execution_mode()
    pipeline_start = time.perf_counter()
    status = "running"
    retry_count = 0
    failed_stage = None
    fallback_used = False
    degraded_mode = False
    error = None
    parse_ms: int | None = None
    diagnosis_ms: int | None = None
    mapping_ms: int | None = None
    scoring_ms: int | None = None

    try:
        stage_start = time.perf_counter()
        with trace_span(
            "medscribe.intake_parser",
            inputs={"raw_input": normalized_input},
            metadata={"stage": "intake_parser", "run_id": run_id, "execution_mode": execution_mode},
            tags=["medscribe", "stage:intake_parser"],
        ) as span:
            update = call_tool("parse_input", state)
            span.set_outputs(update)
            _merge_state(state, update)
        parse_ms = _elapsed_ms(stage_start, time.perf_counter())
    except Exception as exc:
        error = str(exc)
        failed_stage = "parse_input"
        partial_record = _persist_failure(
            persist=persist,
            run_id=run_id,
            timestamp=timestamp,
            input_text=normalized_input,
            state=state,
            execution_mode=execution_mode,
            retry_count=retry_count,
            failed_stage=failed_stage,
            error=error,
            parse_ms=parse_ms,
            diagnosis_ms=diagnosis_ms,
            mapping_ms=mapping_ms,
            scoring_ms=scoring_ms,
            total_ms=_elapsed_ms(pipeline_start, time.perf_counter()),
        )
        setattr(exc, "partial_record", partial_record)
        raise

    try:
        stage_start = time.perf_counter()
        with trace_span(
            "medscribe.triage_engine",
            inputs={"intake_data": state.get("intake_data", {})},
            metadata={"stage": "triage_engine", "run_id": run_id, "execution_mode": execution_mode},
            tags=["medscribe", "stage:triage_engine"],
        ) as span:
            update = triage_engine.run(state)
            span.set_outputs(update)
            _merge_state(state, update)
        with trace_span(
            "medscribe.diagnosis_engine",
            inputs={"intake_data": state.get("intake_data", {}), "triage": state.get("triage", {})},
            metadata={"stage": "diagnosis_engine", "run_id": run_id, "execution_mode": execution_mode},
            tags=["medscribe", "stage:diagnosis_engine"],
        ) as span:
            update = call_tool("generate_diagnosis", state)
            span.set_outputs(update)
            _merge_state(state, update)
        diagnosis_ms = _elapsed_ms(stage_start, time.perf_counter())
    except Exception as exc:
        error = str(exc)
        failed_stage = "generate_diagnosis"
        partial_record = _persist_failure(
            persist=persist,
            run_id=run_id,
            timestamp=timestamp,
            input_text=normalized_input,
            state=state,
            execution_mode=execution_mode,
            retry_count=retry_count,
            failed_stage=failed_stage,
            error=error,
            parse_ms=parse_ms,
            diagnosis_ms=diagnosis_ms,
            mapping_ms=mapping_ms,
            scoring_ms=scoring_ms,
            total_ms=_elapsed_ms(pipeline_start, time.perf_counter()),
        )
        setattr(exc, "partial_record", partial_record)
        raise

    try:
        stage_start = time.perf_counter()
        with trace_span(
            "medscribe.icd_mapper",
            inputs={"diagnoses": state.get("diagnoses", [])},
            metadata={"stage": "icd_mapper", "run_id": run_id, "execution_mode": execution_mode},
            tags=["medscribe", "stage:icd_mapper"],
        ) as span:
            update = call_tool("map_icd", state)
            span.set_outputs(update)
            _merge_state(state, update)
        mapping_ms = _elapsed_ms(stage_start, time.perf_counter())
    except Exception as exc:
        error = str(exc)
        failed_stage = "map_icd"
        partial_record = _persist_failure(
            persist=persist,
            run_id=run_id,
            timestamp=timestamp,
            input_text=normalized_input,
            state=state,
            execution_mode=execution_mode,
            retry_count=retry_count,
            failed_stage=failed_stage,
            error=error,
            parse_ms=parse_ms,
            diagnosis_ms=diagnosis_ms,
            mapping_ms=mapping_ms,
            scoring_ms=scoring_ms,
            total_ms=_elapsed_ms(pipeline_start, time.perf_counter()),
        )
        setattr(exc, "partial_record", partial_record)
        raise

    try:
        stage_start = time.perf_counter()
        with trace_span(
            "medscribe.critic",
            inputs={"icd_mappings": state.get("icd_mappings", [])},
            metadata={"stage": "critic", "run_id": run_id, "execution_mode": execution_mode},
            tags=["medscribe", "stage:critic"],
        ) as span:
            update = call_tool("score_case", state)
            span.set_outputs(update)
            _merge_state(state, update)
        scoring_ms = _elapsed_ms(stage_start, time.perf_counter())
    except Exception as exc:
        error = str(exc)
        failed_stage = "score_case"
        partial_record = _persist_failure(
            persist=persist,
            run_id=run_id,
            timestamp=timestamp,
            input_text=normalized_input,
            state=state,
            execution_mode=execution_mode,
            retry_count=retry_count,
            failed_stage=failed_stage,
            error=error,
            parse_ms=parse_ms,
            diagnosis_ms=diagnosis_ms,
            mapping_ms=mapping_ms,
            scoring_ms=scoring_ms,
            total_ms=_elapsed_ms(pipeline_start, time.perf_counter()),
        )
        setattr(exc, "partial_record", partial_record)
        raise

    try:
        with trace_span(
            "medscribe.governance_policy",
            inputs={"critic_review": state.get("critic_review", {})},
            metadata={
                "stage": "governance_policy",
                "run_id": run_id,
                "execution_mode": execution_mode,
                "governance_inputs_used": [
                    "critic_review.diagnosis_consistency_score",
                    "critic_review.symptom_alignment_score",
                    "critic_review.icd_specificity_score",
                    "critic_review.confidence",
                    "critic_review.recommended_status",
                    "critic_review.reason_codes",
                ],
                "governance_inputs_ignored": [
                    "intake_data.symptoms",
                    "intake_data.severity_descriptors",
                    "intake_data.duration",
                    "triage.level",
                    "triage.rationale",
                    "diagnoses",
                    "icd_mappings",
                ],
                "upstream_context_summary": {
                    "triage_level": state.get("triage", {}).get("level"),
                    "symptom_count": len(state.get("intake_data", {}).get("symptoms", [])),
                    "diagnosis_count": len(state.get("diagnoses", [])),
                    "icd_mapping_count": len(state.get("icd_mappings", [])),
                },
            },
            tags=["medscribe", "stage:governance_policy"],
        ) as span:
            update = governance_policy.run(state)
            span.set_outputs(update)
            _merge_state(state, update)
    except Exception as exc:
        error = str(exc)
        failed_stage = "governance_policy"
        partial_record = _persist_failure(
            persist=persist,
            run_id=run_id,
            timestamp=timestamp,
            input_text=normalized_input,
            state=state,
            execution_mode=execution_mode,
            retry_count=retry_count,
            failed_stage=failed_stage,
            error=error,
            parse_ms=parse_ms,
            diagnosis_ms=diagnosis_ms,
            mapping_ms=mapping_ms,
            scoring_ms=scoring_ms,
            total_ms=_elapsed_ms(pipeline_start, time.perf_counter()),
        )
        setattr(exc, "partial_record", partial_record)
        raise

    total_ms = _elapsed_ms(pipeline_start, time.perf_counter())

    fallback_used = _fallback_used(state)

    if fallback_used:
        status = "degraded"
        degraded_mode = True
    else:
        status = "completed"

    result = _build_record(
        run_id=run_id,
        timestamp=timestamp,
        input_text=normalized_input,
        state=state,
        execution_mode=execution_mode,
        status=status,
        retry_count=retry_count,
        failed_stage=failed_stage,
        fallback_used=fallback_used,
        degraded_mode=degraded_mode,
        error=error,
        parse_ms=parse_ms,
        diagnosis_ms=diagnosis_ms,
        mapping_ms=mapping_ms,
        scoring_ms=scoring_ms,
        total_ms=total_ms,
    )

    if persist:
        storage.append_run(result)
    return result


def run_async(run_id: str, input_text: str) -> None:
    storage.update_run(run_id, {"status": "running"})
    existing_record = storage.get_run(run_id) or {}

    try:
        result = execute(
            input_text,
            run_id=run_id,
            timestamp=existing_record.get("timestamp"),
            persist=False,
        )
        if storage.update_run(run_id, result) is None:
            storage.append_run(result)
    except Exception as exc:
        partial_record = getattr(exc, "partial_record", None)
        if isinstance(partial_record, dict):
            updated_fields = partial_record
        else:
            updated_fields = {
                "status": "failed",
                "error": str(exc),
                "failed_stage": None,
                "fallback_used": False,
                "degraded_mode": False,
                "retry_count": 0,
            }
        if storage.update_run(run_id, updated_fields) is None:
            failure_record = {
                "run_id": run_id,
                "timestamp": existing_record.get("timestamp", _utc_timestamp()),
                "input": input_text.strip(),
            }
            failure_record.update(updated_fields)
            storage.append_run(failure_record)


def start_async_run(run_id: str, input_text: str) -> Thread:
    thread = Thread(target=run_async, args=(run_id, input_text), daemon=True)
    thread.start()
    return thread
