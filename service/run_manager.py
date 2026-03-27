"""Execution wrapper for the existing MED-SCRIBE pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from threading import Thread
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from graph.config import get_execution_mode
from graph.nodes import governance_policy, triage_engine
from graph.state import initial_state, load_schema
from service import storage
from service.tools import call_tool


TRACE = [
    "intake_parser",
    "diagnosis_engine",
    "icd_mapper",
    "critic",
    "policy",
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
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "input": input_text,
        "parsed_input": state.get("intake_data", {}),
        "diagnosis": {
            "diagnoses": state.get("diagnoses", []),
            "triage": state.get("triage", {}),
        },
        "icd_mapping": {
            "mappings": state.get("icd_mappings", []),
        },
        "scores": state.get("critic_review", {}),
        "decision": state.get("governance_result", {}).get("final_status", "FAIL"),
        "summary": _build_summary(state),
        "timing": _timing_snapshot(parse_ms, diagnosis_ms, mapping_ms, scoring_ms, total_ms),
        "trace": list(TRACE),
        "node_diagnostics": node_diagnostics,
        "fallback_nodes": _fallback_nodes(node_diagnostics),
        "fallback_reasons": _fallback_reasons(node_diagnostics),
        "metadata": {
            "pipeline_version": "v1",
            "model_version": "unknown",
            "execution_mode": execution_mode,
            "hybrid_attempted": execution_mode == "hybrid",
        },
        "status": status,
        "retry_count": retry_count,
        "failed_stage": failed_stage,
        "fallback_used": fallback_used,
        "degraded_mode": degraded_mode,
        "error": error,
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
        _merge_state(state, call_tool("parse_input", state))
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
        _merge_state(state, triage_engine.run(state))
        _merge_state(state, call_tool("generate_diagnosis", state))
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
        _merge_state(state, call_tool("map_icd", state))
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
        _merge_state(state, call_tool("score_case", state))
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
        _merge_state(state, governance_policy.run(state))
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
