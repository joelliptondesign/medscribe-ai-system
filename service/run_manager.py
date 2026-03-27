"""Execution wrapper for the existing MED-SCRIBE pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

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


def execute(input_text: str) -> dict[str, Any]:
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("input_text must be a non-empty string")

    load_dotenv()
    load_schema()

    run_id = str(uuid4())
    timestamp = _utc_timestamp()
    state = initial_state(input_text.strip())
    pipeline_start = time.perf_counter()

    stage_start = time.perf_counter()
    _merge_state(state, call_tool("parse_input", state))
    parse_ms = _elapsed_ms(stage_start, time.perf_counter())

    _merge_state(state, triage_engine.run(state))

    stage_start = time.perf_counter()
    _merge_state(state, call_tool("generate_diagnosis", state))
    diagnosis_ms = _elapsed_ms(stage_start, time.perf_counter())

    stage_start = time.perf_counter()
    _merge_state(state, call_tool("map_icd", state))
    mapping_ms = _elapsed_ms(stage_start, time.perf_counter())

    stage_start = time.perf_counter()
    _merge_state(state, call_tool("score_case", state))
    scoring_ms = _elapsed_ms(stage_start, time.perf_counter())

    _merge_state(state, governance_policy.run(state))
    total_ms = _elapsed_ms(pipeline_start, time.perf_counter())

    summary = {
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

    result = {
        "run_id": run_id,
        "timestamp": timestamp,
        "input": input_text.strip(),
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
        "summary": summary,
        "timing": {
            "parse_ms": parse_ms,
            "diagnosis_ms": diagnosis_ms,
            "mapping_ms": mapping_ms,
            "scoring_ms": scoring_ms,
            "total_ms": total_ms,
        },
        "trace": list(TRACE),
        "metadata": {
            "pipeline_version": "v1",
            "model_version": "unknown",
        },
    }

    storage.append_run(result)
    return result
