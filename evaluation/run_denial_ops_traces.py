"""Run live denial workflow traces without creating datasets or evaluator runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DENIAL_DATASET_PATH = REPO_ROOT / "evaluation" / "denial_benchmark_cases.json"
DEFAULT_PROJECT = "medscribe-denial-ops"
VALID_EXECUTION_MODES = {"auto", "deterministic", "hybrid"}
VALID_VARIANTS = {"baseline", "threshold_variant", "routing_sensitivity_variant"}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_cases(path: Path = DENIAL_DATASET_PATH) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("denial benchmark dataset must be a JSON list")
    return [case for case in cases if isinstance(case, dict)]


def _selected_cases(cases: list[dict[str, Any]], case_ids: list[str], limit: int) -> list[dict[str, Any]]:
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in cases if str(case.get("case_id")) in wanted]
        missing = sorted(wanted - {str(case.get("case_id")) for case in selected})
        if missing:
            raise ValueError(f"unknown denial case_id values: {missing}")
    else:
        selected = cases
    return selected[:limit] if limit > 0 else selected


def _provider_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _langsmith_available() -> bool:
    return bool(os.getenv("LANGCHAIN_API_KEY", "").strip() or os.getenv("LANGSMITH_API_KEY", "").strip())


def _execution_mode(requested: str) -> str:
    if requested == "auto":
        return "hybrid" if _provider_available() else "deterministic"
    return requested


def _safe_summary(text: Any, limit: int = 240) -> str:
    summary = " ".join(str(text or "").split())
    if len(summary) > limit:
        return summary[:limit] + "...[truncated]"
    return summary


def _text_blob(case: dict[str, Any]) -> str:
    fields = (
        "denial_type",
        "payer_reason",
        "clinical_summary",
        "documentation_summary",
        "prior_authorization_context",
        "utilization_review_notes",
        "specialist_notes",
        "conflicting_documentation",
        "evidence_strength",
        "missing_required_evidence",
    )
    return " ".join(str(case.get(field, "")).lower() for field in fields)


def _has_signal(case: dict[str, Any], terms: tuple[str, ...]) -> bool:
    text = _text_blob(case)
    return any(term in text for term in terms)


def _operational_summary(case: dict[str, Any]) -> str:
    denial_type = str(case.get("denial_type") or "unclear denial").strip().lower()
    signals: list[str] = []
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    if _has_signal(case, ("conflict", "contradict", "inconsistent", "disagreement")):
        signals.append("conflicting evidence")
    if _has_signal(case, ("partial support", "partially supported", "borderline", "weak but")):
        signals.append("partial support")
    if _has_signal(case, ("specialist", "review candidate")) or metadata.get("specialist_review_candidate"):
        signals.append("specialist review")
    if _has_signal(case, ("timeline", "sequence", "date mismatch", "not aligned")):
        signals.append("timeline conflict")
    if _has_signal(case, ("prior authorization", "prior auth", "authorization mismatch")):
        signals.append("prior auth mismatch")
    if _has_signal(case, ("low evidence", "weak evidence", "missing required", "absent", "not documented")):
        signals.append("low evidence")
    if _has_signal(case, ("modifier", "modifier omitted", "modifier omission")):
        signals.append("modifier issue")
    if metadata.get("boundary_case") or _has_signal(case, ("threshold", "boundary", "borderline")):
        signals.append("threshold boundary")
    if not signals:
        signals.append("standard review")
    return f"{denial_type} | {signals[0]}"


def _case_inputs(case: dict[str, Any]) -> dict[str, Any]:
    return {"summary": _operational_summary(case)}


def _alert_summary(alerts: Any) -> list[str]:
    if not isinstance(alerts, list):
        return []
    summary: list[str] = []
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        alert_class = str(alert.get("class") or "").strip()
        severity = str(alert.get("severity") or "").strip()
        if alert_class:
            summary.append(f"{severity}:{alert_class}" if severity else alert_class)
    return summary


def _first_token_latency(provider_metadata: Any) -> int | None:
    if not isinstance(provider_metadata, list):
        return None
    for item in provider_metadata:
        if not isinstance(item, dict):
            continue
        for key in ("first_token_latency_ms", "time_to_first_token_ms", "ttft_ms"):
            value = item.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return int(value)
        response_metadata = item.get("response_metadata")
        if isinstance(response_metadata, dict):
            for key in ("first_token_latency_ms", "time_to_first_token_ms", "ttft_ms"):
                value = response_metadata.get(key)
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    return int(value)
    return None


def _cost_metadata(provider_metadata: Any) -> dict[str, Any]:
    if not isinstance(provider_metadata, list):
        return {}
    cost: dict[str, Any] = {}
    for item in provider_metadata:
        if not isinstance(item, dict):
            continue
        for key in ("estimated_cost_usd", "cost_usd", "total_cost"):
            if item.get(key) is not None:
                cost[key] = item[key]
        response_metadata = item.get("response_metadata")
        if isinstance(response_metadata, dict):
            for key in ("estimated_cost_usd", "cost_usd", "total_cost"):
                if response_metadata.get(key) is not None:
                    cost[key] = response_metadata[key]
    return cost


def _fallback_reasons(result: dict[str, Any]) -> list[str]:
    diagnostics = result.get("node_diagnostics")
    if not isinstance(diagnostics, list):
        return []
    reasons: list[str] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        reason = str(diagnostic.get("fallback_reason") or "").strip()
        if reason:
            reasons.append(reason)
    return reasons


def _diagnostic_metadata(
    case: dict[str, Any],
    result: dict[str, Any],
    *,
    latency_ms: int,
    token_usage: Any,
    provider_metadata: Any,
    first_token_latency_ms: int | None,
) -> dict[str, Any]:
    record_metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    diagnostic: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "title": case.get("title"),
        "denial_type": case.get("denial_type"),
        "evidence_profile": result.get("evidence_profile"),
        "documentation_gaps": result.get("documentation_gaps"),
        "node_diagnostics": result.get("node_diagnostics"),
        "provider_metadata": provider_metadata if isinstance(provider_metadata, list) and provider_metadata else None,
        "raw_token_usage": token_usage if isinstance(token_usage, dict) and token_usage else None,
        "latency_ms": latency_ms,
        "first_token_latency_ms": first_token_latency_ms,
        "cost_metadata": _cost_metadata(provider_metadata),
        "variant_adjustment": result.get("variant_adjustment"),
        "alerts": result.get("alerts") if isinstance(result.get("alerts"), list) else [],
        "node_timing": record_metadata.get("node_latency_ms"),
        "fallback_reasons": _fallback_reasons(result),
        "operational_metrics": result.get("operational_metrics"),
        "operational_thresholds": result.get("operational_thresholds"),
    }
    return {key: value for key, value in diagnostic.items() if value not in (None, {}, [])}


def _metadata(case: dict[str, Any], result: dict[str, Any], mode: str, latency_ms: int, variant: str) -> dict[str, Any]:
    record_metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    token_usage = record_metadata.get("token_usage") if isinstance(record_metadata, dict) else None
    documentation_gaps = result.get("documentation_gaps") if isinstance(result.get("documentation_gaps"), dict) else {}
    evidence_profile = result.get("evidence_profile") if isinstance(result.get("evidence_profile"), dict) else {}
    provider_metadata = record_metadata.get("provider_metadata")
    first_token_latency_ms = _first_token_latency(provider_metadata)
    provider_call_attempted = bool(record_metadata.get("provider_call_attempted"))
    token_cost_available = bool(record_metadata.get("token_cost_available"))
    first_token_unavailable_reason = None
    if first_token_latency_ms is None:
        if provider_call_attempted:
            first_token_unavailable_reason = "provider_streaming_not_enabled"
        else:
            first_token_unavailable_reason = "no_provider_call"
    metadata: dict[str, Any] = {
        "variant": variant,
        "llm_used": bool(record_metadata.get("llm_used")),
        "fallback_used": bool(result.get("fallback_used")),
        "degraded_mode": bool(result.get("degraded_mode")),
        "alert_count": result.get("alert_count", 0),
        "max_alert_severity": result.get("max_alert_severity"),
        "evidence_strength": evidence_profile.get("evidence_strength") or case.get("evidence_strength") or "unspecified",
        "ambiguity_level": documentation_gaps.get("ambiguity_level"),
        "conflicting_evidence": bool(
            documentation_gaps.get("conflicting_evidence") or evidence_profile.get("has_conflicting_documentation")
        ),
        "specialist_review_candidate": bool(
            documentation_gaps.get("specialist_review_signal")
            or evidence_profile.get("has_specialist_notes")
            or (case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}).get("specialist_review_candidate")
        ),
        "threshold_boundary": bool(
            (case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}).get("boundary_case")
            or _has_signal(case, ("threshold", "boundary", "borderline"))
        ),
        "recoverability": result.get("recoverability"),
        "governance_posture": result.get("governance_posture"),
        "token_cost_available": token_cost_available,
        "first_token_available": first_token_latency_ms is not None,
    }
    token_unavailable_reason = record_metadata.get("token_cost_unavailable_reason")
    if not token_cost_available and token_unavailable_reason:
        metadata["token_cost_unavailable_reason"] = token_unavailable_reason
    if first_token_unavailable_reason:
        metadata["first_token_unavailable_reason"] = first_token_unavailable_reason
    metadata["diagnostic_metadata"] = _diagnostic_metadata(
        case,
        result,
        latency_ms=latency_ms,
        token_usage=token_usage,
        provider_metadata=provider_metadata,
        first_token_latency_ms=first_token_latency_ms,
    )
    return {key: value for key, value in metadata.items() if value is not None}


def _tags(metadata: dict[str, Any], alerts: Any, mode: str) -> list[str]:
    tags = [
        "workflow:denial",
        f"mode:{mode}",
        f"variant:{metadata.get('variant')}" if metadata.get("variant") and metadata.get("variant") != "baseline" else None,
        "contains_phi:false",
        "high_ambiguity" if metadata.get("ambiguity_level") == "high" else None,
        "conflicting_evidence" if metadata.get("conflicting_evidence") else None,
        "specialist_review" if metadata.get("specialist_review_candidate") else None,
        "threshold_boundary" if metadata.get("threshold_boundary") else None,
        "fallback_used" if metadata.get("fallback_used") else None,
        "degraded_mode" if metadata.get("degraded_mode") else None,
    ]
    if isinstance(alerts, list):
        for alert in alerts:
            if isinstance(alert, dict) and alert.get("class"):
                tags.append(str(alert["class"]).lower())
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        if not tag or tag in seen:
            continue
        seen.add(tag)
        result.append(tag)
    return result


def _run_case(case: dict[str, Any], mode: str, variant: str) -> dict[str, Any]:
    from evaluation.langsmith_experiment_runner import apply_denial_variant
    from graph.denial_graph import run_denial_graph
    from graph.tracing import trace_span

    previous_mode = os.environ.get("MEDSCRIBE_EXECUTION_MODE")
    os.environ["MEDSCRIBE_EXECUTION_MODE"] = mode
    started = time.perf_counter()
    try:
        with trace_span(
            str(case.get("case_id") or "unknown_denial_case"),
            inputs=_case_inputs(case),
            metadata={
                "workflow": "denial",
                "trace_type": "operational",
                "execution_mode": mode,
                "variant": variant,
                "workflow_trace_name": f"medscribe.denial.workflow.{mode}",
                "contains_phi": bool(case.get("metadata", {}).get("phi")) if isinstance(case.get("metadata"), dict) else None,
            },
            tags=[
                tag
                for tag in (
                    "workflow:denial",
                    f"mode:{mode}",
                    "contains_phi:false",
                    f"variant:{variant}" if variant != "baseline" else None,
                )
                if tag
            ],
        ) as workflow_span:
            result = run_denial_graph(case)
            result = apply_denial_variant(result, variant, case)
            latency_ms = max(1, int((time.perf_counter() - started) * 1000))
            metadata = _metadata(case, result, mode, latency_ms, variant)
            tags = _tags(metadata, result.get("alerts"), mode)
            output = str(result.get("routing_action") or "").strip().upper() or "MISSING"
            workflow_span.set_outputs(
                {
                    "output": output,
                    "routing_action": result.get("routing_action"),
                    "status": result.get("status"),
                    "error": result.get("error"),
                    "metadata": metadata,
                    "tags": tags,
                }
            )
            return {
                "case_id": case.get("case_id"),
                "title": case.get("title"),
                "input": _operational_summary(case),
                "output": output,
                "routing_action": result.get("routing_action"),
                "governance_posture": result.get("governance_posture"),
                "status": result.get("status"),
                "error": result.get("error"),
                "metadata": metadata,
                "tags": tags,
                "alert_count": result.get("alert_count", 0),
                "max_alert_severity": result.get("max_alert_severity"),
                "node_diagnostic_count": len(result.get("node_diagnostics", [])),
            }
    finally:
        if previous_mode is None:
            os.environ.pop("MEDSCRIBE_EXECUTION_MODE", None)
        else:
            os.environ["MEDSCRIBE_EXECUTION_MODE"] = previous_mode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", default=[], help="Denial benchmark case_id to trace. Repeatable.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum cases to trace when --case-id is not supplied.")
    parser.add_argument("--execution-mode", default="auto", choices=sorted(VALID_EXECUTION_MODES))
    parser.add_argument("--variant", default="baseline", choices=sorted(VALID_VARIANTS))
    parser.add_argument("--project-name", default=DEFAULT_PROJECT)
    parser.add_argument("--no-tracing", action="store_true", help="Run locally without enabling LangSmith tracing.")
    args = parser.parse_args()

    load_dotenv()
    mode = _execution_mode(args.execution_mode)
    if not args.no_tracing and _langsmith_available():
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = args.project_name

    cases = _selected_cases(_load_cases(), args.case_id, args.limit)
    results = [_run_case(case, mode, args.variant) for case in cases]
    summary = {
        "workflow": "denial",
        "trace_type": "operational",
        "variant": args.variant,
        "project_name": args.project_name if not args.no_tracing and _langsmith_available() else None,
        "tracing_requested": not args.no_tracing,
        "tracing_enabled": os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() in {"1", "true", "yes"},
        "execution_mode": mode,
        "provider_available": _provider_available(),
        "case_count": len(results),
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
