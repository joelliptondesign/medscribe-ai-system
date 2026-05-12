"""Local additive denial-management workflow runner."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Callable

from graph.config import get_execution_mode
from graph.operational_alerts import build_layer1_payload
from graph.tracing import trace_span
from graph.nodes.denials import (
    denial_classifier,
    denial_intake_parser,
    documentation_gap_analyzer,
    governance_policy,
    recoverability_analyzer,
    routing_engine,
)


DENIAL_TRACE: tuple[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]], ...] = (
    ("denial_intake_parser", denial_intake_parser.run),
    ("denial_classifier", denial_classifier.run),
    ("recoverability_analyzer", recoverability_analyzer.run),
    ("documentation_gap_analyzer", documentation_gap_analyzer.run),
    ("routing_engine", routing_engine.run),
    ("governance_policy", governance_policy.run),
)


def _diagnostic(node_name: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changed = sorted(key for key in after if before.get(key) != after.get(key))
    hybrid_diagnostics = after.get("_hybrid_node_diagnostics", {})
    hybrid = hybrid_diagnostics.get(node_name, {}) if isinstance(hybrid_diagnostics, dict) else {}
    diagnostic = {
        "node_name": node_name,
        "status": "completed",
        "local_deterministic": True,
        "fallback_triggered": False,
        "changed_fields": changed,
    }
    if hybrid:
        diagnostic.update(hybrid)
        diagnostic["changed_fields"] = changed
        diagnostic["local_deterministic"] = not bool(hybrid.get("hybrid_interpretation_used"))
    return diagnostic


def _provider_metadata(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        metadata = diagnostic.get("provider_metadata")
        if isinstance(metadata, dict) and metadata:
            collected.append(metadata)
    return collected


def _combined_token_usage(provider_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for metadata in provider_metadata:
        token_usage = metadata.get("token_usage")
        if not isinstance(token_usage, dict):
            continue
        for key, value in token_usage.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                combined[key] = combined.get(key, 0) + value
            elif key not in combined:
                combined[key] = value
    return combined


def run_denial_graph(case_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(case_payload, dict):
        raise TypeError("case_payload must be a dict")

    started = time.perf_counter()
    execution_mode = get_execution_mode()
    state: dict[str, Any] = deepcopy(case_payload)
    diagnostics: list[dict[str, Any]] = []
    with trace_span(
        "medscribe.denial_graph",
        inputs={"case_payload": case_payload},
        metadata={"workflow": "denial", "node_count": len(DENIAL_TRACE), "mode": execution_mode, "trace_type": "operational"},
        tags=["medscribe", "workflow:denial", "denial-graph", f"mode:{execution_mode}", "trace_type:operational"],
    ) as graph_span:
        for node_name, node in DENIAL_TRACE:
            before = deepcopy(state)
            with trace_span(
                f"medscribe.denial.{node_name}",
                inputs={"state": before},
                metadata={"workflow": "denial", "node_name": node_name, "mode": execution_mode, "trace_type": "operational_node"},
                tags=["medscribe", "workflow:denial", f"stage:{node_name}", f"mode:{execution_mode}"],
            ) as node_span:
                state = node(state)
                diagnostic = _diagnostic(node_name, before, state)
                diagnostics.append(diagnostic)
                node_span.set_outputs({"changed_fields": diagnostic.get("changed_fields", []), "diagnostic": diagnostic})

        state["node_diagnostics"] = diagnostics
        state["fallback_used"] = any(bool(diagnostic.get("fallback_triggered")) for diagnostic in diagnostics)
        state["degraded_mode"] = bool(state["fallback_used"])
        state["status"] = "degraded" if state["degraded_mode"] else "completed"
        state.pop("_hybrid_node_diagnostics", None)

        for field, default in (
            ("routing_action", "ESCALATE"),
            ("governance_posture", "AMBIGUOUS"),
            ("denial_category", "unclear denial"),
            ("recoverability", "unclear recoverability"),
            ("documentation_gaps", {}),
        ):
            state.setdefault(field, default)
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        state["latency_ms"] = latency_ms
        state["output"] = (
            f"route={state.get('routing_action')} "
            f"posture={state.get('governance_posture')} "
            f"status={state.get('status')}"
        )
        llm_used = any(bool(diagnostic.get("hybrid_interpretation_used")) for diagnostic in diagnostics)
        provider_call_attempted = any(bool(diagnostic.get("live_call_attempted")) for diagnostic in diagnostics)
        provider_metadata = _provider_metadata(diagnostics)
        token_usage = _combined_token_usage(provider_metadata)
        token_cost_available = bool(token_usage)
        metadata = {
            "workflow": "denial",
            "execution_mode": execution_mode,
            "contains_phi": bool(state.get("metadata", {}).get("phi")) if isinstance(state.get("metadata"), dict) else None,
            "routing_action": state.get("routing_action"),
            "governance_posture": state.get("governance_posture"),
            "denial_category": state.get("denial_category"),
            "recoverability": state.get("recoverability"),
            "degraded_mode": state.get("degraded_mode"),
            "fallback_used": state.get("fallback_used"),
            "latency_ms": latency_ms,
            "node_latency_ms": {"total_ms": latency_ms},
            "llm_used": llm_used,
            "provider_call_attempted": provider_call_attempted,
            "token_cost_available": token_cost_available,
            "token_cost_unavailable_reason": None
            if token_cost_available
            else ("provider_metadata_unavailable" if (llm_used or provider_call_attempted) else "no_provider_call"),
        }
        if provider_metadata:
            metadata["provider_metadata"] = provider_metadata
        if token_usage:
            metadata["token_usage"] = token_usage
        if metadata["contains_phi"] is None:
            metadata.pop("contains_phi")
        layer1 = build_layer1_payload(
            workflow="denial",
            status=str(state.get("status")),
            output=state.get("output"),
            metadata=metadata,
            record={**state, "trace": [name for name, _ in DENIAL_TRACE]},
            token_metadata=token_usage,
            expected_trace_count=len(DENIAL_TRACE),
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
        state["metadata"] = {**state.get("metadata", {}), **metadata} if isinstance(state.get("metadata"), dict) else metadata
        state["status"] = layer1["status"]
        state["error"] = layer1["error"]
        state["alerts"] = layer1["alerts"]
        state["alert_count"] = layer1["alert_count"]
        state["max_alert_severity"] = layer1["max_alert_severity"]
        state["operational_metrics"] = layer1["operational_metrics"]
        state["operational_thresholds"] = layer1["operational_thresholds"]
        graph_span.set_outputs(
            {
                "output": state.get("output"),
                "routing_action": state.get("routing_action"),
                "governance_posture": state.get("governance_posture"),
                "status": state.get("status"),
                "fallback_used": state.get("fallback_used"),
                "degraded_mode": state.get("degraded_mode"),
                "alert_count": state.get("alert_count"),
                "max_alert_severity": state.get("max_alert_severity"),
            }
        )
    return state
