"""Layer 1 operational alert helpers for runtime and evaluation payloads."""

from __future__ import annotations

import json
from typing import Any


VALID_EXECUTION_STATUSES = {"completed", "degraded", "failed"}
SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}
DEFAULT_THRESHOLDS: dict[str, int | float] = {
    "latency_warning_ms": 5_000,
    "latency_critical_ms": 15_000,
    "output_chars_warning": 4_000,
    "output_chars_critical": 10_000,
    "token_warning": 8_000,
    "token_critical": 24_000,
    "cost_warning_usd": 1.0,
    "cost_critical_usd": 5.0,
    "first_token_latency_warning_ms": 2_000,
    "first_token_latency_critical_ms": 6_000,
}


def _alert(
    alert_class: str,
    severity: str,
    message: str,
    *,
    observed: Any = None,
    threshold: Any = None,
    unit: str | None = None,
) -> dict[str, Any]:
    alert = {
        "class": alert_class,
        "severity": severity,
        "message": message,
        "observed": observed,
        "threshold": threshold,
    }
    if unit:
        alert["unit"] = unit
    return alert


def _max_severity(alerts: list[dict[str, Any]]) -> str | None:
    if not alerts:
        return None
    return max((str(alert.get("severity", "info")) for alert in alerts), key=lambda item: SEVERITY_ORDER.get(item, 0))


def _primary_error(status: str, alerts: list[dict[str, Any]], existing_error: Any = None) -> str | None:
    if existing_error:
        return str(existing_error)
    critical = [alert for alert in alerts if str(alert.get("severity")) == "critical"]
    if status == "failed" and not critical:
        return "provider_failure"
    if critical:
        return str(critical[0].get("class") or "operational_alert")
    return None


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _token_count(token_metadata: dict[str, Any]) -> int | None:
    for key in ("total_tokens", "total_token_count", "tokens", "input_tokens", "prompt_tokens"):
        value = _to_number(token_metadata.get(key))
        if value is not None:
            if key in {"input_tokens", "prompt_tokens"}:
                output = _to_number(token_metadata.get("output_tokens") or token_metadata.get("completion_tokens")) or 0
                return int(value + output)
            return int(value)
    return None


def _estimated_cost(token_metadata: dict[str, Any]) -> float | None:
    for key in ("total_cost", "estimated_cost_usd", "cost_usd"):
        value = _to_number(token_metadata.get(key))
        if value is not None:
            return float(value)
    return None


def _first_token_latency_ms(token_metadata: dict[str, Any]) -> int | None:
    for key in ("first_token_latency_ms", "time_to_first_token_ms", "ttft_ms"):
        value = _to_number(token_metadata.get(key))
        if value is not None:
            return int(value)
    return None


def _verbosity_bucket(output_char_count: int) -> str:
    if output_char_count >= int(DEFAULT_THRESHOLDS["output_chars_critical"]):
        return "critical"
    if output_char_count >= int(DEFAULT_THRESHOLDS["output_chars_warning"]):
        return "high"
    if output_char_count >= 1_000:
        return "moderate"
    return "normal"


def _output_char_count(output: Any) -> int:
    if output is None:
        return 0
    if isinstance(output, str):
        return len(output)
    try:
        return len(json.dumps(output, sort_keys=True))
    except TypeError:
        return len(str(output))


def _observability_complete(metadata: dict[str, Any], status: str, output: Any) -> bool:
    required = ("workflow", "contains_phi", "latency_ms", "token_cost_available")
    return status in VALID_EXECUTION_STATUSES and output not in {None, ""} and all(key in metadata for key in required)


def _add_threshold_alert(
    alerts: list[dict[str, Any]],
    *,
    value: int | float | None,
    warning: int | float,
    critical: int | float,
    alert_class: str,
    unit: str,
    warning_message: str,
    critical_message: str,
) -> None:
    if value is None:
        return
    if value >= critical:
        alerts.append(
            _alert(alert_class, "critical", critical_message, observed=value, threshold=critical, unit=unit)
        )
    elif value >= warning:
        alerts.append(
            _alert(alert_class, "warning", warning_message, observed=value, threshold=warning, unit=unit)
        )


def build_layer1_payload(
    *,
    workflow: str,
    status: str,
    output: Any,
    metadata: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
    existing_error: Any = None,
    token_metadata: dict[str, Any] | None = None,
    expected_trace_count: int | None = None,
) -> dict[str, Any]:
    metadata = dict(metadata or {})
    record = record if isinstance(record, dict) else {}
    token_metadata = dict(token_metadata or {})
    normalized_status = str(status or record.get("status") or "completed").strip().lower()
    if normalized_status not in VALID_EXECUTION_STATUSES:
        normalized_status = "failed"

    latency_ms = _to_number(metadata.get("latency_ms") or record.get("latency_ms"))
    latency_value = int(latency_ms) if latency_ms is not None else None
    token_count = _token_count(token_metadata)
    estimated_cost = _estimated_cost(token_metadata)
    first_token_latency = _first_token_latency_ms(token_metadata)
    output_chars = _output_char_count(output)
    observability_complete = _observability_complete(metadata, normalized_status, output)
    alerts: list[dict[str, Any]] = []

    _add_threshold_alert(
        alerts,
        value=latency_value,
        warning=DEFAULT_THRESHOLDS["latency_warning_ms"],
        critical=DEFAULT_THRESHOLDS["latency_critical_ms"],
        alert_class="latency_spike",
        unit="ms",
        warning_message="Workflow latency exceeded the Layer 1 warning threshold.",
        critical_message="Workflow latency exceeded the Layer 1 critical threshold.",
    )
    _add_threshold_alert(
        alerts,
        value=output_chars,
        warning=DEFAULT_THRESHOLDS["output_chars_warning"],
        critical=DEFAULT_THRESHOLDS["output_chars_critical"],
        alert_class="verbosity_spike",
        unit="chars",
        warning_message="Output verbosity exceeded the Layer 1 warning threshold.",
        critical_message="Output verbosity exceeded the Layer 1 critical threshold.",
    )
    _add_threshold_alert(
        alerts,
        value=token_count,
        warning=DEFAULT_THRESHOLDS["token_warning"],
        critical=DEFAULT_THRESHOLDS["token_critical"],
        alert_class="token_spike",
        unit="tokens",
        warning_message="Token usage exceeded the Layer 1 warning threshold.",
        critical_message="Token usage exceeded the Layer 1 critical threshold.",
    )
    _add_threshold_alert(
        alerts,
        value=estimated_cost,
        warning=DEFAULT_THRESHOLDS["cost_warning_usd"],
        critical=DEFAULT_THRESHOLDS["cost_critical_usd"],
        alert_class="cost_spike",
        unit="usd",
        warning_message="Estimated provider cost exceeded the Layer 1 warning threshold.",
        critical_message="Estimated provider cost exceeded the Layer 1 critical threshold.",
    )
    _add_threshold_alert(
        alerts,
        value=first_token_latency,
        warning=DEFAULT_THRESHOLDS["first_token_latency_warning_ms"],
        critical=DEFAULT_THRESHOLDS["first_token_latency_critical_ms"],
        alert_class="first_token_latency_spike",
        unit="ms",
        warning_message="First-token latency exceeded the Layer 1 warning threshold.",
        critical_message="First-token latency exceeded the Layer 1 critical threshold.",
    )

    if bool(metadata.get("fallback_used") or record.get("fallback_used")):
        alerts.append(_alert("fallback_used", "warning", "Runtime fallback was used.", observed=True, threshold=False))
    if bool(metadata.get("degraded_mode") or record.get("degraded_mode")):
        alerts.append(_alert("degraded_mode", "warning", "Runtime completed in degraded mode.", observed=True, threshold=False))
    if existing_error or normalized_status == "failed" or record.get("failed_stage"):
        alerts.append(
            _alert(
                "provider_failure",
                "critical",
                "Runtime reported a failed execution path.",
                observed=existing_error or record.get("failed_stage") or normalized_status,
                threshold=None,
            )
        )
    if not observability_complete:
        missing = [
            key for key in ("workflow", "contains_phi", "latency_ms", "token_cost_available") if key not in metadata
        ]
        if missing:
            alerts.append(
                _alert(
                    "missing_required_metadata",
                    "warning",
                    "Layer 1 operational metadata is incomplete.",
                    observed=missing,
                    threshold="workflow,contains_phi,latency_ms,token_cost_available",
                )
            )
    if metadata.get("llm_used") is True and metadata.get("token_cost_available") is False:
        alerts.append(
            _alert(
                "missing_token_cost_metadata",
                "info",
                "Provider-backed execution did not expose token or cost metadata.",
                observed=False,
                threshold=True,
            )
        )
    if output in {None, ""}:
        alerts.append(_alert("invalid_output_schema", "critical", "Primary output is missing.", observed=output, threshold="non-empty output"))
    if workflow == "denial":
        route = metadata.get("routing_action") or record.get("routing_action")
        posture = metadata.get("governance_posture") or record.get("governance_posture")
        if route and route not in {"APPEAL", "RESUBMIT", "WRITE_OFF", "ESCALATE"}:
            alerts.append(_alert("invalid_output_schema", "critical", "Denial routing action is outside the allowed vocabulary.", observed=route, threshold="bounded routing vocabulary"))
        if posture and posture not in {"SUPPORTED", "LOW_CONFIDENCE", "LOW_EVIDENCE", "AMBIGUOUS", "HIGH_RISK"}:
            alerts.append(_alert("invalid_output_schema", "critical", "Denial governance posture is outside the allowed vocabulary.", observed=posture, threshold="bounded posture vocabulary"))
    if record.get("malformed_payload_observed"):
        alerts.append(_alert("malformed_payload", "critical", "Malformed payload was observed in structured runtime output.", observed=True, threshold=False))
    trace = record.get("trace")
    if expected_trace_count is not None and isinstance(trace, list) and len(trace) < expected_trace_count:
        alerts.append(
            _alert(
                "trace_incomplete",
                "warning",
                "Runtime trace has fewer stages than expected.",
                observed=len(trace),
                threshold=expected_trace_count,
                unit="stages",
            )
        )

    metrics = {
        "latency_ms": latency_value,
        "output_char_count": output_chars,
        "verbosity_bucket": _verbosity_bucket(output_chars),
        "observability_complete": observability_complete,
        "token_count": token_count,
        "estimated_cost_usd": estimated_cost,
        "first_token_latency_ms": first_token_latency,
    }
    max_severity = _max_severity(alerts)
    return {
        "status": normalized_status,
        "error": _primary_error(normalized_status, alerts, existing_error),
        "alerts": alerts,
        "alert_count": len(alerts),
        "max_alert_severity": max_severity,
        "operational_metrics": metrics,
        "operational_thresholds": DEFAULT_THRESHOLDS,
    }
