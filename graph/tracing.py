"""Thin LangSmith tracing helpers for MED-SCRIBE runtime stages."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os
from typing import Any, Iterator


MAX_DICT_ITEMS = 40
MAX_LIST_ITEMS = 20
MAX_STRING_LENGTH = 2000
SECRET_KEY_FRAGMENTS = ("api_key", "apikey", "token", "secret", "password", "authorization")


def _tracing_enabled() -> bool:
    enabled = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower()
    api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    return enabled in {"1", "true", "yes"} and bool(api_key)


def _looks_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in SECRET_KEY_FRAGMENTS)


def _sanitize_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return str(value)[:MAX_STRING_LENGTH]

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_DICT_ITEMS:
                sanitized["__truncated__"] = True
                break
            safe_key = str(key)
            if _looks_secret_key(safe_key):
                sanitized[safe_key] = "[REDACTED]"
            else:
                sanitized[safe_key] = _sanitize_value(item, depth=depth + 1)
        return sanitized

    if isinstance(value, list):
        items = [_sanitize_value(item, depth=depth + 1) for item in value[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            items.append({"__truncated__": True, "remaining_count": len(value) - MAX_LIST_ITEMS})
        return items

    if isinstance(value, tuple):
        return [_sanitize_value(item, depth=depth + 1) for item in value[:MAX_LIST_ITEMS]]

    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
            return value[:MAX_STRING_LENGTH] + "...[TRUNCATED]"
        return value

    return str(value)[:MAX_STRING_LENGTH]


def sanitize_payload(payload: Any) -> Any:
    return _sanitize_value(payload)


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    sanitized = sanitize_payload(metadata or {})
    return sanitized if isinstance(sanitized, dict) else {}


class TraceSpanRecorder:
    def __init__(self, run: Any | None = None) -> None:
        self._run = run

    def set_outputs(self, outputs: Any) -> None:
        if self._run is None:
            return
        try:
            self._run.end(outputs=sanitize_payload(outputs))
        except Exception:
            return


@contextmanager
def trace_span(
    name: str,
    *,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[TraceSpanRecorder]:
    if not _tracing_enabled():
        with nullcontext():
            yield TraceSpanRecorder()
        return

    try:
        from langsmith.run_helpers import trace

        with trace(
            name,
            run_type=run_type,
            inputs=sanitize_payload(inputs or {}),
            metadata=_safe_metadata(metadata),
            tags=tags or [],
            project_name=os.getenv("LANGCHAIN_PROJECT", "").strip() or None,
        ) as run:
            yield TraceSpanRecorder(run)
    except Exception:
        yield TraceSpanRecorder()
