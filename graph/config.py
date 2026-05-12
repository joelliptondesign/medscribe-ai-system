"""Execution-mode configuration helpers for MED-SCRIBE."""

from __future__ import annotations

import os


VALID_EXECUTION_MODES = {"deterministic", "hybrid"}
DEFAULT_LANGCHAIN_PROJECT = "medscribe-phase1-runtime"


def get_execution_mode() -> str:
    mode = os.getenv("MEDSCRIBE_EXECUTION_MODE", "deterministic").strip().lower()
    if mode not in VALID_EXECUTION_MODES:
        raise ValueError(
            "Invalid MEDSCRIBE_EXECUTION_MODE: "
            f"{mode!r}. Allowed values are: deterministic, hybrid"
        )
    return mode


def get_model_name() -> str:
    model = os.getenv("MEDSCRIBE_MODEL", "gpt-4o-mini").strip()
    if not model:
        raise ValueError("Invalid MEDSCRIBE_MODEL: value must be a non-empty string")
    return model


def get_model_max_tokens() -> int:
    raw_value = os.getenv("MEDSCRIBE_MAX_TOKENS", "128").strip()
    try:
        max_tokens = int(raw_value)
    except ValueError as exc:
        raise ValueError("Invalid MEDSCRIBE_MAX_TOKENS: value must be an integer") from exc
    if max_tokens < 64 or max_tokens > 1024:
        raise ValueError("Invalid MEDSCRIBE_MAX_TOKENS: value must be between 64 and 1024")
    return max_tokens


def get_langsmith_metadata() -> dict[str, str | bool]:
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() in {"1", "true", "yes"}
    return {
        "langchain_tracing_v2": tracing_enabled,
        "langchain_project": os.getenv("LANGCHAIN_PROJECT", DEFAULT_LANGCHAIN_PROJECT).strip(),
        "langchain_endpoint_configured": bool(os.getenv("LANGCHAIN_ENDPOINT", "").strip()),
        "langchain_api_key_configured": bool(os.getenv("LANGCHAIN_API_KEY", "").strip()),
    }
