"""Execution-mode configuration helpers for MED-SCRIBE."""

from __future__ import annotations

import os


VALID_EXECUTION_MODES = {"deterministic", "hybrid"}


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
