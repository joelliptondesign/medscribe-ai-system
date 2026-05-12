"""Deterministic denial category classification for synthetic cases."""

from __future__ import annotations

from typing import Any


CATEGORIES = {
    "modifier issue": ("modifier", "same-day", "distinct service", "separate service", "59", "25"),
    "coding mismatch": ("coding mismatch", "wrong code", "icd", "cpt", "diagnosis code", "code mismatch"),
    "insufficient documentation": ("insufficient documentation", "missing documentation", "not documented", "omitted", "not included"),
    "unsupported service": ("non-covered", "not covered", "excluded", "unsupported service", "experimental"),
    "medical necessity": ("medical necessity", "not medically necessary", "necessity", "criteria not met", "guideline"),
}


def _combined(state: dict[str, Any]) -> str:
    return " ".join(
        str(state.get(field, "")).lower()
        for field in ("denial_reason", "service_context", "documentation_context", "denial_type")
    )


def run(state: dict[str, Any]) -> dict[str, Any]:
    updated = dict(state)
    text = _combined(updated)
    category = "unclear denial"
    for candidate, keywords in CATEGORIES.items():
        if any(keyword in text for keyword in keywords):
            category = candidate
            break
    updated["denial_category"] = category
    return updated
