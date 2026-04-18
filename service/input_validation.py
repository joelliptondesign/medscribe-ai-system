"""Lightweight input validation for obviously sensitive patterns."""

from __future__ import annotations

import re


_SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "potential email address detected",
    ),
    (
        re.compile(r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b"),
        "potential phone number detected",
    ),
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "potential social security number detected",
    ),
    (
        re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)?\d{2}\b"),
        "potential date of birth detected",
    ),
    (
        re.compile(r"\b\d{8,}\b"),
        "potential long numeric identifier detected",
    ),
]


def validate_input(input_text: str) -> dict[str, bool | str | None]:
    """Return whether the input appears safe enough for demo processing."""
    if not isinstance(input_text, str):
        return {
            "is_safe": False,
            "reason": "input must be a string",
        }

    for pattern, reason in _SENSITIVE_PATTERNS:
        if pattern.search(input_text):
            return {
                "is_safe": False,
                "reason": reason,
            }

    return {
        "is_safe": True,
        "reason": None,
    }
