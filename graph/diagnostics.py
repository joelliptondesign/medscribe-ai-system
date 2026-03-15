"""Helpers for compact non-secret-bearing hybrid diagnostics."""

from __future__ import annotations

from typing import Any


def init_node_diagnostic(node_name: str) -> dict[str, Any]:
    return {
        "node_name": node_name,
        "live_call_attempted": False,
        "live_call_returned": False,
        "parse_succeeded": False,
        "normalization_succeeded": False,
        "fallback_triggered": False,
        "fallback_reason": "",
    }


def append_node_diagnostic(state: dict[str, Any], diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    return list(state.get("node_diagnostics", [])) + [diagnostic]
