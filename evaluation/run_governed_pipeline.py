"""Thin wrapper for executing the governed MED-SCRIBE pipeline without persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from graph.graph_builder import build_graph
from graph.state import initial_state, load_schema


REPO_ROOT = Path(__file__).resolve().parents[1]


def _coerce_raw_input(input_text: str) -> str:
    stripped = input_text.strip()
    if not stripped:
        raise ValueError("input_text must be a non-empty string")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    if isinstance(parsed, dict):
        diagnoses = parsed.get("diagnoses")
        if isinstance(diagnoses, list):
            values = [str(item).strip() for item in diagnoses if str(item).strip()]
            if values:
                return "Diagnoses: " + "; ".join(values)
        return stripped

    if isinstance(parsed, list):
        values = [str(item).strip() for item in parsed if str(item).strip()]
        if values:
            return "Diagnoses: " + "; ".join(values)

    return stripped


def _extract_status(final_output: dict[str, Any], governance_result: dict[str, Any]) -> str:
    status = final_output.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip()

    fallback = governance_result.get("final_status")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()

    return ""


def _extract_escalation(final_output: dict[str, Any], governance_result: dict[str, Any], raw_state: dict[str, Any]) -> bool:
    if "escalation_required" in final_output:
        return bool(final_output.get("escalation_required"))
    if "escalation_required" in governance_result:
        return bool(governance_result.get("escalation_required"))
    return bool(raw_state.get("escalation_required", False))


def _extract_scores(raw_state: dict[str, Any]) -> dict[str, Any]:
    critic_review = raw_state.get("critic_review", {})
    if not isinstance(critic_review, dict):
        return {}

    scores: dict[str, Any] = {}
    for key in ("confidence", "coherence_score", "completeness_score"):
        if key in critic_review:
            scores[key] = critic_review[key]
    return scores


def run_governed_pipeline(input_text: str) -> dict[str, Any]:
    """Execute the full governed pipeline once and return governance-facing artifacts."""
    load_dotenv(REPO_ROOT / ".env")
    load_schema()

    raw_input = _coerce_raw_input(input_text)
    app = build_graph()
    result = app.invoke(initial_state(raw_input))

    final_output = result.get("final_output", {})
    governance_result = result.get("governance_result", {})

    if not isinstance(final_output, dict):
        final_output = {}
    if not isinstance(governance_result, dict):
        governance_result = {}

    return {
        "status": _extract_status(final_output, governance_result),
        "escalation_required": _extract_escalation(final_output, governance_result, result),
        "scores": _extract_scores(result),
        "raw": result,
    }


__all__ = ["run_governed_pipeline"]
