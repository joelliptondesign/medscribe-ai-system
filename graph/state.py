"""Shared workflow state and local schema helpers for MED-SCRIBE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "medscribe_node_schemas.json"


class MedScribeState(TypedDict):
    raw_input: str
    intake_data: dict[str, Any]
    completeness: dict[str, Any]
    triage: dict[str, Any]
    diagnoses: list[str]
    icd_mappings: list[dict[str, Any]]
    critic_review: dict[str, Any]
    governance_result: dict[str, Any]
    final_output: dict[str, Any]
    escalation_required: bool
    errors: list[str]
    node_diagnostics: list[dict[str, Any]]


def load_schema() -> dict[str, Any]:
    """Load the staged node schema from the repo-relative path."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    try:
        data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid schema JSON in {SCHEMA_PATH}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Schema root must be a JSON object: {SCHEMA_PATH}")

    return data


def initial_state(raw_input: str) -> MedScribeState:
    """Create the initial graph state for a single deterministic run."""
    return {
        "raw_input": raw_input,
        "intake_data": {},
        "completeness": {},
        "triage": {},
        "diagnoses": [],
        "icd_mappings": [],
        "critic_review": {},
        "governance_result": {},
        "final_output": {},
        "escalation_required": False,
        "errors": [],
        "node_diagnostics": [],
    }
