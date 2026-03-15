"""Deterministic ICD mapping for the MED-SCRIBE demo.

ICD mapping remains deterministic by design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOOKUP_PATH = Path(__file__).resolve().parents[2] / "data" / "icd_lookup.json"
VALID_STATUSES = {"OK", "PARTIAL_MATCH", "NO_MATCH_FOUND"}


def _load_lookup() -> list[dict[str, str]]:
    return json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))


def run(state: dict[str, Any]) -> dict[str, Any]:
    lookup = _load_lookup()
    indexed = {entry["label"].lower(): entry for entry in lookup}
    symptoms = state["intake_data"].get("symptoms", [])
    diagnoses = state.get("diagnoses", [])

    targets = diagnoses or symptoms
    if not targets and state.get("raw_input", "").strip():
        targets = [state["raw_input"].strip()]
    mappings: list[dict[str, Any]] = []

    for target in targets:
        entry = indexed.get(target.lower())
        if entry:
            mappings.append(
                {
                    "label": target,
                    "icd_code": entry["icd_code"],
                    "icd_label": entry["icd_label"],
                    "status": "OK",
                }
            )
            continue

        matched_symptom = next((symptom for symptom in symptoms if symptom.lower() in indexed), None)
        if matched_symptom:
            partial = indexed[matched_symptom.lower()]
            mappings.append(
                {
                    "label": target,
                    "icd_code": partial["icd_code"],
                    "icd_label": partial["icd_label"],
                    "status": "PARTIAL_MATCH",
                }
            )
        else:
            mappings.append(
                {
                    "label": target,
                    "icd_code": "",
                    "icd_label": "",
                    "status": "NO_MATCH_FOUND",
                }
            )

    return {"icd_mappings": mappings}
