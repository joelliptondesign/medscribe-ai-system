"""Deterministic ICD mapping for the MED-SCRIBE demo.

ICD mapping remains deterministic by design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOOKUP_PATH = Path(__file__).resolve().parents[2] / "data" / "icd_lookup.json"
VALID_STATUSES = {"OK", "PARTIAL_MATCH", "NO_MATCH_FOUND"}
ALIASES = {
    "Arthralgia": {"icd_code": "M25.50", "icd_label": "Pain in unspecified joint"},
    "Dizziness syndrome": {"icd_code": "R42", "icd_label": "Dizziness and giddiness"},
    "Fatigue syndrome": {"icd_code": "R53.83", "icd_label": "Other fatigue"},
    "Influenza-like illness": {"icd_code": "J11.1", "icd_label": "Influenza with other respiratory manifestations, virus not identified"},
    "Nausea/vomiting syndrome": {"icd_code": "R11.2", "icd_label": "Nausea with vomiting, unspecified"},
    "Peripheral edema": {"icd_code": "R60.0", "icd_label": "Localized edema"},
    "Acute cough syndrome": {"icd_code": "R05.9", "icd_label": "Cough, unspecified"},
    "frequent urination": {"icd_code": "R35.0", "icd_label": "Frequency of micturition"},
    "joint pain": {"icd_code": "M25.50", "icd_label": "Pain in unspecified joint"},
    "morning stiffness": {
        "icd_code": "M25.60",
        "icd_label": "Stiffness of unspecified joint, not elsewhere classified",
    },
}


def _load_lookup() -> list[dict[str, str]]:
    return json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))


def run(state: dict[str, Any]) -> dict[str, Any]:
    lookup = _load_lookup()
    indexed = {entry["label"].lower(): entry for entry in lookup}
    intake_data = state.get("intake_data", {})
    symptoms = intake_data.get("symptoms", [])
    diagnoses = state.get("diagnoses", [])
    ambiguity_flag = bool(intake_data.get("ambiguity_flag"))

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

        alias = ALIASES.get(target)
        if alias:
            mappings.append(
                {
                    "label": target,
                    "icd_code": alias["icd_code"],
                    "icd_label": alias["icd_label"],
                    "status": "OK",
                }
            )
            continue

        stricter_symptom_match = next(
            (
                symptom for symptom in symptoms
                if symptom.lower() == target.lower() and symptom.lower() in indexed
            ),
            None,
        )
        if stricter_symptom_match:
            strict = indexed[stricter_symptom_match.lower()]
            mappings.append(
                {
                    "label": target,
                    "icd_code": strict["icd_code"],
                    "icd_label": strict["icd_label"],
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
        elif ambiguity_flag and symptoms:
            fallback_symptom = next((symptom for symptom in symptoms if symptom in ALIASES or symptom.lower() in indexed), None)
            if fallback_symptom:
                fallback = ALIASES.get(fallback_symptom) or indexed.get(fallback_symptom.lower())
                mappings.append(
                    {
                        "label": target,
                        "icd_code": fallback["icd_code"],
                        "icd_label": fallback["icd_label"],
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
