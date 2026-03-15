"""Deterministic edge-case harness for the MED-SCRIBE graph."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.graph_builder import build_graph
from graph.state import initial_state, load_schema


VALID_TRIAGE = {"urgent", "routine", "home_care", "escalate"}
VALID_CRITIC = {"pass", "revise", "fail"}
VALID_ICD_STATUS = {"OK", "PARTIAL_MATCH", "NO_MATCH_FOUND"}

CASES = [
    ("CASE_01", "I have chest pain and shortness of breath."),
    ("CASE_02", "I have a headache."),
    ("CASE_03", ""),
    ("CASE_04", "My elbows feel strange and fizzy."),
    ("CASE_05", "I have fever, cough, sore throat, headache, nausea, back pain, and abdominal pain."),
]


def extract_triage_value(triage: Any) -> Any:
    if isinstance(triage, str):
        return triage
    if isinstance(triage, dict):
        for key in ("level", "recommendation", "value", "triage_level", "status"):
            if key in triage:
                return triage[key]
    return None


def extract_critic_status(critic_review: Any) -> Any:
    if isinstance(critic_review, str):
        return critic_review
    if isinstance(critic_review, dict):
        for key in ("status", "critic_status", "value"):
            if key in critic_review:
                return critic_review[key]
    return None


def safe_diagnoses(result: dict[str, Any]) -> list[Any]:
    diagnoses = result.get("diagnoses")
    return diagnoses if isinstance(diagnoses, list) else []


def safe_icd_mappings(result: dict[str, Any]) -> list[dict[str, Any]]:
    mappings = result.get("icd_mappings")
    return mappings if isinstance(mappings, list) else []


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def run_case(graph: Any, case_id: str, raw_input: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    result = graph.invoke(initial_state(raw_input))

    check(isinstance(result, dict), "output is not dict-like", failures)
    for key in (
        "intake_data",
        "triage",
        "diagnoses",
        "icd_mappings",
        "critic_review",
        "final_output",
        "escalation_required",
        "errors",
    ):
        check(key in result, f"missing top-level section: {key}", failures)

    diagnoses = safe_diagnoses(result)
    mappings = safe_icd_mappings(result)
    triage_value = extract_triage_value(result.get("triage"))
    critic_status = extract_critic_status(result.get("critic_review"))

    check(len(diagnoses) <= 3, "diagnoses length exceeds 3", failures)
    check(critic_status in VALID_CRITIC, f"invalid critic status: {critic_status}", failures)
    check(triage_value in VALID_TRIAGE, f"invalid triage value: {triage_value}", failures)
    check(all(mapping.get("status") in VALID_ICD_STATUS for mapping in mappings), "invalid ICD mapping status found", failures)

    if case_id == "CASE_01":
        check(triage_value in {"urgent", "escalate"}, f"CASE_01 triage was {triage_value}", failures)
        check(isinstance(result.get("escalation_required"), bool), "CASE_01 escalation flag is not boolean", failures)
    elif case_id == "CASE_02":
        check(triage_value in VALID_TRIAGE, "CASE_02 triage not controlled", failures)
    elif case_id == "CASE_03":
        check(critic_status in {"pass", "revise", "fail"}, f"CASE_03 critic status invalid: {critic_status}", failures)
        try:
            json.dumps(result)
        except TypeError as exc:
            failures.append(f"CASE_03 output not JSON-compatible: {exc}")
        check("errors" in result, "CASE_03 errors field missing", failures)
    elif case_id == "CASE_04":
        check(len(diagnoses) <= 3, "CASE_04 diagnoses length exceeds 3", failures)
        check(
            any(mapping.get("status") in VALID_ICD_STATUS for mapping in mappings),
            "CASE_04 has no ICD mapping with a valid status",
            failures,
        )
    elif case_id == "CASE_05":
        check(len(diagnoses) <= 3, "CASE_05 diagnoses length exceeds 3", failures)
        check(critic_status in VALID_CRITIC, f"CASE_05 critic status invalid: {critic_status}", failures)

    return not failures, failures


def main() -> int:
    load_schema()
    graph = build_graph()
    passed = 0
    failed = 0

    for case_id, raw_input in CASES:
        ok, failures = run_case(graph, case_id, raw_input)
        if ok:
            passed += 1
            print(f"{case_id}: PASS")
        else:
            failed += 1
            print(f"{case_id}: FAIL")
            for failure in failures:
                print(f"  - {failure}")

    print("summary:")
    print(f"  total_cases: {len(CASES)}")
    print(f"  passed: {passed}")
    print(f"  failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
