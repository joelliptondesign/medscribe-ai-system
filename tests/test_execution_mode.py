"""Dependency-free execution mode validation for MED-SCRIBE."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.config import get_execution_mode, get_model_name


def run_mode_case(case_id: str, env_value: str | None, expected: str | None, expect_error: bool) -> tuple[bool, str]:
    original = os.environ.get("MEDSCRIBE_EXECUTION_MODE")
    try:
        if env_value is None:
            os.environ.pop("MEDSCRIBE_EXECUTION_MODE", None)
        else:
            os.environ["MEDSCRIBE_EXECUTION_MODE"] = env_value

        if expect_error:
            try:
                get_execution_mode()
            except ValueError as exc:
                return True, str(exc)
            return False, "expected ValueError was not raised"

        actual = get_execution_mode()
        if actual != expected:
            return False, f"expected {expected}, got {actual}"
        return True, actual
    finally:
        if original is None:
            os.environ.pop("MEDSCRIBE_EXECUTION_MODE", None)
        else:
            os.environ["MEDSCRIBE_EXECUTION_MODE"] = original


def run_model_case(case_id: str, env_value: str | None, expected: str | None, expect_error: bool) -> tuple[bool, str]:
    original = os.environ.get("MEDSCRIBE_MODEL")
    try:
        if env_value is None:
            os.environ.pop("MEDSCRIBE_MODEL", None)
        else:
            os.environ["MEDSCRIBE_MODEL"] = env_value

        if expect_error:
            try:
                get_model_name()
            except ValueError as exc:
                return True, str(exc)
            return False, "expected ValueError was not raised"

        actual = get_model_name()
        if actual != expected:
            return False, f"expected {expected}, got {actual}"
        return True, actual
    finally:
        if original is None:
            os.environ.pop("MEDSCRIBE_MODEL", None)
        else:
            os.environ["MEDSCRIBE_MODEL"] = original


def main() -> int:
    cases = [
        ("CASE_DEFAULT", None, "deterministic", False),
        ("CASE_DETERMINISTIC", "deterministic", "deterministic", False),
        ("CASE_HYBRID", "hybrid", "hybrid", False),
        ("CASE_INVALID", "invalid_mode", None, True),
    ]
    model_cases = [
        ("MODEL_DEFAULT", None, "gpt-4o-mini", False),
        ("MODEL_EXPLICIT", "gpt-4o-mini", "gpt-4o-mini", False),
        ("MODEL_INVALID", "   ", None, True),
    ]

    passed = 0
    failed = 0
    for case_id, env_value, expected, expect_error in cases:
        ok, detail = run_mode_case(case_id, env_value, expected, expect_error)
        if ok:
            passed += 1
            print(f"{case_id}: PASS")
        else:
            failed += 1
            print(f"{case_id}: FAIL")
            print(f"  - {detail}")

    for case_id, env_value, expected, expect_error in model_cases:
        ok, detail = run_model_case(case_id, env_value, expected, expect_error)
        if ok:
            passed += 1
            print(f"{case_id}: PASS")
        else:
            failed += 1
            print(f"{case_id}: FAIL")
            print(f"  - {detail}")

    print("summary:")
    print(f"  total_cases: {len(cases) + len(model_cases)}")
    print(f"  passed: {passed}")
    print(f"  failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
