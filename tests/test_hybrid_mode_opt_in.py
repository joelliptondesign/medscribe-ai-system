"""Hybrid mode opt-in checks without requiring live API calls by default."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph import llm_client
from graph.config import get_execution_mode, get_model_name
from graph.graph_builder import build_graph
from graph.state import initial_state, load_schema


def run_graph(raw_input: str) -> dict[str, Any]:
    load_schema()
    graph = build_graph()
    return graph.invoke(initial_state(raw_input))


def with_env(mode: str | None, key: str | None, model: str | None) -> dict[str, str | None]:
    snapshot = {
        "MEDSCRIBE_EXECUTION_MODE": os.environ.get("MEDSCRIBE_EXECUTION_MODE"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "MEDSCRIBE_MODEL": os.environ.get("MEDSCRIBE_MODEL"),
    }
    if mode is None:
        os.environ.pop("MEDSCRIBE_EXECUTION_MODE", None)
    else:
        os.environ["MEDSCRIBE_EXECUTION_MODE"] = mode
    if key is None:
        os.environ.pop("OPENAI_API_KEY", None)
    else:
        os.environ["OPENAI_API_KEY"] = key
    if model is None:
        os.environ.pop("MEDSCRIBE_MODEL", None)
    else:
        os.environ["MEDSCRIBE_MODEL"] = model
    return snapshot


def restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def case_deterministic_bypasses_client() -> tuple[bool, str]:
    snapshot = with_env("deterministic", None, None)
    original = llm_client.get_chat_model
    try:
        def fail_if_called() -> Any:
            raise AssertionError("get_chat_model should not be called in deterministic mode")

        llm_client.get_chat_model = fail_if_called
        result = run_graph("I have a headache.")
        status = result.get("critic_review", {}).get("recommended_status")
        return status in {"pass", "revise", "fail"}, f"critic status: {status}"
    except Exception as exc:
        return False, str(exc)
    finally:
        llm_client.get_chat_model = original
        restore_env(snapshot)


def case_hybrid_missing_key_falls_back() -> tuple[bool, str]:
    snapshot = with_env("hybrid", "", "gpt-4o-mini")
    try:
        result = run_graph("I have fever and cough.")
        errors = result.get("errors", [])
        status = result.get("critic_review", {}).get("recommended_status")
        has_marker = any("hybrid_fallback" in str(item) for item in errors)
        ok = has_marker and status in {"pass", "revise", "fail"}
        return ok, f"errors={errors}, status={status}"
    except Exception as exc:
        return False, str(exc)
    finally:
        restore_env(snapshot)


def case_model_selection_behavior() -> tuple[bool, str]:
    snapshot = with_env("hybrid", "", "gpt-4o-mini")
    try:
        mode = get_execution_mode()
        model = get_model_name()
        return mode == "hybrid" and model == "gpt-4o-mini", f"mode={mode}, model={model}"
    except Exception as exc:
        return False, str(exc)
    finally:
        restore_env(snapshot)


def main() -> int:
    cases = [
        ("CASE_A_DETERMINISTIC_BYPASS", case_deterministic_bypasses_client),
        ("CASE_B_HYBRID_MISSING_KEY", case_hybrid_missing_key_falls_back),
        ("CASE_C_MODEL_SELECTION", case_model_selection_behavior),
    ]
    passed = 0
    failed = 0

    for case_id, fn in cases:
        ok, detail = fn()
        if ok:
            passed += 1
            print(f"{case_id}: PASS")
        else:
            failed += 1
            print(f"{case_id}: FAIL")
            print(f"  - {detail}")

    print("summary:")
    print(f"  total_cases: {len(cases)}")
    print(f"  passed: {passed}")
    print(f"  failed: {failed}")
    return 0 if failed == 0 else 1


def test_hybrid_mode_opt_in() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
