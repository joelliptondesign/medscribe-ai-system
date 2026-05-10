"""Minimal regenerated LangSmith experiment runner for operational benchmarks."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "evaluation" / "operational_benchmark_cases.json"
DEFAULT_DATASET_NAME = "medscribe-operational-benchmark-poc"
EXPERIMENT_LABELS = {"baseline", "threshold_variant"}
THRESHOLD_VARIANT_OVERRIDES = {
    "confidence_min_for_revise": 0.7,
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class EvaluationResult:
    key: str
    score: bool
    comment: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_cases(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("operational benchmark dataset must be a JSON list")
    required = {
        "case_id",
        "title",
        "input_text",
        "operational_theme",
        "expected_high_level_behavior",
        "metadata",
    }
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case at index {index} must be an object")
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"{case.get('case_id', index)} missing required fields: {missing}")
        if not str(case["input_text"]).strip():
            raise ValueError(f"{case['case_id']} input_text must be non-empty")
    return cases


def langsmith_credentials_available() -> bool:
    return bool(os.getenv("LANGCHAIN_API_KEY", "").strip() or os.getenv("LANGSMITH_API_KEY", "").strip())


def _get_langsmith_client() -> Any | None:
    if not langsmith_credentials_available():
        return None
    try:
        from langsmith import Client

        return Client()
    except Exception as exc:
        print(f"langsmith_client_status=unavailable reason={exc}")
        return None


def _dataset_exists(client: Any, dataset_name: str) -> bool:
    try:
        client.read_dataset(dataset_name=dataset_name)
        return True
    except Exception:
        return False


def create_or_reuse_dataset(client: Any | None, cases: list[dict[str, Any]], dataset_name: str) -> str:
    if client is None:
        print("langsmith_dataset_status=skipped credentials_or_client_missing")
        return dataset_name

    try:
        if not _dataset_exists(client, dataset_name):
            client.create_dataset(
                dataset_name=dataset_name,
                description="Synthetic non-PHI MedScribe operational RCM benchmark cases.",
            )
        existing_ids: set[str] = set()
        try:
            for example in client.list_examples(dataset_name=dataset_name):
                metadata = getattr(example, "metadata", None) or {}
                case_id = metadata.get("case_id")
                if case_id:
                    existing_ids.add(str(case_id))
        except Exception:
            existing_ids = set()

        for case in cases:
            if case["case_id"] in existing_ids:
                continue
            client.create_example(
                inputs={"input_text": case["input_text"]},
                outputs={
                    "expected_high_level_behavior": case["expected_high_level_behavior"],
                    "expected_governance_status": case.get("expected_governance_status"),
                    "expected_routing_category": case.get("expected_routing_category"),
                },
                metadata={
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "operational_theme": case["operational_theme"],
                    **case.get("metadata", {}),
                },
                dataset_name=dataset_name,
            )
        print(f"langsmith_dataset_status=ready dataset_name={dataset_name}")
    except Exception as exc:
        print(f"langsmith_dataset_status=degraded dataset_name={dataset_name} reason={exc}")
    return dataset_name


def _case_by_id(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): case for case in cases}


def _case_by_input(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(case["input_text"]): case for case in cases}


def _example_case_id(example: Any) -> str:
    metadata = getattr(example, "metadata", None) or {}
    return str(metadata.get("case_id", "")).strip()


def _list_selected_examples(client: Any, dataset_name: str, cases: list[dict[str, Any]]) -> list[Any]:
    selected_ids = {str(case["case_id"]) for case in cases}
    by_id: dict[str, Any] = {}
    for example in client.list_examples(dataset_name=dataset_name):
        case_id = _example_case_id(example)
        if case_id in selected_ids:
            by_id[case_id] = example
    missing = [case["case_id"] for case in cases if case["case_id"] not in by_id]
    if missing:
        raise ValueError(f"LangSmith dataset is missing selected cases: {missing}")
    return [by_id[str(case["case_id"])] for case in cases]


def _case_from_example(example: Any, cases_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_id = _example_case_id(example)
    if case_id in cases_by_id:
        return cases_by_id[case_id]

    metadata = getattr(example, "metadata", None) or {}
    outputs = getattr(example, "outputs", None) or {}
    inputs = getattr(example, "inputs", None) or {}
    return {
        "case_id": case_id or "unknown",
        "title": metadata.get("title", case_id or "unknown"),
        "input_text": inputs.get("input_text", ""),
        "operational_theme": metadata.get("operational_theme", ""),
        "expected_high_level_behavior": outputs.get("expected_high_level_behavior", ""),
        "expected_governance_status": outputs.get("expected_governance_status"),
        "expected_routing_category": outputs.get("expected_routing_category"),
        "metadata": metadata,
    }


def _record_status(record: dict[str, Any]) -> str:
    return str(record.get("decision") or record.get("summary", {}).get("status") or "").strip().upper()


@contextmanager
def _temporary_threshold_variant(experiment_label: str) -> Any:
    if experiment_label != "threshold_variant":
        yield
        return

    from graph.nodes import governance_policy

    original_load_policy = governance_policy._load_policy

    def load_policy_with_variant() -> dict[str, Any]:
        policy = deepcopy(original_load_policy())
        policy["thresholds"].update(THRESHOLD_VARIANT_OVERRIDES)
        policy["governance_version"] = str(policy.get("governance_version", "")) + "_threshold_variant"
        return policy

    governance_policy._load_policy = load_policy_with_variant
    try:
        yield
    finally:
        governance_policy._load_policy = original_load_policy


def high_level_match(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    if error or not record:
        return EvaluationResult("high_level_match", False, error or "missing record")
    status = str(record.get("status", "")).strip().lower()
    has_core_outputs = bool(record.get("summary")) and bool(record.get("trace"))
    return EvaluationResult(
        "high_level_match",
        status in {"completed", "degraded"} and has_core_outputs,
        f"status={status} theme={case['operational_theme']}",
    )


def governance_status_match(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    expected = str(case.get("expected_governance_status", "")).strip().upper()
    if not expected:
        return EvaluationResult("governance_status_match", True, "no expected governance status")
    if error or not record:
        return EvaluationResult("governance_status_match", False, error or "missing record")
    actual = _record_status(record)
    return EvaluationResult("governance_status_match", actual == expected, f"expected={expected} actual={actual}")


def completed_without_fallback(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    if error or not record:
        return EvaluationResult("completed_without_fallback", False, error or "missing record")
    fallback_used = bool(record.get("fallback_used") or record.get("fallback_nodes"))
    status = str(record.get("status", "")).strip().lower()
    return EvaluationResult(
        "completed_without_fallback",
        status == "completed" and not fallback_used,
        f"status={status} fallback_used={fallback_used}",
    )


def final_status_present(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> EvaluationResult:
    if error or not record:
        return EvaluationResult("final_status_present", False, error or "missing record")
    final_status = _record_status(record)
    return EvaluationResult("final_status_present", bool(final_status), f"final_status={final_status or 'missing'}")


def evaluate_case(case: dict[str, Any], record: dict[str, Any] | None, error: str | None) -> list[EvaluationResult]:
    return [
        high_level_match(case, record, error),
        governance_status_match(case, record, error),
        completed_without_fallback(case, record, error),
        final_status_present(case, record, error),
    ]


def _run_pipeline(case: dict[str, Any], experiment_label: str) -> tuple[dict[str, Any] | None, str | None, int]:
    from service.run_manager import execute

    previous_mode = os.environ.get("MEDSCRIBE_EXECUTION_MODE")
    os.environ["MEDSCRIBE_EXECUTION_MODE"] = "hybrid"
    started = time.perf_counter()
    try:
        with _temporary_threshold_variant(experiment_label):
            record = execute(
                case["input_text"],
                run_id=f"{experiment_label}-{case['case_id']}",
                timestamp=_utc_stamp(),
                persist=False,
            )
        return record, None, max(1, int((time.perf_counter() - started) * 1000))
    except Exception as exc:
        partial = getattr(exc, "partial_record", None)
        record = partial if isinstance(partial, dict) else None
        return record, str(exc), max(1, int((time.perf_counter() - started) * 1000))
    finally:
        if previous_mode is None:
            os.environ.pop("MEDSCRIBE_EXECUTION_MODE", None)
        else:
            os.environ["MEDSCRIBE_EXECUTION_MODE"] = previous_mode


def _run_case_result(case: dict[str, Any], experiment_label: str, dataset_name: str) -> dict[str, Any]:
    record, error, latency_ms = _run_pipeline(case, experiment_label)
    evaluations = evaluate_case(case, record, error)
    token_metadata = {}
    if record:
        token_metadata = record.get("metadata", {}).get("token_usage", {}) or {}
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "experiment_label": experiment_label,
        "dataset_name": dataset_name,
        "threshold_variant_overrides": THRESHOLD_VARIANT_OVERRIDES if experiment_label == "threshold_variant" else {},
        "runtime_status": record.get("status") if record else "error",
        "final_status": _record_status(record or {}),
        "latency_ms": latency_ms,
        "token_metadata": token_metadata,
        "error": error,
        "evaluations": [
            {"key": item.key, "score": item.score, "comment": item.comment}
            for item in evaluations
        ],
        "record": record,
    }


def _console_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "record"}


def _summary(
    *,
    cases: list[dict[str, Any]],
    experiment_label: str,
    dataset_name: str,
    results: list[dict[str, Any]],
    hosted_experiment_name: str | None = None,
    hosted_experiment_url: str | None = None,
    formal_hosted: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_name": dataset_name,
        "experiment_label": experiment_label,
        "case_count": len(cases),
        "generated_at": _utc_stamp(),
        "formal_hosted_experiment": formal_hosted,
        "threshold_variant_overrides": THRESHOLD_VARIANT_OVERRIDES if experiment_label == "threshold_variant" else {},
        "results": [_console_result(result) for result in results],
    }
    if hosted_experiment_name:
        summary["hosted_experiment_name"] = hosted_experiment_name
    if hosted_experiment_url:
        summary["hosted_experiment_url"] = hosted_experiment_url
    return summary


def run_experiment(cases: list[dict[str, Any]], experiment_label: str, dataset_name: str) -> dict[str, Any]:
    if experiment_label not in EXPERIMENT_LABELS:
        raise ValueError(f"experiment_label must be one of {sorted(EXPERIMENT_LABELS)}")

    results = [_run_case_result(case, experiment_label, dataset_name) for case in cases]
    return _summary(
        cases=cases,
        experiment_label=experiment_label,
        dataset_name=dataset_name,
        results=results,
    )


def _result_from_run_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    if "output" in outputs and isinstance(outputs["output"], dict):
        outputs = outputs["output"]
    return dict(outputs)


def _rows_to_results(rows: Iterable[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        run = row.get("run") if isinstance(row, dict) else getattr(row, "run", None)
        outputs = getattr(run, "outputs", None) if run is not None else None
        if isinstance(outputs, dict):
            results.append(_result_from_run_outputs(outputs))
    return results


def _hosted_evaluator(
    key: str,
    cases_by_id: dict[str, dict[str, Any]],
    evaluator_fn: Any,
) -> Any:
    def evaluator(run: Any, example: Any) -> dict[str, Any]:
        outputs = getattr(run, "outputs", None) or {}
        result = _result_from_run_outputs(outputs) if isinstance(outputs, dict) else {}
        record = result.get("record")
        if not isinstance(record, dict):
            record = None
        error = result.get("error")
        case = _case_from_example(example, cases_by_id)
        evaluation = evaluator_fn(case, record, error)
        return {
            "key": key,
            "score": evaluation.score,
            "comment": evaluation.comment,
        }

    evaluator.__name__ = key
    return evaluator


def run_hosted_experiment(
    client: Any,
    cases: list[dict[str, Any]],
    experiment_label: str,
    dataset_name: str,
) -> dict[str, Any]:
    if experiment_label not in EXPERIMENT_LABELS:
        raise ValueError(f"experiment_label must be one of {sorted(EXPERIMENT_LABELS)}")

    from langsmith import evaluate

    cases_by_id = _case_by_id(cases)
    cases_by_input = _case_by_input(cases)
    examples = _list_selected_examples(client, dataset_name, cases)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        input_text = str(inputs.get("input_text", ""))
        case = cases_by_input[input_text]
        return _run_case_result(case, experiment_label, dataset_name)

    target.__name__ = f"medscribe_{experiment_label}_operational_benchmark"

    evaluators = [
        _hosted_evaluator("high_level_match", cases_by_id, high_level_match),
        _hosted_evaluator("governance_status_match", cases_by_id, governance_status_match),
        _hosted_evaluator("completed_without_fallback", cases_by_id, completed_without_fallback),
        _hosted_evaluator("final_status_present", cases_by_id, final_status_present),
    ]
    experiment_prefix = f"medscribe-operational-{experiment_label}"
    hosted_results = evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        description="Regenerated MedScribe operational benchmark experiment.",
        metadata={
            "dataset_name": dataset_name,
            "experiment_label": experiment_label,
            "case_limit": len(cases),
            "runner": "evaluation/langsmith_experiment_runner.py",
        },
        max_concurrency=0,
        client=client,
        blocking=True,
        upload_results=True,
        error_handling="log",
    )
    rows = list(hosted_results)
    results = _rows_to_results(rows)
    hosted_experiment_name = getattr(hosted_results, "experiment_name", None)
    hosted_experiment_url = None
    if hosted_experiment_name:
        try:
            project = client.read_project(project_name=hosted_experiment_name)
            hosted_experiment_url = getattr(project, "url", None)
        except Exception:
            hosted_experiment_url = None
    return _summary(
        cases=cases,
        experiment_label=experiment_label,
        dataset_name=dataset_name,
        results=results,
        hosted_experiment_name=hosted_experiment_name,
        hosted_experiment_url=hosted_experiment_url,
        formal_hosted=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--experiment-label", default="baseline", choices=sorted(EXPERIMENT_LABELS))
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit for smoke runs.")
    parser.add_argument("--skip-runtime", action="store_true", help="Only validate and create/reuse dataset.")
    parser.add_argument("--local-only", action="store_true", help="Run local summary mode without LangSmith evaluate().")
    args = parser.parse_args()

    load_dotenv()
    cases = load_cases()
    if args.limit > 0:
        cases = cases[: args.limit]

    client = _get_langsmith_client()
    dataset_name = create_or_reuse_dataset(client, cases, args.dataset_name)

    if not langsmith_credentials_available():
        print("langsmith_run_status=skipped credentials_absent")
        if args.skip_runtime:
            return 0

    if args.skip_runtime:
        print("runtime_status=skipped")
        return 0

    if client is not None and not args.local_only:
        try:
            summary = run_hosted_experiment(client, cases, args.experiment_label, dataset_name)
        except Exception as exc:
            print(f"langsmith_formal_experiment_status=degraded reason={exc}")
            summary = run_experiment(cases, args.experiment_label, dataset_name)
    else:
        summary = run_experiment(cases, args.experiment_label, dataset_name)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
