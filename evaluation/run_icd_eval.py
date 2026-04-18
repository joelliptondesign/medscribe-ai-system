"""Run deterministic ICD mapping evaluation against baseline and fine-tuned models.

This script is intentionally isolated from production pipeline logic.
It loads evaluation cases, invokes both models three times per case via the
OpenAI API, normalizes outputs, classifies failures, computes aggregate
metrics, and returns an in-memory result structure.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CASES_PATH = REPO_ROOT / "evaluation" / "icd_eval_cases.json"
FINE_TUNE_JOB_PATH = REPO_ROOT / "fine_tuning" / "fine_tune_job.json"
DEFAULT_BASELINE_MODEL = "gpt-4o-mini-2024-07-18"
RUNS_PER_MODEL = 3
SYSTEM_PROMPT = (
    "Map diagnoses to ICD-10 codes. Return only valid JSON following the required schema."
)
VALID_STATUS = {"OK", "REVIEW", "ERROR"}
FAILURE_EMPTY_OUTPUT = "EMPTY_OUTPUT"
FAILURE_SCHEMA_INVALID = "SCHEMA_INVALID"
FAILURE_FIELD_MISSING = "FIELD_MISSING"
FAILURE_FIELD_INVALID = "FIELD_INVALID"
FAILURE_SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"

JSON_SCHEMA = {
    "name": "icd_mapping_response",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string"},
                        "icd_code": {"type": "string"},
                        "icd_label": {"type": "string"},
                        "status": {"type": "string"},
                    },
                    "required": ["label", "icd_code", "icd_label", "status"],
                },
            }
        },
        "required": ["mappings"],
    },
}


def load_repo_api_key() -> str:
    env_path = REPO_ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            api_key = line.split("=", 1)[1].strip()
            if api_key:
                return api_key
            break
    raise RuntimeError("OPENAI_API_KEY missing from repo-local .env")


def load_eval_cases() -> list[dict[str, Any]]:
    payload = json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("evaluation/icd_eval_cases.json must contain a top-level array")
    return payload


def load_model_ids() -> dict[str, str]:
    job_payload = json.loads(FINE_TUNE_JOB_PATH.read_text(encoding="utf-8"))
    fine_tuned_model = job_payload.get("fine_tuned_model")
    if not isinstance(fine_tuned_model, str) or not fine_tuned_model.strip():
        raise ValueError("fine_tuning/fine_tune_job.json does not contain a fine_tuned_model")

    baseline_model = job_payload.get("base_model") or DEFAULT_BASELINE_MODEL
    if not isinstance(baseline_model, str) or not baseline_model.strip():
        baseline_model = DEFAULT_BASELINE_MODEL

    return {
        "baseline": baseline_model,
        "fine_tuned": fine_tuned_model,
    }


def build_user_prompt(diagnoses: list[str]) -> str:
    return json.dumps({"diagnoses": diagnoses}, ensure_ascii=True)


def invoke_model(client: OpenAI, model_name: str, diagnoses: list[str]) -> str:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(diagnoses)},
        ],
    )
    return response.choices[0].message.content or ""


def normalize_mapping(mapping: dict[str, Any]) -> dict[str, str]:
    return {
        "label": mapping["label"].strip(),
        "icd_code": mapping["icd_code"].strip(),
        "icd_label": mapping["icd_label"].strip(),
        "status": mapping["status"].strip(),
    }


def canonicalize_output(payload: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    normalized = [normalize_mapping(mapping) for mapping in payload["mappings"]]
    normalized.sort(key=lambda item: (item["label"], item["icd_code"], item["icd_label"], item["status"]))
    return {"mappings": normalized}


def count_present_required_fields(payload: Any) -> tuple[int, int]:
    total_expected = 1
    present = 1 if isinstance(payload, dict) and "mappings" in payload else 0
    if isinstance(payload, dict) and isinstance(payload.get("mappings"), list):
        for mapping in payload["mappings"]:
            total_expected += 4
            if isinstance(mapping, dict):
                for key in ("label", "icd_code", "icd_label", "status"):
                    if key in mapping:
                        present += 1
            else:
                present += 0
    return present, total_expected


def validate_output(raw_text: str) -> dict[str, Any]:
    if not raw_text or not raw_text.strip():
        return {
            "raw_text": raw_text,
            "parsed_output": None,
            "normalized_output": None,
            "schema_valid": False,
            "field_completeness": 0.0,
            "failure_types": [FAILURE_EMPTY_OUTPUT],
        }

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "raw_text": raw_text,
            "parsed_output": None,
            "normalized_output": None,
            "schema_valid": False,
            "field_completeness": 0.0,
            "failure_types": [FAILURE_SCHEMA_INVALID],
        }

    present_fields, expected_fields = count_present_required_fields(payload)
    field_completeness = present_fields / expected_fields if expected_fields else 1.0
    failure_types: list[str] = []

    if not isinstance(payload, dict):
        failure_types.append(FAILURE_SCHEMA_INVALID)
    elif set(payload.keys()) != {"mappings"}:
        if "mappings" not in payload:
            failure_types.append(FAILURE_FIELD_MISSING)
        else:
            failure_types.append(FAILURE_SCHEMA_INVALID)
    elif not isinstance(payload["mappings"], list):
        failure_types.append(FAILURE_FIELD_INVALID)
    else:
        for mapping in payload["mappings"]:
            if not isinstance(mapping, dict):
                failure_types.append(FAILURE_SCHEMA_INVALID)
                continue
            expected_keys = {"label", "icd_code", "icd_label", "status"}
            actual_keys = set(mapping.keys())
            if expected_keys - actual_keys:
                failure_types.append(FAILURE_FIELD_MISSING)
            if actual_keys - expected_keys:
                failure_types.append(FAILURE_SCHEMA_INVALID)
            for key in expected_keys & actual_keys:
                if not isinstance(mapping[key], str):
                    failure_types.append(FAILURE_FIELD_INVALID)
            if "status" in mapping and isinstance(mapping["status"], str) and mapping["status"] not in VALID_STATUS:
                failure_types.append(FAILURE_FIELD_INVALID)

    failure_types = sorted(set(failure_types))
    schema_valid = not failure_types
    normalized_output = canonicalize_output(payload) if schema_valid else None

    return {
        "raw_text": raw_text,
        "parsed_output": payload if isinstance(payload, dict) else None,
        "normalized_output": normalized_output,
        "schema_valid": schema_valid,
        "field_completeness": field_completeness,
        "failure_types": failure_types,
    }


def extract_expected_codes(expected_output: dict[str, Any]) -> list[str]:
    normalized = canonicalize_output(expected_output)
    return [mapping["icd_code"] for mapping in normalized["mappings"]]


def evaluate_semantics(expected_output: dict[str, Any], normalized_output: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if normalized_output is None:
        return False, []
    expected_codes = extract_expected_codes(expected_output)
    actual_codes = [mapping["icd_code"] for mapping in normalized_output["mappings"]]
    return expected_codes == actual_codes, actual_codes


def evaluate_run(case: dict[str, Any], raw_text: str) -> dict[str, Any]:
    evaluation = validate_output(raw_text)
    semantics_match, actual_codes = evaluate_semantics(
        case["expected_output"],
        evaluation["normalized_output"],
    )
    failure_types = list(evaluation["failure_types"])
    if evaluation["schema_valid"] and not semantics_match:
        failure_types.append(FAILURE_SEMANTIC_MISMATCH)
    failure_types = sorted(set(failure_types))

    return {
        "raw_text": evaluation["raw_text"],
        "parsed_output": evaluation["parsed_output"],
        "normalized_output": evaluation["normalized_output"],
        "schema_valid": evaluation["schema_valid"],
        "field_completeness": evaluation["field_completeness"],
        "mapping_accuracy": 1.0 if semantics_match else 0.0,
        "actual_icd_codes": actual_codes,
        "expected_icd_codes": extract_expected_codes(case["expected_output"]),
        "failure_types": failure_types,
    }


def evaluate_case_runs(case: dict[str, Any], run_results: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_outputs = []
    for run_result in run_results:
        normalized_output = run_result["normalized_output"]
        canonical_outputs.append(
            json.dumps(normalized_output, sort_keys=True, separators=(",", ":"))
            if normalized_output is not None
            else f"INVALID:{','.join(run_result['failure_types'])}"
        )

    distinct_outputs = sorted(set(canonical_outputs))
    return {
        "input": case["input"],
        "expected_output": case["expected_output"],
        "runs": run_results,
        "variance_detected": len(distinct_outputs) > 1,
        "distinct_normalized_outputs": distinct_outputs,
    }


def summarize_model(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_runs = sum(len(case_result["runs"]) for case_result in case_results)
    schema_valid_runs = 0
    field_completeness_sum = 0.0
    mapping_accuracy_sum = 0.0
    varied_case_count = 0
    failure_counter: Counter[str] = Counter()

    for case_result in case_results:
        if case_result["variance_detected"]:
            varied_case_count += 1
        for run_result in case_result["runs"]:
            schema_valid_runs += 1 if run_result["schema_valid"] else 0
            field_completeness_sum += run_result["field_completeness"]
            mapping_accuracy_sum += run_result["mapping_accuracy"]
            for failure_type in run_result["failure_types"]:
                failure_counter[failure_type] += 1

    return {
        "total_cases": len(case_results),
        "total_runs": total_runs,
        "schema_validity_rate": schema_valid_runs / total_runs if total_runs else 0.0,
        "field_completeness": field_completeness_sum / total_runs if total_runs else 0.0,
        "mapping_accuracy": mapping_accuracy_sum / total_runs if total_runs else 0.0,
        "output_variance": {
            "varied_case_count": varied_case_count,
            "varied_case_rate": varied_case_count / len(case_results) if case_results else 0.0,
        },
        "failure_taxonomy": {key: failure_counter.get(key, 0) for key in [
            FAILURE_SCHEMA_INVALID,
            FAILURE_FIELD_MISSING,
            FAILURE_FIELD_INVALID,
            FAILURE_SEMANTIC_MISMATCH,
            FAILURE_EMPTY_OUTPUT,
        ]},
        "case_results": case_results,
    }


def evaluate_model(client: OpenAI, model_name: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_results = []
    for case in cases:
        diagnoses = case["input"]["diagnoses"]
        run_results = []
        for _ in range(RUNS_PER_MODEL):
            raw_text = invoke_model(client, model_name, diagnoses)
            run_results.append(evaluate_run(case, raw_text))
        case_results.append(evaluate_case_runs(case, run_results))
    return summarize_model(case_results)


def run_evaluation() -> dict[str, Any]:
    client = OpenAI(api_key=load_repo_api_key())
    cases = load_eval_cases()
    models = load_model_ids()
    return {
        "baseline": evaluate_model(client, models["baseline"], cases),
        "fine_tuned": evaluate_model(client, models["fine_tuned"], cases),
    }


def main() -> None:
    result = run_evaluation()
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
