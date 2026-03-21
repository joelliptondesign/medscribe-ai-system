import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = (
    REPO_ROOT
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

# === PIPELINE IMPORT (ADJUST IF NEEDED) ===
# Replace with actual pipeline entry point if necessary
try:
    from graph.state import initial_state, load_schema
    from graph.graph_builder import build_graph
except ImportError:
    build_graph = None
    initial_state = None
    load_schema = None


def _format_case_input(input_data):
    symptoms = ", ".join(input_data.get("symptoms", []))
    age = input_data.get("age")
    gender = input_data.get("gender")
    duration = input_data.get("duration", "")
    notes = input_data.get("notes", "")
    return (
        f"{gender} patient age {age} with symptoms: {symptoms}. "
        f"Duration: {duration}. Notes: {notes}"
    )


def run_pipeline(input_data):
    if build_graph is None or initial_state is None or load_schema is None:
        raise ImportError("pipeline entry point not available")

    load_schema()
    app = build_graph()
    result = app.invoke(initial_state(_format_case_input(input_data)))

    return {
        "final_status": result.get("governance_result", {}).get(
            "final_status",
            result.get("final_output", {}).get("status", "UNKNOWN"),
        ),
        "critic_scores": result.get("critic_review", {}),
        "raw_result": result,
    }


DATASET_PATH = Path("evaluation/dataset.json")
OUTPUT_PATH = Path("evaluation/raw_results.json")


def load_dataset():
    with open(DATASET_PATH, "r") as f:
        return json.load(f)["cases"]


def run_case(case):
    result = run_pipeline(case["input"])

    return {
        "case_id": case["case_id"],
        "scenario_type": case["scenario_type"],
        "expected_behavior": case["expected_behavior"],
        "pipeline_output": result
    }


def main():
    cases = load_dataset()

    results = []

    for case in cases:
        result = run_case(case)
        results.append(result)

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
