"""Canonical local hybrid evaluation runner."""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.config import get_execution_mode, get_model_name
from graph.graph_builder import build_graph
from graph.runtime_diagnostics import collect_startup_diagnostics
from graph.state import initial_state


def main() -> None:
    repo_root = REPO_ROOT
    load_dotenv(repo_root / ".env")

    cases = [
        "Patient reports fever and persistent cough for three days.",
        "Sharp abdominal pain with nausea and vomiting.",
        "Mild headache and fatigue after poor sleep.",
        "Chest tightness and shortness of breath.",
        "Runny nose and mild sore throat.",
    ]

    evaluation_dir = repo_root / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    runtime_diagnostic_path = evaluation_dir / "runtime_env_diagnostic.json"

    execution_mode = get_execution_mode()
    model_name = get_model_name()
    startup_diagnostics = collect_startup_diagnostics(repo_root)
    runtime_diagnostic_path.write_text(
        json.dumps(
            {
                "entrypoint": "scripts/local_hybrid_eval.py",
                "startup_diagnostics": startup_diagnostics,
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    compiled_graph = build_graph()
    graph = compiled_graph if hasattr(compiled_graph, "invoke") else compiled_graph.compile()

    runs: list[dict[str, object]] = []
    for case in cases:
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = graph.invoke(initial_state(case))

        diagnoses = result.get("diagnoses", [])
        critic_review = result.get("critic_review", {})
        governance_result = result.get("governance_result", {})
        errors = result.get("errors", [])
        node_diagnostics = result.get("node_diagnostics", [])
        fallback_nodes = [
            item.get("node_name")
            for item in node_diagnostics
            if isinstance(item, dict) and item.get("fallback_triggered") is True and item.get("node_name")
        ]
        fallback_detected = (len(errors) > 0 if isinstance(errors, list) else False) or bool(fallback_nodes)
        first_error = errors[0] if isinstance(errors, list) and errors else None

        record = {
            "run_id": run_id,
            "timestamp": timestamp,
            "execution_mode": execution_mode,
            "model_name": model_name,
            "input_hash": hashlib.sha256(case.encode("utf-8")).hexdigest(),
            "diagnoses_count": len(diagnoses) if isinstance(diagnoses, list) else 0,
            "critic_recommended_status": (
                critic_review.get("recommended_status") if isinstance(critic_review, dict) else None
            ),
            "governance_final_status": (
                governance_result.get("final_status") if isinstance(governance_result, dict) else None
            ),
            "escalation_required": bool(result.get("escalation_required")),
            "error_count": len(errors) if isinstance(errors, list) else 0,
            "fallback_detected": fallback_detected,
            "first_error": first_error,
            "fallback_nodes": fallback_nodes,
        }
        runs.append(record)

        print(f"RUN_ID={run_id}")
        print(f"MODE={execution_mode}")
        print(f"CRITIC={record['critic_recommended_status']}")
        print(f"GOVERNANCE={record['governance_final_status']}")
        print(f"ESCALATION={record['escalation_required']}")

    output_path = evaluation_dir / "local_hybrid_eval_summary.json"
    output_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
