"""Local deterministic demo runner for the MED-SCRIBE graph."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from graph.config import get_execution_mode, get_model_name
from graph.graph_builder import build_graph
from graph.runtime_diagnostics import collect_startup_diagnostics
from graph.state import initial_state, load_schema


DEFAULT_INPUT = "I have had fever, cough, and sore throat for two days."


def main() -> None:
    load_dotenv()
    raw_input = " ".join(sys.argv[1:]).strip() or DEFAULT_INPUT
    mode = get_execution_mode()
    if os.getenv("MEDSCRIBE_ENV_DIAGNOSTIC") == "1":
        evaluation_dir = Path(__file__).resolve().parent / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        (evaluation_dir / "runtime_env_diagnostic_app.json").write_text(
            json.dumps(
                {
                    "entrypoint": "app.py",
                    "startup_diagnostics": collect_startup_diagnostics(Path(__file__).resolve().parent),
                    "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    load_schema()
    app = build_graph()
    result = app.invoke(initial_state(raw_input))
    print(f"Execution mode: {mode}")
    if mode == "hybrid":
        print(f"Model: {get_model_name()}")
    print(json.dumps(result["final_output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
