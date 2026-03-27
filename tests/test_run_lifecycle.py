"""API-level validation for async run lifecycle, degraded fallback, and partial failures."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from graph.llm_client import HybridLLMError
from graph.nodes import critic, diagnosis_engine, intake_parser, triage_engine
from service import retrieval, run_manager, storage
from service.main import app


def _wait_for_terminal_record(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/run/{run_id}")
        assert response.status_code == 200
        record = response.json()
        if record.get("status") in {"completed", "degraded", "failed"}:
            return record
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not reach a terminal state within {timeout} seconds")


def _critic_success_payload() -> dict[str, object]:
    return {
        "diagnosis_consistency_score": 0.9,
        "symptom_alignment_score": 0.9,
        "icd_specificity_score": 0.8,
        "recommended_status": "pass",
        "confidence": 0.85,
        "reason_codes": ["CRITIC_REVIEW_CLEAR"],
        "summary": "Hybrid critic stub accepted the case.",
    }


def _configure_isolated_storage(tmp_path: Path, monkeypatch) -> Path:
    runs_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(storage, "RUNS_PATH", runs_path)
    monkeypatch.setattr(retrieval, "RUNS_PATH", runs_path)
    monkeypatch.setattr(run_manager, "load_dotenv", lambda: None)
    return runs_path


def test_async_happy_path_persists_completed_run(tmp_path: Path, monkeypatch) -> None:
    _configure_isolated_storage(tmp_path, monkeypatch)
    monkeypatch.setenv("MEDSCRIBE_EXECUTION_MODE", "deterministic")

    with TestClient(app) as client:
        response = client.post("/evaluate", json={"input_text": "I have fever and cough."})
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "pending"

        record = _wait_for_terminal_record(client, payload["run_id"])
        assert record["status"] == "completed"
        assert record["fallback_used"] is False
        assert record["degraded_mode"] is False
        assert record["failed_stage"] is None
        assert record["error"] is None
        assert record["metadata"]["execution_mode"] == "deterministic"
        assert record["metadata"]["hybrid_attempted"] is False
        assert record["fallback_nodes"] == []
        assert record["fallback_reasons"] == {}
        assert record["node_diagnostics"] == []


def test_hybrid_fallback_persists_degraded_truth(tmp_path: Path, monkeypatch) -> None:
    _configure_isolated_storage(tmp_path, monkeypatch)
    monkeypatch.setenv("MEDSCRIBE_EXECUTION_MODE", "hybrid")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fail_intake(node_name: str, prompt_text: str, payload: dict, diagnostic: dict | None = None) -> dict:
        if diagnostic is not None:
            diagnostic["live_call_attempted"] = True
        raise HybridLLMError("client_call_failure", "forced_for_test")

    def ok_triage(node_name: str, prompt_text: str, payload: dict, diagnostic: dict | None = None) -> dict:
        if diagnostic is not None:
            diagnostic["live_call_attempted"] = True
            diagnostic["live_call_returned"] = True
            diagnostic["parse_succeeded"] = True
        return {"level": "routine", "rationale": "stub triage"}

    def ok_diagnosis(node_name: str, prompt_text: str, payload: dict, diagnostic: dict | None = None) -> dict:
        if diagnostic is not None:
            diagnostic["live_call_attempted"] = True
            diagnostic["live_call_returned"] = True
            diagnostic["parse_succeeded"] = True
        return {"diagnoses": ["Upper respiratory infection"]}

    def ok_critic(node_name: str, prompt_text: str, payload: dict, diagnostic: dict | None = None) -> dict:
        if diagnostic is not None:
            diagnostic["live_call_attempted"] = True
            diagnostic["live_call_returned"] = True
            diagnostic["parse_succeeded"] = True
        return _critic_success_payload()

    monkeypatch.setattr(intake_parser, "invoke_json", fail_intake)
    monkeypatch.setattr(triage_engine, "invoke_json", ok_triage)
    monkeypatch.setattr(diagnosis_engine, "invoke_json", ok_diagnosis)
    monkeypatch.setattr(critic, "invoke_json", ok_critic)

    with TestClient(app) as client:
        response = client.post(
            "/evaluate",
            json={"input_text": "I have had fever, cough, and sore throat for two days."},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "pending"

        record = _wait_for_terminal_record(client, payload["run_id"])
        assert record["status"] == "degraded"
        assert record["fallback_used"] is True
        assert record["degraded_mode"] is True
        assert record["failed_stage"] is None
        assert record["error"] is None
        assert record["metadata"]["execution_mode"] == "hybrid"
        assert record["metadata"]["hybrid_attempted"] is True
        assert record["fallback_nodes"] == ["intake_parser"]
        assert record["fallback_reasons"] == {"intake_parser": "client_call_failure"}
        assert any(
            diagnostic["node_name"] == "intake_parser" and diagnostic["fallback_triggered"] is True
            for diagnostic in record["node_diagnostics"]
        )


def test_mid_pipeline_failure_persists_partial_record(tmp_path: Path, monkeypatch) -> None:
    _configure_isolated_storage(tmp_path, monkeypatch)
    monkeypatch.setenv("MEDSCRIBE_EXECUTION_MODE", "deterministic")

    original_call_tool = run_manager.call_tool

    def fail_map_icd(name: str, payload: dict) -> dict:
        if name == "map_icd":
            raise RuntimeError("forced map_icd failure")
        return original_call_tool(name, payload)

    monkeypatch.setattr(run_manager, "call_tool", fail_map_icd)

    with TestClient(app) as client:
        response = client.post("/evaluate", json={"input_text": "I have fever and cough."})
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "pending"

        record = _wait_for_terminal_record(client, payload["run_id"])
        assert record["status"] == "failed"
        assert record["failed_stage"] == "map_icd"
        assert record["error"] == "forced map_icd failure"
        assert record["fallback_used"] is False
        assert record["parsed_input"]["symptoms"] == ["fever", "cough"]
        assert record["diagnosis"]["diagnoses"]
        assert record["icd_mapping"]["mappings"] == []
        assert "parse_ms" in record["timing"]
        assert "diagnosis_ms" in record["timing"]
        assert "mapping_ms" not in record["timing"]
        assert record["metadata"]["execution_mode"] == "deterministic"


def test_search_and_compare_tolerate_pending_and_failed_records(tmp_path: Path, monkeypatch) -> None:
    runs_path = _configure_isolated_storage(tmp_path, monkeypatch)
    storage.create_run_shell("pending-run", "I have fever and cough.")
    storage.append_run(
        {
            "run_id": "failed-run",
            "timestamp": "2026-03-27T00:00:00Z",
            "input": "I have cough.",
            "status": "failed",
            "failed_stage": "map_icd",
            "error": "forced failure",
        }
    )
    assert runs_path.exists()

    with TestClient(app) as client:
        compare_response = client.get("/compare", params={"run_id_1": "pending-run", "run_id_2": "failed-run"})
        assert compare_response.status_code == 200
        compare_payload = compare_response.json()
        assert compare_payload["decision_diff"] is False
        assert compare_payload["score_diff"] == {}

        search_response = client.post("/search", json={"query": "fever", "top_k": 5})
        assert search_response.status_code == 200
        search_payload = search_response.json()
        assert {item["run_id"] for item in search_payload["results"]} == {"pending-run", "failed-run"}
