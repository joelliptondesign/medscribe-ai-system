"""Append-only JSONL storage for service run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from datetime import datetime, timezone
from typing import Any


RUNS_PATH = Path(__file__).resolve().parents[1] / "data" / "runs.jsonl"
_RUNS_LOCK = Lock()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_storage_file() -> None:
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNS_PATH.touch(exist_ok=True)


def append_run(record: dict[str, Any]) -> None:
    with _RUNS_LOCK:
        _ensure_storage_file()
        with RUNS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def create_run_shell(run_id: str, input_text: str) -> dict[str, Any]:
    record = {
        "run_id": run_id,
        "timestamp": _utc_timestamp(),
        "input": input_text,
        "status": "pending",
    }
    append_run(record)
    return record


def update_run(run_id: str, updated_fields: dict[str, Any]) -> dict[str, Any] | None:
    with _RUNS_LOCK:
        _ensure_storage_file()
        records: list[dict[str, Any]] = []
        updated_record: dict[str, Any] | None = None

        with RUNS_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                if record.get("run_id") == run_id:
                    record.update(updated_fields)
                    updated_record = record
                records.append(record)

        if updated_record is None:
            return None

        with RUNS_PATH.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        return updated_record


def get_run(run_id: str) -> dict[str, Any] | None:
    with _RUNS_LOCK:
        _ensure_storage_file()
        with RUNS_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                if record.get("run_id") == run_id:
                    return record
    return None


def list_runs() -> list[dict[str, Any]]:
    with _RUNS_LOCK:
        _ensure_storage_file()
        runs: list[dict[str, Any]] = []
        with RUNS_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                runs.append(
                    {
                        "run_id": record.get("run_id"),
                        "timestamp": record.get("timestamp"),
                        "status": record.get("status"),
                    }
                )
    return runs
