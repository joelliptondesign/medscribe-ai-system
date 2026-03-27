"""Vector search over persisted run artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


RUNS_PATH = Path(__file__).resolve().parents[1] / "data" / "runs.jsonl"
EMBED_DIM = 32


def _load_runs() -> list[dict[str, Any]]:
    if not RUNS_PATH.exists():
        return []

    runs: list[dict[str, Any]] = []
    with RUNS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            runs.append(json.loads(raw))
    return runs


def _run_text(record: dict[str, Any]) -> str:
    text = record.get("input")
    if isinstance(text, str) and text.strip():
        return text.strip()

    diagnosis = record.get("diagnosis")
    if isinstance(diagnosis, dict):
        return json.dumps(diagnosis, sort_keys=True)
    return ""


def _embed_text(text: str) -> np.ndarray:
    vector = np.zeros(EMBED_DIM, dtype="float32")
    content = text.strip().lower()
    if not content:
        return vector

    for token in content.split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for idx in range(EMBED_DIM):
            vector[idx] += digest[idx] / 255.0
    return vector


def _build_index() -> tuple[list[dict[str, Any]], faiss.IndexFlatIP | None]:
    runs = _load_runs()
    if not runs:
        return runs, None

    vectors = np.vstack([_embed_text(_run_text(run)) for run in runs]).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    return runs, index


def search(query: str, top_k: int) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    runs, index = _build_index()
    if index is None:
        return []

    query_vector = _embed_text(query).reshape(1, -1).astype("float32")
    faiss.normalize_L2(query_vector)
    limit = min(top_k, len(runs))
    scores, indices = index.search(query_vector, limit)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append(
            {
                "run_id": runs[int(idx)].get("run_id"),
                "score": float(score),
            }
        )
    return results
