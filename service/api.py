"""FastAPI routes for pipeline execution and artifact retrieval."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from service.input_validation import validate_input
from service import retrieval, run_manager, storage
from service.schemas import EvaluateRequest, EvaluateResponse, SearchRequest, SearchResponse, ToolRequest
from service.tools import call_tool


router = APIRouter()


def _build_score_diff(scores_1: dict, scores_2: dict) -> dict:
    diff: dict = {}
    for key in sorted(set(scores_1) | set(scores_2)):
        value_1 = scores_1.get(key)
        value_2 = scores_2.get(key)
        if value_1 != value_2:
            diff[key] = {
                "run_1": value_1,
                "run_2": value_2,
            }
    return diff


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    if not isinstance(request.input_text, str) or not request.input_text.strip():
        raise HTTPException(status_code=400, detail="input_text must be a non-empty string")

    validation = validate_input(request.input_text)
    if not validation["is_safe"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Input contains potentially sensitive information",
                "status": "rejected",
            },
        )

    run_id = str(uuid4())
    storage.create_run_shell(run_id, request.input_text.strip())
    run_manager.start_async_run(run_id, request.input_text)
    return EvaluateResponse(
        run_id=run_id,
        status="pending",
    )


@router.get("/run/{run_id}")
def get_run(run_id: str) -> dict:
    record = storage.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    return record


@router.get("/run/{run_id}/status")
def get_run_status(run_id: str) -> dict:
    record = storage.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": run_id,
        "status": record.get("status", "unknown"),
    }


@router.get("/runs")
def get_runs() -> list[dict]:
    return storage.list_runs()


@router.get("/compare")
def compare_runs(run_id_1: str, run_id_2: str) -> dict:
    run_1 = storage.get_run(run_id_1)
    run_2 = storage.get_run(run_id_2)

    if run_1 is None or run_2 is None:
        raise HTTPException(status_code=404, detail="run not found")

    return {
        "run_id_1": run_id_1,
        "run_id_2": run_id_2,
        "decision_diff": run_1.get("decision") != run_2.get("decision"),
        "score_diff": _build_score_diff(run_1.get("scores", {}), run_2.get("scores", {})),
    }


@router.post("/search", response_model=SearchResponse)
def search_runs(request: SearchRequest) -> SearchResponse:
    try:
        results = retrieval.search(request.query, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SearchResponse(results=results)


@router.post("/tool")
def call_service_tool(request: ToolRequest) -> dict:
    try:
        result = call_tool(request.tool_name, request.payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result}
