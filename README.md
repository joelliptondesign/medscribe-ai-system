# Med Scribe — Governed AI Decision System

## What This Project Is

This project began as a governed AI reasoning pipeline.

It has been extended into a **production-adjacent system** with:

- API-based execution
- persistent run storage
- artifact retrieval (run inspection, no re-execution)
- latency tracking and execution tracing
- run comparison (decision + score diffs)
- vector-based retrieval over past runs (FAISS)
- structured tool-calling layer over pipeline stages

The focus is not just generating outputs, but demonstrating how AI systems can be:

**controlled, observable, and analyzable over time**

---

A production-adjacent AI system that separates **probabilistic reasoning** from **deterministic decision enforcement**, with persistent evaluation, observability, and comparative analysis.

---

## Overview

This system implements a structured, multi-stage pipeline for diagnostic reasoning and evaluation, then enforces outcomes through a deterministic policy layer.

Every run is:

- persisted (append-only)
- inspectable (full artifact retrieval)
- observable (latency + execution trace)
- comparable (run-to-run diffing)

---

## System Capabilities

The system now operates as a stateful service with:

### Execution
- API-driven evaluation (`POST /evaluate`)

### Persistence
- append-only run storage (`data/runs.jsonl`)

### Replay
- retrieval-based run inspection (`GET /run/{run_id}`)

### Observability
- stage-level latency tracking
- execution trace per run

### Analysis
- run comparison (`GET /compare`)

### Retrieval
- FAISS-based similarity search over prior runs (`POST /search`)

### Tooling
- structured tool abstraction over pipeline stages

---

## Core Architecture

Input  
→ Intake Parser  
→ Diagnosis Engine (LLM)  
→ ICD Mapping  
→ Critic (Scoring)  
→ Policy (Deterministic Decision)  
→ Persisted Run Artifact  
→ API Layer (Evaluate / Replay / Compare)

---

## Key Capabilities

### Deterministic Decision Layer
- PASS / REVISE / FAIL enforcement  
- removes ambiguity from LLM outputs  

### Persistent Run Storage
- append-only JSONL (`data/runs.jsonl`)  
- full historical trace of decisions  

### Artifact Replay (Retrieval-Based)
- GET /run/{run_id}  
- no recomputation, no model invocation  

### Observability
- stage-level latency tracking  
- execution trace per run  

### Run Comparison
- GET /compare  
- decision + score-level diffs  

### Vector Retrieval
- similarity search over past runs (FAISS-based)  
- enables retrieval of semantically similar cases  

### Tool Calling
- structured tool abstraction over pipeline stages  
- dispatcher-based execution model  

---

## API

### POST /evaluate
Input:
{
  "input_text": "..."
}

Returns:
{
  "run_id": "...",
  "status": "pending"
}

---

### GET /run/{run_id}
Returns full stored artifact.

Run artifacts now carry job lifecycle state:
- `pending`
- `running`
- `completed`
- `degraded` when hybrid execution fell back on one or more nodes
- `failed` when execution stops mid-pipeline

---

### GET /runs
Lists previous runs.

---

### GET /compare
Compare two runs by ID.

---

### POST /search
Vector similarity search over stored runs.

Request:
{
  "query": "...",
  "top_k": 5
}

---

### POST /tool
Execute a specific pipeline stage via tool dispatcher.

Request:
{
  "tool_name": "...",
  "payload": {...}
}

---

## Tech Stack

- Python  
- FastAPI (service layer)  
- FAISS (vector similarity search)  
- LangChain (LLM orchestration within structured pipeline)  
- JSONL (append-only storage)

---

## Running the System

pip install -r requirements.txt  
uvicorn service.main:app --reload  

To run the API service:

uvicorn service.main:app --reload

Docs: http://localhost:8000/docs

## Local Run Verification

Install dependencies:

`.venv/bin/pip install -r requirements.txt`

Start the service:

`.venv/bin/python -m uvicorn service.main:app --host 127.0.0.1 --port 8000 --reload`

Verify root:

`curl http://127.0.0.1:8000/`

Verify OpenAPI:

`curl http://127.0.0.1:8000/openapi.json`

Submit an evaluation:

`curl -X POST http://127.0.0.1:8000/evaluate -H "Content-Type: application/json" -d '{"input_text":"I have had fever, cough, and sore throat for two days."}'`

Run one-shot lifecycle validation:

`./.venv/bin/python -m pytest tests/test_run_lifecycle.py -q`

Degraded semantics:

- `status=degraded`, `fallback_used=true`, and `degraded_mode=true` mean hybrid mode was active and at least one node fell back to deterministic behavior.
- Stored artifacts include `node_diagnostics`, `fallback_nodes`, `fallback_reasons`, `metadata.execution_mode`, and `metadata.hybrid_attempted`.

Failed-run semantics:

- `status=failed` preserves the partial artifact reached before the exception.
- `failed_stage` and `error` identify where execution stopped.

---

## Design Principles

- separation of reasoning and decision enforcement  
- append-only state for auditability  
- minimal, explicit system boundaries  
- observability by default  

---

## Positioning

This project demonstrates:

- applied AI system design  
- evaluation and governance patterns  
- operational thinking (state, latency, traceability)  
- building production-adjacent systems without overengineering  
