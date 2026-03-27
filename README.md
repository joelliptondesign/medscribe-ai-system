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
  "decision": "PASS",
  "scores": {...},
  "timing": {...},
  "trace": [...]
}

---

### GET /run/{run_id}
Returns full stored artifact.

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
