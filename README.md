# MED-SCRIBE

**Governed AI reasoning pipeline demonstrating how probabilistic LLM outputs can be evaluated, constrained, and converted into deterministic system decisions.**

This repository illustrates a production-style architecture for embedding LLM reasoning inside governed system boundaries using structured pipelines, critic evaluation, and policy arbitration.

MED-SCRIBE
Governed AI Workflow Demo for Clinical Intake Reasoning

OVERVIEW

This project demonstrates a governed AI reasoning pipeline built with LangGraph.

The system simulates a clinical intake workflow where structured reasoning is combined with deterministic policy enforcement.

The goal is to show how probabilistic AI outputs can be evaluated and governed before producing a final decision.

ARCHITECTURE

The pipeline follows this sequence:

intake_parser
↓
triage_engine
↓
diagnosis_engine
↓
icd_mapper
↓
critic
↓
governance_policy
↓
final_formatter

See docs/architecture.md for the visual pipeline diagram.

Design Principles

This demo was built around a small set of architectural constraints designed to make AI reasoning systems safer and more observable.

1. Structured reasoning pipelines

LLM outputs are not treated as final decisions.
Each stage of reasoning is separated into explicit nodes with typed outputs.

2. Deterministic governance layer

Probabilistic outputs from the critic are evaluated by a deterministic policy engine.
Final system decisions are produced only after governance rules are applied.

3. Explicit decision artifacts

Each run produces a structured artifact containing:

• reasoning outputs
• critic evaluation results
• governance policy outcomes
• final decision state

This allows decisions to be inspected, audited, and analyzed, and provides the foundation for replayable execution.

4. Separation of reasoning and arbitration

The system distinguishes between:

• reasoning nodes (LLM or heuristic)
• evaluation nodes (critic scoring)
• governance nodes (policy enforcement)

This separation prevents the LLM from acting as the final authority.

Key design elements:

• structured reasoning pipeline
• critic-mediated evaluation
• deterministic governance layer
• policy-based arbitration
• traceable decision artifacts

RUNNING THE DEMO

Create a virtual environment and install dependencies.

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Then run the demo:

python app.py

You can optionally provide an input phrase:

python app.py "Patient with fever and cough"

EXAMPLE OUTPUT

A governed example artifact is provided:

examples/governed_eval_example.json

This artifact illustrates:

• critic evaluation scores
• governance rule application
• final decision outcome

PROJECT STRUCTURE

app.py — single-run demo entrypoint

graph/ — LangGraph pipeline construction and node logic

governance/ — policy rules and deterministic arbitration

schemas/ — structured node output contracts

data/ — ICD reference dataset

examples/ — canonical governed output artifact

tests/ — deterministic validation tests

NOTES

This repository focuses on demonstrating a governed reasoning workflow architecture.

This repository is intentionally scoped to the primary governed workflow demo surface.

Validation

With this repository state:

• python app.py runs successfully
• pytest passes (3 tests)
