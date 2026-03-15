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
