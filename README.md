# Reliable AI Systems: Fine-Tuning, Evaluation, and Governance

MedScribe is a Python system for making LLM behavior measurable and controllable in a structured workflow. It combines a fixed execution pipeline, persisted run artifacts, offline evaluation, iterative fine-tuning, and a deterministic governance layer. The current repo artifacts show a progression from broken baseline outputs to 95.83% mapping accuracy with fine-tuned v2 on the Phase 1B dataset, while the governed layer reached 100.00% policy compliance and 100.00% decision stability on the latest comparison run.

## Why This Project Matters

Raw LLM outputs are not enough when a system needs consistent behavior.

Reliable AI needs structure, evaluation, and control. This repo shows that loop in practice: measure failures, refine the dataset, fine-tune again, and compare the outcome against a governed execution path.

## Controlled Experiment: Reliability Improvements

| System | Schema Validity | Mapping Accuracy | Decision Stability | Policy Compliance |
| --- | --- | --- | --- | --- |
| Baseline | 4.17% | 0.00% | — | — |
| Fine-Tuned (v1) | 100.00% | 83.33% | — | — |
| Fine-Tuned (v2) | 100.00% | 95.83% | — | — |
| Governed | — | — | 100.00% | 100.00% |

Values come from `evaluation/icd_eval_v2_comparison.json` and the persisted evaluation summaries in `evaluation/`.

## Example: Before vs After

Baseline output on the sample comparison set returned non-standard status values. For `Pharyngitis`:

```json
{"mappings":[{"label":"Pharyngitis","icd_code":"J02.9","icd_label":"Acute pharyngitis, unspecified","status":"active"}]}
```

Fine-tuned v2 on the same input returned valid JSON in the required format:

```json
{"mappings":[{"label":"Pharyngitis","icd_code":"J02.9","icd_label":"Acute pharyngitis, unspecified","status":"OK"}]}
```

Governed outcome on the latest Phase 1B comparison run: policy compliance was `1.00`, decision stability was `1.00`, escalation rate was `1.00`, and override rate was `0.00`.

## Iterative Improvement Loop

The baseline system failed first at structure. Fine-tuned v1 fixed that and reached 100.00% schema validity, but evaluation still exposed semantic mismatches in otherwise valid JSON.

The v2 dataset kept the original training set and added 30 targeted examples focused on exact ICD selection, mixed-status outputs, and cases that should remain `REVIEW` instead of becoming overconfident `OK`.

That second pass moved Phase 1B accuracy from 83.33% to 95.83% while preserving 100.00% schema validity.

## What This Demonstrates

- Structured output enforcement
- Evaluation-driven development
- Iterative model improvement
- Governance / decision control
- Reproducible system design

## Technical Proof Surface

A reviewer can verify the results directly from the repo artifacts:

- `evaluation/icd_eval_summary.json`
- `evaluation/full_system_comparison.json`
- `evaluation/icd_eval_v2_comparison.json`
- `evaluation/icd_eval_scale_summary.json`
- `fine_tuning/sample_outputs.json`
- `fine_tuning/sample_outputs_v2.json`
- `fine_tuning/fine_tune_job.json`
- `fine_tuning/fine_tune_job_v2.json`
- `evaluation/icd_eval_results_governed.json`

## Tech Stack

- Python
- OpenAI fine-tuning + inference
- LangGraph
- FastAPI
- JSON / JSONL artifact storage
- Custom evaluation framework

## System Overview

The runtime pipeline is:

`Input -> Intake Parser -> Triage Engine -> Diagnosis Engine -> ICD Mapping -> Critic -> Governance Policy`

The repo contains two operating surfaces:

- Service runtime: asynchronous execution, persisted run records, retrieval, comparison, and search
- Offline evaluation: dataset-driven execution, scoring, fine-tuning comparisons, and governed comparisons

Key outputs include:

- structured intake data
- triage decisions
- diagnosis lists
- ICD mappings
- critic scores and recommendations
- governance-enforced final decisions
- persisted run and evaluation artifacts

## Architecture

Primary service boundary:

- `service/main.py`: FastAPI app entry point
- `service/api.py`: request handlers
- `service/run_manager.py`: execution orchestration
- `service/storage.py`: runtime artifact persistence
- `service/retrieval.py`: stored-run search and retrieval

Pipeline implementation:

- `graph/state.py`: shared state
- `graph/graph_builder.py`: graph definition
- `graph/nodes/intake_parser.py`
- `graph/nodes/triage_engine.py`
- `graph/nodes/diagnosis_engine.py`
- `graph/nodes/icd_mapper.py`
- `graph/nodes/critic.py`
- `graph/nodes/governance_policy.py`

Evaluation and proof artifacts:

- `evaluation/eval_runner.py`
- `evaluation/score_runner.py`
- `evaluation/`
- `fine_tuning/`
- `runs/`

## API

Main endpoints:

- `POST /evaluate`
- `GET /run/{run_id}`
- `GET /run/{run_id}/status`
- `GET /runs`
- `GET /compare`
- `POST /search`
- `POST /tool`

Supported tool calls:

- `parse_input`
- `generate_diagnosis`
- `map_icd`
- `score_case`

## Example Run Artifact

```json
{
  "run_id": "example-run",
  "status": "completed",
  "input": "Patient reports fever and sore throat.",
  "diagnosis": {
    "diagnoses": ["Pharyngitis"],
    "triage": {
      "level": "home_care",
      "rationale": "Symptoms fit a simple upper-respiratory pattern."
    }
  },
  "icd_mapping": {
    "mappings": [
      {
        "label": "Pharyngitis",
        "icd_code": "J02.9",
        "icd_label": "Acute pharyngitis, unspecified",
        "status": "OK"
      }
    ]
  },
  "scores": {
    "diagnosis_consistency_score": 1.0,
    "symptom_alignment_score": 1.0,
    "icd_specificity_score": 1.0,
    "recommended_status": "pass",
    "confidence": 1.0
  },
  "decision": "PASS",
  "timing": {
    "total_ms": 1280
  }
}
```

## Running the System

```bash
pip install -r requirements.txt
uvicorn service.main:app --reload
```

Optional local verification:

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/openapi.json
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"input_text":"I have had fever, cough, and sore throat for two days."}'
```

Offline evaluation flow:

```bash
python evaluation/eval_runner.py
python evaluation/score_runner.py
python scripts/test_run_aggregator.py --run_id <RUN_ID>
```

## Security & Safety Considerations

- This system uses synthetic, non-PHI data.
- It is not intended for clinical or production PHI usage.
- It is designed for reliability, evaluation, and system control experimentation.

### Input Validation

A lightweight input validation layer rejects common sensitive identifiers at the API boundary. The current checks cover email addresses, phone numbers, SSN-like patterns, and date-of-birth formats.

Verified behavior:

- unsafe inputs -> HTTP 400 rejected, no pipeline execution, no run created
- safe inputs -> normal execution path, run created

- Unsafe Input: `Contact me at john@example.com` -> rejected (400)
- Safe Input: `I have a headache and mild fever` -> accepted, run created

This behavior was manually verified using controlled test inputs.

### Production Considerations

- PHI detection / redaction not implemented
- Encryption and access control not implemented
- Adversarial input hardening not implemented
- Intended as a controlled system demonstration, not a production deployment

## Threat Model

### Trust Boundaries

- User input -> untrusted
- LLM output -> untrusted until evaluated
- Persisted artifacts -> sensitive

### Primary Risks

- Accidental PHI input
- Malformed or adversarial input
- Secret leakage
- Misuse of model outputs

### Current Controls

- Structured schema enforcement
- Critic scoring layer
- Governance enforcement layer
- Deterministic output constraints
- Persisted artifacts for auditability

### Known Gaps

- No PHI redaction layer
- No encryption or access control
- No adversarial input protection
- Governance is deterministic but not policy-rich

### Safety Note

Basic input validation is included for demonstration purposes and does not replace production-grade safeguards.
