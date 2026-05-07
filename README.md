# MedScribe: Governed AI System Reliability Workflows

MedScribe is a reliability-focused AI systems engineering project that makes LLM behavior measurable, inspectable, and governed inside a structured execution pipeline. It combines a multi-stage LLM workflow, persisted run artifacts, offline evaluation, fine-tuning comparisons, LangSmith observability, synthetic incident testing, trace-driven RCA, and deterministic governance controls.

The project is built around a practical AI engineering question: how do you turn raw model outputs into measurable system behavior that can be evaluated, debugged, compared, and governed?

Current verified artifacts show:

- Fine-tuned v2 ICD mapping reached 95.83% mapping accuracy on the Phase 1B comparison dataset.
- Fine-tuned v1 and v2 preserved 100.00% schema validity.
- The governed layer reached 100.00% policy compliance and 100.00% decision stability on the latest persisted governed comparison run.
- Hybrid synthetic incident runs produced hosted LangSmith traces with nested runs, stage outputs, child model spans, token usage, cost visibility, and realistic latency.

Values come from `evaluation/icd_eval_v2_comparison.json`, `evaluation/icd_eval_results_governed.json`, and the persisted evaluation summaries under `evaluation/`.

MedScribe uses synthetic, non-PHI data. It is not a production clinical system.

## Why This Project Matters

Modern AI systems need more than prompt quality. They need operational reliability workflows:

- measurable behavior across runs
- structured outputs and validation boundaries
- evaluator-driven iteration
- traceable multi-stage execution
- governed final decisions
- RCA workflows when behavior diverges
- comparison discipline when model-backed stages vary

MedScribe demonstrates those capabilities in a compact, auditable Python system.

## Controlled Experiment: Reliability Improvements

| System | Schema Validity | Mapping Accuracy | Decision Stability | Policy Compliance |
| --- | --- | --- | --- | --- |
| Baseline | 4.17% | 0.00% | - | - |
| Fine-Tuned (v1) | 100.00% | 83.33% | - | - |
| Fine-Tuned (v2) | 100.00% | 95.83% | - | - |
| Governed | - | - | 100.00% | 100.00% |

The baseline failed first at structure. Fine-tuned v1 fixed schema validity but still exposed semantic mismatches. Fine-tuned v2 added targeted examples for exact ICD selection, mixed-status outputs, and cases that should remain reviewable instead of overconfident.

## System Overview

Runtime pipeline:

```text
Input -> Intake Parser -> Triage Engine -> Diagnosis Engine -> ICD Mapper -> Critic -> Governance Policy
```

The pipeline produces:

- structured intake data
- triage decisions
- diagnosis candidates
- ICD mappings
- critic scores and recommendations
- governance-enforced final decisions
- persisted run and evaluation artifacts

The repo contains two main operating surfaces:

- Service runtime: asynchronous execution, persisted run records, retrieval, comparison, and search.
- Offline evaluation: dataset-driven execution, scoring, fine-tuning comparison, governed comparison, and synthetic incident testing.

## Operational Reliability Workflows

MedScribe includes reliability workflows that mirror production AI engineering practices:

- Synthetic incident testing: known failure modes are exercised through controlled non-PHI cases.
- Trace-driven RCA: hosted traces and local artifacts are used to localize failures across stages.
- Evaluator refinement: coarse summary checks are expanded into behavior-specific reporting fields.
- Governance attribution: final decisions expose direct rule inputs, ignored upstream signals, fail drivers, and rule evaluations.
- Policy simulation: bounded policy changes are compared against baseline governed behavior.
- Before/after comparison workflows: incident summaries and trace outputs are used to compare behavior after targeted changes.
- Hybrid vs deterministic execution observations: deterministic runs provide fast local behavior; hybrid runs expose model spans, token/cost accounting, and live-model variability.

These workflows are intentionally operational. They focus on whether system behavior can be measured, explained, and compared.

## Observability And RCA

MedScribe integrates LangSmith tracing around the governed runtime and synthetic incident runner.

Observed trace surfaces include:

- synthetic incident root runs
- nested governed runtime runs
- stage spans for intake, triage, diagnosis, ICD mapping, critic, and governance
- `ChatOpenAI` child model spans in hybrid mode
- stage inputs and outputs
- runtime metadata and incident tags
- token, cost, and latency visibility in hybrid runs
- governance attribution metadata

The RCA workflow uses traces to answer operational questions:

- Did the runtime complete or fall back?
- Which stage changed the structured payload?
- Did the model return valid structured output?
- Did critic scoring match the evidence?
- Which governance rule drove the final status?
- Did reporting semantics match runtime behavior?
- Did an apparent comparison difference come from policy logic or upstream drift?

## Synthetic Incident Testing

The synthetic incident pack exercises reliability and debugging behaviors, not production clinical coverage.

Current incident classes include:

- Schema-valid semantic failure: JSON shape can be valid while semantics remain wrong.
- ICD specificity mismatch: coding specificity can drive critic and governance outcomes.
- Governance override: urgent triage can coexist with documentation or coding failure.
- Critic false-positive: low-risk narrative context can be weakened by structured representation loss and downstream broadening.
- Ambiguity / overconfidence: ambiguous evidence should not become an unqualified confident diagnosis.
- Policy-change divergence: policy metadata is visible, but causal comparison requires frozen upstream replay.
- Malformed-instruction resilience: malformed instruction text did not propagate as malformed downstream structure in inspected runs.
- Token/cost visibility workflows: deterministic synthetic runs lacked model accounting; hybrid runs exposed model spans, tokens, cost, and realistic latency.

The latest synthetic summary includes behavior-specific reporting fields for several incidents, such as ambiguity preservation, representation-loss caution amplification, malformed payload observation, and governance-vs-triage divergence.

## Governed Decision Pipeline

Governance is deterministic and rule-based. It consumes critic metrics and recommendations, then applies policy thresholds from `governance/policy_rules.json`.

The governance layer records:

- policy version
- governance version
- applied rules
- final status
- escalation requirement
- reason codes
- inputs used
- upstream signals ignored as direct governance inputs
- fail drivers
- rule evaluations
- upstream context summary

This makes final decisions inspectable without changing the underlying model output.

## What This Demonstrates For Production AI Systems

MedScribe demonstrates core AI systems engineering capabilities:

- AI reliability engineering: behavior is measured through datasets, summaries, and trace artifacts.
- Evaluation-loop design: failures produce targeted examples, comparisons, and reporting improvements.
- Trace-driven debugging: multi-stage behavior can be inspected from raw input through governance.
- Governed decision pipelines: model outputs are not accepted as final decisions without critic and policy layers.
- Measurable AI behavior: schema validity, mapping accuracy, stability, compliance, and incident summaries are persisted.
- Operational introspection: traces expose stage outputs, model spans, metadata, diagnostics, and governance attribution.
- Structured-output validation: node normalizers enforce expected output shapes and fallback paths.
- Debugging multi-stage AI systems: RCA artifacts show how upstream representation loss can affect downstream scoring.
- Observability-aware development: instrumentation and reporting were improved without changing runtime semantics.
- Production-style RCA workflows: investigations separate symptoms, code paths, trace evidence, hypotheses, fix surfaces, and validation strategy.

## Example: Before vs After Fine-Tuning

Baseline output on the sample comparison set returned non-standard status values. For `Pharyngitis`:

```json
{"mappings":[{"label":"Pharyngitis","icd_code":"J02.9","icd_label":"Acute pharyngitis, unspecified","status":"active"}]}
```

Fine-tuned v2 on the same input returned valid JSON in the required format:

```json
{"mappings":[{"label":"Pharyngitis","icd_code":"J02.9","icd_label":"Acute pharyngitis, unspecified","status":"OK"}]}
```

Governed outcome on the latest Phase 1B comparison run: policy compliance was `1.00`, decision stability was `1.00`, escalation rate was `1.00`, and override rate was `0.00`.

## Technical Proof Surface

A reviewer can verify the core reliability results from repo artifacts:

- `evaluation/icd_eval_summary.json`
- `evaluation/full_system_comparison.json`
- `evaluation/icd_eval_v2_comparison.json`
- `evaluation/icd_eval_scale_summary.json`
- `evaluation/icd_eval_results_governed.json`
- `fine_tuning/sample_outputs.json`
- `fine_tuning/sample_outputs_v2.json`
- `fine_tuning/fine_tune_job.json`
- `fine_tuning/fine_tune_job_v2.json`
- `evaluation/synthetic_incidents/last_run_summary.json`

Selected public RCA case studies are under `docs/case_studies/`. The curated ICD specificity evaluator case study shows how trace evidence, critic signals, governance output, and synthetic incident reporting were used to correct an evaluator mismatch without changing runtime behavior.

## Tech Stack

- Python
- FastAPI
- LangGraph
- LangSmith tracing
- OpenAI fine-tuning and inference
- JSON / JSONL artifact storage
- Custom evaluation framework
- Deterministic governance policy layer

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
- `graph/tracing.py`: fail-open trace wrapper and output capture
- `graph/llm_client.py`: hybrid LLM invocation surface

Evaluation and proof artifacts:

- `evaluation/eval_runner.py`
- `evaluation/score_runner.py`
- `evaluation/run_synthetic_incidents.py`
- `evaluation/`
- `fine_tuning/`
- `runs/`
- `docs/`

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

## Running The System

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

Synthetic incident run:

```bash
python evaluation/run_synthetic_incidents.py
```

Hybrid traced synthetic run, when credentials are configured:

```bash
MEDSCRIBE_EXECUTION_MODE=hybrid LANGCHAIN_TRACING_V2=true python evaluation/run_synthetic_incidents.py
```

## Security And Safety Considerations

- This system uses synthetic, non-PHI data.
- It is not intended for clinical use or production PHI workflows.
- It is designed for reliability, evaluation, governance, and observability experimentation.

### Input Validation

A lightweight input validation layer rejects common sensitive identifiers at the API boundary. The current checks cover email addresses, phone numbers, SSN-like patterns, and date-of-birth formats.

Verified behavior:

- unsafe inputs -> HTTP 400 rejected, no pipeline execution, no run created
- safe inputs -> normal execution path, run created

Examples:

- Unsafe input: `Contact me at john@example.com` -> rejected with 400
- Safe input: `I have a headache and mild fever` -> accepted, run created

### Threat Model

Trust boundaries:

- User input -> untrusted
- LLM output -> untrusted until parsed, normalized, evaluated, and governed
- Persisted artifacts -> sensitive

Primary risks:

- accidental PHI input
- malformed or adversarial input
- secret leakage
- misuse of model outputs
- over-trust in model-generated clinical text

Current controls:

- structured schema enforcement
- node-specific normalization
- critic scoring layer
- deterministic governance layer
- input validation demo layer
- persisted artifacts for auditability
- LangSmith trace output sanitization

Known gaps:

- no production PHI redaction layer
- no production encryption or access-control layer
- no adversarial security hardening
- no deterministic replay harness for causal policy comparison

Basic input validation is included for demonstration purposes and does not replace production-grade safeguards.
