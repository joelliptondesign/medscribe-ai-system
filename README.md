# MedScribe: Production-Style AI Reliability Engineering

MedScribe is a practical AI systems engineering project for building, observing, evaluating, and governing a multi-stage LLM workflow. It is intentionally framed around operational reliability rather than model novelty: trace-driven debugging, structured evaluation, governance controls, synthetic incidents, regression analysis, and the tradeoffs between regenerated reruns and controlled comparison.

The system uses synthetic, non-PHI medical-style inputs to exercise realistic AI operations workflows. It is not a clinical product and does not claim production clinical readiness.

## Project Overview

MedScribe turns unstructured intake text into a governed, auditable workflow:

```text
Input
  -> Intake Parser
  -> Triage Engine
  -> Diagnosis Engine
  -> ICD Mapper
  -> Critic
  -> Governance Policy
  -> Persisted Artifacts + LangSmith Traces
```

The repository demonstrates:

- FastAPI service runtime with persisted run artifacts
- LangGraph orchestration for staged execution
- hybrid LLM execution with opt-in LangSmith tracing
- deterministic governance over model-backed stages
- evaluation harnesses for schema validity, ICD mapping, policy compliance, and decision stability
- synthetic incident testing for operational failure modes
- trace-driven RCA reports and public case-study artifacts
- regeneration-only workflow studies covering observability, runtime stabilization, regression, spillover, and policy intervention

## Why This Project Matters

Production AI systems fail in ways that do not show up in a single happy-path demo. They need operational workflows that can answer:

- Did the model return valid structured output?
- Which stage changed the payload?
- Did token usage explain latency?
- Did governance change because policy changed, or because upstream model output drifted?
- Did aggregate evaluator metrics hide internal decision movement?
- Did a local fix spill over into adjacent incident classes?

MedScribe demonstrates those questions with implementation depth: traceable runtime stages, persisted evidence, evaluation summaries, synthetic incidents, governance attribution, and regression-oriented documentation.

## Controlled Experiment / Reliability Metrics

Verified metrics are preserved from repository artifacts under `evaluation/`.

| System | Schema Validity | Mapping Accuracy | Decision Stability | Policy Compliance | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 4.17% | 0.00% | - | - | `evaluation/icd_eval_summary.json` |
| Fine-Tuned v1 | 100.00% | 83.33% | - | - | `evaluation/icd_eval_summary.json` |
| Fine-Tuned v2 | 100.00% | 95.83% | - | - | `evaluation/icd_eval_v2_comparison.json` |
| Governed | - | - | 100.00% | 100.00% | `evaluation/icd_eval_results_governed.json` |

Fine-tuning is included as one reliability experiment, not the central thesis. The broader project focuses on what happens after a model produces output: validation, scoring, governance, traceability, regression testing, and operational RCA.

## System Overview

The pipeline produces and persists:

- structured intake data
- triage decision and rationale
- diagnosis candidates
- ICD mappings
- critic metrics and recommendation
- deterministic governance result
- run lifecycle status
- comparison and evaluation artifacts
- trace metadata and stage outputs when hybrid tracing is enabled

Two execution modes are supported:

- deterministic mode for local development and predictable test execution
- hybrid mode for model-backed stages, token/cost accounting, and hosted trace inspection

## Operational Reliability Workflow

The repository models a mature intervention sequence used in AI operations:

| Stage | Goal | Typical Actions | Operational Risks |
| --- | --- | --- | --- |
| Stage 1: Observability | Make behavior inspectable | Add trace metadata, stage timing, critic/governance snapshots, output-shape summaries | More fields can be over-interpreted as causality |
| Stage 2: Runtime Stabilization | Reduce verbosity, token/cost spread, and output noise | Tighten prompts, set output caps, add model metadata | Semantic drift, prompt-token overhead, reduced debugging detail |
| Stage 2.5: Regression & Spillover | Detect movement outside the target incident | Run full synthetic pack, compare decisions, reason codes, latency, token/cost | Stable aggregate metrics can hide internal drift |
| Stage 3: Policy/Governance Intervention | Change downstream interpretation boundary | Adjust a narrow governance rule or threshold | Spillover, reviewer workload shifts, false-positive/false-negative movement |

This workflow is evidence-first: observe, stabilize, regress, then intervene semantically only after blast-radius inspection.

## Observability and RCA

MedScribe integrates LangSmith around the governed runtime and synthetic incident runner.

Observed trace surfaces include:

- synthetic incident root runs
- nested governed runtime runs
- stage spans for intake, triage, diagnosis, ICD mapping, critic, and governance
- `ChatOpenAI` child spans in hybrid mode
- stage inputs and outputs
- incident metadata and run IDs
- token, cost, and latency data
- governance attribution metadata

RCA workflows use these traces to inspect:

- fallback and degraded execution paths
- structured-output validity
- stage-level payload drift
- critic metric changes
- governance rule evaluations
- token/cost versus latency relationships
- policy metadata versus policy-causal attribution

## Synthetic Incident Testing

Synthetic incidents are non-PHI operational test cases. They are designed to exercise reliability and debugging behavior, not clinical coverage.

| Incident | Operational Theme | Key Finding |
| --- | --- | --- |
| `MS-SYN-001` | Schema-valid semantic failure | Valid JSON can still miss the clinically important emphasis. |
| `MS-SYN-002` | ICD specificity and latency/token anomaly | Same-token triage spans showed materially different latency; later runtime changes complicated historical reproduction. |
| `MS-SYN-003` | Governance override | Urgent triage and coding/documentation failure can diverge. |
| `MS-SYN-004` | Critic false positive | Representation loss can amplify caution downstream. |
| `MS-SYN-005` | Ambiguity / overconfidence | Ambiguous evidence should remain reviewable rather than overconfident. |
| `MS-SYN-006` | Policy-change divergence | Regenerated reruns made policy attribution visible but not causally isolated. |
| `MS-SYN-007` | Malformed downstream payload resilience | Malformed instruction language did not propagate as malformed structured output in inspected runs. |

The synthetic pack supports trace-driven RCA, regression checks, spillover analysis, and operational comparison after runtime or governance changes.

## Evaluation and Governance

Evaluation surfaces include:

- ICD mapping comparison
- schema validity scoring
- governed pipeline comparison
- synthetic incident high-level matching
- behavior-specific synthetic reporting
- smoke and lifecycle tests

Governance is deterministic and rule-based. It consumes critic metrics and recommendations, then applies thresholds from `governance/policy_rules.json`.

The governance layer records:

- policy version
- governance version
- applied rules
- final status
- escalation requirement
- reason codes
- inputs used
- upstream inputs ignored as direct governance inputs
- fail drivers
- rule evaluations
- upstream context summary

This makes final decisions inspectable without treating raw model output as the final authority.

## Replay & Regeneration Findings

Phase 4 explored regeneration-only debugging across policy-divergence and latency/token incidents.

Operational findings:

- Regenerated reruns are useful for live operational realism.
- Isolated reruns reduce pack-level noise but do not create replay.
- Observability improves inspection speed but does not prove causality.
- Runtime stabilization can reduce verbosity while shifting semantic distributions.
- Regression runs must inspect incident-level decisions, not only aggregate pass/fail counts.
- Policy intervention can improve a target incident while spilling over to adjacent incident classes.
- Historical latency incidents are difficult to reinterpret after runtime and governance mutations.

Replay/comparison limits observed:

- upstream state was not frozen
- critic metrics and reason-code wording drifted across regenerated runs
- same-token latency variation persisted without a trustworthy causal explanation
- stable aggregate metrics hid internal governance movement
- current reruns reflected the current runtime, not necessarily the historical runtime

Artifact-assisted workflows previously improved clarity by holding upstream artifacts constant. Regeneration-only workflows were better for current-state realism, weaker for controlled causal comparison.

## Production-Style Engineering Practices

This repository demonstrates production-relevant engineering practices:

- asynchronous FastAPI runtime
- LangGraph staged orchestration
- schema-oriented node contracts
- fallback diagnostics
- persisted run records
- retrieval, comparison, and search surfaces
- deterministic governance layer
- opt-in hybrid model execution
- LangSmith trace instrumentation
- evaluation datasets and scored artifacts
- synthetic incident pack
- regression and spillover reports
- public RCA case studies
- test coverage for edge cases, execution modes, hybrid opt-in, and run lifecycle

The project is intentionally bounded: it demonstrates reliability workflows with synthetic data rather than claiming production healthcare deployment.

## Technical Architecture

Primary service boundary:

- `service/main.py`: FastAPI app entry point
- `service/api.py`: request handlers
- `service/run_manager.py`: execution orchestration
- `service/storage.py`: runtime artifact persistence
- `service/retrieval.py`: stored-run search and retrieval
- `service/tools.py`: tool-call surface

Pipeline implementation:

- `graph/graph_builder.py`: LangGraph pipeline definition
- `graph/state.py`: shared workflow state
- `graph/config.py`: execution mode and model configuration
- `graph/llm_client.py`: hybrid LLM invocation surface
- `graph/tracing.py`: fail-open trace wrapper and output capture
- `graph/nodes/`: intake, triage, diagnosis, ICD mapping, critic, governance

Evaluation and evidence:

- `evaluation/eval_runner.py`
- `evaluation/score_runner.py`
- `evaluation/run_icd_eval.py`
- `evaluation/run_governed_pipeline.py`
- `evaluation/run_synthetic_incidents.py`
- `evaluation/synthetic_incidents/`
- `docs/case_studies/`
- `runs/`

## API Surface

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

Example request:

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"input_text":"I have had fever, cough, and sore throat for two days."}'
```

## Running The System

Install and start the API:

```bash
pip install -r requirements.txt
uvicorn service.main:app --reload
```

Basic local checks:

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/openapi.json
```

Offline evaluation:

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

Selected verification:

```bash
python tests/test_execution_mode.py
python tests/test_hybrid_mode_opt_in.py
python tests/test_run_lifecycle.py
```

## Safety and Scope

MedScribe is a synthetic reliability engineering project.

Scope boundaries:

- synthetic, non-PHI data only
- not intended for clinical use
- not a production PHI workflow
- not a substitute for clinical review
- not a production security or access-control implementation

Current controls:

- structured schema enforcement
- node-specific normalization
- fallback diagnostics
- critic scoring layer
- deterministic governance layer
- input validation demo layer
- persisted artifacts for auditability
- trace instrumentation and metadata

Known gaps:

- no production PHI redaction layer
- no production encryption or access-control layer
- no adversarial security hardening
- no deterministic replay harness for causal policy comparison

The project is best read as a compact proof of operational AI engineering discipline: make behavior observable, evaluate it with artifacts, govern it deterministically, test incidents synthetically, and document uncertainty when regenerated workflows cannot provide causal proof.
