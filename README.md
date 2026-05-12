# MedScribe: Operational AI Systems Engineering Sandbox

MedScribe is a production-style AI systems engineering sandbox for building, observing, evaluating, and governing multi-stage LLM workflows over synthetic, non-PHI medical-style inputs. It is not a clinical product. It is a compact demonstration of operational AI maturity: LLM-assisted interpretation, deterministic routing and governance, trace-driven debugging, synthetic incidents, hosted/local evaluation, and regression-oriented experimentation.

The repo contains two separate domain workflows:

- CDI/coding support: a governed intake-to-ICD workflow with critic scoring and rule-based downstream governance.
- Denial operations: an additive denial-management graph with bounded routing actions, governance postures, operational alerts, and live operations traces.

Both workflows are designed to make modern AI operations questions inspectable: what changed, where it changed, whether execution was healthy, whether routing moved, and whether a governance boundary stayed deterministic.

## What This Demonstrates

- Operational AI systems engineering for governed multi-stage workflows.
- Hybrid LLM workflows with LLM-assisted interpretation where it is useful.
- Deterministic routing and rule-based downstream governance after interpretation.
- Observability-first AI ops with LangSmith-compatible traces and stage diagnostics.
- Layer 1/2/3 operational analysis for execution health, pairwise routing usefulness, and routing-distribution movement.
- Hosted and local regenerated operational experimentation.
- Operational queue semantics for live denial trace inspection.
- Regression tests, synthetic incidents, benchmark corpora, and documented operational findings.

## Architecture At A Glance

```text
CDI input text
  -> intake parser
  -> triage engine
  -> diagnosis engine
  -> ICD mapper
  -> critic
  -> deterministic governance policy
  -> persisted run record + optional LangSmith traces

Denial case payload
  -> denial intake parser
  -> denial classifier
  -> recoverability analyzer
  -> documentation-gap analyzer
  -> deterministic routing engine
  -> deterministic governance policy
  -> route/posture/status payload + optional operational traces
```

Shared reliability conventions include structured stage outputs, deterministic local execution, hybrid LLM execution where configured, fail-open tracing, node diagnostics, fallback visibility, synthetic benchmark corpora, and smoke/regression tests.

## CDI Workflow

The CDI workflow turns unstructured synthetic intake text into a governed run record:

- parsed intake data
- triage output
- diagnosis candidates
- ICD mappings
- critic metrics and recommendation
- deterministic governance result
- persisted runtime/evaluation artifacts

Primary implementation points:

- `service/run_manager.py`
- `graph/graph_builder.py`
- `graph/nodes/`
- `governance/policy_rules.json`
- `data/runs.jsonl`

The CDI path supports local deterministic execution for controlled evaluation and hybrid LLM workflows for model-backed stages when configured.

## Denial Operations Workflow

The denial workflow is additive and separate from the CDI workflow. It operates on synthetic denial cases and produces bounded operational routing outputs.

Routing actions:

- `APPEAL`
- `RESUBMIT`
- `WRITE_OFF`
- `ESCALATE`

Governance postures:

- `SUPPORTED`
- `LOW_CONFIDENCE`
- `LOW_EVIDENCE`
- `AMBIGUOUS`
- `HIGH_RISK`

Primary implementation points:

- `graph/denial_graph.py`
- `graph/nodes/denials/`
- `graph/operational_alerts.py`
- `evaluation/denial_benchmark_cases.json`
- `evaluation/run_denial_ops_traces.py`

LLM-assisted denial interpretation is limited to recoverability and documentation-gap analysis. Routing and governance remain deterministic.

## LLM-Assisted Interpretation

MedScribe keeps LLM-assisted interpretation upstream and deterministic control downstream.

In denial operations, `recoverability_analyzer` and `documentation_gap_analyzer` may call the configured provider in hybrid mode. They produce structured intermediate evidence signals only. They do not choose the final route or governance posture.

If provider access, network access, JSON parsing, or response validation fails, the workflow uses deterministic fallback behavior and records fallback/degraded diagnostics. A degraded fallback is not automatically a workflow defect when the route completes and runtime `error` remains null.

## Deterministic Routing And Governance

Final operational decisions are not raw model output.

CDI governance is rule-based over critic and pipeline outputs. Denial routing and governance are deterministic nodes over structured denial state. This keeps the final operational boundary inspectable even when upstream interpretation uses an LLM.

Runtime status is separate from workflow quality:

- `completed`: execution completed without degraded fallback.
- `degraded`: execution completed with fallback or degraded behavior.
- `failed`: execution did not complete a recoverable path.

Conservative outcomes such as `ESCALATE` or `AMBIGUOUS` are not execution errors by themselves.

## Layer 1 / 2 / 3 Operational Model

MedScribe uses a three-layer model for operational AI analysis:

| Layer | Purpose | Examples |
| --- | --- | --- |
| Layer 1 operational debugging | Execution health and observability | latency, fallback, degraded mode, provider/runtime visibility, trace completeness, token/cost availability |
| Layer 2 pairwise routing evaluation | Safer or more useful routing/governance comparison before human review | baseline vs variant preference, ambiguity handling, recoverability alignment, unsafe write-off avoidance |
| Layer 3 routing-distribution experimentation | Aggregate routing movement and spillover inspection | route/posture distribution, threshold sensitivity, escalation saturation, boundary-case movement |

Observed denial-ops findings in the repo include latency/fallback visibility, escalation saturation, threshold sensitivity, limited LLM-assisted route movement at boundary cases, and distribution movement under strengthened variants. These are operational observations over synthetic corpora, not clinical or payer-outcome claims.

## LangSmith Tracing And Evaluation

MedScribe separates operational traces from evaluator traces.

Operational traces show workflow execution: inputs, stage spans, route/posture/status outputs, diagnostics, alerts, timing, provider metadata when available, and fallback/degraded behavior.

Evaluator traces show scoring feedback. Hosted LangSmith evaluator rows receive `{run, example}` and emit evaluator-specific score/comment outputs. They are not the workflow graph and should not be used as the primary trace for node-level debugging.

Relevant tooling:

- `evaluation/langsmith_experiment_runner.py`: hosted/local benchmark execution, variants, and pairwise summaries.
- `evaluation/run_denial_ops_traces.py`: live denial operational traces without dataset/evaluator rows.
- `docs/LANGSMITH_DATASET_EXPERIMENT_POC.md`
- `docs/LANGSMITH_TRACE_DIAGNOSTIC_20260511.md`

## Operational Queue Semantics

Live denial operations traces are for RCM operations engineering, system-health investigation, and routing/governance debugging. They are not claim adjudication queues.

Queue-facing semantics:

- Name: denial case id, such as `DENIAL-001`.
- Input: concise operational denial summary, such as `medical necessity | conflicting evidence`.
- Output: routing action only.
- Error: unrecovered execution anomaly only.
- Tags: operational investigation cues such as ambiguity, conflicting evidence, specialist review, threshold boundary, fallback/degraded mode, and alert classes.
- Metadata: compact operational signals and availability flags.
- `diagnostic_metadata`: drill-down details such as evidence profile, documentation gaps, node diagnostics, alerts, provider metadata when available, token usage when available, timing, fallback reasons, and operational thresholds.

This split keeps live trace tables readable while preserving debugging evidence.

## Hosted And Local Experimentation

Local deterministic execution supports controlled operational evaluation, regression testing, and smoke checks without requiring provider or LangSmith access.

Hosted execution uses LangSmith-compatible dataset/evaluate paths when credentials, SDK support, and network access are available. Hosted readiness should be checked before interpreting hybrid smoke results. Provider or network failures should be distinguished from workflow regressions.

Experiment surfaces include:

- CDI and denial benchmark corpora.
- Synthetic incident packs.
- Hosted formal experiments when available.
- Local summaries when hosted execution is unavailable.
- Pairwise routing evaluation.
- Routing-distribution reporting.

Regenerated runs may vary between executions. They are useful for current-state operational evidence, observability, and regression awareness.

## Evidence And Metrics

Verified CDI reliability metrics are preserved from repository artifacts under `evaluation/`.

| System | Schema Validity | Mapping Accuracy | Decision Stability | Policy Compliance | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 4.17% | 0.00% | - | - | `evaluation/icd_eval_summary.json` |
| Fine-Tuned v1 | 100.00% | 83.33% | - | - | `evaluation/icd_eval_summary.json` |
| Fine-Tuned v2 | 100.00% | 95.83% | - | - | `evaluation/icd_eval_v2_comparison.json` |
| Governed | - | - | 100.00% | 100.00% | `evaluation/icd_eval_results_governed.json` |

Denial operational findings are documented in `docs/HOSTED_DENIAL_EXPERIMENT_ANALYSIS.md` and `docs/LAYERED_AI_OPS_SYNTHESIS.md`, including routing distributions, governance posture distributions, pairwise preferences, latency warnings, token metadata availability, and boundary-case movement.

## Repository Map

- `service/`: FastAPI runtime, run lifecycle, storage, retrieval, tool surface.
- `graph/`: CDI graph, denial graph, LLM client, tracing, diagnostics, operational alerts.
- `graph/nodes/`: CDI workflow nodes.
- `graph/nodes/denials/`: denial workflow nodes.
- `evaluation/`: benchmark runners, hosted/local experiment runner, denial ops trace runner, corpora, scored artifacts.
- `evaluation/synthetic_incidents/`: synthetic incident datasets and summaries.
- `docs/`: architecture, data contracts, LangSmith diagnostics, operational analyses, case studies.
- `governance/`: policy rules for deterministic governance.
- `tests/`: regression tests for edge cases, execution modes, hybrid opt-in, lifecycle, and denial graph behavior.
- `runs/` and `data/`: persisted example artifacts and runtime records.

`REPO_TREE.txt` lists the public repository inventory. Local-only SITRECs and telemetry remain outside public artifacts.

## Running Workflows

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn service.main:app --reload
```

Run offline CDI evaluation:

```bash
python evaluation/eval_runner.py
python evaluation/score_runner.py
```

Run CDI synthetic incidents:

```bash
.venv/bin/python evaluation/run_synthetic_incidents.py --workflow cdi
```

Run deterministic denial operations smoke:

```bash
.venv/bin/python evaluation/run_denial_ops_traces.py --limit 1 --execution-mode deterministic --no-tracing
```

Run hybrid denial operations smoke:

```bash
.venv/bin/python evaluation/run_denial_ops_traces.py --limit 1 --execution-mode hybrid --no-tracing
```

Run hosted/local operational benchmark tooling:

```bash
.venv/bin/python evaluation/langsmith_experiment_runner.py --workflow denial --experiment-label baseline --limit 2 --local-only
.venv/bin/python evaluation/langsmith_experiment_runner.py --workflow denial --pairwise --pairwise-variant routing_sensitivity_variant --limit 2 --local-only
```

Run focused regression checks:

```bash
.venv/bin/python -m pytest tests/test_edge_cases.py tests/test_execution_mode.py tests/test_hybrid_mode_opt_in.py tests/test_run_lifecycle.py tests/test_denial_graph.py
```

## Boundaries

MedScribe is an operational experimentation environment, not production healthcare software.

Current boundaries:

- synthetic, non-PHI datasets only
- no production PHI handling layer
- no production clinical decisioning claim
- no clinical correctness claim
- no payer outcome simulation
- no downstream human behavior simulation
- no deterministic replay implementation
- no production security or access-control implementation

Hosted traces improve inspection and debugging. Local deterministic execution supports controlled evaluation. Regenerated operational experiments provide current-state evidence, not frozen historical reproduction.

## Why This Repo Exists

Modern AI systems fail in operationally specific ways: malformed outputs, hidden routing drift, degraded provider paths, trace/evaluator confusion, latency and token visibility gaps, and governance movement hidden by aggregate scores.

MedScribe exists to make those failure modes concrete and inspectable. It shows how an AI engineering team can combine LLM-assisted interpretation with deterministic downstream control, then use traces, evaluations, synthetic incidents, pairwise comparison, and distribution reporting to understand behavior before treating a workflow as reliable.
