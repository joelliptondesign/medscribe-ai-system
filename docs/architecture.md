# MedScribe Architecture

MedScribe is an operational AI systems engineering project for observing, evaluating, and governing multi-stage LLM workflows over synthetic, non-PHI medical-style inputs. The repo now contains two bounded workflow surfaces:

- CDI/coding-support workflow: a governed intake-to-ICD pipeline with critic scoring and policy governance.
- Denial operations workflow: an additive denial-management graph with routing, governance posture, operational alerts, and live trace support.

Both workflows are framed around reliability engineering rather than clinical production use.

## Workflow Overview

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
  -> operational route/posture/status payload + optional LangSmith traces
```

## CDI Workflow

The CDI workflow is orchestrated through the service runtime and graph nodes:

- `service/run_manager.py` coordinates run lifecycle and stage execution.
- `graph/graph_builder.py` defines the staged graph structure.
- `graph/nodes/` implements intake parsing, triage, diagnosis, ICD mapping, critic scoring, and governance.
- `governance/policy_rules.json` supplies deterministic governance thresholds.
- `data/runs.jsonl` and `runs/` hold persisted runtime and evaluation artifacts.

The CDI path can run deterministically for local development and tests, or in hybrid mode when model-backed stages are enabled. The final governed decision is produced by deterministic policy logic over structured upstream outputs.

## Denial Workflow

The denial workflow is additive and separate from the CDI pipeline:

- `graph/denial_graph.py` runs the denial node sequence.
- `graph/nodes/denials/denial_intake_parser.py` extracts denial context and evidence signals.
- `graph/nodes/denials/denial_classifier.py` classifies denial category.
- `graph/nodes/denials/recoverability_analyzer.py` interprets recoverability.
- `graph/nodes/denials/documentation_gap_analyzer.py` interprets documentation gaps.
- `graph/nodes/denials/routing_engine.py` produces the bounded operational route.
- `graph/nodes/denials/governance_policy.py` produces the governance posture.

Denial routing actions are bounded to `APPEAL`, `RESUBMIT`, `WRITE_OFF`, and `ESCALATE`. Governance postures are bounded to `SUPPORTED`, `LOW_CONFIDENCE`, `LOW_EVIDENCE`, `AMBIGUOUS`, and `HIGH_RISK`.

## Shared Reliability Conventions

Both workflows share reliability-oriented conventions:

- structured stage outputs
- deterministic local execution mode
- hybrid LLM workflows where configured
- fail-open trace helpers
- node diagnostics and fallback visibility
- synthetic benchmark corpora
- regression tests and smoke checks
- hosted/local evaluation support

The shared reliability layer is intentionally practical: it makes current behavior inspectable and testable without claiming production clinical readiness.

## LLM-Assisted Interpretation And Deterministic Control

LLM-assisted interpretation is used upstream, where the system can benefit from model-backed parsing or evidence interpretation. Deterministic routing and governance remain the downstream control boundary.

In the denial workflow, model use is limited to recoverability and documentation-gap interpretation. Routing and governance do not call the model. If provider access or response parsing fails, hybrid nodes fall back to deterministic interpretation and record the fallback in node diagnostics and Layer 1 operational debugging alerts.

In the CDI workflow, hybrid mode can support model-backed stages while governance remains deterministic and policy-driven.

## Operational Tracing

Tracing is implemented through `graph/tracing.py` and LangSmith-compatible run helpers. The helper is fail-open: workflow execution can continue even when tracing is disabled or unavailable.

Trace surfaces include:

- CDI governed runtime trace: `medscribe.governed_run`
- CDI stage spans: intake, triage, diagnosis, ICD mapping, critic, governance
- Denial graph trace: `medscribe.denial_graph`
- Denial node spans: denial intake, classifier, recoverability, documentation gaps, routing, governance
- Denial live operations traces created by `evaluation/run_denial_ops_traces.py`
- Hosted experiment and evaluator traces created by `evaluation/langsmith_experiment_runner.py`

Provider spans appear only when hybrid provider calls execute and LangChain/LangSmith tracing captures them.

## Operational Traces Versus Evaluator Traces

Operational traces show workflow execution. They contain case-level or input-level payloads, stage outputs, diagnostics, routing/governance decisions, alerts, and timing metadata.

Evaluator traces show scoring feedback. In hosted LangSmith experiments, evaluator rows receive `{run, example}` inputs and emit evaluator scores/comments. They are not the workflow graph and should not be used as the primary trace for diagnosing node behavior.

The live denial operations path avoids dataset/evaluator rows. It emits first-class operational traces for queue-style debugging.

## Layer 1 / 2 / 3 Operations Model

MedScribe uses a three-layer operational model:

- Layer 1 operational debugging: execution reliability, provider/runtime observability, completion status, fallback/degraded behavior, alerts, latency, token/cost availability, metadata completeness, and trace visibility.
- Layer 2 pairwise routing evaluation: operational usefulness before human review, comparing which output is safer or more useful for routing/governance review.
- Layer 3 routing-distribution experimentation: routing movement, governance distribution, threshold sensitivity, escalation saturation, and spillover behavior.

This model keeps execution health separate from workflow quality. For example, `ESCALATE` or `AMBIGUOUS` can be a valid conservative outcome, while `fallback_used` or `latency_spike` is an execution-health signal.

## Denial Operations Queue Semantics

The dedicated live denial operations runner uses operational queue semantics for RCM operations engineering and system-health investigation:

- Name: denial case id, such as `DENIAL-001`.
- Input: concise operational summary, such as `medical necessity | conflicting evidence`.
- Output: routing action only.
- Error: unrecovered execution anomaly only.
- Top-level metadata: investigation signals and availability flags.
- Diagnostic metadata: full drill-down payload, including evidence profile, documentation gaps, node diagnostics, provider metadata when available, raw token usage when available, alerts, timing, fallback reasons, and operational thresholds.
- Tags: workflow, mode, variant, ambiguity, conflicting evidence, specialist review, threshold boundary, fallback/degraded mode, and alert classes.

Governance posture is retained as diagnostic context and should not be treated as the same field as routing action.

## Hosted And Local Execution

Local execution supports deterministic development, regression tests, smoke checks, local summaries, and pairwise reports.

Hosted execution uses LangSmith-compatible dataset/evaluate paths when credentials, SDK support, and network access are available. Hosted readiness should be checked before interpreting hybrid smoke results. Provider or network failure must be distinguished from workflow regression: degraded fallback caused by provider/network failure is not automatically a workflow defect if deterministic fallback completes and runtime `error` remains null.

## Boundaries

This repository does not implement payer outcome simulation, downstream human behavior simulation, or production clinical decisioning. Results are synthetic operational evidence for AI reliability engineering.
