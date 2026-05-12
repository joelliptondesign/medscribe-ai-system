# Data Contracts

MedScribe uses lightweight, inspectable data contracts for runtime records, operational traces, evaluator payloads, and benchmark artifacts. The contracts are intentionally practical: they make current workflow behavior observable without claiming production clinical readiness.

## Service Runtime Artifacts

Path:
`data/runs.jsonl`

Format:
append-only JSON Lines. Each line is one service run record with lifecycle state and persisted execution fields available for that run.

Primary CDI runtime fields include:

- parsed intake data
- triage output
- diagnosis candidates
- ICD mappings
- critic scores and recommendation
- deterministic governance decision
- run status and timing metadata

## Evaluation Run Artifacts

Path:
`runs/{run_id}/scored_results.json`

Structure:

```json
{
  "results": []
}
```

Each result item is a scored evaluation case. Aggregation tooling reads these artifacts independently from the service runtime store.

## CDI Workflow Contract

The CDI workflow consumes raw synthetic input text and emits a governed run record. The record is built from staged outputs:

- intake parser output
- triage output
- diagnosis output
- ICD mapping output
- critic metrics
- governance summary

Governance is deterministic and policy-driven. Raw model-backed outputs are upstream evidence, not the final authority.

## Denial Workflow Contract

The denial workflow consumes a synthetic denial case payload and emits a bounded operational routing payload.

Primary outcome fields:

- `routing_action`: operational route
- `governance_posture`: confidence/evidence posture
- `status`: execution-health status
- `error`: unrecovered execution anomaly, or null
- `alerts`: Layer 1 operational alerts
- `metadata`: top-level operational metadata
- `node_diagnostics`: per-node execution diagnostics

Routing actions are bounded:

- `APPEAL`
- `RESUBMIT`
- `WRITE_OFF`
- `ESCALATE`

Governance postures are bounded:

- `SUPPORTED`
- `LOW_CONFIDENCE`
- `LOW_EVIDENCE`
- `AMBIGUOUS`
- `HIGH_RISK`

These fields are distinct. `routing_action` is the next operational disposition. `governance_posture` explains evidence/confidence posture. `status` describes execution health.

## Runtime Status Semantics

Runtime status values describe execution health:

- `completed`: workflow completed without degraded fallback.
- `degraded`: workflow completed with fallback or degraded execution behavior.
- `failed`: workflow did not complete a recoverable execution path.

Execution health is separate from routing quality. A conservative route such as `ESCALATE` is not an execution error by itself.

## Layer 1 Alert Semantics

Layer 1 alerts are reliability signals. Current alert classes include latency, verbosity, token, cost, first-token latency, fallback, degraded mode, provider failure, missing metadata, invalid output schema, malformed payload, and incomplete trace signals.

Warning and informational alerts remain in `alerts`, metadata, and tags. They do not populate runtime `error` unless execution truly fails or a critical unrecovered anomaly is present.

## Operational Trace Payloads

Operational traces represent workflow execution. For denial live operations, table-facing fields are intentionally concise:

- trace name: case id, such as `DENIAL-001`
- input: concise operational summary
- output: routing action only
- error: unrecovered execution anomaly only
- tags: investigation signals
- top-level metadata: availability flags and operational signals
- diagnostic metadata: full drill-down details

Diagnostic metadata can include full case context, evidence profile, documentation gaps, node diagnostics, provider metadata when available, raw token usage when available, latency, first-token availability, cost metadata when available, fallback reasons, alert objects, and operational thresholds.

## Evaluator Payloads

Evaluator payloads are not operational traces. Hosted evaluator rows receive the target run and dataset example, commonly shaped as `{run, example}`, and emit feedback such as:

```json
{
  "key": "denial_routing_action_match",
  "score": true,
  "comment": "expected and actual routing matched"
}
```

Evaluator comments should not be interpreted as workflow node output. Operational route/posture/status fields remain on the target run output and workflow traces.

## Operational Metadata Versus Diagnostic Metadata

Top-level operational metadata is optimized for filtering and queue inspection. It includes fields such as workflow, variant, `llm_used`, fallback/degraded flags, alert count, max alert severity, evidence strength, ambiguity level, conflicting-evidence signal, specialist-review signal, threshold-boundary signal, recoverability, governance posture, token/cost availability, and first-token availability.

Diagnostic metadata is optimized for debugging. It can contain larger payloads such as full case details, node diagnostics, provider metadata, raw usage metadata, timing, alert objects, fallback reasons, and thresholds.

This separation keeps queue rows readable while preserving the evidence needed for trace-driven investigation.

## LLM-Assisted Interpretation Outputs

LLM-assisted interpretation may produce structured recoverability or documentation-gap outputs when provider-backed calls succeed. When provider access, network access, or response normalization fails, the workflow uses deterministic fallback behavior and records fallback diagnostics.

Provider token, cost, and first-token fields are preserved only when exposed by provider/tooling. Missing provider telemetry is represented through explicit availability flags rather than inferred values.

## Deterministic Governance Boundaries

Deterministic governance consumes structured upstream outputs and applies bounded rules. In the denial workflow, routing and governance nodes do not invoke provider calls. In the CDI workflow, final governance is rule-based over critic and pipeline outputs.

This boundary keeps LLM-assisted interpretation upstream and final operational control deterministic.

## Separate Stores

The repo intentionally keeps stores and payloads separate:

- service runtime records: `data/runs.jsonl`
- evaluation artifacts: `runs/` and `evaluation/*results*.json`
- benchmark corpora: `evaluation/operational_benchmark_cases.json`, `evaluation/denial_benchmark_cases.json`, and synthetic incident datasets
- local-only operational continuity: `.local_sitrecs/`
- local-only telemetry: `telemetry/`

Operational traces can inform future curated datasets, but traces are not automatically promoted into benchmark corpora.
