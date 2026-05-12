# LangSmith Trace Diagnostic 2026-05-11

## Purpose

Diagnose current denial-workflow LangSmith trace structure, evaluator/workflow separation, LLM-assisted interpretation visibility, output-column behavior, tags/metadata visibility, and provider telemetry capture without changing workflow behavior.

## Current Observed Problem

Some LangSmith rows that look like denial workflow rows are evaluator rows. Their run names include values such as `denial_final_route_present`, `denial_completed_without_fallback`, `denial_governance_posture_match`, `denial_routing_action_match`, and `denial_high_level_match`. These rows receive `{run, example}` as evaluator inputs and emit evaluator feedback such as score/comment. They are not the operational denial graph trace.

Separately, alert classes such as `latency_spike` can appear as evaluator output/comment when Layer 1 warning alerts are copied into the benchmark result's `error` field. The hosted evaluator wrappers treat any non-null `result["error"]` as an evaluator failure input.

## Trace Type Taxonomy

### Hosted Experiment/Evaluate Runs

- Code path: `evaluation/langsmith_experiment_runner.py::run_hosted_experiment`.
- Hosted experiment prefix: `medscribe-<workflow>-operational-<experiment_label>`.
- Target function name: `medscribe_<workflow>_<experiment_label>_operational_benchmark`.
- Input shape for denial target: `{"case_payload": <case>}`.
- Output shape for target: the full `_run_case_result(...)` dictionary, including `output`, `routing_action`, `governance_posture`, `status`, `error`, `alerts`, `metadata`, `tags`, and nested `record`.
- Metadata shape: experiment-level metadata includes dataset name, workflow, experiment label, case limit, and runner path.
- Expected table behavior: experiment rows show target outputs and evaluator feedback. If multiple scalar output fields exist, the UI may prioritize or display fields differently from the intended operational `output`.
- Child spans: target execution can contain child traces if runtime tracing is enabled and child spans are linked by LangSmith.
- LLM spans: possible only during LLM-assisted denial interpretation when provider calls execute and LangChain tracing captures them.

### Operational Denial Workflow Traces

- Code path: `graph/denial_graph.py::run_denial_graph`.
- Current workflow-level trace name: `medscribe.denial_graph`.
- Current node trace names: `medscribe.denial.denial_intake_parser`, `medscribe.denial.denial_classifier`, `medscribe.denial.recoverability_analyzer`, `medscribe.denial.documentation_gap_analyzer`, `medscribe.denial.routing_engine`, and `medscribe.denial.governance_policy`.
- Input shape for workflow trace: `{"case_payload": <case>}`.
- Node input shape: `{"state": <state before node>}`.
- Workflow output shape: route, governance posture, status, degraded/fallback flags, and alert count.
- Node output shape: changed fields and node diagnostic.
- Metadata shape: workflow and node names; current metadata is concise and does not include the full benchmark result metadata.
- Tag shape: `medscribe`, `workflow:denial`, `denial-graph`, and per-node `stage:<node>`.
- Expected table behavior: these are operational traces and should be the rows used to inspect the denial task graph.
- Child spans: yes, the graph span should contain the node spans when tracing links correctly.
- LLM spans: only expected under LLM-assisted interpretation nodes if provider calls execute and LangChain tracing links the model run.

### CDI Runtime Traces

- Code path: `service/run_manager.py::execute`.
- Workflow-level trace name: `medscribe.governed_run`.
- Stage names: `medscribe.intake_parser`, `medscribe.triage_engine`, `medscribe.diagnosis_engine`, `medscribe.icd_mapper`, `medscribe.critic`, `medscribe.governance_policy`.
- Input shape: raw input text.
- Output shape: full CDI run record.
- Child spans: yes, CDI stages are manually traced.
- LLM spans: possible inside hybrid tool calls if LangChain tracing captures provider calls.

### Synthetic Incident Traces

- CDI synthetic incident root: `medscribe.synthetic_incident`.
- Denial synthetic incident root: `medscribe.denial_synthetic_incident`.
- Input shape: incident id plus input text or denial case payload.
- Output shape: incident summary, primary output, Layer 1 metadata, tags, alerts, and high-level match flag.
- Child spans: CDI incidents call `execute(...)`, which can create child stage spans. Denial incidents call `run_denial_graph(...)`, which can create denial graph/node spans.

### Evaluator Traces

- Code path: hosted evaluator closures in `evaluation/langsmith_experiment_runner.py`.
- Evaluator names include `denial_high_level_match`, `denial_routing_action_match`, `denial_governance_posture_match`, `denial_completed_without_fallback`, and `denial_final_route_present`.
- Input shape: LangSmith evaluator input, effectively `{run, example}`.
- Output shape: `{"key": <evaluator_name>, "score": <bool>, "comment": <string>}`.
- Metadata/tag shape: governed by LangSmith evaluate feedback machinery, not by denial graph code.
- Expected table behavior: these rows show evaluator feedback, not the denial task graph.
- Child spans: no denial graph child spans should be expected under evaluator rows.
- LLM spans: none expected for current deterministic evaluator helpers.

## Evaluator vs Workflow Diagnosis

Rows named `denial_final_route_present`, `denial_completed_without_fallback`, `denial_governance_posture_match`, `denial_routing_action_match`, and `denial_high_level_match` are evaluator traces/feedback runs. Their input appears as `{example, run}` because LangSmith evaluators are invoked with the target run and dataset example. They do not show the denial task graph because they are not the target workflow run and do not call `run_denial_graph`.

The operational denial workflow trace is currently named `medscribe.denial_graph`, not `denial.workflow.hybrid` or `run_denial_graph`.

## Operational Workflow Trace Visibility

The code now emits a workflow-level manual trace at `graph/denial_graph.py::run_denial_graph` using `trace_span("medscribe.denial_graph", ...)`.

The exact instrumentation point exists inside `run_denial_graph`, wrapping the node loop. Node-level spans are emitted around each denial node call.

Hosted smoke output confirmed formal hosted experiment creation with a target function name of `medscribe_denial_baseline_operational_benchmark` and experiment prefix `medscribe-denial-operational-baseline`. A direct post-run `list_runs` query for the hosted project was blocked by LangSmith authentication, so child-span visibility was not independently enumerated from the API in this pass.

Expected location:

- Project/session created by LangSmith evaluate, named with `medscribe-denial-operational-baseline-*`.
- Target run: `medscribe_denial_baseline_operational_benchmark`.
- Operational child trace: `medscribe.denial_graph`.
- Node child traces: `medscribe.denial.<node_name>`.

## LLM Span Visibility Diagnosis

LLM-assisted interpretation calls are attempted only in:

- `graph/nodes/denials/recoverability_analyzer.py`
- `graph/nodes/denials/documentation_gap_analyzer.py`

Local direct diagnostics showed:

- deterministic mode: both LLM-assisted interpretation nodes report `execution_mode=deterministic`, `live_call_attempted=false`, and no fallback.
- hybrid mode without provider access: both nodes report `execution_mode=hybrid`, `live_call_attempted=true`, `live_call_returned=false`, `fallback_triggered=true`, and deterministic fallback.
- hosted hybrid smoke: benchmark result reported `llm_used=true`, `routing_action=APPEAL`, and `governance_posture=SUPPORTED`.

Provider calls are made through `graph/llm_client.py::invoke_json`, which calls `ChatOpenAI.invoke(...)` with LangChain tags and metadata. This is not wrapped with the local `trace_span` helper. The local `graph/tracing.py` helper creates manual spans only. Provider spans depend on LangChain/LangSmith tracing integration observing `ChatOpenAI.invoke`.

Current wrapper behavior:

- `invoke_json` returns only parsed JSON.
- It does not return the raw LangChain response.
- It does not extract `response_metadata`, token usage, cost, or first-token latency.
- Any provider usage metadata exposed by the model response is discarded before the denial graph result is built.

## Output-Column Bug Diagnosis

The intended operational output is present as `result["output"]`, with denial preferring `routing_action`.

The observed alert-class output issue is caused by evaluator plumbing, not by routing logic:

1. `_run_case_result` stores Layer 1 primary alert/error information in `result["error"]`.
2. Hosted evaluator wrappers pass `error = result.get("error")` into evaluator functions.
3. Evaluators treat any non-null `error` as an evaluator failure condition.
4. For a hosted hybrid run with a Layer 1 warning such as `latency_spike`, evaluator comments can become `latency_spike`.
5. LangSmith evaluator rows then show that evaluator comment/output, making `latency_spike` appear in an Output-like column.

This is separate from the operational target output, which remains `APPEAL` for the hosted hybrid smoke.

Correct target behavior:

- Operational target output: route/posture/status, e.g. `APPEAL`.
- Evaluator output: score/comment about evaluation criteria.
- Alerts: metadata/tags and structured alert fields, not evaluator failure input unless the workflow truly failed.

## Latency, Token, Cost, And First-Token Diagnosis

Latency shown as `0.00s` in evaluator rows is expected for evaluator rows because evaluator functions are small local scoring functions. It is not a reliable measure of the operational denial graph or provider call latency.

Operational latency is present in result metadata as `latency_ms`. Hosted hybrid smoke reported a Layer 1 latency warning at 5287 ms.

Tokens are empty because `invoke_json` discards the raw `ChatOpenAI` response and does not copy any usage metadata into the node diagnostic, record metadata, or result token metadata.

Cost is empty because no cost field is produced by the wrapper and no cost calculation is implemented.

First-token latency is empty because no streaming or first-token timing is captured.

Classification:

- Evaluator latency: expected for evaluator rows.
- Operational latency: available in `latency_ms` metadata.
- Token/cost/first-token fields: wrapper discarding or not capturing provider metadata; do not fabricate.
- Provider metadata: may be unavailable from tooling, but current code does not preserve it even if present on the response.

## Tag And Metadata Visibility Diagnosis

Operational result dictionaries include `tags` and `metadata`, including `workflow:*`, `variant:*`, route/posture tags, status, severity, and alert tags. Manual `trace_span` calls also pass tags for graph and node spans.

LangSmith evaluator rows may not show the same tags because they are evaluator feedback runs controlled by LangSmith evaluate, not operational target traces.

The hosted experiment target output includes `tags` as a field in the result object, but that does not guarantee LangSmith uses those values as run-level tags for the target run. The current target function returns tags; it does not explicitly set tags on the hosted evaluate target run itself.

## Recommended Fix Plan

1. Separate operational runtime error from Layer 1 warning alert class.
   - Keep `result["error"]` null for warning-only alerts such as `latency_spike`.
   - Add a distinct field such as `operational_error` or `primary_alert_class` for alert filtering.
   - Hosted evaluators should receive only true runtime failure as `error`.

2. Normalize hosted target output for table display.
   - Keep `output` as the primary route/posture/status value.
   - Move alerts to metadata and structured details.
   - Avoid allowing warning alert classes to become evaluator comments.

3. Add explicit run-level tags/metadata to hosted target traces if LangSmith evaluate supports target run metadata injection.
   - Returning a `tags` field is not equivalent to setting LangSmith run tags.

4. Preserve provider response metadata in `invoke_json`.
   - Capture response usage metadata when available.
   - Store token/cost/first-token fields only if exposed.
   - Keep unavailable fields explicitly unavailable.

5. Make provider span visibility explicit.
   - Confirm whether LangChain auto-traces `ChatOpenAI.invoke` under `medscribe.denial.<node>`.
   - If not, wrap provider calls with a manual child span around `invoke_json` while preserving LangChain provider tracing.

6. Consider renaming the operational graph span in a later pass if dashboard clarity requires it.
   - Current name: `medscribe.denial_graph`.
   - Possible clearer name: `denial.workflow.hybrid` or `denial.workflow`.
   - Do not rename evaluator runs.

## Success Criteria For Next Implementation Pass

- Hosted experiment target row Output column shows `route=<routing_action> posture=<governance_posture> status=<status>` or a status value, not `latency_spike`.
- Evaluator rows remain clearly labeled as evaluator rows and show score/comment only.
- Warning alerts do not cause evaluator failure comments.
- Operational denial workflow trace is easy to find by name and contains child node spans.
- Hybrid hosted run with provider access shows LLM/provider child activity under documentation-gap and recoverability interpretation nodes, or the diagnostic explicitly records provider tracing unavailable.
- `routing_engine` and `governance_policy` remain deterministic and contain no provider calls.
- Token, cost, and first-token telemetry are populated only when available from provider/tooling; otherwise unavailable reasons remain explicit.
- Tags/metadata for workflow, variant, route, posture, status, severity, and alerts are visible on operational target traces, not only inside returned JSON.

## Implementation Update

The next implementation pass corrected the diagnosed alert/output issue without changing deterministic denial routing or governance:

- Runtime `error` is reserved for failed execution, invalid schema, unrecovered provider failure, or critical operational failure.
- Warning and informational Layer 1 alerts remain in `alerts`, `alert_count`, `max_alert_severity`, metadata, and tags.
- Denial operational output now uses `route=<routing_action> posture=<governance_posture> status=<status>` for table readability.
- Hosted evaluator wrappers ignore warning-only alert classes when constructing evaluator comments.
- LLM-assisted provider calls preserve response and usage metadata when exposed by the provider response.
- LLM-assisted provider invocations are wrapped in manual `medscribe.denial.<node>.llm` spans under the documentation-gap and recoverability node executions when tracing is enabled.

## Dedicated Live Denial Operations Trace Path

An additive live operations entrypoint now exists at `evaluation/run_denial_ops_traces.py`.

- Purpose: run selected synthetic denial benchmark cases as first-class operational traces, not as datasets, hosted experiments, or evaluator feedback runs.
- Default project: `medscribe-denial-ops`.
- Workflow trace name in the live operations queue: the denial `case_id`, such as `DENIAL-001`.
- Workflow identity is stored in metadata as `workflow_trace_name`, with values such as `medscribe.denial.workflow.hybrid` or `medscribe.denial.workflow.deterministic`.
- Default execution mode: `auto`, which selects hybrid when provider access is available and deterministic otherwise.
- Input shape: a concise operational denial summary such as `medical necessity | conflicting evidence` or `modifier issue | partial support`.
- Output shape: the routing action only: `APPEAL`, `RESUBMIT`, `WRITE_OFF`, or `ESCALATE`.
- Error shape: null unless execution has an unrecovered operational anomaly.
- Metadata: workflow, trace type, execution mode, case id, routing action, governance posture, denial category, recoverability, evidence strength, ambiguity, documentation gaps, alerts, token usage, provider metadata when available, latency, node timing, fallback/degraded flags, and `contains_phi:false`.
- Tags: operational investigation signals such as `workflow:denial`, `mode:hybrid`, `high_ambiguity`, `conflicting_evidence`, `specialist_review`, `threshold_boundary`, `fallback_used`, `degraded_mode`, `latency_spike`, and `token_spike`. Route, posture, and error duplication are avoided in tags.

This path preserves evaluator separation. Evaluator rows created by LangSmith `evaluate(...)` still use evaluator inputs such as `{run, example}` and evaluator outputs such as score/comment. Live denial operations traces use case-level inputs and contain the denial workflow waterfall.

Operational traces are not automatically added to LangSmith datasets. Live traces are operational debugging evidence; curated datasets remain benchmark/evaluation inputs; experiments produce evaluator traces and feedback.

The live operations queue is intended for RCM operations engineering triage, not denials analyst review. Routing action is the primary output because it is the next operational disposition. Governance posture remains secondary diagnostic context because it explains confidence and evidence posture without replacing the route.

## Table-Facing Metadata Update

Live denial operations traces now separate table-facing metadata from diagnostic metadata:

- Top-level metadata prioritizes investigation signals: `llm_used`, `fallback_used`, `degraded_mode`, `alert_count`, `max_alert_severity`, `evidence_strength`, `ambiguity_level`, `conflicting_evidence`, `specialist_review_candidate`, `threshold_boundary`, `recoverability`, `governance_posture`, `token_cost_available`, `token_cost_unavailable_reason`, `first_token_available`, and `first_token_unavailable_reason`.
- Top-level metadata avoids duplicating first-class table columns such as case id, primary output, runtime error, token totals, latency, and cost fields.
- Full drill-down data is preserved under `diagnostic_metadata`, including case id, title, denial type, evidence profile, documentation gaps, node diagnostics, provider metadata, raw token usage, latency, first-token latency if exposed, cost metadata if exposed, alert objects, node timing, fallback reasons, operational metrics, and thresholds.
- First-token latency is passed through only when a real provider or streaming metadata field exposes it. The current non-streaming `ChatOpenAI.invoke` path did not expose first-token timing in hosted smoke verification, so live traces report `first_token_available=false` with `first_token_unavailable_reason=provider_streaming_not_enabled` when provider calls occur without first-token metadata.
