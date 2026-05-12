# Hosted Denial Experiment Analysis

## Purpose

This note records hosted LangSmith denial-management experiment findings for the synthetic denial workflow. It captures operational observability, routing distribution, governance posture distribution, and pairwise comparison behavior for baseline, threshold-variant, and routing-sensitivity-variant runs.

## Execution Mode

Runtime execution used `.venv/bin/python` from the repository virtual environment.

Hosted readiness was checked through the existing evaluation runner without printing secret values. An initial sandboxed readiness attempt was degraded by connectivity limits. An approved network-enabled readiness check succeeded, and the existing runner reported the denial benchmark dataset ready.

## Hosted Readiness Status

- LangSmith client construction: ready during network-enabled preflight.
- Hosted denial dataset: ready.
- Hosted experiment mode: available.
- Hosted experiment URL presence: true for the completed hosted runs.
- Hosted trace behavior: hosted experiment submission completed through the existing runner path.

Operational interpretation note:

- Hosted readiness should be checked before interpreting hybrid smoke results.
- Provider or network failure should be distinguished from workflow regression.
- Degraded fallback caused by provider/network failure is not automatically a workflow defect when deterministic fallback completes and runtime `error` remains null.

## Baseline Findings

The hosted denial baseline run completed over 24 synthetic non-PHI denial cases.

Routing action counts:

- `APPEAL`: 2
- `RESUBMIT`: 6
- `WRITE_OFF`: 1
- `ESCALATE`: 15

Governance posture counts:

- `SUPPORTED`: 2
- `LOW_CONFIDENCE`: 6
- `LOW_EVIDENCE`: 1
- `AMBIGUOUS`: 11
- `HIGH_RISK`: 4

Operational distribution observations:

- Low-evidence routing counts: `ESCALATE` 4, `RESUBMIT` 6, `WRITE_OFF` 1, `APPEAL` 0.
- Ambiguity routing counts: `ESCALATE` 13, other routing actions 0.
- Conflicting-evidence routing counts: `ESCALATE` 6, other routing actions 0.
- Specialist-review candidate routing counts: `ESCALATE` 3, other routing actions 0.
- Timeline inconsistency routing counts: `ESCALATE` 2, other routing actions 0.
- Runtime status counts: `completed` 24.

Degraded/fallback observations:

- Degraded count: 0.
- Fallback count: 0.

Token/cost observations:

- Token and cost metadata was not available from the local denial scaffold output attached to the hosted experiment summary.

## Threshold Variant Findings

The hosted denial `threshold_variant` run completed over 24 synthetic non-PHI denial cases.

Routing action counts:

- `APPEAL`: 2
- `RESUBMIT`: 6
- `WRITE_OFF`: 1
- `ESCALATE`: 15

Governance posture counts:

- `SUPPORTED`: 2
- `LOW_CONFIDENCE`: 6
- `LOW_EVIDENCE`: 1
- `AMBIGUOUS`: 11
- `HIGH_RISK`: 4

Movement versus baseline:

- Routing distribution movement: none observed.
- Governance posture distribution movement: none observed.
- Degraded count: 0.
- Fallback count: 0.
- Notable divergent cases: none observed in pairwise comparison.

## Routing-Sensitivity Variant Findings

The hosted denial `routing_sensitivity_variant` run completed over 24 synthetic non-PHI denial cases.

Routing action counts:

- `APPEAL`: 2
- `RESUBMIT`: 6
- `WRITE_OFF`: 1
- `ESCALATE`: 15

Governance posture counts:

- `SUPPORTED`: 2
- `LOW_CONFIDENCE`: 6
- `LOW_EVIDENCE`: 1
- `AMBIGUOUS`: 11
- `HIGH_RISK`: 4

Movement versus baseline:

- Routing distribution movement: none observed.
- Governance posture distribution movement: none observed.
- Degraded count: 0.
- Fallback count: 0.
- Notable divergent cases: none observed in pairwise comparison.

## Pairwise Findings

Pairwise comparison used the existing local pairwise evaluator path because the current runner's pairwise mode is local-only.

Baseline versus `threshold_variant`:

- Case count: 24.
- Baseline preferred: 0.
- Variant preferred: 0.
- Ties: 24.
- Status-changed count: 0.
- Notable divergent cases: none.

Baseline versus `routing_sensitivity_variant`:

- Case count: 24.
- Baseline preferred: 0.
- Variant preferred: 0.
- Ties: 24.
- Status-changed count: 0.
- Notable divergent cases: none.

Explanation summary:

- The evaluated variants produced the same denial routing and governance outcomes as baseline for this corpus.
- The comparison path therefore preferred neither variant over baseline.
- Ambiguous, conflicting-evidence, specialist-review, and timeline-sensitive cases remained routed toward escalation.

## Routing Distribution Findings

Across hosted baseline and both hosted variants, the distribution was stable:

- `ESCALATE` was the dominant routing action with 15 of 24 cases.
- `RESUBMIT` appeared in 6 of 24 cases.
- `APPEAL` appeared in 2 of 24 cases.
- `WRITE_OFF` appeared in 1 of 24 cases.
- Ambiguity and conflicting evidence were routed to `ESCALATE` in the observed distribution.
- Specialist-review candidate cases were routed to `ESCALATE`.

## Trace And Evaluator Feedback Observations

- Hosted experiment submission completed through the existing LangSmith-compatible runner path.
- The runner summary exposed hosted experiment URL presence but did not emit a separate evaluator feedback aggregate in the local JSON summary.
- Node-level denial outputs remained inspectable through routing action, governance posture, denial category, recoverability, degraded mode, fallback flag, and routing distribution fields.

## Dashboard Interpretation Notes

- Denial dashboard rows should use `route=<routing_action> posture=<governance_posture> status=<status>` as the primary operational output when present.
- Governance posture remains available as structured detail and should not be interpreted as the same field as routing action.
- Reason codes and documentation gaps are supporting evidence for debugging, not the primary route outcome.
- Millisecond-level `latency_ms` metadata is the preferred timing field for short local scaffold executions.
- Token and cost availability is explicit. For local denial scaffold executions, `llm_used` is false, token/cost availability is false, and the unavailable reason is `no_provider_call`.
- Traces and runs are operational evidence. Curated denial benchmark datasets remain the evaluation inputs and are not automatically populated from arbitrary traces.
- Layer 1 alert metadata is execution-health metadata. Warning alerts such as `latency_spike`, `fallback_used`, `degraded_mode`, and `missing_token_cost_metadata` do not populate runtime `error` when execution completes. `LOW_EVIDENCE`, `AMBIGUOUS`, `ESCALATE`, and conservative denial routing are not execution errors.
- Alert tags such as `alert:latency_spike`, `severity:warning`, and `status:degraded` are intended for operational debugging of reliability signals such as fallback, degraded mode, malformed payloads, missing metadata, trace incompleteness, and threshold breaches.
- Hybrid denial mode limits LLM-assisted interpretation to documentation-gap and recoverability nodes. Final routing and governance remain deterministic.
- Hosted hybrid traces should be interpreted as regenerated operational evidence. Provider token/cost/first-token fields are preserved when exposed by provider/tooling and are not inferred. Manual provider spans are expected under hybrid documentation-gap and recoverability node spans when tracing is enabled.

## Corrected Evaluator/Workflow Separation

- Operational target rows carry route/posture/status output plus structured metadata, tags, alerts, and node diagnostics.
- Evaluator rows carry evaluator `score` and evaluator-specific comments such as expected-versus-actual routing or posture.
- Non-critical alert classes remain in metadata/tags and are not passed to normal route/posture evaluators as failure comments.

## Dedicated Live Operations Trace Path

- `evaluation/run_denial_ops_traces.py` runs denial cases as live operational traces without creating LangSmith dataset examples, hosted experiment rows, or evaluator feedback.
- The default live operations project is `medscribe-denial-ops`.
- The top-level live workflow trace is named with the operational case id, such as `DENIAL-001`; workflow identity remains in metadata as `medscribe.denial.workflow.hybrid` or `medscribe.denial.workflow.deterministic`.
- Hybrid is selected automatically when provider access is available; deterministic mode remains available through `--execution-mode deterministic` and local fallback behavior.
- The live operations table input is a concise problem summary such as `medical necessity | conflicting evidence`.
- The live operations table output is the routing action only: `APPEAL`, `RESUBMIT`, `WRITE_OFF`, or `ESCALATE`.
- Governance posture is retained in metadata/details and tags are reserved for investigation signals rather than duplicating the route.
- Runtime `error` remains empty for ambiguity, low evidence, high risk, fallback warnings, degraded-but-completed execution, escalation routing, and latency warnings.
- Alerts remain metadata/tags and do not replace primary output.
- The expected waterfall is the live workflow trace containing `medscribe.denial_graph`, each denial node span, and provider/LLM spans under recoverability and documentation-gap analyzer nodes when provider calls execute and tracing is enabled.
- Live traces are operational debugging evidence. Selected observations can later inform curated datasets, but no automatic trace-to-dataset promotion is performed.

Live operations smoke verification:

- Project: `medscribe-denial-ops`.
- Top-level trace observed: `DENIAL-001`.
- Input observed: concise denial summary.
- Child workflow trace observed: `medscribe.denial_graph`.
- Node spans observed for intake parsing, classification, recoverability, documentation gaps, routing, and governance.
- Provider spans observed for hybrid recoverability and documentation-gap interpretation.
- Output observed as routing action only with runtime `error` null.
- Governance posture remained present in metadata/details.
- Token usage fields were present when exposed by the provider response.

Updated queue semantics:

- Name: case id.
- Input: concise denial problem summary.
- Output: routing action only.
- Error: unrecovered execution anomaly only.
- Tags: operational investigation signals such as ambiguity, conflicting evidence, specialist review, threshold boundary, fallback/degraded mode, and alert classes.
- Top-level metadata: ops signals and availability flags rather than duplicated table columns.
- Diagnostic metadata: full case, evidence, node, provider, token, latency, first-token, cost, alert, timing, fallback, and operational-metric drill-down.
- First-token latency: not fabricated. It is present only if exposed by real provider/streaming metadata; otherwise `first_token_available=false` and an explicit unavailable reason is recorded.

Hosted metadata verification:

- Top-level metadata began with operational signals and availability flags.
- Top-level metadata did not duplicate case id, route, error, token totals, or latency.
- Raw token usage, provider metadata, and latency remained available under `diagnostic_metadata`.
- First-token latency was not exposed by the current non-streaming provider path; `first_token_available=false` and `first_token_unavailable_reason=provider_streaming_not_enabled` were recorded.

## Limitations

- Hosted observations are regenerated operational evidence, not frozen historical reproduction.
- The denial workflow remains a local deterministic scaffold executed through hosted experiment orchestration; it is not a clinical reasoning engine.
- Pairwise comparison is heuristic and local-only in the current runner.
- Token and cost metadata was unavailable in the denial scaffold summary.
- No payer outcome, downstream human behavior, or clinical outcome simulation was performed.

## Hosted Hybrid Operations Batch 2026-05-12

Execution mode:

- Runner: `evaluation/run_denial_ops_traces.py`.
- Project: `medscribe-denial-ops`.
- Case count: 12 synthetic denial cases.
- Execution mode: hybrid.
- Trace type: live operational traces, not evaluator rows and not dataset-only execution.

Layer 1 operational findings:

- Runtime status: `completed` for 12 of 12 cases.
- Runtime errors: 0.
- Fallback/degraded executions: 0.
- Alert distribution: `latency_spike` warning on 3 of 12 cases.
- Malformed or invalid output frequency: 0 observed.
- Provider failure frequency: 0 observed.
- Verbosity distribution: `normal` for 12 of 12 cases.
- Latency range: 3208 ms to 6041 ms.
- Mean latency: 4167.92 ms.
- Token metadata availability: 12 of 12 cases.
- Total-token range: 1047 to 1092 tokens.
- Mean total tokens: 1069.83.
- First-token availability: 0 of 12 cases. The current non-streaming provider path did not expose first-token timing, so `provider_streaming_not_enabled` remained the precise unavailable reason.

Layer 2 pairwise findings:

- Baseline versus `threshold_variant`, 12 cases: baseline preferred 1, threshold variant preferred 1, ties 10.
- Baseline versus `routing_sensitivity_variant`, 12 cases: routing-sensitivity variant preferred 1, ties 11.
- Notable divergent case: `DENIAL-002`.
- Threshold comparison moved `DENIAL-002` from `RESUBMIT`/`LOW_CONFIDENCE` to `ESCALATE`/`AMBIGUOUS`, with baseline preferred by the heuristic.
- Routing-sensitivity comparison moved `DENIAL-002` from `ESCALATE`/`AMBIGUOUS` to `RESUBMIT`/`LOW_CONFIDENCE`, with routing-sensitivity variant preferred by the heuristic.
- LLM-assisted interpretation now produced limited routing disagreement, but most cases remained ties.

Layer 3 routing-distribution observations:

- Hosted live ops batch routing distribution: `APPEAL` 2, `RESUBMIT` 2, `WRITE_OFF` 1, `ESCALATE` 7.
- Hosted live ops governance posture distribution: `SUPPORTED` 2, `LOW_CONFIDENCE` 2, `LOW_EVIDENCE` 1, `AMBIGUOUS` 5, `HIGH_RISK` 2.
- Threshold comparison baseline routes: `APPEAL` 2, `RESUBMIT` 3, `WRITE_OFF` 1, `ESCALATE` 6.
- Threshold variant routes: `APPEAL` 2, `RESUBMIT` 2, `WRITE_OFF` 1, `ESCALATE` 7.
- Routing-sensitivity baseline routes: `APPEAL` 2, `RESUBMIT` 2, `WRITE_OFF` 1, `ESCALATE` 7.
- Routing-sensitivity variant routes: `APPEAL` 2, `RESUBMIT` 3, `WRITE_OFF` 1, `ESCALATE` 6.
- Ambiguous cases remained escalated in aggregate: threshold comparison moved ambiguity escalation from 5 to 6 cases; routing-sensitivity moved it from 6 to 5 cases.
- Conflicting-evidence cases remained escalated in both pairwise comparisons: 2 cases in baseline and variants.
- Specialist-review candidate cases remained escalated: 1 case in baseline and variants.
- Evidence-strength distribution was unchanged across variants: conflicting 2, low 3, moderate 3, partial 2, strong 2.
- Recoverability-boundary routing moved by one case between `ESCALATE` and `RESUBMIT` depending on variant.
- Escalation saturation remained visible: `ESCALATE` was still the dominant route, but saturation was no longer completely static under LLM-assisted execution.
- Threshold sensitivity is beginning to emerge at the boundary-case level, but the current variants remain weak at producing broad aggregate movement.

Operational interpretation:

- LLM-assisted interpretation materially influenced downstream deterministic routing for a small number of boundary cases.
- Routing and governance remained deterministic after interpretation: movement came from changed intermediate evidence interpretation, not LLM calls in routing or governance.
- Results remain regenerated operational evidence, not frozen historical reproduction.

## Strengthened Variant Operations Batch 2026-05-12

Execution mode:

- Runner: `evaluation/run_denial_ops_traces.py`.
- Project: `medscribe-denial-ops`.
- Case count: 24 synthetic denial cases per batch.
- Execution mode: hybrid.
- Variants: `baseline`, `threshold_variant`, and `routing_sensitivity_variant`.
- Trace type: live operational traces, not evaluator-only rows and not dataset-only execution.

Strengthened variant semantics:

- `threshold_variant` applies stricter deterministic overlay rules for high ambiguity, low evidence, specialist-review signals, unsupported-service risk, and recoverability boundaries.
- `routing_sensitivity_variant` applies deterministic overlay rules that reduce avoidable escalation for recoverable documentation gaps and partial-support cases while preserving high-risk, conflicting-evidence, timeline, and unsupported-service escalation.
- Baseline denial graph semantics, deterministic routing node behavior, and deterministic governance node behavior were not changed.

Layer 1 operational findings:

- Baseline: 24 of 24 completed, runtime errors 0, fallback/degraded 0, latency warnings 9.
- `threshold_variant`: 24 of 24 completed, fallback/degraded 0, latency alerts 8, including 1 critical latency alert at 15869 ms.
- `routing_sensitivity_variant`: 24 of 24 completed, runtime errors 0, fallback/degraded 0, latency warnings 8.
- Token metadata was available for 24 of 24 cases in each batch.
- First-token metadata was unavailable for all cases because the current provider path did not expose streaming first-token timing.
- Verbosity bucket remained `normal` for all cases in all three batches.

Latency and token observations:

- Baseline latency: 3229 ms minimum, 7819 ms maximum, 4884.96 ms mean.
- `threshold_variant` latency: 3377 ms minimum, 15870 ms maximum, 5216.50 ms mean.
- `routing_sensitivity_variant` latency: 3026 ms minimum, 7138 ms maximum, 4778.42 ms mean.
- Baseline total-token range: 1048 to 1101, mean 1076.83.
- `threshold_variant` total-token range: 1047 to 1105, mean 1075.75.
- `routing_sensitivity_variant` total-token range: 1047 to 1095, mean 1077.00.

Layer 2 pairwise observations from hosted operational outputs:

- Baseline versus `threshold_variant`: baseline preferred 11, threshold variant preferred 5, ties 8.
- Baseline versus `routing_sensitivity_variant`: routing-sensitivity variant preferred 3, baseline preferred 3, ties 18.
- Threshold comparison produced 14 route or posture movements.
- Routing-sensitivity comparison produced 6 route or posture movements.
- Threshold notable movements included recoverable or partial-support cases moving from `RESUBMIT`/`LOW_CONFIDENCE` to `ESCALATE`/`AMBIGUOUS`.
- Routing-sensitivity notable movements included recoverable ambiguity moving from `ESCALATE`/`AMBIGUOUS` to `RESUBMIT`/`LOW_CONFIDENCE` in selected boundary cases.

Layer 3 routing-distribution observations:

- Baseline routes: `APPEAL` 2, `RESUBMIT` 8, `WRITE_OFF` 1, `ESCALATE` 13.
- `threshold_variant` routes: `APPEAL` 4, `RESUBMIT` 0, `WRITE_OFF` 1, `ESCALATE` 19.
- `routing_sensitivity_variant` routes: `APPEAL` 2, `RESUBMIT` 11, `WRITE_OFF` 2, `ESCALATE` 9.
- Baseline governance postures: `SUPPORTED` 2, `LOW_CONFIDENCE` 8, `LOW_EVIDENCE` 1, `AMBIGUOUS` 9, `HIGH_RISK` 4.
- `threshold_variant` governance postures: `SUPPORTED` 4, `LOW_CONFIDENCE` 0, `LOW_EVIDENCE` 1, `AMBIGUOUS` 11, `HIGH_RISK` 8.
- `routing_sensitivity_variant` governance postures: `SUPPORTED` 2, `LOW_CONFIDENCE` 11, `LOW_EVIDENCE` 2, `AMBIGUOUS` 5, `HIGH_RISK` 4.
- Escalation saturation increased under the stricter threshold variant and decreased under the routing-sensitivity variant.
- Routing movement is now visible at aggregate distribution level, not only at one boundary case.
- LLM-assisted interpretation remained upstream of deterministic overlay/routing/governance behavior.

Operational limitations:

- These observations are regenerated operational evidence, not frozen historical reproduction.
- Pairwise scoring remains a deterministic local heuristic applied to hosted operational outputs.
- The strengthened variants are operational sensitivity perturbations, not payer outcome prediction or clinical correctness claims.
