# Layered AI Ops Synthesis

This tracked synthesis note records the operational evaluation model used by the MedScribe regenerated workflows. It applies to both the CDI/coding-support workflow and the additive denial-operations workflow.

## Three Layers

Layer 1 operational debugging covers execution reliability and observability: runtime completion, hosted or local mode, fallback/degraded behavior, node diagnostics, evaluator feedback, trace visibility, latency, verbosity, malformed payloads, missing operational metadata, provider/runtime visibility, and token/cost availability semantics.

Layer 2 pairwise routing evaluation covers operational usefulness before human review. CDI workflows compare review-routing usefulness. Denial workflows compare which routing/governance outcome is safer and more operationally useful before human review.

Layer 3 routing-distribution experimentation covers routing movement, threshold sensitivity, escalation saturation, and spillover behavior. CDI workflows summarize final status buckets and governance reason-code movement. Denial workflows summarize APPEAL, RESUBMIT, WRITE_OFF, ESCALATE, governance posture distribution, low-evidence routing counts, ambiguity-routing counts, specialist-review candidate routing counts, denial category counts, and recoverability observations.

For RCM operations debugging, Layer 1 answers whether the run and trace were healthy enough to inspect, Layer 2 answers which output is more useful before human review, and Layer 3 answers whether aggregate routing/system behavior is shifting.

## Denial Workflow Extension

The denial workflow is additive and selectable through the existing evaluation runner with `--workflow denial`. It preserves the existing CDI/coding-support workflow and uses the separate local denial graph.

Denial Layer 1 observability exposes node diagnostics, degraded/fallback visibility, routing action, governance posture, denial category, recoverability, and documentation gaps.

Layer 1 status and alerts are separate from governance posture and routing quality. `completed`, `degraded`, and `failed` describe execution health only. Alerts such as `latency_spike`, `fallback_used`, `degraded_mode`, `missing_required_metadata`, `malformed_payload`, and `trace_incomplete` indicate operational reliability signals. They do not mean that `LOW_EVIDENCE`, `AMBIGUOUS`, `ESCALATE`, or conservative denial routing is wrong.

Denial Layer 2 comparison uses an inspectable local heuristic. It favors ambiguity escalation, recoverable routes over premature write-off, low-evidence caution, unsupported certainty avoidance, and visible operational rationale.

Denial Layer 3 reporting is an operational distribution summary only. It does not simulate payer acceptance or downstream human outcomes.

Richer denial evidence fields now let the local scaffold model operational evidence tension: partial imaging support, conflicting specialist notes, weak but non-zero necessity rationale, conservative treatment ambiguity, timeline inconsistency, utilization review disagreement, prior authorization mismatch, conflicting coding specificity, incomplete longitudinal evidence, and specialist-review escalation. These fields increase benchmark realism for evaluation and reporting, but they remain synthetic operational signals rather than clinical simulation.

Hybrid denial execution may use LLM-assisted interpretation only in documentation-gap and recoverability nodes. Routing and governance stay deterministic, and hybrid failures use deterministic fallback behavior with Layer 1 visibility. Hybrid traces are regenerated operational evidence, not frozen historical reproduction.

## Observability Scope

These workflows provide regenerated observability. They do not freeze upstream state or prove causal reproducibility. Local-only denial execution and hosted CDI experiments can both produce useful current-state operational evidence.

## Local And Hosted Execution

Local-only mode emits local summaries and pairwise reports without formal hosted experiment rows. Hosted mode preserves the existing LangSmith integration pattern when credentials, SDK support, and network access are available. Denial hosted support follows the same dataset/evaluator pattern where practical, while local denial execution remains the bounded default scaffold for this phase.

Hosted readiness should be checked before interpreting hybrid smoke results. Provider or network failure should be separated from workflow regression: degraded fallback caused by provider/network failure is not automatically a workflow defect when deterministic fallback completes and runtime `error` remains null.

## Hosted Hybrid Denial Batch Observation 2026-05-12

A 12-case hosted hybrid denial operations batch ran through the dedicated `medscribe-denial-ops` trace path. These were live operational traces, not evaluator-only rows and not dataset-only execution.

Layer 1 findings:

- 12 of 12 cases completed.
- Runtime errors: 0.
- Fallback/degraded count: 0.
- Alerts: 3 latency warning alerts.
- Token metadata was available for all 12 cases.
- First-token metadata was unavailable for all 12 cases because the current provider path is non-streaming and did not expose first-token timing.

Layer 2 findings:

- Baseline versus `threshold_variant`: baseline preferred 1, threshold variant preferred 1, ties 10.
- Baseline versus `routing_sensitivity_variant`: routing-sensitivity variant preferred 1, ties 11.
- `DENIAL-002` was the notable boundary case in both comparisons.

Layer 3 findings:

- Hosted live ops routing distribution: `APPEAL` 2, `RESUBMIT` 2, `WRITE_OFF` 1, `ESCALATE` 7.
- Hosted live ops governance posture distribution: `SUPPORTED` 2, `LOW_CONFIDENCE` 2, `LOW_EVIDENCE` 1, `AMBIGUOUS` 5, `HIGH_RISK` 2.
- Escalation remained the dominant route, but LLM-assisted interpretation produced limited route movement at boundary cases.
- Threshold variants are beginning to show boundary-level sensitivity, but they remain too weak to produce broad aggregate distribution movement.

This batch remains regenerated operational evidence. It does not provide payer outcome simulation, downstream human behavior simulation, or clinical correctness proof.

## Strengthened Variant Sensitivity Observation 2026-05-12

The denial evaluation and live operations runners now apply denial-specific variant overlays outside the baseline denial graph:

- `threshold_variant` increases sensitivity to ambiguity, low evidence, specialist review, unsupported-service risk, and recoverability-boundary uncertainty.
- `routing_sensitivity_variant` reduces avoidable escalation for recoverable documentation gaps and partial-support cases while preserving high-risk escalation.
- Routing and governance node implementations remain deterministic and unchanged; variant movement is deterministic runner-scoped operational sensitivity behavior.

A 24-case hosted hybrid denial operations batch was run for each variant in the `medscribe-denial-ops` project.

Layer 1 observations:

- Baseline completed 24 of 24 cases with 9 latency warnings, no fallback/degraded executions, and no runtime errors.
- `threshold_variant` completed 24 of 24 cases with 8 latency alerts, including 1 critical latency alert, and no fallback/degraded executions.
- `routing_sensitivity_variant` completed 24 of 24 cases with 8 latency warnings, no fallback/degraded executions, and no runtime errors.
- Token metadata was available for all 72 hosted operational traces.
- First-token metadata remained unavailable because the current provider path did not expose streaming first-token timing.

Layer 2 observations:

- Baseline versus `threshold_variant`: baseline preferred 11, threshold variant preferred 5, ties 8.
- Baseline versus `routing_sensitivity_variant`: routing-sensitivity variant preferred 3, baseline preferred 3, ties 18.
- Threshold comparison produced 14 route or posture movements.
- Routing-sensitivity comparison produced 6 route or posture movements.

Layer 3 observations:

- Baseline routes: `APPEAL` 2, `RESUBMIT` 8, `WRITE_OFF` 1, `ESCALATE` 13.
- `threshold_variant` routes: `APPEAL` 4, `RESUBMIT` 0, `WRITE_OFF` 1, `ESCALATE` 19.
- `routing_sensitivity_variant` routes: `APPEAL` 2, `RESUBMIT` 11, `WRITE_OFF` 2, `ESCALATE` 9.
- Escalation saturation increased under stricter threshold sensitivity and decreased under routing sensitivity.
- Aggregate routing movement is now visible, but results remain regenerated operational evidence rather than payer-outcome simulation.
