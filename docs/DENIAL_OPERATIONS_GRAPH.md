# Denial Operations Graph

The denial operations graph is an additive, local-only scaffold for synthetic non-PHI denial-management cases. It does not replace or modify the existing CDI/coding-support workflow.

## Pipeline

The standalone runner `run_denial_graph(case_payload: dict) -> dict` executes:

1. `denial_intake_parser`
2. `denial_classifier`
3. `recoverability_analyzer`
4. `documentation_gap_analyzer`
5. `routing_engine`
6. `governance_policy`

The runner returns `routing_action`, `governance_posture`, `denial_category`, `recoverability`, `documentation_gaps`, `node_diagnostics`, `status`, `degraded_mode`, and `fallback_used`.

By default, denial execution remains deterministic. When `MEDSCRIBE_EXECUTION_MODE=hybrid` or the evaluation runner is called with `--execution-mode hybrid`, only `recoverability_analyzer` and `documentation_gap_analyzer` attempt LLM-assisted interpretation. If the provider is unavailable, returns malformed JSON, violates the schema, or returns an invalid value, the node falls back to the deterministic heuristic path and records fallback/degraded diagnostics.

## Evaluation Integration

`evaluation/langsmith_experiment_runner.py --workflow denial` loads the denial benchmark corpus and runs the denial graph through the existing regenerated operational-analysis workflow.

Layer 1 operational debugging reports completion status, fallback/degraded flags, node diagnostics, routing action, governance posture, denial category, recoverability, and documentation gaps.

Layer 1 execution-health fields include `status`, `error`, `alerts`, `alert_count`, and `max_alert_severity`. `status` is limited to `completed`, `degraded`, or `failed`. `error` is null when no operational anomaly exists and contains one primary execution anomaly when an alert-worthy runtime health issue is present.

Layer 1 alerts are operational reliability signals, not denial-routing judgments. `LOW_EVIDENCE`, `AMBIGUOUS`, `ESCALATE`, and conservative routing are not execution errors. Alert classes cover latency, token/cost availability, verbosity, fallback/degraded execution, missing metadata, invalid schema, malformed payload, provider failure, trace incompleteness, and evaluator failure where applicable.

Layer 2 pairwise routing evaluation asks: which routing/governance outcome is safer and more operationally useful before human review? The local heuristic considers ambiguity escalation quality, low-evidence handling, recoverability alignment, unsafe write-off avoidance, unsupported certainty avoidance, and operational inspectability.

Layer 3 routing-distribution experimentation summarizes APPEAL, RESUBMIT, WRITE_OFF, ESCALATE, governance posture distribution, low-evidence routing counts, ambiguity-routing counts, specialist-review candidate routing counts, denial category counts, threshold sensitivity, escalation saturation, and recoverability observations.

Richer denial reporting also tracks evidence strength, conflicting-evidence routing, partial-support routing, timeline-inconsistency routing, and recoverability-boundary routing. These are operational realism signals, not clinical truth labels.

LangSmith-facing denial outputs expose `routing_action` as the preferred primary `output` value for dashboard scanning. Fallback display order is governance posture, denial category, final status, then runtime status. Structured metadata keeps routing action separate from governance posture and includes available evidence strength, ambiguity flags, conflicting-evidence flags, specialist-review candidate flags, degraded/fallback flags, millisecond latency, `llm_used`, and token/cost availability semantics.

Local denial scaffold runs do not call an LLM provider. Their metadata reports `llm_used` as false, `token_cost_available` as false, and `token_cost_unavailable_reason` as `no_provider_call`. No token or cost values are inferred when unavailable.

Denial traces and runs are operational debugging evidence. Denial datasets remain curated benchmark inputs created or reused through explicit experiment setup, not automatic collections of traces.

Denial evaluation preserves regenerated operational experimentation semantics. It observes current workflow behavior and does not prove payer outcomes.

## Node Responsibilities

`denial_intake_parser` extracts denial reason, service context, documentation context, and simple documentation signals.

It also accepts optional synthetic evidence fields such as `imaging_summary`, `lab_summary`, `treatment_history`, `medication_context`, `prior_authorization_context`, `utilization_review_notes`, `specialist_notes`, `timeline_flags`, `conflicting_documentation`, `evidence_strength`, `missing_required_evidence`, `conservative_treatment_history`, `medical_necessity_rationale`, and `coding_specificity_flags`. Not every case is expected to contain every field.

`denial_classifier` classifies the denial as medical necessity, insufficient documentation, coding mismatch, modifier issue, unclear denial, or unsupported service.

`recoverability_analyzer` estimates whether the denial is likely recoverable, partially recoverable, low recoverability, or unclear recoverability. In hybrid mode, it may use an LLM to interpret recoverability evidence into structured intermediate fields only: recoverability, recoverability factors, uncertainty, and rationale. It does not choose final routing or governance posture.

`documentation_gap_analyzer` identifies missing evidence, unsupported specificity, documentation insufficiency, and ambiguity.

For richer synthetic cases, it also surfaces partial support, timeline inconsistency, and specialist-review signals. In hybrid mode, it may use an LLM to interpret documentation gaps into structured intermediate fields only: missing evidence, documentation insufficiency, unsupported specificity, conflicting evidence, partial support, ambiguity level, specialist-review candidate, and rationale. It does not choose final routing or governance posture.

`routing_engine` produces a routing action only from `APPEAL`, `RESUBMIT`, `WRITE_OFF`, and `ESCALATE`.

`governance_policy` produces a descriptive governance posture only from `SUPPORTED`, `LOW_CONFIDENCE`, `LOW_EVIDENCE`, `AMBIGUOUS`, and `HIGH_RISK`.

`routing_engine` and `governance_policy` remain deterministic and do not call an LLM.

## Benchmark Corpus

`evaluation/denial_benchmark_cases.json` contains synthetic non-PHI denial cases covering appealability ambiguity, documentation insufficiency, coding mismatch, modifier omission, unsupported medical necessity, recoverable documentation gaps, conflicting payer rationale, borderline appealability, specialist-review-worthy ambiguity, low-evidence denial, low recoverability, and boundary-sensitive routing cases.

The corpus includes richer operational scenarios for partially supported imaging findings, conflicting specialist documentation, weak but non-zero medical necessity evidence, conservative treatment ambiguity, timeline inconsistency, utilization review disagreement, borderline modifier justification, prior authorization mismatch ambiguity, conflicting coding specificity, incomplete longitudinal evidence, low-evidence high-risk denials, and recoverability tension.

`evaluation/synthetic_incidents/denial_incidents.json` contains a small synthetic non-PHI incident pack for malformed denial parsing, ambiguity escalation collapse, unsafe write-off tendency, routing instability, fallback/degraded execution visibility, conflicting evidence interpretation, conflicting evidence collapse, unsupported certainty escalation, ambiguity routing oscillation, evidence extraction failure, partial-documentation instability, specialist escalation instability, and Layer 1 alert probes for latency spike, missing metadata, malformed payload, and trace incompleteness.

## Current Phase Limitations

This phase is a bounded workflow scaffold. It uses deterministic local keyword and field checks suitable for synthetic benchmark cases only. Richer evidence fields model operational ambiguity and evidence tension; they do not constitute clinical adjudication, payer acceptance prediction, or real-world medical necessity determination.

LLM-assisted denial interpretation is regenerated and may vary with provider behavior. Token, cost, and first-token metadata are reported only when the provider/tooling exposes them and are not fabricated.

## Non-Goals

- No physician-behavior simulation.
- No downstream human override simulation.

## Relationship To Existing CDI Workflow

The denial graph is additive and separate from the existing CDI/coding-support graph. Both workflows reuse the current operational reliability conventions for tracing, diagnostics, evaluation, and deterministic downstream control.
