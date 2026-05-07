# LangSmith Phase 3 Malformed Reporting - 2026-05-07T20:02:37Z

## Scope

Implemented a reporting/evaluator clarification for `MS-SYN-007`. Runtime behavior, prompts, governance logic, routing logic, thresholds, datasets, model settings, and incident identity were not changed.

## Reporting Clarification Added

File changed:

- `evaluation/run_synthetic_incidents.py`

For `malformed_downstream_payload` summaries, the runner now adds:

- `malformed_instruction_detected`
- `malformed_payload_observed`
- `incident_behavior_classification`
- `expected_shape`
- `observed_output_shape`
- `shape_validation_result`
- `malformed_payload_stage`
- `structured_outputs_valid`
- `fallback_triggered_by_malformed_payload`
- `operational_interpretation`

## Before/After Interpretation

Before:

- `MS-SYN-007` reported runtime completion, governed decision, fallback status, failure-localization clue, high-level match, and trace stage count.
- The summary did not state whether a malformed downstream payload was actually observed.
- The incident class could be read as true malformed payload propagation even when stage outputs stayed well-formed.

After:

- `MS-SYN-007` still reports runtime `completed`, governed decision `FAIL`, and high-level match `true`.
- The summary now states that malformed instruction language was detected.
- The summary now states that malformed downstream payload was not observed.
- The summary now reports expected and observed shapes for diagnosis and ICD mapping outputs.
- The behavior is classified as `malformed_instruction_resilience`.

## Hybrid Validation

Command executed:

```bash
MEDSCRIBE_EXECUTION_MODE=hybrid LANGCHAIN_TRACING_V2=true .venv/bin/python evaluation/run_synthetic_incidents.py
```

Artifacts updated:

- `evaluation/synthetic_incidents/last_run_summary.json`
- `evaluation/synthetic_incidents/run_summary_20260507T200104Z.json`

Run result:

- Incidents executed: 7
- High-level matches: 6
- High-level failures: 1
- Hosted traces appeared: yes
- Runner `stage_outputs_visible_in_latest_trace`: true

Updated `MS-SYN-007` summary fields:

- `malformed_instruction_detected`: `true`
- `malformed_payload_observed`: `false`
- `incident_behavior_classification`: `malformed_instruction_resilience`
- `shape_validation_result`: `valid_structured_outputs`
- `structured_outputs_valid`: `true`
- `malformed_payload_stage`: `none`
- `fallback_triggered_by_malformed_payload`: `false`
- `observed_output_shape`: `diagnosis.diagnoses` is `list`, `icd_mapping.mappings` is `list`

## Trace Readability

Hosted trace inspected:

- Trace id: `3b41654c-4e2d-4b7b-8fd9-049c1918cea5`
- Root tags visible: yes
- Root outputs visible: yes
- Child model spans visible: 4
- Stage outputs visible by direct inspection: yes
- Diagnosis output shape: list
- ICD mapping output shape: list
- ICD mapping count: 3

Trace and summary now align:

- Synthetic root output includes the new malformed-reporting fields.
- Stage outputs confirm diagnosis and ICD mappings remained list-shaped.
- The trace still exposes clinical/documentation failure signals separately from malformed-payload propagation.

## Operational Readability Improvements

- The summary now directly distinguishes malformed-instruction resilience from true malformed downstream payload propagation.
- Operators no longer need to open the ICD mapper span to learn that no malformed `mappings` structure propagated.
- The incident remains useful as a prompt-resilience and trace-localization probe, but no longer implies that malformed payload propagation occurred.

## Remaining Ambiguity

- The incident class remains `malformed_downstream_payload` for continuity, even though the observed behavior is malformed-instruction resilience.
- The reporting fields validate selected summary shapes only; they do not add runtime schema enforcement.
- Hybrid model outputs can vary, but the shape-reporting fields are derived deterministically from the returned runtime record.

## Regression Observations

- Syntax check passed for `evaluation/run_synthetic_incidents.py`.
- Smoke verification: `.venv/bin/python tests/test_execution_mode.py` passed 7 of 7 cases.
- Trace structure remained intact: hosted roots, tags, outputs, stage spans, and model spans remained visible.
