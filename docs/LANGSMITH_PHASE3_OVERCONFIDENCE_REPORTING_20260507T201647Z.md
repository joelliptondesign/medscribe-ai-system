# LangSmith Phase 3 Overconfidence Reporting Clarification

Timestamp: 2026-05-07T20:16:47Z

## Scope

Implemented a reporting/evaluator clarification for `MS-SYN-005` only. No runtime behavior, prompts, parser behavior, governance logic, routing logic, thresholds, datasets, model settings, or incident definitions were changed.

## Reporting Clarification Added

File modified: `evaluation/run_synthetic_incidents.py`

Added a summary-only helper for `ambiguous_case_overconfidence` incidents. The runner now adds these derived fields to `MS-SYN-005` summary output:

- `ambiguity_detected`
- `ambiguity_reasons`
- `diagnosis_count`
- `critic_confidence`
- `critic_recommended_status`
- `single_diagnosis_on_ambiguous_input`
- `insufficient_evidence_detected`
- `low_confidence_detected`
- `overconfidence_observed`
- `ambiguity_preserved`
- `incident_behavior_classification`
- `operational_interpretation`

These fields are derived from existing runtime outputs: parsed intake, diagnosis list, critic scores, governance decision, and reason codes.

## Before Interpretation

Before this clarification, `MS-SYN-005` reporting showed:

- runtime completed
- governance decision
- fallback status
- failure localization clue
- high-level expected match
- trace stage count

The high-level expected match confirmed that ambiguity was present or diagnoses existed, but it did not explicitly distinguish ambiguity preservation from actual overconfidence behavior.

## After Interpretation

Hybrid validation command:

```bash
MEDSCRIBE_EXECUTION_MODE=hybrid LANGCHAIN_TRACING_V2=true .venv/bin/python evaluation/run_synthetic_incidents.py
```

Generated summary artifact:

- `evaluation/synthetic_incidents/run_summary_20260507T201527Z.json`
- `evaluation/synthetic_incidents/last_run_summary.json`

Updated `MS-SYN-005` summary fields:

```json
{
  "governed_decision": "FAIL",
  "ambiguity_detected": true,
  "ambiguity_reasons": ["low_evidence_input"],
  "diagnosis_count": 0,
  "critic_confidence": 0.0,
  "critic_recommended_status": "fail",
  "single_diagnosis_on_ambiguous_input": false,
  "insufficient_evidence_detected": true,
  "low_confidence_detected": true,
  "overconfidence_observed": false,
  "ambiguity_preserved": true,
  "incident_behavior_classification": "ambiguity_preservation"
}
```

Operational interpretation now states that ambiguity was detected and preserved through low-confidence critic scoring and a non-pass governance outcome.

## Ambiguity Preservation vs Overconfidence

The updated report distinguishes:

- Ambiguity preservation: ambiguity is detected, insufficient evidence remains visible, critic confidence is low, and governance does not pass the case.
- True overconfidence: ambiguity is detected, but the run passes or produces a single high-confidence diagnosis on ambiguous evidence.

For the validation run, `MS-SYN-005` was classified as `ambiguity_preservation`, not true overconfidence.

## Trace Observations

Latest hosted `MS-SYN-005` trace inspected after validation:

- Root id: `3eef53f2-90b5-4a3d-bb8e-e8da12474189`
- Root start time: `2026-05-07 20:15:56.228343`
- Synthetic root outputs included the new reporting fields.
- Tags remained visible, including `incident_id:MS-SYN-005`, `incident_class:ambiguous_case_overconfidence`, and `ambiguous-case-overconfidence`.
- Stage outputs were visible for intake, triage, diagnosis, ICD mapper, critic, and governance.
- Child `ChatOpenAI` model spans remained visible for intake, triage, diagnosis, and critic.
- Token usage remained visible on child model spans.

The runner-level LangSmith visibility summary reported hosted roots visible and incident tags expected. Its generic `stage_outputs_visible_in_latest_trace` value was `false` because it checks the latest synthetic root overall, not the `MS-SYN-005` trace directly. Direct `MS-SYN-005` trace inspection showed all required stage outputs visible.

## Regression Observations

Smoke verification command:

```bash
.venv/bin/python tests/test_execution_mode.py
```

Result:

- `total_cases`: 7
- `passed`: 7
- `failed`: 0

Hybrid pack result:

- incident count: 7
- passed high level: 6
- failed high level: 1
- `MS-SYN-005`: completed, `FAIL`, high-level match true, ambiguity preserved true, overconfidence observed false

## Remaining Ambiguity

The clarification does not define product-level clinical semantics for how ambiguous evidence should be represented in the diagnosis stage. It only reports whether the existing runtime output looks like ambiguity preservation or true overconfidence. A future runtime-level uncertainty schema would require a separate brief.

## Implementation Status

Reporting clarification implemented. No runtime ambiguity semantics were introduced.
