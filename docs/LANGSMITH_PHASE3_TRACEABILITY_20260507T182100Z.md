# LangSmith Phase 3 Traceability - 2026-05-07T18:21:00Z

## Scope

Implemented a bounded governance traceability enhancement only. No governance thresholds, rule semantics, routing logic, prompts, datasets, model settings, or pass/revise/fail decision code were changed.

## Attribution Metadata Added

Files changed:

- `graph/nodes/governance_policy.py`
- `service/run_manager.py`

Governance output now includes `governance_result.governance_attribution` with:

- `governance_inputs_used`
- `governance_inputs_ignored`
- `governance_fail_drivers`
- `governance_rule_evaluations`
- `upstream_context_summary`

The `medscribe.governance_policy` LangSmith span metadata now also includes compact attribution fields:

- `governance_inputs_used`
- `governance_inputs_ignored`
- `upstream_context_summary`

## Governance Inputs Used

The governance policy explicitly reports these direct inputs:

- `critic_review.diagnosis_consistency_score`
- `critic_review.symptom_alignment_score`
- `critic_review.icd_specificity_score`
- `critic_review.confidence`
- `critic_review.recommended_status`
- `critic_review.reason_codes`

## Governance Inputs Ignored

The governance policy explicitly reports these available upstream signals as not directly consumed:

- `intake_data.symptoms`
- `intake_data.severity_descriptors`
- `intake_data.duration`
- `triage.level`
- `triage.rationale`
- `diagnoses`
- `icd_mappings`

## Hybrid Validation

Command executed:

```bash
MEDSCRIBE_EXECUTION_MODE=hybrid LANGCHAIN_TRACING_V2=true .venv/bin/python evaluation/run_synthetic_incidents.py
```

Run result:

- Summary artifact: `evaluation/synthetic_incidents/last_run_summary.json`
- Timestamped summary: `evaluation/synthetic_incidents/run_summary_20260507T181913Z.json`
- Incidents executed: 7
- High-level matches: 6
- High-level failures: 1
- `MS-SYN-003` decision: `FAIL`
- Hosted synthetic roots appeared: yes

## Updated MS-SYN-003 Trace Readability

Updated hosted trace inspected:

- Trace id: `1f896484-9a5b-4148-a2b5-0f5a95f6d761`
- Root tags visible: yes
- Root outputs visible: yes
- Stage outputs visible by direct inspection: yes
- Child model spans visible: 4

Observed `MS-SYN-003` governance attribution:

- Triage level in upstream context: `urgent`
- Governance final status: `FAIL`
- Attribution present in governance output: yes
- Attribution present in governance span metadata: yes
- Governance explicitly lists `triage.level` and `triage.rationale` as ignored direct inputs.
- Governance fail drivers list one fail driver:
  - input: `critic_review.icd_specificity_score`
  - score: `0.2`
  - applied rule: `RULE_ICD_SPECIFICITY_FAIL`
  - reason code: `LOW_ICD_SPECIFICITY`

Rule evaluations visible in trace:

- `critic_review.diagnosis_consistency_score`: score `0.5`, `REVISE`, `RULE_DIAGNOSIS_CONSISTENCY_REVISE`
- `critic_review.symptom_alignment_score`: score `0.8`, `PASS`, `RULE_SYMPTOM_ALIGNMENT_PASS`
- `critic_review.icd_specificity_score`: score `0.2`, `FAIL`, `RULE_ICD_SPECIFICITY_FAIL`
- `critic_review.confidence`: score `0.6`, `REVISE`, `RULE_CONFIDENCE_REVISE`
- `critic_review.recommended_status`: value `revise`, `REVISE`, `RULE_CRITIC_RECOMMENDATION_REVISE`

Upstream context summary visible in trace:

- symptoms: `chest pain`, `shortness of breath`
- severity descriptors: `sudden`, `severe`, `mild`
- duration: `null`
- diagnosis count: 3
- ICD mapping statuses: `PARTIAL_MATCH`, `PARTIAL_MATCH`, `PARTIAL_MATCH`
- triage rationale: urgent evaluation for sudden/severe chest pain and shortness of breath

## Operational Debugging Improvements

- The trace now separates direct governance inputs from upstream context that exists but is not directly consumed.
- The trace now shows that urgent triage was present but ignored by the governance policy as a direct input.
- The trace now identifies the precise fail-level rule input driving the final `FAIL`.
- The trace now makes it easier to distinguish a policy/rule failure from a missing trace-output problem.

## Remaining Ambiguity

- The final status still does not distinguish urgent safety escalation from coding/documentation failure.
- The attribution does not change whether governance should consume triage directly; it only makes current behavior explicit.
- The runner's coarse `stage_outputs_visible_in_latest_trace` field remained `false`, while direct hosted trace inspection showed stage outputs visible.
- Hybrid model outputs can vary between runs; the attribution structure is deterministic once stage outputs exist.

## Regression Observations

- Syntax check passed for `graph/nodes/governance_policy.py` and `service/run_manager.py`.
- Smoke verification: `.venv/bin/python tests/test_execution_mode.py` passed 7 of 7 cases.
- Trace structure remained intact: synthetic root, governed runtime path, stage spans, tags, outputs, and model spans remained visible.
