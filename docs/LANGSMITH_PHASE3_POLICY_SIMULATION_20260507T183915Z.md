# LangSmith Phase 3 Policy Simulation - 2026-05-07T18:39:15Z

## Scope

Performed one bounded policy-simulation experiment for `MS-SYN-003`. This was not a broad governance redesign. Prompts, datasets, routing architecture, model settings, incident classes, traceability enhancements, and reporting enhancements were preserved.

## Baseline Semantics

Baseline artifact:

- `evaluation/synthetic_incidents/last_run_summary.json` from run started at `2026-05-07T18:27:46Z`

Baseline `MS-SYN-003` behavior:

- Triage level: `urgent`
- Urgent triage detected: `true`
- Governance ignored triage directly: `true`
- Documentation/coding failure: `true`
- Governance failure reason: `coding_or_documentation_specificity`
- Governance-vs-triage divergence: `true`
- Final governed decision: `FAIL`
- High-level expected match: `true`

Baseline interpretation:

- The case was triaged as urgent, but final governance status was `FAIL`.
- The failure was explained by coding/documentation specificity signals, not direct triage urgency.
- Prior traceability work showed the fail driver as `critic_review.icd_specificity_score` via `RULE_ICD_SPECIFICITY_FAIL`.

## Experimental Policy Semantics

Experimental behavior added:

If all of the following are true:

- triage level is `urgent` or `escalate`
- baseline governance status would be `FAIL`
- the critic recommendation is `revise`
- there is exactly one fail driver
- the fail driver is `critic_review.icd_specificity_score`

Then:

- return final status `REVISE_ESCALATED`
- preserve `escalation_required = true`
- preserve critic metrics
- preserve governance attribution metadata
- preserve existing trace structure and stage outputs

Exact code surfaces changed:

- `graph/nodes/governance_policy.py`
- `evaluation/run_synthetic_incidents.py`

The evaluator/reporting update records `experimental_escalated_review` for governance-override summaries and treats `REVISE_ESCALATED` as an escalated divergent governance state for reporting.

## Hybrid Comparison Run

Command executed:

```bash
MEDSCRIBE_EXECUTION_MODE=hybrid LANGCHAIN_TRACING_V2=true .venv/bin/python evaluation/run_synthetic_incidents.py
```

Artifacts:

- `evaluation/synthetic_incidents/last_run_summary.json`
- `evaluation/synthetic_incidents/run_summary_20260507T183727Z.json`

Run result:

- Incidents executed: 7
- High-level matches: 6
- High-level failures: 1
- Hosted roots visible: yes
- Runner `stage_outputs_visible_in_latest_trace`: true

## Before/After Governance Outputs

`MS-SYN-003` before:

- Final status: `FAIL`
- Escalation required: true
- Fail driver: `critic_review.icd_specificity_score`
- Rule: `RULE_ICD_SPECIFICITY_FAIL`
- Reporting: urgent triage detected, coding/documentation failure detected, governance-vs-triage divergence detected.

`MS-SYN-003` after:

- Final status: `REVISE_ESCALATED`
- Escalation required: true
- Applied rules included:
  - `RULE_DIAGNOSIS_CONSISTENCY_REVISE`
  - `RULE_SYMPTOM_ALIGNMENT_PASS`
  - `RULE_ICD_SPECIFICITY_FAIL`
  - `RULE_CONFIDENCE_PASS`
  - `RULE_CRITIC_RECOMMENDATION_REVISE`
  - `EXPERIMENTAL_RULE_URGENT_TRIAGE_ICD_SPECIFICITY_ESCALATED_REVIEW`
- Reason codes included `EXPERIMENTAL_URGENT_TRIAGE_CODING_REVIEW`.
- Policy simulation metadata reported:
  - `enabled: true`
  - `applied: true`
  - `experimental_status: REVISE_ESCALATED`
  - `preserves_escalation_required: true`

## Trace Observations

Updated `MS-SYN-003` trace inspected:

- Trace id: `eb014e8a-34bc-4d29-a3d0-dae67c80e7fc`
- Root tags visible: yes
- Root outputs visible: yes
- Child model spans visible: 4
- Stage outputs visible by direct inspection: yes
- Triage level: `urgent`
- Governance final status: `REVISE_ESCALATED`
- Escalation required: true
- Fail driver remained `critic_review.icd_specificity_score`, score `0.2`, rule `RULE_ICD_SPECIFICITY_FAIL`.
- Experimental policy input used: `triage.level`.

## Evaluator And Reporting Observations

Updated `MS-SYN-003` summary:

- `governed_decision`: `REVISE_ESCALATED`
- `urgent_triage_detected`: true
- `governance_ignored_triage`: false
- `documentation_or_coding_failure`: true
- `governance_failure_reason`: `coding_or_documentation_specificity`
- `governance_vs_triage_divergence`: true
- `experimental_escalated_review`: true

Interpretation change:

- Before, the summary showed urgent triage plus final `FAIL`.
- After, the summary shows urgent triage plus an experimental escalated review status, while still preserving the coding/documentation-specific failure driver.

## Unrelated Incident Observations

Observed decisions in the after run:

- `MS-SYN-001`: `FAIL`
- `MS-SYN-002`: `FAIL`
- `MS-SYN-003`: `REVISE_ESCALATED`
- `MS-SYN-004`: `FAIL`
- `MS-SYN-005`: `FAIL`
- `MS-SYN-006`: `FAIL`
- `MS-SYN-007`: `FAIL`

Compared with the immediate pre-experiment summary, `MS-SYN-004` changed from `REVISE` to `FAIL`. The experimental policy condition was not targeted at `MS-SYN-004`; the observed difference is consistent with live hybrid model-output variability or shared critic/governance inputs, and should be treated as an observed comparison-run consequence rather than proof of policy-simulation impact.

## Operational Interpretation Changes

- `REVISE_ESCALATED` made the urgent safety dimension easier to see than plain `FAIL`.
- The trace still preserves the coding/documentation fail driver, so the experiment did not hide the ICD-specificity issue.
- The new status is more descriptive for this incident, but it introduces a fourth status that downstream consumers may not understand.

## Workflow And Tooling Observations

- LangSmith trace attribution made the simulation easier to inspect because fail drivers and upstream triage context were visible in the same governance output.
- The synthetic summary made before/after comparison easier because `experimental_escalated_review` appeared at the incident root.
- Hybrid comparison remains noisy: model-backed stage outputs can vary between runs, making it difficult to attribute unrelated incident changes to policy code alone without deterministic replay.
- Existing smoke tests do not validate final-status vocabulary beyond execution-mode configuration.

## Regression Verification

- Syntax check passed for `graph/nodes/governance_policy.py` and `evaluation/run_synthetic_incidents.py`.
- Smoke verification: `.venv/bin/python tests/test_execution_mode.py` passed 7 of 7 cases.

## Limitations

- This experiment introduced a nonstandard status label and did not update downstream consumers beyond the synthetic reporting layer.
- The experiment did not determine whether `REVISE_ESCALATED` is product-correct.
- No broader governance redesign was attempted.
- No deterministic replay harness was added, so live hybrid comparison remains subject to model variability.
