# LangSmith Phase 3 Reporting Clarification - 2026-05-07T18:29:26Z

## Scope

Implemented a reporting/evaluator clarification for `MS-SYN-003`-style governance override incidents. No governance behavior, thresholds, prompts, routing logic, datasets, model settings, incident classes, or pass/revise/fail outcomes were changed.

## Reporting Fields Added

File changed:

- `evaluation/run_synthetic_incidents.py`

For `governance_override` summaries, the runner now adds:

- `triage_level`
- `urgent_triage_detected`
- `governance_ignored_triage`
- `documentation_or_coding_failure`
- `governance_failure_reason`
- `governance_vs_triage_divergence`
- `operational_interpretation`

## Before/After Interpretation

Before:

- `MS-SYN-003` summary showed runtime status, governed decision, reason-code localization, high-level match, and trace stage count.
- The summary did not explicitly distinguish urgent triage from coding/documentation-driven governance failure.

After:

- `MS-SYN-003` still reports governed decision `FAIL`.
- `high_level_expected_match` remains `true`.
- The summary now reports urgent triage detected, direct triage input ignored by governance, documentation/coding failure detected, and governance-vs-triage divergence detected.

## Hybrid Validation

Command executed:

```bash
MEDSCRIBE_EXECUTION_MODE=hybrid LANGCHAIN_TRACING_V2=true .venv/bin/python evaluation/run_synthetic_incidents.py
```

Artifacts updated:

- `evaluation/synthetic_incidents/last_run_summary.json`
- `evaluation/synthetic_incidents/run_summary_20260507T182746Z.json`

Run result:

- Incidents executed: 7
- High-level matches: 6
- High-level failures: 1
- `MS-SYN-003` governed decision: `FAIL`
- `MS-SYN-003` high-level match: `true`

Updated `MS-SYN-003` summary fields:

- `triage_level`: `urgent`
- `urgent_triage_detected`: `true`
- `governance_ignored_triage`: `true`
- `documentation_or_coding_failure`: `true`
- `governance_failure_reason`: `coding_or_documentation_specificity`
- `governance_vs_triage_divergence`: `true`
- `operational_interpretation`: urgent triage was detected, while final governance failure is explained by coding/documentation specificity signals rather than direct triage urgency.

## Trace Readability

Hosted trace inspected:

- Trace id: `6112c18d-3969-4acc-ad3f-f308359f035b`
- Root tags visible: yes
- Root outputs visible: yes
- Child model spans visible: 4
- Stage outputs visible by direct inspection: yes

Trace and summary now align:

- Synthetic root output includes the new reporting fields.
- Governance attribution remains visible in the governance stage output.
- Triage remains `urgent`.
- Governance attribution still lists `triage.level` and `triage.rationale` as ignored direct inputs.
- Governance fail driver remains `critic_review.icd_specificity_score` via `RULE_ICD_SPECIFICITY_FAIL`.

## Operational Readability Improvements

- The synthetic summary now communicates the CDI/RCM distinction directly: the case can be urgent while the governance failure is documentation/coding-specific.
- The incident no longer requires opening every stage span to see the governance-vs-triage distinction.
- The reporting layer now preserves final governance status while adding operational interpretation around why the status can coexist with urgent triage.

## Remaining Ambiguity

- The runtime final status still does not encode separate categories for urgent safety escalation and documentation/coding failure.
- The reporting layer marks `governance_ignored_triage` based on current governance-policy behavior; it does not alter that behavior.
- Hybrid model outputs can vary between runs, but the reporting fields are derived deterministically from the returned runtime record.
- The runner's coarse `stage_outputs_visible_in_latest_trace` field remained `false`, while direct hosted trace inspection showed stage outputs visible.

## Regression Observations

- Syntax check passed for `evaluation/run_synthetic_incidents.py`.
- Smoke verification: `.venv/bin/python tests/test_execution_mode.py` passed 7 of 7 cases.
- Trace structure remained intact: hosted roots, tags, outputs, stage spans, and model spans remained visible.
