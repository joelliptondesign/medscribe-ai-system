# LangSmith Phase 3 Semantic RCA - 2026-05-07T16:18:08Z

## RCA Target Selected

- Incident: `MS-SYN-002`
- Incident class: `incorrect_icd_specificity`
- Target type: evaluator mismatch in the existing synthetic incident runner.

## Pre-Fix Target Note

The selected target is the synthetic evaluator's high-level match logic for ICD specificity. In the latest hybrid run, `MS-SYN-002` was marked `high_level_expected_match: true` even though the hosted trace showed `NO_MATCH_FOUND` ICD output and the governed decision was `FAIL`.

No code changes had been made at the time this target was selected.

## Observed Symptom

- Before reference artifact: `evaluation/synthetic_incidents/last_run_summary.json` from hybrid run started at `2026-05-07T16:09:55Z`.
- Before summary for `MS-SYN-002`: runtime `completed`, governed decision `FAIL`, failure clue `NO_ICD_MATCH_FOUND`, and `high_level_expected_match: true`.
- Hosted trace inspected: `e9714d84-aba0-416e-a8ba-b7cb81dcef4e`.
- The trace showed an ICD mapper output with one mapping whose status was `NO_MATCH_FOUND` and whose `icd_code` and `icd_label` were empty.
- The critic output included `NO_ICD_MATCH_FOUND`, zero diagnosis consistency, zero symptom alignment, zero ICD specificity, and recommended status `fail`.
- The governance output was `FAIL` with `RULE_ICD_SPECIFICITY_FAIL`.

## Trace Observations

- Incident tags and metadata were visible on the selected hosted root.
- The trace contained 12 runs, including 4 model spans.
- Stage outputs were visible for diagnosis, ICD mapper, critic, governance, and the synthetic incident root.
- The diagnosis stage returned an empty diagnoses list for the selected case.
- The ICD mapper localized the issue to a non-specific/no-match mapping rather than a parser crash, routing failure, or governance-only policy decision.

## Root Cause Hypothesis

The failure pattern originated in the synthetic evaluator logic. The `incorrect_icd_specificity` high-level match returned true whenever `record["icd_mapping"]["mappings"]` was present, even when that list contained only a `NO_MATCH_FOUND` mapping with empty ICD code and label.

Classification: evaluator-driven mismatch.

Reproducibility: deterministic at the evaluator layer. Given the before run record, the old condition returns true because the mappings field exists; the revised condition returns false unless a specific `OK` mapping with non-empty code and label exists.

## Bounded Fix Applied

- File changed: `evaluation/run_synthetic_incidents.py`.
- Scope: synthetic incident evaluator logic only.
- Change: `incorrect_icd_specificity` now requires at least one mapping with:
  - `status` equal to `OK`
  - non-empty `icd_code`
  - non-empty `icd_label`
- No runtime architecture, prompts, routing, governance logic, model settings, or incident classes were changed.
- Syntax check: `PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m py_compile evaluation/run_synthetic_incidents.py` completed successfully; the temporary compile cache was removed afterward.

## Before/After Comparison Summary

Before reference:

- Existing hybrid summary started at `2026-05-07T16:09:55Z`.
- `MS-SYN-002`: decision `FAIL`, high-level match `true`.
- Hosted trace showed `NO_MATCH_FOUND` ICD mapping and governance `FAIL`.

After command:

```bash
MEDSCRIBE_EXECUTION_MODE=hybrid LANGCHAIN_TRACING_V2=true .venv/bin/python evaluation/run_synthetic_incidents.py
```

After artifacts:

- `evaluation/synthetic_incidents/last_run_summary.json`
- `evaluation/synthetic_incidents/run_summary_20260507T161850Z.json`

After summary:

- Incidents executed: 7.
- High-level matches: 6.
- High-level failures: 1.
- `MS-SYN-002`: decision `FAIL`, high-level match `false`.
- The other six incidents retained `high_level_expected_match: true`.

Hosted after-run inspection:

- Hosted roots visible since after-run start: 7.
- Roots with child model spans: 7.
- Child model spans visible: 28.
- Incident tags/metadata visible: yes.
- Root outputs visible: yes.
- Stage outputs visible by direct hosted trace inspection: yes.
- Target after-run trace for `MS-SYN-002`: `a071fd48-f2b8-49b9-ba4e-a12c050de0eb`.

Note: the runner's coarse `langsmith_visibility.stage_outputs_visible_in_latest_trace` field was `false` in the after summary, but direct trace inspection across the seven after-run roots found stage outputs visible.

## Regression Observations

- Smoke verification: `.venv/bin/python tests/test_execution_mode.py` passed 7 of 7 cases.
- High-level synthetic regression surface: no unrelated incident changed from matched to unmatched. Only `MS-SYN-002`, the selected target, moved from matched to unmatched.
- Obvious operational degradation: not observed. The after run completed all seven incidents and hosted traces retained model spans, tags, root outputs, and stage outputs.
- Runtime decisions are model-backed and may vary between hybrid runs. The evaluator correction is deterministic for records containing no specific `OK` ICD mapping.

## Operational Lessons

- A synthetic probe can appear green if its evaluator checks only structural presence rather than semantic adequacy.
- ICD specificity checks need to distinguish an existing mapping container from a clinically usable mapping.
- Hosted trace outputs are useful for grounding evaluator corrections because they show the exact stage where the mismatch appears.

## Remaining Limitations

- This cycle corrected one evaluator mismatch only.
- The runtime still produced `NO_MATCH`/low-specificity behavior for the selected live hybrid case; that runtime semantic behavior was not changed in this cycle.
- The runner's built-in LangSmith visibility summary remains coarse and can disagree with direct trace inspection on stage-output visibility.
