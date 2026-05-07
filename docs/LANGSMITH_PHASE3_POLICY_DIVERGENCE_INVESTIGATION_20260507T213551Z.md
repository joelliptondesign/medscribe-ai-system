# LangSmith Phase 3 Policy Divergence Investigation

Timestamp: 2026-05-07T21:35:51Z

## Scope

Investigation-only RCA for the existing policy-change divergence synthetic incident. No runtime behavior, prompts, governance logic, routing logic, evaluators, thresholds, datasets, model settings, or incident definitions were modified. No validation or regression reruns were performed.

## RCA Target Selected

- Incident id: `MS-SYN-006`
- Incident class: `policy_change_divergence`
- Input: `Synthetic demo case: adult reports joint pain and morning stiffness for six weeks, symptoms are persistent but not severe, and prior over-the-counter treatment helped only slightly.`
- Expected behavior: runtime completes and governance output exposes `policy_version`, `governance_version`, applied rules, and final status for later policy comparison.
- Expected failure mode: a future policy threshold change could diverge from the baseline governed decision.
- Latest local summary inspected: `evaluation/synthetic_incidents/last_run_summary.json`
- Latest hosted trace inspected: `fecbf84e-5bef-4f82-ac37-56129a487bae`, started `2026-05-07 20:57:04.610016`

## Current Evaluator Result

The synthetic evaluator treats `policy_change_divergence` as a high-level match when either `policy_version` or `governance_version` is present in the runtime summary. This confirms policy/governance metadata is visible, but it does not establish that a policy change caused any observed difference.

Latest local summary:

- `runtime_status`: `completed`
- `governed_decision`: `FAIL`
- `fallback_used`: `false`
- `high_level_expected_match`: `true`
- `failure_localization_clue`: `reason_codes:MISSING_SYMPTOMS,LOW_EVIDENCE_INPUT,NO_ICD_MATCH,LOW_DIAGNOSIS_CONSISTENCY,LOW_SYMPTOM_ALIGNMENT`

## Hosted Trace Observations

Six recent hosted `MS-SYN-006` synthetic roots were inspected:

- `fecbf84e-5bef-4f82-ac37-56129a487bae`, `2026-05-07 20:57:04.610016`: final `FAIL`
- `75fe607e-4ec1-4fa4-9abc-a9ed5e9b4181`, `2026-05-07 20:16:03.114601`: final `FAIL`
- `ca36e8ea-af1b-4ab9-a00c-75dbbb5b6c4f`, `2026-05-07 20:01:41.929840`: final `FAIL`
- `702a2315-a663-452f-8069-c04b693d7910`, `2026-05-07 18:38:00.493267`: final `FAIL`
- `bc629cce-d433-4044-b611-801f1b9bfe2e`, `2026-05-07 18:28:20.811088`: final `FAIL`
- `db460cb6-8613-489f-b5fc-5cb45d073eed`, `2026-05-07 18:19:54.464578`: final `FAIL`

All sampled roots completed, retained `MS-SYN-006` tags, reported no fallback, and returned high-level expected match true. The final status was reproducibly `FAIL` across the sampled hosted roots, but reason-code wording varied slightly across runs.

Latest trace visibility:

- realistic root latency: about `5.99s`
- child `ChatOpenAI` spans visible for intake, triage, diagnosis, and critic
- token usage visible on child model spans
- stage outputs visible for intake, triage, diagnosis, ICD mapper, critic, governance, and governed run

## Policy Divergence Observations

The latest hosted trace does not show a clean isolated policy-threshold divergence. It shows a stable fail-level governance result caused by fail-level critic inputs that originate upstream.

Latest trace intake output:

- `symptoms`: `[]`
- `duration`: `six weeks`
- `medications`: `["over-the-counter treatment"]`
- `severity_descriptors`: `["persistent", "not severe"]`
- `missing_fields`: `["symptoms"]`
- `ambiguity_flag`: `true`
- `ambiguity_reasons`: `["low_evidence_input"]`

Triage output:

- `level`: `escalate`
- rationale: intake lacks specific symptoms despite persistent duration, and ambiguity warrants further evaluation

Diagnosis output:

- `diagnoses`: `[]`

ICD mapping output:

- one `NO_MATCH_FOUND` mapping against the raw synthetic input text

Critic output:

- `diagnosis_consistency_score`: `0.2`
- `symptom_alignment_score`: `0.1`
- `icd_specificity_score`: `0.0`
- `confidence`: `0.3`
- `recommended_status`: `fail`
- `reason_codes`: `MISSING_SYMPTOMS`, `LOW_EVIDENCE_INPUT`, `NO_ICD_MATCH`

Governance output:

- `final_status`: `FAIL`
- `policy_version`: `policy_v1`
- `governance_version`: `governance_v1`
- applied rules: diagnosis consistency fail, symptom alignment fail, ICD specificity fail, confidence fail, critic recommendation fail
- governance fail drivers: all four critic metrics and critic recommendation
- policy simulation metadata present, with `applied: false`

## Runtime And Policy Code Observations

`service.run_manager.execute` records `policy_version` and `governance_version` in the summary and persists critic scores, governance decision, trace stage list, node diagnostics, and LangSmith metadata.

`graph.nodes.governance_policy.run` loads `governance/policy_rules.json`, evaluates four critic metrics against static pass/revise/fail thresholds, then applies critic recommended status. It also reports governance attribution showing direct governance inputs and ignored upstream context.

Current `governance/policy_rules.json` contains one active policy surface:

- `policy_version`: `policy_v1`
- `governance_version`: `governance_v1`
- pass/revise thresholds for diagnosis consistency, symptom alignment, ICD specificity, and confidence

The governance stage exposes policy metadata clearly, but there is no deterministic replay mechanism that applies alternative policy thresholds to the exact same frozen upstream stage outputs.

## Reproducibility And Comparison Limitations

The final `FAIL` behavior appears reproducible in recent hosted runs. However, that reproducibility does not make the policy comparison causally trustworthy because the failure is already overdetermined by upstream inputs:

- symptoms normalize to an empty list
- diagnosis is empty
- ICD mapping has no match
- critic recommends `fail`
- all governance metrics are in fail bands

Under these conditions, many plausible policy threshold changes would still return `FAIL`. A comparison run that differs later could be caused by upstream model/runtime variability rather than policy change.

The incident definition itself notes that this case captures a baseline governed trace for later policy comparison without implementing replay. Current tooling therefore supports baseline observability, not causal policy attribution.

## Operational Impact

Operational risk:

- Teams may misinterpret differences between live hybrid runs as policy drift when upstream model output or structured representation changed.
- The policy metadata is visible, but the workflow does not freeze upstream inputs for apples-to-apples comparison.
- A policy comparison can be contaminated by intake normalization drift, diagnosis variability, ICD mapping outcomes, or critic scoring changes.

Likely issue categories:

- replay limitation: high evidence
- attribution limitation: high evidence
- evaluator-driven limitation: high evidence
- representation drift: high evidence
- model variability: medium evidence
- governance-rule-driven fail: high evidence
- policy-threshold-driven divergence: low evidence in inspected traces

## Debugging Signal Assessment

Strong signals:

- Policy and governance versions are visible in runtime summaries.
- Governance attribution shows direct critic inputs, ignored upstream context, rule evaluations, fail drivers, and policy simulation metadata.
- Stage outputs are visible in hosted traces.
- Node diagnostics show live calls returned and no fallback was used.
- Child model spans and token usage are visible.

Weak or missing signals:

- No replay harness freezes intake, triage, diagnosis, ICD mapping, and critic outputs before applying alternative policy thresholds.
- No summary field states whether a policy comparison is causally interpretable.
- The evaluator marks the incident as matched when metadata exists, not when divergence is policy-caused.
- The policy metadata does not include a hash or snapshot of threshold values in the synthetic summary.
- Operators must manually compare upstream stage outputs across runs to determine whether a policy difference is contaminated by upstream drift.

## RCA Hypotheses

Primary hypothesis: current `MS-SYN-006` is useful as a governed baseline trace, but it does not provide causally trustworthy policy-divergence comparison because there is no deterministic replay or frozen upstream-output simulation.

Confidence: high.

Evidence:

- Incident notes explicitly say it captures a baseline governed trace for later policy comparison without implementing replay.
- Evaluator checks only that policy/governance version metadata is present.
- Hosted traces show policy metadata and governance attribution, but no alternate-policy application to the same upstream payload.

Secondary hypothesis: observed final `FAIL` is governance-rule-driven from critic fail metrics, not primarily policy-threshold divergence.

Confidence: high.

Evidence:

- Latest trace critic scores were `0.2`, `0.1`, `0.0`, and `0.3`, all fail-band values under current thresholds.
- Critic recommended `fail`.
- Governance fail drivers included all four critic metrics plus critic recommendation.

Secondary hypothesis: upstream representation drift contaminates policy interpretation.

Confidence: high.

Evidence:

- The raw input contains `joint pain` and `morning stiffness`, but the latest normalized intake had `symptoms: []`.
- Diagnosis output was empty and ICD mapping returned `NO_MATCH_FOUND`.
- Governance attribution shows those upstream signals in context, but governance only directly consumed critic metrics.

Secondary hypothesis: live hybrid variability remains a comparison risk even when final status is stable.

Confidence: medium.

Evidence:

- Six sampled hosted roots were all final `FAIL`, but reason-code wording varied across runs.
- Prior policy-simulation documentation noted hybrid comparison noise and absence of deterministic replay.

## Recommended Future Fix Surface

Smallest plausible future mutation surface:

- `evaluation/run_synthetic_incidents.py`
- optionally summary JSON artifacts under `evaluation/synthetic_incidents/`

Recommended reporting-only fields for `policy_change_divergence` summaries:

- `policy_version`
- `governance_version`
- `policy_threshold_snapshot`
- `critic_metric_snapshot`
- `governance_fail_driver_count`
- `policy_comparison_ready`
- `upstream_drift_risk_detected`
- `replay_required_for_causal_attribution`
- `policy_divergence_causally_trustworthy`
- `operational_interpretation`

Expected operational impact:

- Clarifies that current `MS-SYN-006` is a baseline trace, not a causal policy comparison.
- Separates policy metadata visibility from policy-caused divergence.
- Reduces risk of attributing live-run differences to policy changes when upstream outputs changed.

Expected regression risk:

- Low if limited to synthetic reporting fields.
- Medium if future work introduces an actual replay/simulation harness because it must define frozen payload contracts.

Likely validation strategy for a future reporting-only fix:

- Use existing `MS-SYN-006`; do not create a new incident.
- Run one hybrid traced synthetic incident pack.
- Confirm summary fields expose policy version, governance version, critic metric snapshot, fail drivers, and replay/causality flags.
- Inspect hosted `MS-SYN-006` trace for retained outputs/tags/model spans.
- Run `tests/test_execution_mode.py`.

Likely validation strategy for a future replay/simulation fix:

- Capture one frozen governed-run state immediately before governance.
- Apply current and candidate policy rules to that same frozen state.
- Compare only governance outputs and attribution.
- Keep upstream model calls out of the comparison loop.

## Remaining Unknowns

- No alternate policy threshold set was applied during this investigation.
- No deterministic replay artifact exists for the latest hosted trace.
- The investigation did not isolate why `joint pain` and `morning stiffness` were not retained as normalized symptoms in the latest hybrid trace.
- Product-level expectations for policy drift workflow are not fully defined.

## Implementation Status

No fixes were implemented. No runtime, evaluator, prompt, governance, routing, threshold, dataset, or model-setting changes were made.
