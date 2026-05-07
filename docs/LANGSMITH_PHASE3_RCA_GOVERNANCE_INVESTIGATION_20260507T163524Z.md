# LangSmith Phase 3 Governance RCA Investigation - 2026-05-07T16:35:24Z

## RCA Target Selected

- Incident: `MS-SYN-003`
- Incident class: `governance_override`
- Target type: governance/policy divergence between urgent triage evidence and final governance status.
- Selection reason: the incident is operationally plausible, observable in hosted traces, and suitable for a later bounded before/after comparison because multiple existing traces show different governance outcomes across deterministic and live hybrid executions.

No fixes or behavior changes were performed in this investigation.

## Trace Observations

Primary hosted trace inspected:

- Trace id: `9ed34233-b771-4ba2-ab36-7daec5136283`
- Root name: `medscribe.synthetic_incident`
- Root start: `2026-05-07 16:19:05.784514`
- Tags included `governance-override`, `incident_class:governance_override`, `incident_id:MS-SYN-003`, `medscribe`, `phase2`, and `synthetic-incident`.
- Metadata included dataset name/version, expected failure mode, incident class, incident id, and non-PHI marker.
- Trace contained 12 runs and 4 child model spans.

Stage outputs:

- Intake parsed symptoms as `chest pain` and `shortness of breath`, severity descriptors as `sudden`, `severe`, and `mild`, and missing field `duration`.
- Triage returned level `urgent` with rationale that sudden/severe chest pain and shortness of breath require immediate attention.
- Diagnosis returned `acute coronary syndrome`, `pulmonary embolism`, and `aortic dissection`.
- ICD mapper mapped all three diagnoses to `R07.9` / `Chest pain, unspecified` with status `PARTIAL_MATCH`.
- Critic returned recommended status `revise`, confidence `0.6`, diagnosis consistency `0.5`, symptom alignment `0.8`, ICD specificity `0.2`, and reason codes `DIAGNOSIS_CONFLICT`, `ICD_CODE_MISMATCH`, and `MISSING_DURATION`.
- Governance returned final status `FAIL`, escalation required `true`, policy version `policy_v1`, governance version `governance_v1`, and applied rules:
  - `RULE_DIAGNOSIS_CONSISTENCY_REVISE`
  - `RULE_SYMPTOM_ALIGNMENT_PASS`
  - `RULE_ICD_SPECIFICITY_FAIL`
  - `RULE_CONFIDENCE_REVISE`
  - `RULE_CRITIC_RECOMMENDATION_REVISE`

Existing trace comparison for reproducibility:

- `9ed34233-b771-4ba2-ab36-7daec5136283`, started `2026-05-07 16:19:05.784514`: triage `urgent`, critic `revise`, ICD specificity score `0.2`, governance `FAIL`.
- `0c9a6ada-e858-48a1-b079-3392fb6919f2`, started `2026-05-07 16:10:14.714302`: triage `urgent`, critic `revise`, ICD specificity score `0.2`, governance `FAIL`.
- `76ad3112-d39b-4768-ba46-245d0a5879ef`, started `2026-05-07 14:39:46.688250`: triage `urgent`, critic `pass`, ICD specificity score `1.0`, governance `PASS`.

## Execution-Flow Observations

- `service.run_manager.execute` runs a fixed linear pipeline: intake parser, triage engine, diagnosis engine, ICD mapper, critic, and governance policy.
- The governance stage receives `critic_review` as its traced input.
- `graph.nodes.governance_policy.run` loads `governance/policy_rules.json`, evaluates four critic metrics against static pass/revise thresholds, then applies critic recommended status.
- The inspected governance code does not directly consume triage level, urgent symptoms, red-flag terms, diagnosis labels, or incident metadata.
- The synthetic incident runner classifies `governance_override` as a high-level match when policy/governance version fields are present. That evaluator logic checks traceability of governance output, not clinical desirability of the final status.

## Governance/Policy Observations

- The final `FAIL` is governance-rule-driven from critic evidence, especially ICD specificity.
- Triage correctly identifies the case as urgent, but that urgent routing evidence is not a direct governance policy input.
- The governance policy escalates monotonically: any metric in the fail band can raise the final status to `FAIL`, and later pass-band metrics do not reduce that status.
- In the primary trace, symptom alignment passed, but ICD specificity score `0.2` was below the revise threshold `0.4`, so `RULE_ICD_SPECIFICITY_FAIL` set the status to `FAIL`.
- Critic recommended `revise`, not `fail`, so the critic recommendation alone did not cause the final `FAIL`.
- Missing duration and partial ICD specificity contributed to the final result even though the clinical risk signal was already urgent.

## Root-Cause Hypotheses

Primary hypothesis: governance-rule-driven divergence from urgent triage context.

- Confidence: high.
- Evidence: both recent live hybrid `MS-SYN-003` traces show triage `urgent`, critic `revise`, ICD specificity `0.2`, and governance `FAIL`; governance code only evaluates critic metrics and recommendation, not triage level or urgent symptom semantics.
- Why current behavior occurs: the policy treats low ICD specificity as a fail-level governance condition regardless of whether the case is already being escalated as urgent.
- Why this may be undesirable operationally: for a high-risk chest-pain/shortness-of-breath case, `FAIL` may conflate documentation/coding insufficiency with urgent safety escalation. An operator may need to distinguish "urgent escalation due to red flags" from "failed due to mapping specificity" even when both require attention.

Secondary hypothesis: upstream ICD mapping specificity creates a governance failure cascade.

- Confidence: medium-high.
- Evidence: all three serious diagnoses mapped to the nonspecific chest pain code `R07.9` with `PARTIAL_MATCH`, producing low ICD specificity and critic reason `ICD_CODE_MISMATCH`.
- Why current behavior occurs: the critic scores and governance rules penalize partial ICD matches strongly.
- Why this may be undesirable operationally: ICD specificity failure can dominate the final governance status and obscure whether the risk response itself was appropriate.

Secondary hypothesis: synthetic evaluator is too coarse for governance override quality.

- Confidence: medium.
- Evidence: `MS-SYN-003` remains `high_level_expected_match: true` as long as governance metadata/version fields exist, regardless of whether final status is `PASS`, `REVISE`, or `FAIL`.
- Why current behavior occurs: the evaluator currently validates that governance output is present and inspectable, not that the override semantics are operationally desirable.
- Why this may be undesirable operationally: the incident can remain green even when the investigation target is the quality and interpretability of the override decision.

## Missing Or Difficult-To-Interpret Information

- The governance output does not state which upstream evidence source caused each critic metric score.
- The trace does not include a separate explicit routing-policy decision beyond triage level.
- The final status vocabulary does not distinguish safety escalation from quality/coding failure.
- The synthetic evaluator summary does not capture governance-vs-triage disagreement.
- No UI screenshots were captured; this investigation used hosted LangSmith API trace inspection and local code/artifact inspection.

## Bounded Recommended Fix Surface

Smallest plausible future mutation surface:

- `evaluation/run_synthetic_incidents.py` if the next phase only needs better detection/reporting of governance override quality in the synthetic pack.
- `graph.nodes.governance_policy` and `governance/policy_rules.json` only if the next phase explicitly changes policy semantics for urgent-triage cases.
- A lightweight governance trace/evaluation metadata addition only if the next phase needs clearer attribution without changing final decisions.

Expected operational impact by option:

- Evaluator-only change: improves incident detection and before/after comparison quality without changing runtime behavior.
- Governance metadata-only change: improves trace interpretability with low behavioral risk.
- Policy semantic change: can distinguish urgent safety escalation from coding-specific failure, but carries higher regression risk because it affects governed final statuses.

Expected regression risk:

- Low for evaluator-only reporting.
- Low to medium for metadata-only attribution.
- Medium to high for policy semantics because fail/revise/pass decisions may change across unrelated governed runs.

Expected trace/eval behavior after a future fix:

- Traces should still show `medscribe.synthetic_incident`, `medscribe.governed_run`, six stage spans, and child model spans in hybrid mode.
- The selected incident should expose whether urgent triage was preserved into governance attribution.
- A governance-focused evaluator should mark the case as divergent when urgent triage and final policy failure are not separately explained.

## Expected Validation Approach For Next Phase

- Use existing `MS-SYN-003` as the before reference; do not create a new incident class.
- Run exactly one after comparison in hybrid mode if a future fix is approved.
- Compare triage level, critic scores, governance applied rules, final status, reason codes, root outputs, tags, and child model-span visibility.
- Run the existing smoke test after any future mutation.
- Check that unrelated synthetic incidents do not change at a high level unless the approved fix explicitly targets shared governance semantics.

## Remaining Unknowns

- Whether the desired product behavior is to keep `FAIL` for urgent high-risk cases with low ICD specificity or to separate urgent escalation from documentation/coding failure.
- Whether policy consumers interpret `FAIL` as clinical escalation, quality failure, or both.
- Whether the ICD mapper should produce diagnosis-specific codes for high-risk differentials or whether nonspecific symptom coding is acceptable for this demo workflow.
- Whether governance should consume triage level directly or only consume critic-derived scores.
