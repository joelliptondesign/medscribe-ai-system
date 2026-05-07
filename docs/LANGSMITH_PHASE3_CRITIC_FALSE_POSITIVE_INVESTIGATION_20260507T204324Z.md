# LangSmith Phase 3 Critic False-Positive Investigation

Timestamp: 2026-05-07T20:43:24Z

## Scope

Investigation-only RCA for the existing critic false-positive synthetic incident. No runtime behavior, prompts, governance logic, routing logic, evaluators, thresholds, datasets, model settings, or incident definitions were modified. No validation or regression reruns were performed.

## RCA Target Selected

- Incident id: `MS-SYN-004`
- Incident class: `critic_false_positive`
- Input: `Synthetic demo case: adult reports mild headache after skipped lunch, no fever, no neurologic symptoms, no trauma, and symptoms improved after rest.`
- Expected behavior: runtime completes and critic output is inspectable for potentially excessive concern or low-confidence scoring.
- Expected failure mode: critic may flag a low-risk case as problematic despite coherent benign evidence.
- Latest local summary inspected: `evaluation/synthetic_incidents/last_run_summary.json`
- Latest hosted trace inspected: `ad36daec-ff9e-4257-b577-4c1a12369e8f`, started `2026-05-07 20:15:48.661919`

## Current Evaluator Result

The current synthetic evaluator marks `critic_false_positive` as a high-level match when `record.get("scores") is not None`. This confirms critic output exists, but it does not classify whether the critic behavior is a true false-positive, a justified caution, or a borderline revise.

Latest local summary:

- `runtime_status`: `completed`
- `governed_decision`: `REVISE`
- `fallback_used`: `false`
- `high_level_expected_match`: `true`
- `failure_localization_clue`: `reason_codes:AMBIGUITY_IN_DATA,PARTIAL_MATCH_ICD,MISSING_DURATION,LOW_DIAGNOSIS_CONSISTENCY,LOW_SYMPTOM_ALIGNMENT`

## Hosted Trace Observations

Five recent hosted `MS-SYN-004` synthetic roots were inspected:

- `ad36daec-ff9e-4257-b577-4c1a12369e8f`, `2026-05-07 20:15:48.661919`: final `REVISE`
- `d246c19f-8451-4d12-8990-b01b69b262f6`, `2026-05-07 20:01:29.013394`: final `FAIL`
- `ba36a464-5fc5-472d-9217-51fcb5b3190a`, `2026-05-07 18:37:45.512444`: final `FAIL`
- `830a19d9-7157-4b55-8c4b-9d432ede54bf`, `2026-05-07 18:28:06.608362`: final `REVISE`
- `6ed7faa3-cc94-4665-b3f8-65947eba576b`, `2026-05-07 18:19:39.648930`: final `FAIL`

All sampled roots completed, retained incident tags, and reported no fallback. The final status varied between `REVISE` and `FAIL`, indicating that the broad false-positive pattern is reproducible but the severity is not deterministic under live hybrid execution.

Latest trace `ad36daec-ff9e-4257-b577-4c1a12369e8f` showed:

- realistic root latency: about `7.57s`
- child `ChatOpenAI` spans visible for intake, triage, diagnosis, and critic
- token usage visible on child model spans
- stage outputs visible for intake, triage, diagnosis, ICD mapper, critic, governance, and governed run

## Critic And Stage Observations

Latest trace intake output:

- `symptoms`: `["headache"]`
- `severity_descriptors`: `["mild"]`
- `duration`: `null`
- `missing_fields`: `["duration"]`
- `ambiguity_flag`: `true`
- `ambiguity_reasons`: `["low_evidence_input"]`

Triage output:

- `level`: `routine`
- rationale: mild headache, missing duration, not severe enough for urgent care, routine follow-up appropriate

Diagnosis output:

- `diagnoses`: `["tension headache", "migraine", "sinus headache"]`

ICD mapping output:

- `tension headache` -> `G44.209`, `Tension-type headache, unspecified`, `OK`
- `migraine` -> `R51.9`, `Headache, unspecified`, `PARTIAL_MATCH`
- `sinus headache` -> `R51.9`, `Headache, unspecified`, `PARTIAL_MATCH`

Critic output:

- `diagnosis_consistency_score`: `0.7`
- `symptom_alignment_score`: `0.6`
- `icd_specificity_score`: `0.5`
- `confidence`: `0.65`
- `recommended_status`: `revise`
- `reason_codes`: `AMBIGUITY_IN_DATA`, `PARTIAL_MATCH_ICD`, `MISSING_DURATION`
- summary: data requires clarification on symptom duration and diagnosis specificity

Governance output:

- `final_status`: `REVISE`
- `escalation_required`: `true`
- applied revise-level rules for diagnosis consistency, symptom alignment, ICD specificity, confidence, and critic recommendation
- no governance fail drivers in the latest trace
- governance attribution showed direct inputs came from critic metrics/recommendation/reason codes, while raw intake, triage, diagnoses, and ICD mappings were ignored as direct policy inputs

## False-Positive Assessment

The latest trace looks like a borderline or over-strict `REVISE`, not a clean catastrophic false-positive:

- The input describes a low-risk presentation with mild headache, no fever, no neurologic symptoms, no trauma, and improvement after rest.
- The intake normalizer discarded or did not preserve several reassuring negatives and contextual details as structured fields.
- Missing duration remained visible even though the narrative contained temporal context about skipped lunch and improvement after rest.
- The diagnosis stage produced three possible headache diagnoses from a mild single-symptom case.
- Two of three ICD mappings were partial matches.
- The critic flagged uncertainty and specificity issues rather than directly inventing a high-risk concern.

Operationally, the critic's caution may be defensible from the structured payload it received, but it is plausibly a false-positive relative to the full raw narrative because reassuring context is not preserved in the structured stage inputs that the critic evaluates.

## Code Path Observations

`evaluation/run_synthetic_incidents.py` marks `critic_false_positive` as matched when critic scores exist. That evaluator checks inspectability, not false-positive classification.

`graph/nodes/diagnosis_engine.py` supports a single-symptom headache rule in deterministic mode and hybrid mode accepts any normalized diagnosis list up to three items. The latest hybrid trace produced three headache-related diagnoses.

`graph/nodes/icd_mapper.py` maps exact lookup hits as `OK`, then falls back to symptom-based partial matches. In the latest trace, `tension headache` mapped as `OK`, while `migraine` and `sinus headache` became partial headache-code matches.

`graph/nodes/critic.py` penalizes low-evidence presentations when symptom count is one or fewer and triage is not urgent. It also penalizes partial ICD mappings and computes confidence from diagnosis, symptom, and ICD scores.

`graph/nodes/governance_policy.py` evaluates only critic metrics and critic recommendation against static thresholds. In the latest trace, the critic's scores were above fail thresholds but below pass thresholds, so governance preserved the issue as `REVISE`. Earlier hosted traces show the same incident sometimes escalated to `FAIL`.

## Operational Impact

The operational risk is false escalation or unnecessary review for a low-risk, improving case. This can reduce trust in the critic because the pipeline summary says revise or fail without clearly showing how much of that decision came from genuine clinical concern versus structured-data loss, diagnosis proliferation, or ICD partial-match artifacts.

Likely issue categories:

- critic calibration: medium evidence
- evaluator weakness: high evidence
- ICD mapping ambiguity: medium evidence
- symptom/diagnosis alignment mismatch: medium evidence
- governance amplification: medium evidence across sampled traces, but low evidence in the latest trace because it remained `REVISE`
- reporting ambiguity: high evidence

## Debugging Signal Assessment

Strong signals:

- Hosted trace roots and child stage spans are visible.
- Critic scores, reason codes, recommendation, and confidence are visible.
- Governance attribution clearly shows the critic metrics that drove `REVISE`.
- ICD mapping outputs expose which diagnoses received `OK` versus `PARTIAL_MATCH`.
- Node diagnostics show live calls returned and no fallback was used.

Weak or missing signals:

- The summary does not classify whether the critic issue is likely false-positive or justified.
- Reassuring negative evidence from the raw input is not represented as structured fields for downstream comparison.
- The summary does not report diagnosis count, partial ICD count, low-risk narrative markers, or whether governance amplified critic concern from `revise` to `fail`.
- The evaluator treats critic inspectability as success and does not score false-positive detection quality.
- Operators must manually correlate raw narrative, parsed intake omissions, diagnosis proliferation, ICD partial matches, critic scores, and governance thresholds.

## RCA Hypotheses

Primary hypothesis: the apparent false-positive originates from structured representation loss plus diagnosis/ICD broadening, with the critic responding to the structured payload rather than the full low-risk narrative.

Confidence: medium-high.

Evidence:

- Latest trace parsed only `headache`, `mild`, missing duration, and ambiguity, while reassuring details such as no fever, no neurologic symptoms, no trauma, skipped lunch, and improvement after rest were not preserved as structured fields.
- Diagnosis produced three possible headache labels.
- ICD mapper produced two partial matches.
- Critic reason codes focused on ambiguity, partial ICD match, and missing duration.

Secondary hypothesis: the synthetic evaluator is too coarse for critic false-positive detection.

Confidence: high.

Evidence:

- For `critic_false_positive`, high-level match is true when `scores` exists, regardless of whether the critic passed, revised, failed, or whether the concern was justified.

Secondary hypothesis: governance can amplify critic false-positive behavior in some live runs.

Confidence: medium.

Evidence:

- The latest trace preserved the critic result as `REVISE`.
- Recent sampled hosted traces include both `REVISE` and `FAIL` final statuses for the same incident, all with high-level match true.
- Governance is monotonic over critic metric bands and recommendation; it does not independently review low-risk raw narrative context.

## Recommended Future Fix Surface

Smallest plausible future mutation surface:

- `evaluation/run_synthetic_incidents.py`

Recommended reporting-only fields for `critic_false_positive` summaries:

- `critic_recommended_status`
- `critic_confidence`
- `critic_reason_codes`
- `diagnosis_count`
- `partial_icd_mapping_count`
- `low_risk_context_present`
- `governance_amplified_critic`
- `false_positive_risk_observed`
- `incident_behavior_classification`
- `operational_interpretation`

Expected operational impact:

- Makes false-positive review faster without changing runtime decisions.
- Separates critic inspectability from likely false-positive behavior.
- Shows whether governance preserved or amplified critic concern.

Expected regression risk:

- Low if limited to synthetic summary/reporting fields.
- Medium if future changes alter critic scoring, diagnosis generation, or intake schema.

Likely validation strategy for a future fix:

- Use existing `MS-SYN-004`; do not create a new incident.
- Run one hybrid traced synthetic incident pack after a reporting-only change.
- Confirm the summary exposes critic reason codes, diagnosis count, ICD partial-match count, and governance amplification status.
- Inspect hosted `MS-SYN-004` trace for retained outputs/tags/model spans.
- Run `tests/test_execution_mode.py`.

## Remaining Unknowns

- Product-level tolerance for critic caution on low-risk but incomplete structured input is not defined.
- The investigation did not determine whether diagnosis generation should avoid multiple headache labels for this case; that would require a separate runtime or prompt brief.
- The exact cause of `REVISE` versus `FAIL` variance across live hybrid traces was not isolated because no new validation rerun was performed.

## Implementation Status

No fixes were implemented. No runtime, evaluator, prompt, governance, routing, threshold, dataset, or model-setting changes were made.
