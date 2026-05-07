# LangSmith Phase 3 Malformed Payload Investigation - 2026-05-07T19:56:42Z

## RCA Target Selected

- Incident id: `MS-SYN-007`
- Incident class: `malformed_downstream_payload`
- Synthetic input contains the phrase: `mappings should be a string not an array`.
- Latest summary reference: `evaluation/synthetic_incidents/last_run_summary.json`, started `2026-05-07T18:37:27Z`.
- Latest summary result: runtime `completed`, governed decision `FAIL`, fallback used `false`, fallback nodes `[]`, high-level expected match `true`.

No fixes, reruns, or behavior changes were performed in this investigation.

## Existing Trace Observations

Hosted traces inspected:

- `1e64652f-4363-43c0-a28f-2cffce3a39a0`, started `2026-05-07 18:38:07.142353`
- `2c9d86fc-87a5-42ba-83b7-2fa9edc74ee5`, started `2026-05-07 18:28:28.622849`
- `2106d8ee-dae2-4e48-b1d6-15210e993cc2`, started `2026-05-07 18:20:02.503018`
- `22464f3e-be50-4cc0-93b9-b9fa886b4187`, started `2026-05-07 16:19:31.734166`

Trace visibility:

- Synthetic roots were visible.
- Incident tags and metadata were visible.
- Stage outputs were visible.
- Four child model spans were visible in each inspected live hybrid trace.
- Node diagnostics were visible and reported live calls returned with parsing and normalization succeeded.

## Stage Propagation Observations

Latest inspected trace `1e64652f-4363-43c0-a28f-2cffce3a39a0`:

- Intake parser extracted symptoms `fever` and `cough`.
- Intake parser did not preserve the phrase `mappings should be a string not an array` as a structured field.
- Intake parser marked duration missing and did not trigger fallback.
- Triage returned `escalate` due to missing duration and need for more information.
- Diagnosis returned `Upper Respiratory Infection`, `Influenza`, and `COVID-19`.
- ICD mapper returned a normal list under `icd_mappings`, including one `OK` mapping and two `PARTIAL_MATCH` mappings.
- Critic returned recommended status `revise` with reason codes `MISSING_DURATION`, `UNCLEAR_SYMPTOM_DURATION`, and `PARTIAL_ICD_MATCH`.
- Governance returned final status `FAIL` with escalation required.
- The governance fail driver in the latest inspected trace was `critic_review.symptom_alignment_score` via `RULE_SYMPTOM_ALIGNMENT_FAIL`.

Earlier inspected traces showed similar high-level behavior:

- Runtime completed without fallback.
- `icd_mappings` remained list-shaped.
- Reason codes centered on missing duration, partial ICD mapping, multiple diagnoses, diagnosis consistency, symptom alignment, and ICD specificity.
- Triage varied across traces (`routine` in one older trace, `escalate` in later traces), consistent with live hybrid variability.

## Downstream Propagation Behavior

- The malformed-language phrase did not create an actual malformed downstream payload.
- No inspected trace showed `icd_mappings` as a string.
- No inspected trace showed malformed list/dict shape propagation from ICD mapper into critic or governance.
- No fallback node was reported.
- No parse or normalization failure was reported in node diagnostics.
- Downstream stages degraded operationally through low confidence, missing-duration, partial-match, and governance escalation semantics, not through structural exception handling.

## Runtime Code Path Observations

- `graph.llm_client.invoke_json` requires provider output to parse as a JSON object; non-object output raises `HybridLLMError("contract_rejection")`.
- `graph.nodes.intake_parser._normalize_intake_output` normalizes model output into a fixed `intake_data` shape and drops unsupported free-text instructions unless they map to known fields.
- `graph.nodes.diagnosis_engine._normalize_diagnosis_output` requires `diagnoses` to be a list and falls back on invalid shape.
- `graph.nodes.icd_mapper.run` is deterministic and always constructs `icd_mappings` as a Python list.
- `graph.nodes.critic._normalize_critic_output` enforces exact critic keys and valid reason-code shape for hybrid critic responses.
- `service.run_manager` merges each stage update into state and records fallback diagnostics, but no fallback was triggered for the inspected incident traces.

## Operational Brittleness Observations

Strong debugging signals:

- LangSmith stage outputs make it clear that `icd_mappings` remained a list.
- Node diagnostics show whether live calls returned, parsing succeeded, normalization succeeded, and fallback was triggered.
- Governance attribution shows which critic scores drove the final status.
- The synthetic root summary shows completion, governed decision, fallback status, and failure-localization clues.

Weak or missing debugging signals:

- The incident summary does not explicitly say that no malformed payload actually occurred.
- The incident class name implies downstream payload malformation, but the current incident is only a text-level injection-style prompt.
- The runner does not compare expected malformed structure against observed structure.
- The trace does not explicitly classify malformed-language instructions as ignored, sanitized, or not propagated.
- There is no per-stage shape contract report for key outputs such as `diagnoses` and `icd_mappings`.

Operational symptoms:

- The incident appears green at the high-level evaluator layer because the runtime completed and did not crash.
- A real debugging workflow could spend time searching for malformed payload propagation even though the traces show no structural malformation.
- The most visible failure is clinical/documentation quality (`FAIL`) rather than malformed-structure handling.

## Root-Cause Hypotheses

Primary hypothesis: the synthetic incident does not actually inject or force a malformed downstream structure; it only includes malformed-output language in patient text.

- Confidence: high.
- Evidence: all inspected traces completed without fallback, `icd_mappings` remained list-shaped, node diagnostics reported successful parsing/normalization, and the deterministic ICD mapper constructs list outputs in code.
- Why malformed payload does not propagate: the text phrase is consumed as ordinary input, while node normalizers and deterministic mapping code produce fixed structured outputs.

Secondary hypothesis: structured-output guardrails are strong enough for this scenario, but observability does not clearly label the guardrail success.

- Confidence: medium-high.
- Evidence: model-backed nodes returned valid normalized outputs; deterministic ICD mapper returned valid list-shaped mappings; however, summaries do not state that malformed payload did not materialize.

Secondary hypothesis: live hybrid variability changes clinical outputs but not structural behavior.

- Confidence: medium.
- Evidence: inspected traces varied in triage level and diagnosis labels, but all retained structured outputs and completed without fallback.

## Classification

- Parser-driven: partially relevant; JSON/object parsing and node normalization prevent malformed model output from entering state.
- Schema-driven: relevant; node-specific normalizers enforce expected list/object shapes.
- Serialization-driven: not observed.
- Model-output-driven: relevant to variable clinical content, not observed as structural malformed payload.
- Validation-driven: relevant; normalization and contract checks would fallback on malformed model outputs.
- Propagation-driven: not observed; malformed shape did not propagate.

## Current Observability Sufficiency

Current observability is sufficient to determine that no malformed `icd_mappings` structure propagated in inspected traces.

Current observability is not sufficient to make that conclusion obvious from the root summary alone. A larger system would likely need explicit per-stage shape validation telemetry or expected-vs-observed structure reporting to avoid confusing "malformed instruction in input text" with "malformed downstream payload emitted by a node."

## Bounded Recommended Future Fix Surface

Smallest plausible future mutation surface:

- `evaluation/run_synthetic_incidents.py`

Recommended future reporting-only fix:

- Add malformed-payload-specific summary fields such as `malformed_payload_observed`, `malformed_payload_stage`, `expected_shape`, `observed_shape`, and `shape_validation_result`.

Potential runtime validation surface if behavior changes are later authorized:

- Node-specific normalizers in `graph.nodes.*`
- A lightweight shared shape-report helper used by the synthetic runner or trace metadata

Expected operational impact:

- Reporting-only change would reduce ambiguity without changing runtime behavior.
- Runtime validation changes would improve explicit failure localization but carry higher regression risk if downstream consumers rely on current permissive normalization.

Expected regression risk:

- Low for synthetic reporting-only fields.
- Medium for runtime validation/normalization behavior changes.

Likely validation strategy:

- Use existing `MS-SYN-007` only.
- Run one hybrid traced comparison if a future fix is approved.
- Confirm summaries explicitly report whether malformed structure was observed.
- Confirm `icd_mappings` remains list-shaped or any malformed stage is explicitly localized.
- Confirm traces, tags, outputs, model spans, and node diagnostics remain visible.

## Remaining Unknowns

- Whether `MS-SYN-007` is intended to test prompt-injection-style malformed instructions or actual malformed node output.
- Whether the synthetic pack should include expected observed-shape assertions for malformed payload incidents.
- Whether production consumers need root-level shape validation summaries or whether stage outputs are enough.
- Whether live model output could produce malformed node responses under other prompts, since this specific incident did not trigger that behavior.
