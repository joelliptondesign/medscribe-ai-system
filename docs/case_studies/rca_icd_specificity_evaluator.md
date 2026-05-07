# RCA Case Study: ICD Specificity Evaluator Mismatch

## Problem

A synthetic incident for ICD specificity initially appeared to pass the high-level evaluator even though the governed runtime returned a final `FAIL`. The trace showed that the system produced a structurally valid ICD mapping container, but the mapping itself was not clinically usable: it had `NO_MATCH_FOUND` status and empty ICD code and label fields.

The issue was not a crash, schema failure, or missing trace. It was an evaluator/reporting mismatch: the incident summary treated the presence of a mapping list as success, even when the mapping contents showed failure.

## System Context

MedScribe runs a structured pipeline:

```text
Input -> Intake Parser -> Triage Engine -> Diagnosis Engine -> ICD Mapper -> Critic -> Governance Policy
```

The synthetic incident pack exercises known reliability risks using non-PHI cases. In this case, the target incident was `MS-SYN-002`, classified as `incorrect_icd_specificity`.

The relevant runtime surfaces were:

- ICD mapper output
- critic scores and reason codes
- deterministic governance status
- synthetic incident summary fields
- LangSmith trace spans for each stage

## Trace Observations

The hosted trace localized the failure to semantic adequacy rather than structure:

- The ICD mapper emitted a mapping object, so the output shape was valid.
- The mapping status was `NO_MATCH_FOUND`.
- The ICD code and ICD label were empty.
- The critic emitted failure signals including `NO_ICD_MATCH_FOUND`.
- Governance returned `FAIL` through the ICD specificity rule path.
- Stage outputs, incident metadata, and model spans remained visible in the trace.

This made the operational issue clear: the runtime behaved cautiously, but the evaluator summary was too permissive.

## Root Cause

The evaluator logic for the ICD specificity incident checked whether a mappings collection existed. It did not require at least one successful, specific ICD mapping.

That created a false high-level pass:

- `mappings` present -> evaluator marked expected behavior as matched
- `NO_MATCH_FOUND` with empty ICD fields -> runtime and governance correctly treated the case as failed

The root cause was evaluator-driven, not model-output parsing, routing, governance, or tracing.

## Bounded Fix

The synthetic incident evaluator was tightened for the `incorrect_icd_specificity` case.

The evaluator now requires at least one mapping with:

- `status` equal to `OK`
- non-empty `icd_code`
- non-empty `icd_label`

This changed only reporting/evaluation semantics for the incident summary. Runtime behavior, prompts, routing, model settings, policy thresholds, and governance decisions were not changed.

## Before/After Comparison

Before:

- Incident: `MS-SYN-002`
- Runtime completed.
- Governance returned `FAIL`.
- ICD mapper returned `NO_MATCH_FOUND`.
- High-level incident summary reported `high_level_expected_match: true`.

After:

- Runtime completed.
- Governance still returned `FAIL`.
- ICD mapper failure remained visible.
- High-level incident summary reported `high_level_expected_match: false`.
- Other synthetic incidents retained their expected high-level classifications.

The comparison showed that the fix improved evaluator truthfulness without changing the governed pipeline.

## Regression Verification

The focused smoke test passed after the change:

```bash
.venv/bin/python tests/test_execution_mode.py
```

Result: 7 of 7 execution-mode checks passed.

The synthetic incident run still produced visible trace roots, model spans, stage outputs, tags, and runtime metadata in hybrid traced mode.

## Operational Lessons

- Valid JSON is not the same as valid system behavior.
- Evaluators should check semantic adequacy, not just container presence.
- Trace outputs are most useful when paired with incident summaries that preserve the same operational meaning.
- Governance failure and evaluator failure are different signals and should not be collapsed.
- RCA is easier when traces expose stage outputs, critic reason codes, governance rule drivers, and incident metadata together.

## What This Demonstrates For Production AI Systems

This case study demonstrates a production-relevant AI systems pattern: reliability work often happens at the boundary between model behavior, structured outputs, evaluation logic, and governed final decisions.

The value of the workflow was not only finding a bad output. It was identifying that the runtime had already surfaced the issue while the evaluator summary misrepresented it. The fix made the measurement layer better aligned with actual governed behavior, which is essential for trustworthy AI evaluation loops.
