# LangSmith Phase 2 Synthetic Incidents - 2026-05-07T14:41:05Z

## Incident Inventory

| Incident ID | Incident Class | High-Level Result |
| --- | --- | --- |
| MS-SYN-001 | schema_valid_semantic_failure | completed, decision FAIL |
| MS-SYN-002 | incorrect_icd_specificity | completed, decision REVISE |
| MS-SYN-003 | governance_override | completed, decision PASS |
| MS-SYN-004 | critic_false_positive | completed, decision REVISE |
| MS-SYN-005 | ambiguous_case_overconfidence | completed, decision FAIL |
| MS-SYN-006 | policy_change_divergence | completed, decision FAIL |
| MS-SYN-007 | malformed_downstream_payload | completed, decision PASS |

The incident pack contains synthetic non-PHI input text only. No baseline evaluation datasets were modified.

## Runner Command

```bash
.venv/bin/python evaluation/run_synthetic_incidents.py
```

## Run Summary

- Dataset: `medscribe_synthetic_incidents_phase2`
- Dataset version: `2026-05-07`
- Incidents executed: 7
- High-level expected matches: 7
- High-level failures/errors: 0
- Summary artifact: `evaluation/synthetic_incidents/last_run_summary.json`
- Timestamped summary artifact: `evaluation/synthetic_incidents/run_summary_20260507T143946Z.json`
- LangSmith project: `medscribe-phase1-runtime`

## Trace Visibility

- Hosted synthetic incident traces appeared: yes
- Recent synthetic incident root count: 7
- Latest synthetic incident trace URL: https://smith.langchain.com/o/54f1d5cc-5981-44bf-910c-cd63c6ffae87/projects/p/6e278d0a-972a-40ac-9bd2-78dabae50145/r/133ce484-ba60-4e80-b447-e56fe4e6fd55?trace_id=133ce484-ba60-4e80-b447-e56fe4e6fd55&start_time=2026-05-07T14:39:46.694280
- Incident tags and metadata appeared in hosted traces: yes
- Stage outputs remained visible on the latest queried trace: yes

## Debugging Workflow Observations

- Each incident run has an incident-level wrapper span named `medscribe.synthetic_incident`.
- The existing runtime root span `medscribe.governed_run` remains nested under the incident-level span.
- MedScribe stage spans remain visible for intake parser, triage engine, diagnosis engine, ICD mapper, critic, and governance policy.
- The runner summary records runtime status, governed decision, fallback nodes, failure localization clue, and a coarse high-level expected-match flag.
- Failure localization clues came from governed reason codes for all completed cases in this run.

## UI Reconnaissance Notes

- The hosted project accepted incident-tagged traces.
- Stage output visibility remained available after incident wrapping.
- Incident metadata is attached through the existing lightweight tracing wrapper.
- No screenshots were captured.

## Limitations And Blockers

- The runner performs lightweight high-level matching only; it does not implement RCA iteration loops or deterministic replay.
- Incident cases are synthetic probes for trace and evaluation workflow validation, not production clinical tests.
- A first visibility query occurred before synthetic incident roots were available in the hosted list; the generated summary was updated after a follow-up hosted query confirmed seven incident roots.
