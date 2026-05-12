# Runtime Experiment Change Classification

Timestamp: 2026-05-08T20:08:50Z

Purpose: classify the remaining uncommitted changes after README publication. No runtime, prompt, governance, or evaluator content was changed during this classification pass.

## Current Uncommitted Files

| File | Classification | Commit Recommendation |
| --- | --- | --- |
| `evaluation/run_synthetic_incidents.py` | observability-support | Candidate after review with the observability/reporting changes; adds policy-divergence summary fields and comparison annotations. |
| `service/run_manager.py` | observability-support | Candidate after review; adds operational observability snapshots and trace metadata without changing final decision semantics. |
| `graph/config.py` | experimental-stage2 | Keep experimental until broader latency/token regression validates the `MEDSCRIBE_MAX_TOKENS=128` default. |
| `graph/llm_client.py` | experimental-stage2 | Keep experimental; changes model call behavior through `max_tokens` and stabilization metadata. |
| `prompts/critic_prompt.txt` | experimental-stage2 | Keep experimental; reduces critic verbosity and may alter reason-code/confidence behavior. |
| `prompts/diagnosis_engine_prompt.txt` | experimental-stage2 | Keep experimental; changes diagnosis output ceiling and wording constraints. |
| `prompts/intake_parser_prompt.txt` | experimental-stage2 | Keep experimental; adds compactness constraints that may alter structured intake shape. |
| `prompts/triage_engine_prompt.txt` | experimental-stage2 | Keep experimental; tightens rationale length and may affect triage output distribution. |
| `governance/policy_rules.json` | needs-review-before-commit | Requires explicit review before commit; changes governance version and adds a low-evidence boundary rule ID. |
| `graph/nodes/governance_policy.py` | needs-review-before-commit | Requires explicit review before commit; changes final decision semantics for low-evidence boundary cases. |

## Public-Safety Findings

No secrets, API keys, local absolute paths, private LangSmith URLs, OPORD/AAR references, MedFox/FoxCore references, or internal-only commentary were found in the inspected diffs.

The scan surfaced environment variable names only:

- `OPENAI_API_KEY`
- `LANGCHAIN_API_KEY`

These are configuration variable names, not secret values.

## Recommended Next Action

Review and decide whether to split the remaining changes into separate commits:

1. Observability/reporting support: `evaluation/run_synthetic_incidents.py`, `service/run_manager.py`.
2. Stage 2 latency/runtime experiment: `graph/config.py`, `graph/llm_client.py`, and prompt files.
3. Stage 3 governance semantics: `governance/policy_rules.json`, `graph/nodes/governance_policy.py`.

Do not combine governance semantics with observability-only changes unless the intent is to promote the full experimental runtime state together.
