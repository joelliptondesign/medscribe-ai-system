# LangSmith Phase 3 RCA Token Cost - 2026-05-07T15:57:57Z

## Observed Symptom

- Synthetic incident root runs show missing token and cost accounting and near-zero latency.
- Governed live hybrid runs show model-backed trace behavior, including `ChatOpenAI` spans.

## Code Paths Inspected

- `docs/LANGSMITH_PHASE1_RECON_20260507T005043Z.md`
- `docs/LANGSMITH_PHASE2_SYNTHETIC_INCIDENTS_20260507T144105Z.md`
- `evaluation/run_synthetic_incidents.py`
- `evaluation/synthetic_incidents/last_run_summary.json`
- `service/run_manager.py`
- `graph/tracing.py`
- `graph/llm_client.py`
- `graph/config.py`
- `telemetry/executor_heartbeat.jsonl`

## Governed Live Execution Path Findings

- The inspected Phase 1 live validation records a governed hybrid run executed with process-level `LANGCHAIN_TRACING_V2=true` and `MEDSCRIBE_EXECUTION_MODE=hybrid`.
- The governed service path is `service.run_manager.execute`.
- `service.run_manager.execute` opens a root `medscribe.governed_run` span through `graph.tracing.trace_span`.
- `service.run_manager._execute_pipeline` opens stage spans for `medscribe.intake_parser`, `medscribe.triage_engine`, `medscribe.diagnosis_engine`, `medscribe.icd_mapper`, `medscribe.critic`, and `medscribe.governance_policy`.
- Live model calls occur through `graph.llm_client.invoke_json`, which constructs `ChatOpenAI` and calls `model.invoke(...)`.
- The `ChatOpenAI` invocation attaches LangChain config tags `medscribe`, `stage:<node_name>`, and `hybrid-llm`, plus metadata containing `node_name` and model name.
- Output capture is manual for MedScribe chain spans: `TraceSpanRecorder.set_outputs(...)` calls `run.end(outputs=sanitize_payload(outputs))`.
- Phase 1 docs record that the live hybrid governed trace contained four `ChatOpenAI` model invocation spans.
- No inspected code explicitly copies token or cost usage from model spans onto `medscribe.governed_run`; any root-level usage display would be LangSmith-side aggregation, not runtime-authored metadata.

## Synthetic Incident Runner Findings

- The documented Phase 2 runner command is `.venv/bin/python evaluation/run_synthetic_incidents.py`.
- `evaluation/run_synthetic_incidents.py` loads `.env` and sets `LANGCHAIN_TRACING_V2=true` only when `LANGCHAIN_API_KEY` is configured.
- The synthetic runner does not set `MEDSCRIBE_EXECUTION_MODE=hybrid`.
- `graph.config.get_execution_mode()` defaults to `deterministic` when `MEDSCRIBE_EXECUTION_MODE` is unset.
- Each synthetic case opens a manual incident-level span named `medscribe.synthetic_incident`.
- Inside that incident span, each case calls the same `service.run_manager.execute(case["input_text"], run_id=..., persist=False)` path.
- The existing runtime root span `medscribe.governed_run` is therefore nested under the incident-level span.
- The runner attaches only an operational incident summary to the incident root span, including status, decision, fallback nodes, failure localization clue, and high-level match.
- The Phase 2 summary completed seven incidents in about two seconds, from `2026-05-07T14:39:46Z` to `2026-05-07T14:39:48Z`.
- The persisted synthetic summary records `trace_stage_count: 6` for each incident and no evidence of model-span counts, model usage, tokens, or cost.
- The visibility helper in the runner queries only chain runs and stage output visibility. It does not query or report `ChatOpenAI` child spans, token usage, cost, or latency aggregation.

## Trace Semantics Comparison

- Governed live validation used hybrid mode and produced `ChatOpenAI` model spans.
- Synthetic incident execution used the same governed service path but, by documented command and runner code, did not request hybrid mode.
- The synthetic root `medscribe.synthetic_incident` is a manual chain span whose output is a locally computed operational summary, not a model response.
- The nested runtime and stage spans are also manual chain spans. They expose sanitized inputs, metadata, and outputs, but they do not create token or cost accounting by themselves.
- Child `ChatOpenAI` spans are expected only if the runtime is actually in hybrid mode and reaches `graph.llm_client.invoke_json`.

## Likely Root Cause

Classification: synthetic incidents bypass live LLM calls.

The primary observed cause is that the synthetic incident runner does not enable `MEDSCRIBE_EXECUTION_MODE=hybrid`, while the execution mode defaults to deterministic. The Phase 2 command also lacks a hybrid-mode environment prefix. As a result, synthetic incident traces are operational chain traces around deterministic/local runtime work, not model-backed runs. Without `ChatOpenAI` child spans, LangSmith has no model token or cost usage to attribute to those synthetic roots.

A secondary trace semantics limitation is that manual chain spans do not author token or cost metadata, and the inspected code does not propagate model usage onto root span outputs or metadata. If synthetic incidents are later run in hybrid mode, token/cost should be checked first on child `ChatOpenAI` spans and then separately verified for any root-level LangSmith aggregation behavior.

## Confidence Level

High for the deterministic-mode bypass cause.

Medium for root-level token/cost inheritance behavior, because the inspected code shows no explicit usage propagation, but this RCA did not perform a fresh hosted LangSmith query.

## Optional Validation Run

Not performed.

Reason: running `evaluation/run_synthetic_incidents.py` would write `evaluation/synthetic_incidents/last_run_summary.json` and a timestamped summary under `evaluation/synthetic_incidents/`, which is outside the allowed mutation surfaces for this RCA brief.

## Recommended Bounded Fix Options

- Add an explicit hybrid-mode execution path for a future synthetic validation brief, using process-level `MEDSCRIBE_EXECUTION_MODE=hybrid` without changing runner code.
- Extend a future non-RCA validation query to report child `ChatOpenAI` span presence, token usage, cost usage, and root aggregation status for synthetic traces.
- If root-level usage is required after hybrid validation, add a bounded tracing-only enhancement in a separate brief to record model usage summaries without changing prompts, governance, routing, datasets, or model settings.

## Recommended Next Executor Brief Scope

- Run exactly one existing synthetic incident pack in live hybrid mode with hosted LangSmith tracing enabled.
- Do not create or edit incident data.
- Capture whether each synthetic incident root contains nested `ChatOpenAI` spans.
- Capture token and cost accounting location: child model spans, synthetic root, nested governed root, or absent.
- Write only a docs validation note and one telemetry heartbeat unless an explicit Phase 3 fix brief follows.

## Implementation Changes

No implementation changes were performed.

## Hybrid Synthetic Validation - 2026-05-07T16:12:43Z

- Credential/config presence check: `LANGCHAIN_API_KEY` present, `OPENAI_API_KEY` present, `LANGCHAIN_PROJECT` present as `medscribe-phase1-runtime`, and `LANGCHAIN_TRACING_V2` configured in `.env` as `false`. No secret values were printed or recorded.
- Hybrid-mode command executed with process-level overrides:

```bash
MEDSCRIBE_EXECUTION_MODE=hybrid LANGCHAIN_TRACING_V2=true .venv/bin/python evaluation/run_synthetic_incidents.py
```

- Initial sandboxed attempt executed the same command but could not upload/query hosted LangSmith traces because DNS resolution for `api.smith.langchain.com` failed. It wrote `evaluation/synthetic_incidents/run_summary_20260507T160634Z.json`.
- Approved network run completed and wrote `evaluation/synthetic_incidents/last_run_summary.json` and `evaluation/synthetic_incidents/run_summary_20260507T160955Z.json`.
- Run result: 7 incidents executed, 7 high-level matches, 0 high-level failures.
- Hosted roots visible: 7 synthetic incident roots since summary start time `2026-05-07T16:09:55Z`.
- Hosted trace ids inspected: `6cf5c21d-3f63-4088-b313-81b202580970`, `ca3a096f-2cd2-4fd3-8be8-3406e28605b9`, `34907a0a-3629-4262-9b09-ec9a2d8cdb7f`, `1f4e90d5-5a30-47d5-91f3-1599427a2d3e`, `0c9a6ada-e858-48a1-b079-3392fb6919f2`, `e9714d84-aba0-416e-a8ba-b7cb81dcef4e`, and `41da657a-97ce-4b7a-8af1-ac3fc880eec1`.
- Child model spans visible: 28 hosted child model spans, all with run type `llm`.
- Token visibility: token usage appeared on 28 of 28 child model spans.
- Cost visibility: cost appeared on 28 of 28 child model spans.
- Root-level aggregation: token usage appeared on 7 of 7 synthetic roots; cost appeared on 7 of 7 synthetic roots.
- Latency visibility: root latency appeared on 7 of 7 synthetic roots, with observed root latency range 5.886301 to 21.396997 seconds. Child model span latency range was 0.551492 to 9.468798 seconds.
- Incident tags/metadata remained visible: yes.
- Stage outputs remained visible: yes, all six MedScribe stage outputs were present in inspected traces.
- Previous RCA status: confirmed. The missing token/cost and near-zero latency symptom was caused by the prior synthetic run not being executed in live hybrid mode. When the existing pack was run with `MEDSCRIBE_EXECUTION_MODE=hybrid`, model spans, token usage, cost usage, and realistic latency appeared.
- Remaining limitation: this validation used hosted LangSmith API inspection, not UI screenshots. The runner's built-in visibility summary still reports only coarse chain/stage visibility and does not itself summarize token/cost/model-span counts.
- Smoke verification: `.venv/bin/python tests/test_execution_mode.py` passed 7 of 7 cases.
