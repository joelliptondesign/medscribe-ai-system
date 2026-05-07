# LangSmith Phase 1 Recon - 2026-05-07T00:50:43Z

## Integration Surfaces

- Runtime entry path: `service.main:app` exposes the FastAPI app.
- Primary orchestration path: `service.run_manager.execute` executes the governed service path.
- Runtime stage order: intake parser, triage engine, diagnosis engine, ICD mapper, critic, governance policy.
- Tool dispatch path: `service.tools.call_tool` dispatches parse, diagnosis, ICD mapping, and critic calls.
- Direct governance call: `service.run_manager.execute` calls `graph.nodes.governance_policy.run`.
- Hybrid LLM surface: `graph.llm_client.invoke_json` uses `ChatOpenAI.invoke` for hybrid node calls.
- LangGraph surface exists in `graph.graph_builder.build_graph`, used by evaluation/governed pipeline helpers, but the service runtime path uses `service.run_manager.execute`.

## Instrumentation Approach Used

- Added `graph.tracing.trace_span` as a thin, fail-open LangSmith wrapper around `langsmith.run_helpers.trace`.
- Added a root runtime span around `service.run_manager.execute` through a wrapper that preserves the existing function signature.
- Added stage spans for intake parser, triage engine, diagnosis engine, ICD mapper, critic, and governance policy.
- Added metadata for run id, execution mode, pipeline version, stage name, and LangSmith environment readiness.
- Added LangChain tags and metadata to the existing hybrid LLM invocation surface.
- Added keyless environment support for `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`, and `LANGCHAIN_ENDPOINT`.

## Successful Local Run Observations

- Deterministic governed run completed with status `completed`.
- Final governed decision was `PASS`.
- Runtime record preserved the existing trace stage list: `intake_parser`, `triage_engine`, `diagnosis_engine`, `icd_mapper`, `critic`, `governance_policy`.
- With `LANGCHAIN_TRACING_V2=true` and no API key, the run still completed and reported `langchain_api_key_configured: False`.
- The tracing wrapper did not alter runtime output or governance decision in the local verification run.

## Operational Friction

- No `LANGCHAIN_API_KEY` was present in the shell or repo-local `.env`.
- A visible LangSmith-hosted root trace could not be confirmed without credentials.
- UI reconnaissance could not be performed because no connected LangSmith project/run was available.
- The runtime emitted an existing urllib3 LibreSSL warning during local Python execution.

## UI Observations

- Run table structure: not observed; LangSmith UI access was not available in this environment.
- Trace timeline layout: not observed; no hosted trace was created without credentials.
- Expandable trace panels: not observed.
- Metadata display patterns: not observed in UI; local runtime metadata includes LangSmith readiness fields.
- Payload viewers: not observed.
- Filtering/search affordances: not observed.
- Evaluator result layouts: not observed.
- Screenshots captured: none.

## Limitations Encountered

- Hosted trace visibility remains unverified until `LANGCHAIN_API_KEY` is configured with `LANGCHAIN_TRACING_V2=true`.
- Evaluation surfaces appear traceable through the existing LangGraph path and shared node functions, but hosted evaluation trace visibility was not confirmed.
- The instrumentation is limited to lightweight span boundaries and metadata; it does not add replay, RCA loops, incident assets, routing changes, or governance changes.


## Hosted Trace Validation - 2026-05-07T01:28:57Z

- Current environment inspection found `LANGCHAIN_API_KEY` present, `LANGCHAIN_TRACING_V2=false` in `.env`, `LANGCHAIN_PROJECT=medscribe-phase1-runtime`, and no `OPENAI_API_KEY`.
- One governed deterministic run was executed with process-level `LANGCHAIN_TRACING_V2=true`; runtime completed with status `completed` and decision `PASS`.
- Hosted LangSmith trace became visible in project `medscribe-phase1-runtime`.
- Root run name: `medscribe.governed_run`.
- Root run id: `4203ea5a-6340-4e60-b842-7d9a71711fde`.
- Trace URL: https://smith.langchain.com/o/54f1d5cc-5981-44bf-910c-cd63c6ffae87/projects/p/6e278d0a-972a-40ac-9bd2-78dabae50145/r/4203ea5a-6340-4e60-b842-7d9a71711fde?trace_id=4203ea5a-6340-4e60-b842-7d9a71711fde&start_time=2026-05-07T01:28:57.575066
- Six child spans were visible: `medscribe.intake_parser`, `medscribe.triage_engine`, `medscribe.diagnosis_engine`, `medscribe.icd_mapper`, `medscribe.critic`, and `medscribe.governance_policy`.
- Root tags were readable as `governed-runtime` and `medscribe`; root extra keys included `metadata` and `runtime`.
- Model invocation spans did not appear in the deterministic run because the runtime used local deterministic paths.
- A separate no-persist hybrid credential check degraded through LLM-backed nodes with `client_call_failure:RuntimeError` and fallback nodes `intake_parser`, `triage_engine`, `diagnosis_engine`, and `critic`.
- Explicit `OPENAI_API_KEY` is required for live hybrid model inference; a blank `OPENAI_API_KEY=` placeholder was added to local `.env` because the key was absent.
- `.env.example` already contained a non-secret `OPENAI_API_KEY` placeholder.
- Operational friction: sandboxed network execution could not resolve `api.smith.langchain.com`; rerunning with approved network access validated hosted trace visibility.


## Live Model Trace Validation - 2026-05-07T01:32:44Z

- Credential presence check found `LANGCHAIN_API_KEY` present, `OPENAI_API_KEY` present, `LANGCHAIN_PROJECT=medscribe-phase1-runtime`, and `LANGCHAIN_TRACING_V2=false` in `.env`. No secret values were recorded.
- One governed hybrid run was executed with process-level `LANGCHAIN_TRACING_V2=true` and `MEDSCRIBE_EXECUTION_MODE=hybrid`; no `.env` values were overwritten.
- Runtime status: `completed`.
- Governed decision: `FAIL`.
- LLM-backed nodes succeeded without fallback: `intake_parser`, `triage_engine`, `diagnosis_engine`, and `critic` each attempted a live call, received a response, parsed JSON, and normalized output.
- Hosted LangSmith trace became visible in project `medscribe-phase1-runtime`.
- Root run name: `medscribe.governed_run`.
- Root run id: `e3a56a5b-fc85-47e3-9f2a-c7b7d13086fa`.
- Trace URL: https://smith.langchain.com/o/54f1d5cc-5981-44bf-910c-cd63c6ffae87/projects/p/6e278d0a-972a-40ac-9bd2-78dabae50145/r/e3a56a5b-fc85-47e3-9f2a-c7b7d13086fa?trace_id=e3a56a5b-fc85-47e3-9f2a-c7b7d13086fa&start_time=2026-05-07T01:32:44.722035
- Direct child spans visible: 6, covering `medscribe.intake_parser`, `medscribe.triage_engine`, `medscribe.diagnosis_engine`, `medscribe.icd_mapper`, `medscribe.critic`, and `medscribe.governance_policy`.
- Full trace run count: 11.
- Model invocation spans appeared: 4 `ChatOpenAI` spans.
- Root tags remained readable: `governed-runtime` and `medscribe`; root extra keys included `metadata` and `runtime`.
- Failure localization status: no provider or model errors occurred; the governed `FAIL` decision was a runtime output, not a tracing or provider failure.
- Operational friction: `.env` still has `LANGCHAIN_TRACING_V2=false`, so tracing was enabled at process scope for validation.
- Phase 1 completion condition for hosted live model-backed trace visibility is satisfied by this run.


## Trace Output Capture Fix - 2026-05-07T14:32:27Z

- Cause of missing outputs: the lightweight `trace_span` wrapper opened LangSmith trace contexts with inputs, tags, and metadata, but no call site passed returned values to `run.end(outputs=...)`.
- Output capture fix applied in `graph.tracing.trace_span` through a fail-open recorder that sanitizes payloads and redacts secret-like keys before attaching outputs.
- `service.run_manager.execute` now attaches the final governed runtime result to the root span after the existing execution path returns.
- Stage spans now attach existing returned update payloads for intake parser, triage engine, diagnosis engine, ICD mapper, critic, and governance policy before those updates are merged into state.
- Runtime behavior verification: one live hybrid governed run completed with status `completed`, decision `FAIL`, and no fallback nodes.
- Hosted LangSmith trace became visible in project `medscribe-phase1-runtime`.
- Root run name: `medscribe.governed_run`.
- Root run id: `4bf54dc3-b90c-4a73-9edd-441e857c6808`.
- Trace URL: https://smith.langchain.com/o/54f1d5cc-5981-44bf-910c-cd63c6ffae87/projects/p/6e278d0a-972a-40ac-9bd2-78dabae50145/r/4bf54dc3-b90c-4a73-9edd-441e857c6808?trace_id=4bf54dc3-b90c-4a73-9edd-441e857c6808&start_time=2026-05-07T14:32:27.280361
- Root output visibility: visible, with keys including `status`, `decision`, `parsed_input`, `diagnosis`, `icd_mapping`, `scores`, `summary`, `metadata`, and timing fields.
- Stage output visibility: visible for `medscribe.intake_parser`, `medscribe.triage_engine`, `medscribe.diagnosis_engine`, `medscribe.icd_mapper`, `medscribe.critic`, and `medscribe.governance_policy`.
- Model invocation spans still appeared: 4 `ChatOpenAI` spans.
- Tags and metadata remained readable: root tags `governed-runtime` and `medscribe`; root extra keys included `metadata` and `runtime`.
- Smoke verification result: `.venv/bin/python tests/test_execution_mode.py` passed 7 of 7 cases.
- Remaining limitation: output payloads are intentionally sanitized and truncated to avoid secret exposure and excessive trace payload expansion.
