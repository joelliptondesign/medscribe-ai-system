# LangSmith Dataset Experiment POC

## Dataset Concept

`evaluation/operational_benchmark_cases.json` defines a synthetic, non-PHI operational benchmark for CDI and coding-support review-routing workflows. Cases cover insufficient ICD specificity, unsupported specificity, documentation ambiguity, borderline documentation sufficiency, low-confidence coding support, recoverable documentation gaps, conflicting documentation, coder-review-worthy cases, conservative revise routing, specialist-review-worthy ambiguity, low-evidence boundary cases, adjacent-case spillover checks, and over-cautious escalation candidates.

Each case carries expected high-level behavior and practical expected routing or governance fields where useful. These fields support lightweight regenerated operational evaluation, not frozen historical output comparison.

## Experiment Concept

`evaluation/langsmith_experiment_runner.py` loads the dataset, optionally creates or reuses a LangSmith dataset, runs the existing MedScribe hybrid LLM workflow with `persist=False`, and preserves three additive evaluation layers.

The runner also supports a selectable denial-management workflow with `--workflow denial`. Denial mode loads `evaluation/denial_benchmark_cases.json`, executes the additive local denial graph, and emits the same three-layer operational structure without replacing the CDI/coding-support workflow.

Layer 1 operational debugging is the existing execution-health and observability evaluation. It attaches four lightweight evaluators:

- `high_level_match`
- `governance_status_match`
- `completed_without_fallback`
- `final_status_present`

Layer 2 pairwise routing evaluation is a local governance usefulness comparison for baseline vs `threshold_variant` outputs. It asks which output provides the safer, clearer, and more operationally useful CDI review-routing outcome before human review.

Layer 3 routing-distribution experimentation summarizes final status buckets, governance reason-code movement, low-evidence routing volume, specialist-review-oriented volume, adjacent-case spillover indicators, threshold sensitivity, escalation saturation, latency when available, and token/cost availability when exposed by runtime metadata.

For denial workflows, Layer 1 reports local completion, fallback/degraded flags, node diagnostics, routing action, governance posture, denial category, recoverability, and documentation gaps. Layer 2 asks which routing/governance outcome is safer and more operationally useful before human review, using deterministic local heuristics for ambiguity escalation quality, low-evidence handling, recoverability alignment, unsafe write-off avoidance, unsupported certainty avoidance, and operational inspectability. Layer 3 reports APPEAL, RESUBMIT, WRITE_OFF, and ESCALATE distributions, governance posture distribution, low-evidence routing counts, ambiguity-routing counts, specialist-review candidate routing counts, denial category counts, threshold sensitivity, escalation saturation, and recoverability observations.

The default dataset name is `medscribe-operational-benchmark-poc`. Supported experiment labels are `baseline`, `threshold_variant`, and `routing_sensitivity_variant`.

When LangSmith credentials are available, the runner uses LangSmith's formal `evaluate(...)` workflow. That creates a hosted experiment associated with selected dataset examples and uploads the evaluator scores as hosted feedback. The console summary is still printed locally in the same operational structure.

The runner emits a non-secret LangSmith preflight status before runtime execution. It reports key presence, tracing flag presence, endpoint presence, client construction, and hosted readiness without printing secret values. Hosted readiness uses a non-mutating dataset-list check before dataset setup or hosted experiment creation. `--local-only` keeps the older trace/local-summary path, skips hosted dataset setup, and does not create formal hosted experiment rows.

The pairwise workflow is local. It uses a deterministic, inspectable heuristic rather than an LLM-as-judge because the current runner already supports regenerated formal single-label experiments and adding hosted pairwise judge feedback would expand the workflow beyond this POC pass.

## Operational Traces vs Formal Experiments

Trace-only runs call the runtime and rely on tracing hooks to show root and child traces in the LangSmith project. They are useful for inspecting execution details but do not create the hosted experiment table rows, per-example evaluation entries, or comparison-oriented experiment surfaces.

Formal experiment runs call LangSmith `evaluate(...)` with selected dataset examples. They create a hosted experiment, associate each target run with an example, attach evaluator feedback, and make the run eligible for LangSmith experiment comparison workflows.

Dedicated denial operations traces use `evaluation/run_denial_ops_traces.py`. This path is separate from datasets, experiments, and evaluator feedback. It runs selected synthetic denial cases as live operational traces in the `medscribe-denial-ops` project by default, names the live queue row with the denial `case_id`, and keeps evaluator rows out of the live operations table. Workflow identity remains available in metadata as `medscribe.denial.workflow.hybrid` or `medscribe.denial.workflow.deterministic`.

The runner distinguishes hosted states in output:

- `local_only_requested`
- `pairwise_local_only`
- `hosted_succeeded`
- `hosted_requested_local_summary_produced`
- `hosted_unavailable_local_summary_produced`

With the formal path, the LangSmith UI should show:

- a hosted experiment row named with the `medscribe-operational-<label>` prefix
- per-example target runs
- evaluator feedback for `high_level_match`, `governance_status_match`, `completed_without_fallback`, and `final_status_present`
- trace links from the experiment run into runtime execution details when tracing is enabled

## Regenerated Execution Workflow

Run a baseline validation without executing runtime calls:

```bash
python evaluation/langsmith_experiment_runner.py --skip-runtime
```

Run a small regenerated baseline smoke test:

```bash
python evaluation/langsmith_experiment_runner.py --experiment-label baseline --limit 2
```

Run the same smoke test without creating formal hosted experiment rows:

```bash
python evaluation/langsmith_experiment_runner.py --experiment-label baseline --limit 2 --local-only
```

Run a threshold variant after a local threshold change:

```bash
python evaluation/langsmith_experiment_runner.py --experiment-label threshold_variant
```

Run the runner-local routing sensitivity profile:

```bash
python evaluation/langsmith_experiment_runner.py --experiment-label routing_sensitivity_variant --local-only
```

Run an inspectable local pairwise comparison:

```bash
python evaluation/langsmith_experiment_runner.py --pairwise --limit 2 --local-only
```

Run a local denial-management smoke test:

```bash
python evaluation/langsmith_experiment_runner.py --workflow denial --experiment-label baseline --limit 2 --local-only
```

Run a local denial-management hybrid smoke test:

```bash
python evaluation/langsmith_experiment_runner.py --workflow denial --experiment-label baseline --limit 2 --local-only --execution-mode hybrid
```

Run a live denial operations trace without creating dataset examples or evaluator feedback:

```bash
.venv/bin/python evaluation/run_denial_ops_traces.py --limit 1
```

Run the same path in deterministic local mode:

```bash
.venv/bin/python evaluation/run_denial_ops_traces.py --limit 1 --execution-mode deterministic --no-tracing
```

Run a local denial-management pairwise comparison:

```bash
python evaluation/langsmith_experiment_runner.py --workflow denial --pairwise --pairwise-variant routing_sensitivity_variant --limit 2 --local-only
```

The `threshold_variant` label applies a temporary in-memory governance threshold override inside the experiment runner. The policy file is not rewritten. The current variant raises `confidence_min_for_revise` to `0.7` only while a `threshold_variant` case is executing. Baseline runs and normal runtime calls continue to load thresholds from `governance/policy_rules.json`.

The `routing_sensitivity_variant` label is also runner-local. It lowers the existing metric pass/revise floors for diagnosis consistency, symptom alignment, ICD specificity, and confidence to `0.0` only inside the experiment runner. Its purpose is to expose whether routing is moving because of threshold bands or being dominated by non-threshold signals such as degraded fallback, critic recommendation, and low-evidence boundary handling. It is a sensitivity profile, not a product policy change.

When LangSmith credentials are absent, hosted dataset creation and hosted trace visibility are skipped without failing the script. Local runtime execution can still proceed if dependencies and local environment allow it.

## Baseline vs Variant Comparisons

The baseline label represents the current repo runtime and governance configuration. The threshold variant label represents a regenerated run after an intentional local threshold or policy experiment. Comparisons should focus on high-level operational movement: final status, fallback use, completion, latency, and evaluator pass rates.

The threshold comparison is regenerated. It reruns the current pipeline over the same dataset examples while applying the temporary runner-local threshold override for the variant label.

Aggregate evaluator behavior and internal routing behavior can diverge. A run can preserve Layer 1 evaluator pass rates while still changing final status buckets, reason-code counts, low-evidence volume, specialist-review-oriented volume, or pairwise preference signals. The console summary includes `routing_distribution` for each single-label run and `routing_distribution_movement` for local pairwise comparisons to make those internal movements inspectable.

The original `threshold_variant` can be structurally valid while producing no meaningful routing movement. In the current degraded corpus, cases can be dominated by fallback outputs, zero core metric scores, critic `fail` recommendations, and low-evidence boundary handling. When those dominant signals are unchanged, raising only `confidence_min_for_revise` may change an internal threshold band without changing final status buckets or governance reason-code counts.

Layer 3 reporting includes dominance diagnostics for this distinction: dominant reason codes, degraded count, fallback count, critic recommended-status counts, zero metric counts, and blocked-movement indicators when status and reason-code distributions are identical. Structural variant validation means the runner applied the intended temporary thresholds. Meaningful routing movement means final status buckets, reason-code distributions, or other routing diagnostics changed.

The pairwise comparison criteria are:

- preserves reviewability
- avoids unsupported specificity
- handles ambiguity conservatively
- identifies documentation insufficiency appropriately
- avoids unsafe PASS behavior
- provides operationally useful rationale
- routes borderline cases appropriately

Pairwise output is emitted example-by-example under `pairwise_preferences`, with status movement, reason codes, governance snapshots, critic snapshots, scoring signals, and the preferred output. Hosted LangSmith feedback is not attached for the pairwise layer in this pass.

The routing distribution layer is operationally framed. It reports observed review-routing movement across regenerated outputs; it does not assert clinical correctness, payer acceptance, downstream human behavior, or downstream outcome.

Denial routing distribution reporting is also operationally framed. It does not simulate payer acceptance or downstream human outcomes. Routing actions remain separate from governance posture.

## Dashboard Payload Normalization

LangSmith-facing benchmark outputs include a top-level `output` field for the dashboard primary Output column. Denial runs prefer `routing_action`, then fall back to `governance_posture`, `denial_category`, `final_status`, and `status`. CDI/runtime runs prefer the governed operational decision or status such as `PASS`, `REVISE`, `FAIL`, or `ESCALATE`. Reason codes remain available in structured record details and metadata instead of occupying the primary output field.

Benchmark results also include structured operational metadata when available: `workflow`, `case_id`, `variant`, routing action, governance posture, denial category, recoverability, evidence strength, ambiguity flags, conflicting-evidence flags, specialist-review candidate flags, degraded/fallback flags, `contains_phi`, execution mode, `llm_used`, token/cost availability, token/cost unavailable reason, `latency_ms`, and node-level latency details.

Tags are normalized for filtering with concise values such as `workflow:denial`, `workflow:cdi`, `variant:baseline`, `routing_action:escalate`, `governance_posture:ambiguous`, `contains_phi:false`, `degraded_mode:false`, and `fallback_used:false`.

Latency is reported as measured millisecond-level metadata. If the LangSmith UI rounds very short local scaffold runs to `0.00s`, `latency_ms` remains the operational timing field to inspect.

Token and cost fields are availability semantics, not estimates. Deterministic/local scaffold executions set `llm_used` to false, `token_cost_available` to false, and `token_cost_unavailable_reason` to `no_provider_call`. Provider-backed or hybrid executions preserve token/cost metadata when the runtime exposes it; otherwise they report availability as false with an explicit unavailable reason.

Traces and runs are operational debugging evidence. Datasets are curated benchmark and evaluation inputs. The runner intentionally creates or reuses benchmark examples only through the explicit experiment dataset setup path and does not add arbitrary traces or runs to datasets.

Live denial operations traces are not automatically promoted into datasets. The intended lifecycle is: live traces support operational debugging, selected traces can later inform curated synthetic benchmark cases, curated datasets feed experiments, and experiments produce evaluator traces and feedback.

Live denial operations queue semantics are intentionally different from analyst review and evaluator tables. Name is the case id. Input is a concise problem summary such as `documentation insufficiency | specialist review`. Output is the routing action only. Governance posture, evidence strength, recoverability, ambiguity, alerts, token usage, latency, node timing, fallback/degraded flags, and provider metadata remain in metadata/details. Error is reserved for unrecovered execution anomalies, not conservative routing, escalation, ambiguity, low evidence, fallback warnings, degraded-but-completed execution, or latency warnings.

Live operations metadata is split by purpose. Top-level metadata is for quick triage and prioritizes non-duplicative signals such as LLM use, fallback/degraded flags, alert count/severity, evidence strength, ambiguity, conflicting evidence, specialist-review candidacy, recoverability, governance posture, and token/first-token availability. Token totals, latency, cost, case identifiers, route, and runtime error are not duplicated as leading metadata because they are either first-class table columns or drill-down details. Full diagnostic data remains nested under `diagnostic_metadata`.

First-token latency is not estimated from total latency. It is captured only if the provider or streaming path exposes a real first-token timing field. With the current non-streaming hybrid denial provider path, traces record `first_token_available=false` with a precise unavailable reason when first-token metadata is not exposed.

## Denial Hybrid Interpretation

Denial hybrid mode is limited to LLM-assisted evidence interpretation nodes. `documentation_gap_analyzer` and `recoverability_analyzer` may call the configured LLM provider to produce structured intermediate signals. `routing_engine` and `governance_policy` remain deterministic and do not call an LLM.

Hybrid prompts require strict JSON, prohibit PHI invention, use only provided synthetic content, avoid payer outcome prediction, and avoid final routing or governance assignment. Invalid provider output triggers deterministic fallback and Layer 1 fallback/degraded visibility.

Hosted hybrid denial runs should show the denial graph span and per-node spans, with provider activity nested under the LLM-assisted interpretation node when tracing support is available. Hybrid observations remain regenerated operational evidence from LLM-assisted interpretation, not frozen historical reproduction.

## Layer 1 Operational Alerting

Layer 1 status is execution health only: `completed`, `degraded`, or `failed`. It does not encode whether a governance decision was conservative, ambiguous, low evidence, escalated, or different from an expected route.

Layer 1 `error` is null unless execution fails, structured output is invalid, provider failure is unrecovered, or a critical operational failure is present. Warning and informational alerts, including latency warnings, fallback warnings, degraded-mode warnings, and missing token/cost metadata notices, remain in `alerts` and do not populate runtime `error`. Governance posture values such as `LOW_EVIDENCE` or `AMBIGUOUS`, denial routing actions such as `ESCALATE`, and pairwise disagreements are not execution errors.

Layer 1 alerts are emitted as structured objects with `class`, `severity`, `message`, `observed`, `threshold`, and optional `unit`. Supported classes include latency, cost, token, verbosity, first-token latency, fallback, degraded mode, missing metadata, missing token/cost metadata, invalid output schema, malformed payload, provider failure, trace incompleteness, and evaluator failure where applicable.

Alert thresholds are local heuristic thresholds for operational debugging. The current payload exposes latency, verbosity, token, cost, and first-token latency thresholds in `operational_thresholds`. Metrics are emitted under `operational_metrics`, including `latency_ms`, `output_char_count`, `verbosity_bucket`, `observability_complete`, and token/cost/first-token fields only when available.

LangSmith-facing metadata and tags include alert visibility fields such as `alert:<class>`, `severity:<level>`, and `status:<completed|degraded|failed>` where applicable. These tags are intended for RCM operations debugging filters, not for judging clinical correctness or denial appeal validity.

Hosted experiment denial outputs use the table-oriented form `route=<routing_action> posture=<governance_posture> status=<status>`. Dedicated live operations denial traces use the route alone as primary output. Evaluator rows remain separate and emit evaluator-specific `score` and `comment` fields. Alert classes are metadata/tags and should not appear as the primary operational output or as evaluator comments unless an evaluator is explicitly testing alert behavior.

## Regenerated Operational Scope

This POC is regenerated operational testing. It calls the current runtime against benchmark inputs and observes current behavior. Runs may vary between executions, so comparisons should be interpreted as current operational evidence rather than frozen historical reproduction.

## Usefulness For Modern AI Evaluation Workflows

The workflow gives a compact way to compare current model-mediated behavior against operational expectations. It is useful for detecting broad regressions, routing drift, fallback increase, missing final statuses, and threshold sensitivity. It also gives LangSmith a stable dataset surface for repeated regenerated experiments.

## Limitations

The evaluator logic is intentionally lightweight and does not prove clinical correctness, coding correctness, payer correctness, downstream human behavior, or appeal validity. Regenerated model outputs may vary. Hosted LangSmith results require credentials and network availability. Token metadata is captured only if exposed by the runtime or tracing provider.

Formal experiment creation depends on the installed LangSmith SDK and hosted API availability. If the formal API path fails, the runner reports the degraded status and falls back to the local summary path.

LangSmith preflight diagnostics are availability checks only. Successful client construction does not prove hosted experiment rows or runtime child traces will be created; hosted runs still depend on API reachability, dataset/example access, SDK behavior, and tracing configuration.

For hosted experiment runs, the runner enables `LANGCHAIN_TRACING_V2=true` after LangSmith API readiness succeeds so child runtime traces can be attached. If readiness fails, tracing is not force-enabled and the runner falls back to a local summary without attempting hosted dataset setup.

Hosted regenerated experiments can differ materially from sandboxed local summaries. In network-restricted execution, provider or LangSmith reachability can push cases into degraded fallback and flatten Layer 2/Layer 3 comparisons. In network-enabled hosted execution, the same corpus can complete without fallback, create experiment rows, attach evaluator feedback, and expose child traces. Even with successful traces, policy comparisons remain regenerated observations: final routing and reason-code movement can reflect both threshold changes and model-output variance.

Layer 2 pairwise comparison uses a deterministic local heuristic in this pass. It is meant to surface inspectable review-routing preference signals, not to replace reviewer-labeled evaluation. Layer 3 distribution reporting reports unavailable token or cost fields as unavailable rather than fabricating telemetry.

## Future Scaling Considerations

Future work can add more case families, clearer routing taxonomies, evaluator calibration, reviewer-labeled expected outcomes, trend reports, and separate datasets for high-dollar, deadline-sensitive, coding-specific, and documentation-specific workflows. Scaling should preserve non-PHI synthetic inputs unless a governed data handling process is established.

## Recommended Next Steps

Run a two-case baseline smoke test with credentials available, inspect LangSmith dataset/example creation, then run the full baseline. After that, run a threshold variant only after the intended local threshold change is isolated and documented.
