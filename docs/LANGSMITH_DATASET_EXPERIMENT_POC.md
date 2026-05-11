# LangSmith Dataset Experiment POC

## Dataset Concept

`evaluation/operational_benchmark_cases.json` defines a synthetic, non-PHI operational benchmark for CDI and coding-support review-routing workflows. Cases cover insufficient ICD specificity, unsupported specificity, documentation ambiguity, borderline documentation sufficiency, low-confidence coding support, recoverable documentation gaps, conflicting documentation, coder-review-worthy cases, conservative revise routing, specialist-review-worthy ambiguity, low-evidence boundary cases, adjacent-case spillover checks, and over-cautious escalation candidates.

Each case carries expected high-level behavior and practical expected routing or governance fields where useful. These fields support lightweight regenerated evaluation, not deterministic output replay.

## Experiment Concept

`evaluation/langsmith_experiment_runner.py` loads the dataset, optionally creates or reuses a LangSmith dataset, runs the existing MedScribe hybrid pipeline with `persist=False`, and preserves three additive evaluation layers.

Layer 1 is the existing operational telemetry evaluation. It attaches four lightweight evaluators:

- `high_level_match`
- `governance_status_match`
- `completed_without_fallback`
- `final_status_present`

Layer 2 is a local pairwise governance usefulness comparison for baseline vs `threshold_variant` outputs. It asks which output provides the safer, clearer, and more operationally useful CDI review-routing outcome before human review.

Layer 3 is routing distribution reporting. It summarizes final status buckets, governance reason-code movement, low-evidence routing volume, specialist-review-oriented volume, adjacent-case spillover indicators, latency when available, and token/cost availability when exposed by runtime metadata.

The default dataset name is `medscribe-operational-benchmark-poc`. Supported experiment labels are `baseline`, `threshold_variant`, and `routing_sensitivity_variant`.

When LangSmith credentials are available, the runner uses LangSmith's formal `evaluate(...)` workflow. That creates a hosted experiment associated with selected dataset examples and uploads the evaluator scores as hosted feedback. The console summary is still printed locally in the same operational structure.

The runner emits a non-secret LangSmith preflight status before runtime execution. It reports key presence, tracing flag presence, endpoint presence, client construction, and hosted readiness without printing secret values. Hosted readiness uses a non-mutating dataset-list check before dataset setup or hosted experiment creation. `--local-only` keeps the older trace/local-summary path, skips hosted dataset setup, and does not create formal hosted experiment rows.

The pairwise workflow is local. It uses a deterministic, inspectable heuristic rather than an LLM-as-judge because the current runner already supports regenerated formal single-label experiments and adding hosted pairwise judge feedback would expand the workflow beyond this POC pass.

## Trace-Only Runs vs Formal Experiments

Trace-only runs call the runtime and rely on tracing hooks to show root and child traces in the LangSmith project. They are useful for inspecting execution details but do not create the hosted experiment table rows, per-example evaluation entries, or comparison-oriented experiment surfaces.

Formal experiment runs call LangSmith `evaluate(...)` with selected dataset examples. They create a hosted experiment, associate each target run with an example, attach evaluator feedback, and make the run eligible for LangSmith experiment comparison workflows.

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

The `threshold_variant` label applies a temporary in-memory governance threshold override inside the experiment runner. The policy file is not rewritten. The current variant raises `confidence_min_for_revise` to `0.7` only while a `threshold_variant` case is executing. Baseline runs and normal runtime calls continue to load thresholds from `governance/policy_rules.json`.

The `routing_sensitivity_variant` label is also runner-local. It lowers the existing metric pass/revise floors for diagnosis consistency, symptom alignment, ICD specificity, and confidence to `0.0` only inside the experiment runner. Its purpose is to expose whether routing is moving because of threshold bands or being dominated by non-threshold signals such as degraded fallback, critic recommendation, and low-evidence boundary handling. It is a sensitivity profile, not a product policy change.

When LangSmith credentials are absent, hosted dataset creation and hosted trace visibility are skipped without failing the script. Local runtime execution can still proceed if dependencies and local environment allow it.

## Baseline vs Variant Comparisons

The baseline label represents the current repo runtime and governance configuration. The threshold variant label represents a regenerated run after an intentional local threshold or policy experiment. Comparisons should focus on high-level operational movement: final status, fallback use, completion, latency, and evaluator pass rates.

The threshold comparison is regenerated. It does not replay the baseline policy artifacts or prior model outputs. It reruns the current pipeline over the same dataset examples while applying the temporary runner-local threshold override for the variant label.

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

The routing distribution layer is operationally framed. It reports observed review-routing movement across regenerated outputs; it does not assert clinical correctness, payer acceptance, reimbursement impact, physician behavior, downstream human override behavior, or downstream human outcome.

## Distinction From Deterministic Replay

This POC is regenerated testing. It does not implement deterministic replay or artifact replay. It does not replay old artifacts, old model outputs, prior traces, or frozen intermediate state. It calls the current runtime against the benchmark inputs and observes current behavior.

## Usefulness For Modern AI Evaluation Workflows

The workflow gives a compact way to compare current model-mediated behavior against operational expectations. It is useful for detecting broad regressions, routing drift, fallback increase, missing final statuses, and threshold sensitivity. It also gives LangSmith a stable dataset surface for repeated regenerated experiments.

## Limitations

The evaluator logic is intentionally lightweight and does not prove clinical correctness, coding correctness, payer correctness, reimbursement economics, physician behavior, downstream human override behavior, or appeal validity. Regenerated model outputs may vary. Hosted LangSmith results require credentials and network availability. Token metadata is captured only if exposed by the runtime or tracing provider.

Formal experiment creation depends on the installed LangSmith SDK and hosted API availability. If the formal API path fails, the runner reports the degraded status and falls back to the local summary path.

LangSmith preflight diagnostics are availability checks only. Successful client construction does not prove hosted experiment rows or runtime child traces will be created; hosted runs still depend on API reachability, dataset/example access, SDK behavior, and tracing configuration.

For hosted experiment runs, the runner enables `LANGCHAIN_TRACING_V2=true` after LangSmith API readiness succeeds so child runtime traces can be attached. If readiness fails, tracing is not force-enabled and the runner falls back to a local summary without attempting hosted dataset setup.

Hosted regenerated experiments can differ materially from sandboxed local summaries. In network-restricted execution, provider or LangSmith reachability can push cases into degraded fallback and flatten Layer 2/Layer 3 comparisons. In network-enabled hosted execution, the same corpus can complete without fallback, create experiment rows, attach evaluator feedback, and expose child traces. Even with successful traces, policy comparisons remain regenerated observations: final routing and reason-code movement can reflect both threshold changes and model-output variance.

Layer 2 pairwise comparison uses a deterministic local heuristic in this pass. It is meant to surface inspectable review-routing preference signals, not to replace reviewer-labeled evaluation. Layer 3 distribution reporting reports unavailable token or cost fields as unavailable rather than fabricating telemetry.

## Future Scaling Considerations

Future work can add more case families, clearer routing taxonomies, evaluator calibration, reviewer-labeled expected outcomes, trend reports, and separate datasets for high-dollar, deadline-sensitive, coding-specific, and documentation-specific workflows. Scaling should preserve non-PHI synthetic inputs unless a governed data handling process is established.

## Recommended Next Steps

Run a two-case baseline smoke test with credentials available, inspect LangSmith dataset/example creation, then run the full baseline. After that, run a threshold variant only after the intended local threshold change is isolated and documented.
