# LangSmith Dataset Experiment POC

## Dataset Concept

`evaluation/operational_benchmark_cases.json` defines a synthetic, non-PHI operational benchmark for mundane RCM-style MedScribe workflows. Cases cover routine review, documentation gaps, resubmission, appeals, write-off candidates, ambiguous evidence, escalation edges, recovery value, ICD specificity, modifier issues, and low documentation confidence.

Each case carries expected high-level behavior and practical expected routing or governance fields where useful. These fields support lightweight regenerated evaluation, not deterministic output replay.

## Experiment Concept

`evaluation/langsmith_experiment_runner.py` loads the dataset, optionally creates or reuses a LangSmith dataset, runs the existing MedScribe hybrid pipeline with `persist=False`, and attaches four lightweight evaluators:

- `high_level_match`
- `governance_status_match`
- `completed_without_fallback`
- `final_status_present`

The default dataset name is `medscribe-operational-benchmark-poc`. Supported experiment labels are `baseline` and `threshold_variant`.

When LangSmith credentials are available, the runner uses LangSmith's formal `evaluate(...)` workflow. That creates a hosted experiment associated with selected dataset examples and uploads the evaluator scores as hosted feedback. The console summary is still printed locally in the same operational structure.

`--local-only` keeps the older trace/local-summary path and does not create formal hosted experiment rows.

## Trace-Only Runs vs Formal Experiments

Trace-only runs call the runtime and rely on tracing hooks to show root and child traces in the LangSmith project. They are useful for inspecting execution details but do not create the hosted experiment table rows, per-example evaluation entries, or comparison-oriented experiment surfaces.

Formal experiment runs call LangSmith `evaluate(...)` with selected dataset examples. They create a hosted experiment, associate each target run with an example, attach evaluator feedback, and make the run eligible for LangSmith experiment comparison workflows.

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

The `threshold_variant` label applies a temporary in-memory governance threshold override inside the experiment runner. The policy file is not rewritten. The current variant raises `confidence_min_for_revise` to `0.7` only while a `threshold_variant` case is executing. Baseline runs and normal runtime calls continue to load thresholds from `governance/policy_rules.json`.

When LangSmith credentials are absent, hosted dataset creation and hosted trace visibility are skipped without failing the script. Local runtime execution can still proceed if dependencies and local environment allow it.

## Baseline vs Variant Comparisons

The baseline label represents the current repo runtime and governance configuration. The threshold variant label represents a regenerated run after an intentional local threshold or policy experiment. Comparisons should focus on high-level operational movement: final status, fallback use, completion, latency, and evaluator pass rates.

The threshold comparison is regenerated. It does not replay the baseline policy artifacts or prior model outputs. It reruns the current pipeline over the same dataset examples while applying the temporary runner-local threshold override for the variant label.

## Distinction From Deterministic Replay

This POC is regenerated testing. It does not replay old artifacts, old model outputs, prior traces, or frozen intermediate state. It calls the current runtime against the benchmark inputs and observes current behavior.

## Usefulness For Modern AI Evaluation Workflows

The workflow gives a compact way to compare current model-mediated behavior against operational expectations. It is useful for detecting broad regressions, routing drift, fallback increase, missing final statuses, and threshold sensitivity. It also gives LangSmith a stable dataset surface for repeated regenerated experiments.

## Limitations

The evaluator logic is intentionally lightweight and does not prove clinical correctness, coding correctness, payer correctness, or appeal validity. Regenerated model outputs may vary. Hosted LangSmith results require credentials and network availability. Token metadata is captured only if exposed by the runtime or tracing provider.

Formal experiment creation depends on the installed LangSmith SDK and hosted API availability. If the formal API path fails, the runner reports the degraded status and falls back to the local summary path.

## Future Scaling Considerations

Future work can add more case families, clearer routing taxonomies, evaluator calibration, reviewer-labeled expected outcomes, trend reports, and separate datasets for high-dollar, deadline-sensitive, coding-specific, and documentation-specific workflows. Scaling should preserve non-PHI synthetic inputs unless a governed data handling process is established.

## Recommended Next Steps

Run a two-case baseline smoke test with credentials available, inspect LangSmith dataset/example creation, then run the full baseline. After that, run a threshold variant only after the intended local threshold change is isolated and documented.
