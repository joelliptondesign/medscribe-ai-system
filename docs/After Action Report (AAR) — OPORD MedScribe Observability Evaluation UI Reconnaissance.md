# After Action Report (AAR): OPORD MedScribe Observability Evaluation UI Reconnaissance

Date: 2026-05-07

## Mission Summary

The mission evaluated MedScribe observability, synthetic evaluation, RCA, comparison, and policy-simulation workflows using the existing governed runtime, synthetic incident pack, and hosted LangSmith traces. The work stayed bounded to instrumentation, incident execution, investigation, reporting clarification, and documentation. No new incidents were created during Phase 4, and no additional runtime behavior was changed for this AAR.

The reconnaissance showed that the current stack is strong at making runs inspectable: hosted roots, nested governed runs, stage spans, model spans, metadata, tags, token/cost accounting, and stage outputs can all be made visible. The deeper operational limitation is not basic observability. It is causal interpretation: once model-backed stage outputs vary, policy comparisons, evaluator conclusions, and governance/debugging claims require deterministic replay or frozen upstream state to be trusted.

## Completed Phases

Phase 1 established runtime and trace instrumentation. The governed service path is `service.run_manager.execute`, with stage order intake parser, triage engine, diagnosis engine, ICD mapper, critic, and governance policy. Hosted traces became visible after credential and tracing configuration, and output capture was added so root and stage outputs appeared in LangSmith.

Phase 2 created and ran a seven-incident synthetic pack. Each incident received an incident-level `medscribe.synthetic_incident` root, nested governed runtime span, stage spans, tags, metadata, and summary output. The initial runner was useful for traceability but used coarse high-level matching.

Phase 3 explored specific RCA and reporting cycles:

- Token/cost RCA confirmed deterministic synthetic runs lacked model spans, while hybrid synthetic runs exposed `ChatOpenAI` spans, token usage, cost, and realistic latency.
- Governance RCA showed urgent triage could coexist with governance `FAIL` driven by critic/ICD specificity.
- Governance attribution added explicit inputs used, ignored upstream signals, rule evaluations, fail drivers, and upstream context.
- Reporting clarifications improved `MS-SYN-003`, `MS-SYN-004`, `MS-SYN-005`, and `MS-SYN-007` interpretability without changing runtime behavior.
- Policy simulation for `MS-SYN-003` showed the value and risk of introducing an experimental status, `REVISE_ESCALATED`.
- Malformed payload, overconfidence, critic false-positive, and policy-divergence investigations exposed recurring weaknesses in evaluator semantics, representation preservation, and causal comparison.

Phase 4 synthesized those results into this AAR and a grounded UI/component inventory for future MedFox-oriented product exploration.

## Operational Findings

Trace visibility was effective once the runtime attached outputs explicitly. Root spans, nested governed spans, stage spans, child model spans, tags, metadata, and node diagnostics gave enough evidence to localize most observed behaviors.

Evaluation visibility started coarse. Early summaries reported status, decision, fallback nodes, failure-localization clue, high-level match, and trace stage count. That was enough to prove the synthetic pack ran, but not enough to interpret whether a behavior was operationally desirable.

RCA workflows were most productive when traces exposed both final outcomes and intermediate structured payloads. The strongest investigations correlated raw input, parsed intake, triage, diagnosis, ICD mapping, critic metrics, governance rules, and synthetic summary fields.

Experiment comparison remained noisy in live hybrid mode. The `MS-SYN-003` policy simulation changed the target incident as intended, but unrelated incident behavior also changed between runs. That made causal attribution difficult without replay.

Reporting clarification was low-risk and high-value. Several incidents were not runtime failures; they were ambiguous summaries. Adding derived fields clarified operational meaning while preserving runtime behavior.

## Tooling Strengths

LangSmith and the current tracing setup did well at:

- Showing nested execution structure across synthetic root, governed root, stage spans, and child model spans.
- Preserving tags and incident metadata for filtering and trace lookup.
- Displaying stage outputs after explicit output capture was added.
- Showing model-span token usage, cost, and latency in hybrid mode.
- Supporting manual RCA by exposing payloads at each stage.
- Making governance attribution readable once metadata was added to governance outputs.

The current MedScribe artifacts did well at:

- Producing repeatable synthetic incident roots.
- Capturing non-PHI incident inputs and high-level summaries.
- Keeping reporting-only clarifications separate from runtime behavior changes.
- Making investigation artifacts auditable over time.

## Tooling Weaknesses

Several limitations remained operational rather than infrastructural:

- Causal attribution is weak when upstream model outputs vary between runs.
- The runner’s high-level evaluator often confirms inspectability rather than quality.
- Root summaries initially hid distinctions that were obvious only after opening multiple spans.
- Policy comparison lacks frozen upstream replay.
- Confidence is represented mostly as a critic scalar, not a pipeline-wide uncertainty object.
- Representation loss can occur before critic/governance, but summaries need explicit fields to show it.
- The runner’s coarse `stage_outputs_visible_in_latest_trace` field can disagree with direct incident-specific trace inspection because it checks the latest synthetic root overall.

These are not failures of tracing itself. They are limits of evaluation semantics, replay design, and reporting granularity.

## Debugging Workflow Observations

The most reliable debugging flow was:

1. Start at the synthetic incident root summary.
2. Confirm runtime status, governed decision, fallback usage, high-level match, and tags.
3. Open the nested governed run.
4. Read stage outputs in order: intake, triage, diagnosis, ICD mapper, critic, governance.
5. Compare critic scores and reason codes to governance rule evaluations.
6. Use governance attribution to separate direct policy inputs from ignored upstream context.
7. Return to the root summary to decide whether reporting adequately communicates the finding.

The weakest point was the manual correlation step. For complex incidents, the operator had to infer whether a non-pass outcome was caused by runtime semantics, representation loss, model variability, policy thresholds, evaluator mismatch, or reporting ambiguity.

## Policy Simulation Observations

The `MS-SYN-003` policy simulation demonstrated that a narrow policy experiment can be made visible in traces and summaries. `REVISE_ESCALATED` made urgent safety escalation more legible while preserving the coding/documentation fail driver.

The experiment also showed the cost of live hybrid comparison. One unrelated incident changed status between comparison runs. That does not prove policy impact; it demonstrates comparison contamination. Policy simulation is operationally useful only when the compared policy variants receive the same frozen upstream state.

Governance attribution improved policy simulation readability. It showed the fail driver, the ignored triage signals, and the experimental input used. That made the experiment interpretable even though the broader comparison workflow remained noisy.

## Reproducibility Observations

Some outcomes were stable:

- Hybrid runs consistently produced hosted model spans, token usage, cost, and realistic latency.
- `MS-SYN-005` repeatedly avoided the harmful overconfidence pattern in inspected traces.
- `MS-SYN-006` repeatedly ended as `FAIL` across sampled hosted roots.

Some outcomes varied:

- `MS-SYN-004` varied between `REVISE` and `FAIL`.
- `MS-SYN-007` varied in clinical outputs while retaining valid structured payload shapes.
- Reason-code wording varied across live hybrid traces.

The key finding is that final-status reproducibility is not enough. A run can reproduce the same final status while still varying in upstream structured outputs or reason-code phrasing. Conversely, unrelated incidents can change between comparison runs without being caused by the targeted policy mutation.

## Evaluator And Reporting Observations

The synthetic evaluator initially answered narrow questions:

- Did governance metadata exist?
- Did critic scores exist?
- Did the runtime complete?
- Did ambiguity appear somewhere?

Those checks were useful for traceability validation but fragile for RCA. They could mark an incident as successful while missing the operational distinction being tested.

Reporting clarifications improved:

- `MS-SYN-003`: urgent triage vs coding/documentation-driven governance failure.
- `MS-SYN-004`: representation-loss-driven caution amplification vs standalone critic false-positive.
- `MS-SYN-005`: ambiguity preservation vs true overconfidence.
- `MS-SYN-007`: malformed instruction resilience vs true malformed downstream payload propagation.

The pattern is consistent: root summaries should not just report final status. They should classify the operational behavior under test.

## Observability vs Reproducibility

Observability answers: what happened in this run?

Reproducibility answers: does the same behavior happen again under controlled conditions?

Causal comparison answers: did this specific policy, evaluator, or runtime change cause the difference?

The mission demonstrated that current traces can answer the first question well. They can partially answer the second question by sampling multiple hosted roots. They cannot fully answer the third question without frozen upstream replay or deterministic simulation.

This distinction matters for policy-divergence work. `MS-SYN-006` showed policy metadata and governance attribution clearly, but the latest failure was already overdetermined by upstream representation loss: symptoms normalized to empty, diagnosis was empty, ICD mapping was `NO_MATCH_FOUND`, critic recommended `fail`, and governance applied fail-level rules. That is observable, but not a clean policy comparison.

## Current-Industry Capability Map

Current tooling does well at run observability, trace nesting, payload inspection, tag-based retrieval, model-span usage accounting, and manual RCA support.

Current tooling remains human-driven for interpreting whether a trace difference is caused by model variability, representation drift, evaluator mismatch, governance policy, or reporting semantics.

Observability is strongest at the individual run level. It becomes weaker when comparing live hybrid runs because upstream stages can change.

Causal attribution becomes weak when the same policy is not evaluated against the exact same upstream state.

Replay and simulation semantics become noisy when comparison requires rerunning LLM-backed stages.

Evaluator design becomes fragile when high-level match means “artifact exists” rather than “the target behavior was correctly classified.”

Runtime behavior and reporting semantics diverge when runtime outputs are technically correct but root summaries imply the wrong operational story.

## Current Tooling Stops vs Deterministic Governance/Replay Concepts Begin

Current tooling stops at rich trace evidence and manual comparison. It can show that a governance rule fired, that a model span used tokens, or that a malformed payload did not propagate. It does not automatically prove that a policy change caused an observed difference across live runs.

Deterministic governance/replay concepts begin where the team needs causal comparison:

- Freeze the upstream state immediately before governance.
- Apply multiple policy versions to the same frozen critic and context payload.
- Compare only governance outputs, rule evaluations, and attribution.
- Keep model-backed intake, diagnosis, ICD, and critic calls out of the policy comparison loop.

Observed limitations that motivate this boundary:

- Hybrid comparison noise changed unrelated incident outcomes.
- Upstream drift contaminated policy-divergence interpretation.
- Replay was absent for `MS-SYN-006`.
- Policy-divergence attribution remained untrustworthy without frozen state.
- Representation loss amplified critic caution in `MS-SYN-004`.
- Evaluator/reporting ambiguity made root summaries initially too shallow.
- Governance attribution metadata materially improved interpretability by naming inputs used, ignored, and responsible for fail drivers.

This is not a claim that current tooling is inadequate. It is a boundary statement: traceability can reveal evidence; deterministic replay is needed to support causal policy claims.

## UI And Component Inventory

### Run And Trace Tables

Operational purpose: list runs, incidents, statuses, latency, tags, project, and timestamps.

Why useful: they help operators find the relevant incident root and compare recent run outcomes.

MedFox stance: borrow and adapt. A minimal demo needs dense run tables with filters for incident id, final status, policy version, model-backed vs deterministic mode, and high-level classification.

### Trace Timelines

Operational purpose: show stage duration, nesting, and model calls over time.

Why useful: they reveal latency hotspots and distinguish chain spans from model spans.

MedFox stance: borrow. Keep it compact and operational rather than decorative. Include token/cost badges for model spans.

### Nested Execution Trees

Operational purpose: show synthetic root, governed root, stage spans, and child model spans.

Why useful: they make pipeline structure inspectable without reading code.

MedFox stance: borrow. This is essential for a credible observability demo.

### Expandable Detail Panels

Operational purpose: let operators inspect inputs, outputs, metadata, diagnostics, and attribution only when needed.

Why useful: RCA requires detail, but run tables become unusable if every payload is always expanded.

MedFox stance: borrow. Use panels for stage payloads, critic metrics, governance rule evaluations, and evaluator classifications.

### Metadata Displays

Operational purpose: expose incident id, incident class, dataset version, policy version, governance version, execution mode, and tracing status.

Why useful: metadata determines whether comparisons are valid.

MedFox stance: borrow and strengthen. Metadata should include policy snapshot and replay/frozen-state identifiers when available.

### Payload Viewers

Operational purpose: inspect structured inputs and outputs for each stage.

Why useful: most RCA depended on seeing what was preserved, dropped, normalized, or broadened.

MedFox stance: borrow. Add shape and schema cues for structured outputs.

### Evaluator Summaries

Operational purpose: classify whether an incident target behavior occurred, not just whether a run completed.

Why useful: root summaries reduce the need to manually traverse every trace.

MedFox stance: adapt. Summaries should distinguish inspectability from behavioral classification.

### Comparison Cards

Operational purpose: show before/after status, driver changes, confidence, policy version, and affected incidents.

Why useful: they make experiments readable.

MedFox stance: adapt carefully. Cards must disclose whether comparison used live rerun or frozen replay.

### Policy And Version Comparison Surfaces

Operational purpose: compare thresholds, rule ids, policy versions, governance versions, and rule evaluations.

Why useful: policy-divergence work depends on knowing what changed and what stayed fixed.

MedFox stance: borrow and extend. Include “causal trust” indicators when upstream state is frozen.

### Incident Detail Layouts

Operational purpose: combine input text, expected behavior, expected failure mode, latest result, trace link, evaluator summary, and RCA notes.

Why useful: synthetic incidents are operational stories, not just test rows.

MedFox stance: borrow. This is essential for a demo that explains why each incident matters.

### Lightweight Operational Dashboards

Operational purpose: summarize incident counts, pass/fail/high-level matches, trace visibility, model-span visibility, and unresolved ambiguities.

Why useful: the user needs mission-level state without opening every artifact.

MedFox stance: borrow. Keep dashboard metrics tied to evidence, not vanity counts.

### Governance Attribution Surfaces

Operational purpose: show inputs used, inputs ignored, fail drivers, rule evaluations, and upstream context.

Why useful: governance RCA became much clearer after attribution metadata was added.

MedFox stance: strongly borrow and adapt. This is one of the most valuable patterns observed.

### Debugging And RCA Flows

Operational purpose: walk from symptom to trace evidence to hypothesis to bounded fix surface.

Why useful: the Phase 3 workflow produced repeatable investigation artifacts.

MedFox stance: adapt. A credible demo should support an RCA workspace that links incidents, traces, hypotheses, evidence, and recommended fix surfaces.

## MedFox Product-Layer Implications

Operationally strong patterns:

- Nested traces with visible stage outputs.
- Incident-tagged roots.
- Governance attribution.
- Root-level operational interpretation fields.
- Direct model-span usage visibility.
- Documentation artifacts tied to exact incident ids and trace ids.

Shallow or brittle patterns:

- High-level expected match fields that only confirm output presence.
- Root summaries without behavior classification.
- Live before/after comparison without replay.
- Policy metadata without threshold snapshots or frozen-state references.
- Confidence reported as one critic scalar without cross-stage uncertainty semantics.

Workflows that scaled poorly:

- Manually correlating raw input, parsed fields, diagnoses, ICD mappings, critic metrics, and governance outputs.
- Determining whether a comparison difference was caused by policy code or upstream model variability.
- Explaining why a green high-level match could still require RCA.

Where observability ended and manual reasoning began:

- Deciding whether `MS-SYN-004` was a simple critic false-positive or representation-loss amplification.
- Deciding whether `MS-SYN-005` avoided overconfidence or merely exposed ambiguity.
- Deciding whether `MS-SYN-006` represented policy drift or upstream contamination.
- Deciding whether `MS-SYN-007` was malformed payload propagation or malformed instruction resilience.

Surfaces that would benefit from deterministic attribution/replay:

- Policy comparison.
- Governance threshold experiments.
- Critic calibration comparisons.
- Representation-loss tracing.
- Evaluator before/after claims.

Minimal credible MedFox demo UI surfaces:

- Incident run table.
- Nested trace tree.
- Stage payload viewer.
- Governance attribution panel.
- Evaluator summary panel.
- Before/after comparison card with replay status.
- Policy/version panel.
- RCA note panel with evidence and confidence.

## Recommended Future Directions

Bounded future exploration areas:

- Replay/simulation workflows for governance and evaluator comparison.
- Frozen upstream evaluation comparison, especially pre-governance state replay.
- Governance attribution as a first-class UI and artifact surface.
- Evaluator calibration for incident-specific behavior classification.
- Representation-loss tracing from raw narrative to structured fields.
- Confidence semantics across intake, diagnosis, critic, and governance.
- Structured-output validation and expected-vs-observed shape reporting.
- Comparison trustworthiness indicators for live rerun vs frozen replay.
- Operational semantics separation, especially safety escalation vs documentation/coding failure.

Recommended next technical direction:

- Add reporting-only causal-readiness fields for `policy_change_divergence`.
- Then design a small frozen-state governance replay harness in a separate brief.
- Keep runtime model behavior out of policy comparison until frozen replay exists.

## Final Mission Assessment

The mission met its success conditions. MedScribe traces became visible, model-backed hybrid runs were inspected, synthetic incidents were executed, RCA workflows were exercised, reporting ambiguity was reduced for multiple incident classes, and policy simulation exposed both the value and limitations of current comparison workflows.

The clearest operational lesson is that observability is necessary but not sufficient. The team can now see what happened. The next maturity step is controlling what is being compared.

Current tooling is strong enough to support manual RCA and trace-driven investigation. It is not enough by itself to support causally trustworthy policy-divergence claims when upstream model-backed stages are rerun. For that, deterministic replay and frozen upstream state become the critical boundary.

This AAR closes the OPORD MedScribe observability/evaluation UI reconnaissance mission and records the reusable UI and workflow patterns most relevant to a future MedFox-oriented demo.
