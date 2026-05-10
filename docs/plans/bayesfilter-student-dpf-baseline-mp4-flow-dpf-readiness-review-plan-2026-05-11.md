# Plan: MP4 flow and DPF readiness review

## Date

2026-05-11

## Status

Active next-phase plan for the quarantined student DPF experimental-baseline
stream.  Tightened before execution on 2026-05-11.

This plan implements MP4 from:
`docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`.

## Lane Boundary

Owned paths:

- `experiments/student_dpf_baselines/`;
- student-baseline plans, audits, reports, and reset memo under `docs/plans/`.

Out of scope:

- DPF monograph writing files;
- DPF monograph reset memo;
- `docs/chapters/ch19*.tex`;
- `docs/references.bib`;
- production `bayesfilter/` code;
- editing vendored student code in place;
- copying student implementation code into production;
- treating student flow, DPF, HMC, neural OT, or differentiable resampling code
  as production evidence.

## Goal

MP4 determines whether any flow or DPF path in the two student snapshots is
ready for a later bounded comparison panel.

The phase does not attempt to fix or productionize student code.  It should
produce a readiness map with enough evidence to decide:

- which flow and DPF entry points exist in each snapshot;
- which entry points can be imported and called in the current environment;
- which entry points have compatible target semantics for comparison;
- which entry points are blocked by missing dependencies, missing assumptions,
  long runtime, notebooks, plotting, pretrained artifacts, or unclear metrics;
- whether exactly one bounded EDH/LEDH-family candidate should proceed to a
  later adapter/comparison phase.

## Current Evidence

The student-baseline stream has already completed:

- provenance capture for both vendored snapshots;
- linear Kalman adapter and reference checks;
- MLCOE BPF particle diagnostics on linear fixtures;
- nonlinear Gaussian range-bearing reference/proxy panel;
- advanced kernel PFF debug gate.

The MP3 kernel PFF result is important for MP4:

- reduced scalar and matrix kernel PFF runs completed quickly;
- every time step hit `max_iterations=40`;
- loose tolerance did not reduce iteration counts;
- kernel PFF remains `excluded_pending_debug` for routine panels.

Therefore MP4 should not select a kernel PFF path as the first flow comparison
candidate unless the result is explicitly a debug-only follow-up.

## Remaining Gaps

### Gap 1: Flow and DPF entry points are not cataloged

Both snapshots contain flow, DPF, and differentiable resampling code, but the
current harness only validates Kalman, bootstrap/BPF, nonlinear proxy, and
kernel PFF debug paths.

Closure target:

- create an inventory of advanced and MLCOE flow/DPF entry points with file,
  class/function, expected target, dependencies, likely fixture, and readiness
  class.

### Gap 2: Names do not guarantee comparable semantics

Both snapshots use labels such as EDH, LEDH, PFF, PFPF, and DPF, but matching
names may hide different likelihoods, flow schedules, resampling assumptions,
or training requirements.

Closure target:

- classify comparability by target semantics, not by name alone;
- separate deterministic EDH/LEDH particle flows from kernel PFF,
  stochastic-flow PFF, neural DPF, and HMC/parameter-inference code.

### Gap 3: Importability and call signatures are unknown

Some paths may be importable but require project-specific model objects,
TensorFlow modules, pretrained weights, notebooks, plotting, or long experiment
scripts.

Closure target:

- run static inventory plus import and signature probes only;
- do not instantiate filter classes during MP4;
- do not run large experiments during inventory;
- record blockers as structured evidence.

### Gap 4: First comparison candidate is not selected

The next useful comparison should be narrow and reproducible.  A broad
flow/DPF sweep would mix runnable filters with blocked neural or notebook-heavy
workflows.

Closure target:

- select at most one bounded candidate for a later comparison phase, preferably
  EDH or LEDH on the existing Gaussian range-bearing or linear-Gaussian fixture;
- require target, fixture, metric, expected runtime, and blocker plan before
  adapter work.

### Gap 5: DPF and neural/OT paths may be artifact-dependent

The DPF, neural OT, differentiable resampling, and HMC paths may require
trained models, checkpoint files, notebook state, or experiment-specific loss
definitions.

Closure target:

- classify those paths separately as reproduction-gate candidates, not as
  immediate comparison candidates, unless a small command and metric already
  exist.

## Hypotheses

### F1: EDH/LEDH is the most plausible first flow candidate

At least one EDH or LEDH-family path can be imported from each snapshot and
mapped to a shared Gaussian target without vendored-code edits.

Primary criterion:

- advanced and MLCOE each expose at least one EDH/LEDH-family entry point with
  inspectable constructor and run/update methods.

Veto diagnostics:

- either implementation requires notebook state, plotting, or broad experiment
  scripts before a minimal call surface is visible;
- the two paths use incompatible target semantics that cannot be labeled.

### F2: kernel PFF remains excluded from first flow comparison

Given MP3, kernel PFF should not be selected for routine comparison until a
separate debug plan resolves max-iteration behavior.

Primary criterion:

- MP4 classifies kernel PFF entries as `excluded_pending_debug` or
  `debug_only`, not as the first bounded comparison candidate.

Veto diagnostics:

- MP4 selects kernel PFF without explaining why MP3's hit-max diagnostics are
  no longer relevant.

### F3: DPF/neural/OT paths require reproduction gates

DPF, neural OT, differentiable resampling, and HMC paths are unlikely to be
ready for immediate comparison because their metrics depend on training,
pretrained artifacts, or parameter-inference objectives.

Primary criterion:

- those paths are classified with explicit reproduction prerequisites rather
  than merged into flow comparison.

Veto diagnostics:

- DPF or neural/OT paths are compared only because their names resemble the
  target program vocabulary.

### F4: import/signature probes are enough for readiness classification

Most MP4 decisions can be made without running large experiments or editing
vendored code.

Primary criterion:

- an experiment-owned inventory runner can import candidate modules, inspect
  constructors and public methods, and write JSON/Markdown readiness records.

Veto diagnostics:

- inventory requires modifying student snapshots;
- inventory requires large binary artifacts, generated plots, or notebooks as
  first-step evidence.
- inventory requires instantiating filters or calling filter/update methods.

### F5: a later bounded comparison can reuse existing fixtures

If an EDH/LEDH candidate is selected, the later comparison should reuse the
existing linear-Gaussian or nonlinear Gaussian range-bearing fixtures rather
than introduce a new benchmark immediately.

Primary criterion:

- the selected candidate has a declared fixture and proxy/reference metrics:
  latent-state RMSE, ESS if applicable, runtime, finite output checks, and
  agreement with Kalman/EKF/UKF references where meaningful.

Veto diagnostics:

- the selected candidate requires a new target with no reference or proxy
  metric.

## Phase MP4.0: Preflight and Lane Guard

Actions:

1. Record current Git status.
2. Confirm student reset memo, master program, and MP4 plan are present.
3. Confirm no production import boundary:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

4. Confirm no vendored student files are dirty.
5. Write an independent plan audit before implementation work.

Exit gate:

- proceed only if edits can remain in the student-baseline lane.

## Phase MP4.1: Static Inventory

Actions:

1. Add an experiment-owned inventory runner under
   `experiments/student_dpf_baselines/runners/`.
2. Catalog candidate modules and entry points, including:
   - advanced `EDHFlow`, `EDHParticleFilter`, `LEDHFlow`,
     `LEDHParticleFilter`, `StochasticPFFlow`,
     `StochasticPFParticleFilter`, `TFDifferentiableParticleFilter`, and
     kernel PFF classes;
   - MLCOE `ParticleFlowFilter`, `EDH`, `LEDH`, `PFPF_EDH`, `PFPF_LEDH`,
     `KPFF`, `DPF`, flow solvers, and resampling modules.
3. Record for each entry:
   - file path;
   - import path;
   - class or function name;
   - constructor signature if inspectable;
   - public method names relevant to filtering;
   - declared or inferred target type;
   - dependencies and artifact requirements;
   - initial readiness class.

Exit gate:

- inventory includes both snapshots and distinguishes flow, DPF, kernel PFF,
  neural/OT, HMC, and experiment-script paths.

## Phase MP4.2: Import and Signature Probe

Actions:

1. Run import-only probes for candidate modules in the current environment.
2. Inspect signatures for constructors and likely call/update/filter methods.
3. Avoid executing notebook, plotting, training, HMC, long-running, or
   large-artifact code.
4. Do not instantiate candidate classes and do not call filter, step, update,
   train, HMC, notebook, or experiment-script entry points.
5. Convert exceptions to structured blockers.

Exit gate:

- every candidate has one of:
  - `importable`;
  - `candidate_for_bounded_comparison`;
  - `adapter_internal_only`;
  - `blocked_missing_dependency`;
  - `blocked_missing_artifact`;
  - `blocked_missing_assumption`;
  - `blocked_import_side_effect`;
  - `blocked_unclear_api`;
  - `excluded_pending_debug`;
  - `reproduction_gate_required`;
  - `not_comparable`.

An `importable` module is not automatically comparison-ready.  The report must
separately state whether the imported path is suitable for a later bounded
comparison, needs a reproduction gate, or should remain excluded.

## Phase MP4.3: Candidate Selection

Actions:

1. Compare readiness classes against hypotheses F1-F5.
2. Select at most one later comparison candidate.
3. For the selected candidate, specify:
   - implementation path in each snapshot;
   - fixture;
   - metric;
   - expected runtime cap;
   - adapter scope;
   - blocker plan.
4. If no candidate satisfies the gates, classify MP4 as a blocker-only phase
   and recommend the next reproduction gate.

Exit gate:

- MP4 produces either one bounded comparison candidate or a precise reason why
  no flow/DPF comparison should proceed.

## Phase MP4.4: Report, Audit, Tidy, Reset Memo

Actions:

1. Write JSON inventory and summary outputs under
   `experiments/student_dpf_baselines/reports/outputs/`.
2. Write a Markdown report under `experiments/student_dpf_baselines/reports/`.
3. Run syntax checks for edited experiment-owned modules.
4. Run import-boundary search over production code and tests.
5. Run `git diff --check`.
6. Check generated artifact sizes.
7. Update the student reset memo with:
   - goals;
   - gaps;
   - hypothesis outcomes;
   - candidate decision;
   - whether the next phase is justified.

Exit gate:

- MP4 evidence is recorded and lane-scoped;
- no monograph, production, or vendored student files are modified.

## Stop Rules

Stop and ask for direction if:

- inventory requires edits outside the student-baseline lane;
- inventory requires editing vendored student code;
- a candidate requires production `bayesfilter/` changes;
- a candidate requires copying student implementation code into production;
- import probes trigger long-running experiments, plotting, training, or large
  artifact generation;
- no independent or proxy metric can be stated for a proposed comparison;
- monograph-writing and student-baseline files cannot be cleanly staged
  separately.
