# Plan: MP1 MLCOE particle adapter gate

## Date

2026-05-10

## Status

Active next-phase plan for the quarantined student DPF experimental-baseline
stream.

This plan implements MP1 from:
`docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`.

## Lane Boundary

This is student-baseline experimental work only.

Owned paths:

- `experiments/student_dpf_baselines/`;
- student-baseline plans and audits under `docs/plans/`;
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`.

Out of scope:

- DPF monograph writing files;
- DPF monograph reset memo;
- `docs/chapters/ch19*.tex`;
- `docs/references.bib`;
- production `bayesfilter/` code;
- editing vendored student code in place;
- treating student code as production or correctness authority.

Final staging must be path-scoped to the student-baseline lane.

## Goal for the Next Phase

The next phase should close the largest current asymmetry in the linear
student-baseline harness: `advanced_particle_filter` exposes bootstrap
particle-filter diagnostics on the existing linear fixtures, while `2026MLCOE`
currently exposes only the Kalman smoke path through its adapter.

The goal is to add a narrow, quarantined MLCOE BPF adapter gate and determine
whether MLCOE particle diagnostics can be compared to the advanced
bootstrap-PF diagnostics on the already validated linear fixtures.

This phase does not attempt GSMC, UPF, flow filters, DPF, HMC, neural OT, or
nonlinear correctness.  Those remain later phases.

## Current Evidence

Completed evidence:

- both student snapshots are fixed plain-source snapshots under
  `experiments/student_dpf_baselines/vendor/`;
- both Kalman paths match independent Kalman references to numerical precision
  on small and stress linear-Gaussian fixtures;
- `advanced_particle_filter` bootstrap-PF diagnostics are available through
  `advanced_particle_filter_adapter.py`;
- advanced bootstrap-PF diagnostics degrade under the low-noise stress fixture
  at smaller particle counts;
- nonlinear smoke is feasible but not reference-consistent;
- advanced kernel PFF is classified as
  `algorithm_test_sensitivity_and_long_runtime`.

Relevant implementation facts:

- MLCOE BPF lives in
  `experiments/student_dpf_baselines/vendor/2026MLCOE/src/filters/particle.py`.
- MLCOE BPF requires a model with:
  - `state_dim`;
  - `F`;
  - `Q_filter`;
  - `R_inv_filter`;
  - `h_func`.
- MLCOE BPF exposes:
  - particle state variable `X`;
  - particle weights `W`;
  - effective sample size variable `ess`;
  - one-step posterior mean returned by `step(z)`.
- MLCOE BPF does not directly expose:
  - filtering log likelihood;
  - normalized likelihood estimate;
  - explicit resampling indicator;
  - per-step predictive covariance.

## Gaps to Close

### Gap 1: MLCOE BPF is not wrapped

The current MLCOE adapter only calls `src.filters.classical.KF`.  The harness
therefore cannot compare MLCOE particle-filter behavior to the advanced
bootstrap-PF behavior.

Closure target:

- add a quarantined adapter path that calls MLCOE `src.filters.particle.BPF`
  on existing linear-Gaussian fixtures.

### Gap 2: Particle diagnostics are asymmetric

The advanced adapter reports particle means, particle covariances, ESS,
resampling count, and a particle-filter log-likelihood diagnostic.  MLCOE BPF
does not expose the same full set of diagnostics.

Closure target:

- expose only defensible MLCOE BPF diagnostics:
  - posterior particle mean trajectory;
  - posterior weighted particle covariance trajectory;
  - ESS trajectory;
  - approximate resampling count inferred from the documented BPF threshold;
  - runtime.
- keep unavailable log-likelihood fields as null.

### Gap 3: Reference comparison for MLCOE BPF is undefined

Kalman reference agreement is already known for Kalman paths, but particle
paths require different metrics.

Closure target:

- compare MLCOE BPF particle mean trajectory to the independent Kalman filtered
  mean trajectory;
- compare MLCOE BPF particle covariance trajectory to the independent Kalman
  filtered covariance trajectory where shape permits;
- compare ESS and runtime to the advanced bootstrap-PF diagnostics without
  implying exact semantic identity.

### Gap 4: TensorFlow seeding and graph behavior may block reproducibility

MLCOE BPF uses TensorFlow sampling and a `tf.function` step.  Determinism and
runtime may differ from NumPy-side advanced bootstrap-PF runs.

Closure target:

- set TensorFlow and NumPy seeds explicitly for each run;
- keep the first gate small and bounded;
- record runtime and failures as structured adapter results.

### Gap 5: MLCOE particle diagnostics could expose model-assumption mismatch

The existing fixtures are NumPy linear-Gaussian fixtures.  MLCOE BPF expects a
TensorFlow model object with specific field names.

Closure target:

- create the fixture-to-MLCOE model bridge inside the adapter layer only;
- do not patch vendored MLCOE model code;
- record a structured blocker if the bridge cannot preserve fixture semantics.

## Hypotheses

### H1: MLCOE BPF adapter feasibility

MLCOE BPF can run on at least one existing linear-Gaussian fixture through a
quarantined adapter without modifying vendored MLCOE code.

Test:

- construct a minimal adapter-local TensorFlow model from a validated fixture;
- initialize MLCOE BPF with fixture prior `m0, P0`;
- run BPF over fixture observations for a small particle count.

Primary criterion:

- at least one run returns `status="ok"` with particle mean trajectory and ESS
  trajectory.

Veto diagnostics:

- requires vendored-code edits;
- requires production code edits;
- fails because fixture semantics cannot be represented by the MLCOE BPF model
  contract.

### H2: Diagnostic extraction

MLCOE BPF exposes enough particle state and weight information after each step
to compute weighted particle mean, weighted particle covariance, ESS, and
threshold-based resampling diagnostics.

Test:

- after each BPF step, read `X`, `W`, and `ess`;
- compute weighted covariance in the adapter;
- infer resampling events from `ess < 0.1 * num_particles`, matching the
  MLCOE BPF implementation threshold.

Primary criterion:

- output includes non-null `particle_means`, `particle_covariances`,
  `ess_by_time`, `resampling_count`, and runtime.

Veto diagnostics:

- TensorFlow graph behavior prevents reliable access to post-step state;
- covariance computation is numerically invalid;
- inferred resampling count is ambiguous and cannot be documented.

### H3: Linear stress particle behavior

MLCOE BPF will show the same broad qualitative pressure observed in the
advanced bootstrap-PF diagnostics: lower observation noise and smaller particle
counts will reduce ESS and increase state-estimation error against the Kalman
reference.

Test:

- run MLCOE BPF over the existing stress fixtures:
  - `lgssm_1d_long`;
  - `lgssm_cv_2d_long`;
  - `lgssm_cv_2d_low_noise`;
- use seeds `[0, 1, 2]` and particle counts `[64, 128, 512]` for the first
  bounded gate;
- compare particle mean and covariance RMSE against the independent Kalman
  reference;
- summarize ESS and threshold-based resampling count.

Primary criterion:

- results can classify H3 as supported, partially supported, or not supported
  with fixture-level evidence.

Veto diagnostics:

- run time is too high for the bounded gate;
- all MLCOE BPF runs fail or return invalid numerical values;
- metrics cannot be interpreted without fabricating likelihood estimates.

### H4: Cross-student particle comparison remains comparison-only

Once MLCOE BPF diagnostics are available, cross-student particle comparisons
can be summarized as experimental evidence without implying production
correctness or exact semantic equivalence.

Test:

- compare MLCOE BPF and advanced bootstrap-PF on matching fixture, seed, and
  particle-count groups;
- report:
  - particle mean RMSE between implementations;
  - particle covariance RMSE where shapes align;
  - ESS summaries;
  - runtime summaries;
  - missing log-likelihood status for MLCOE BPF.

Primary criterion:

- the report clearly separates reference metrics, cross-student metrics, and
  unavailable metrics.

Veto diagnostics:

- report claims correctness from student agreement;
- report compares log likelihoods when MLCOE BPF has no defensible likelihood
  output;
- report hides missing metrics instead of recording them.

## Phase MP1.0: Preflight and Lane Guard

### Goal

Confirm the phase can proceed without touching monograph or production files.

### Actions

1. Record `git status --short --branch`.
2. Confirm the active master program and student reset memo exist.
3. Confirm no production import boundary currently exists:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

4. Record current unrelated dirty files in the reset memo if the phase is
   executed.

### Exit Gate

Proceed only if new edits can remain under:

- `experiments/student_dpf_baselines/`;
- student-baseline files under `docs/plans/`.

## Phase MP1.1: Adapter API Probe

### Goal

Determine the minimal MLCOE BPF call sequence on one fixture before modifying
the broader panel runner.

### Actions

1. Add an adapter-local fixture model bridge in
   `experiments/student_dpf_baselines/adapters/mlcoe_adapter.py`.
2. Add a new function, for example `run_bpf_fixture(...)`, that:
   - prepends the MLCOE snapshot path;
   - imports `src.filters.particle.BPF`;
   - builds TensorFlow constants from fixture arrays;
   - initializes BPF from `m0, P0`;
   - runs observations step by step;
   - records particle means, covariances, ESS, and runtime.
3. Keep the existing Kalman adapter behavior unchanged.

### Test Command

Use a small direct smoke command or a temporary runner during development:

```bash
python -m experiments.student_dpf_baselines.runners.run_mlcoe_particle_gate
```

The final runner will be added in MP1.2.

### Exit Gate

Proceed if one `lgssm_1d_short` BPF run succeeds or fails with a precise
structured blocker.

## Phase MP1.2: Bounded Linear Particle Panel

### Goal

Run a bounded MLCOE BPF panel against the existing linear references.

### Actions

1. Add:

```text
experiments/student_dpf_baselines/runners/run_mlcoe_particle_gate.py
```

2. Use fixtures:
   - `lgssm_1d_short`;
   - `lgssm_cv_2d_short`;
   - `lgssm_1d_long`;
   - `lgssm_cv_2d_long`;
   - `lgssm_cv_2d_low_noise`.
3. Use seeds `[0, 1, 2]` and particle counts `[64, 128, 512]`.
4. For each run, write records containing:
   - MLCOE BPF adapter result;
   - independent Kalman reference metrics;
   - particle mean RMSE against Kalman;
   - particle covariance RMSE against Kalman;
   - average/min ESS;
   - threshold-based resampling count;
   - runtime;
   - missing log-likelihood status.
5. Produce:

```text
experiments/student_dpf_baselines/reports/outputs/mlcoe_particle_gate_2026-05-10.json
experiments/student_dpf_baselines/reports/outputs/mlcoe_particle_gate_summary_2026-05-10.json
experiments/student_dpf_baselines/reports/student-dpf-baseline-mlcoe-particle-gate-result-2026-05-10.md
```

### Exit Gate

Proceed if the panel either:

- completes with interpretable records; or
- produces structured blockers explaining why MLCOE BPF cannot enter routine
  comparison.

## Phase MP1.3: Cross-Student Particle Comparison

### Goal

Compare available MLCOE BPF diagnostics to existing advanced bootstrap-PF
diagnostics without overstating equivalence.

### Actions

1. Reuse the advanced adapter on the same fixtures, seeds, and particle counts
   when runtime remains bounded.
2. Summarize matched groups by:
   - fixture;
   - seed;
   - particle count.
3. Compare:
   - particle mean RMSE between implementations;
   - particle covariance RMSE where shapes align;
   - average/min ESS;
   - resampling count;
   - runtime.
4. Keep likelihood comparison one-sided:
   - advanced bootstrap-PF log-likelihood may be reported against Kalman;
   - MLCOE BPF log-likelihood remains null unless a defensible estimator is
     added in a later plan.

### Exit Gate

Proceed if the report clearly distinguishes:

- reference-backed metrics;
- cross-student comparison metrics;
- unavailable metrics;
- implementation-specific semantics.

## Phase MP1.4: Audit, Tidy, and Reset Memo Update

### Goal

Make the phase auditable and decide whether MP2 or another phase is justified.

### Actions

1. Run import-boundary check:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

2. Syntax-check edited experiment modules:

```bash
python -m py_compile \
  experiments/student_dpf_baselines/adapters/mlcoe_adapter.py \
  experiments/student_dpf_baselines/runners/run_mlcoe_particle_gate.py
```

3. Run whitespace check on edited files:

```bash
git diff --check -- \
  docs/plans/bayesfilter-student-dpf-baseline-mp1-mlcoe-particle-adapter-plan-2026-05-10.md \
  docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md \
  experiments/student_dpf_baselines/adapters/mlcoe_adapter.py \
  experiments/student_dpf_baselines/runners/run_mlcoe_particle_gate.py
```

4. Check generated output sizes.
5. Update the student reset memo with:
   - phase result;
   - hypothesis outcomes;
   - interpretation;
   - next justified phase.

### Exit Gate

The phase is complete when:

- all generated evidence is under `experiments/student_dpf_baselines/reports/`;
- the reset memo records the result and next decision;
- no monograph or production files are staged;
- a scoped commit can be made if requested.

## Decision Rules

### Continue to MP2: Nonlinear Reference and Proxy-Metric Spine

Continue to MP2 if:

- MLCOE BPF runs on at least the short and stress linear fixtures; and
- particle diagnostics are interpretable; and
- no vendored-code edits were required.

Rationale:

- a balanced linear particle baseline would then exist, making nonlinear proxy
  metrics the next largest gap.

### Run a Narrow MP1 Follow-Up Instead

Run a narrow MP1 follow-up if:

- MLCOE BPF runs only on 1D fixtures;
- covariance or ESS extraction is partially blocked;
- runtime is acceptable but metrics need adapter refinement;
- BPF works but GSMC/UPF are now the obvious missing MLCOE particle paths.

Rationale:

- extending a partially successful particle gate is lower risk than moving to
  nonlinear work.

### Stop and Ask for Direction

Stop if:

- MLCOE BPF cannot run without vendored-code edits;
- TensorFlow graph behavior prevents structured diagnostics;
- all BPF runs fail or produce invalid numerical values;
- the phase requires production `bayesfilter/` changes;
- the comparison cannot avoid fabricated likelihood claims.

## Expected Interpretation

The most likely useful outcome is not exact agreement between particle filters.
The useful outcome is a balanced diagnostic panel showing how two independent
student particle implementations behave under the same linear stress fixtures,
with independent Kalman references as the anchor.

The key question is whether MLCOE BPF can supply reliable ESS, state-estimation,
covariance, runtime, and resampling diagnostics.  If yes, the student-baseline
lane can move from Kalman-only MLCOE evidence to a genuine cross-student
particle baseline.  If no, the failure itself becomes structured evidence that
MLCOE particle paths are not ready for routine comparison.
