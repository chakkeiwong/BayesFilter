# Master program: student DPF experimental baseline

## Date

2026-05-10

## Status

Active governing program for the quarantined student DPF experimental-baseline
stream.

This document consolidates the prior consolidation, gap-closure, and
hypothesis-closure cycles into one controlling program.  It is not a production
implementation plan and it is not part of the reader-facing DPF monograph
writing stream.

## Governing Purpose

The purpose of this stream is to turn two student implementations into
reproducible, quarantined comparison artifacts.  The outputs should help
BayesFilter developers understand algorithm behavior, implementation
assumptions, failure modes, and baseline performance.  The outputs must not be
used as production code, correctness certificates, or API authority.

Student code is `comparison_only`.

The program succeeds when it produces:

1. fixed, provenance-recorded snapshots of both student repositories;
2. isolated adapters that call student code through a stable result contract;
3. reference-backed linear-Gaussian comparison panels;
4. nonlinear and flow comparison panels only after independent references or
   explicit proxy metrics exist;
5. structured blocker records for non-runnable or long-running student code;
6. reset-memo updates that preserve lane separation from monograph writing and
   production BayesFilter work.

## Lane Boundary

Owned surfaces:

- `experiments/student_dpf_baselines/`;
- `experiments/controlled_dpf_baseline/`, if a later controlled-reference
  harness is explicitly created for this stream;
- student-baseline plans, audits, reports, and this master program under
  `docs/plans/`;
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`.

Explicitly out of scope:

- reader-facing DPF monograph chapter writing;
- `docs/chapters/ch19*.tex` and later monograph chapter drafts;
- `docs/references.bib`;
- DPF monograph rebuild plans and reset memo;
- production `bayesfilter/` code;
- production HMC target construction;
- promotion of student code into production modules.

Path-scoped staging is mandatory.  A student-baseline commit must not include
dirty monograph files or production implementation files.

## Governing Documents

This master program controls the student-baseline lane.

Supporting records:

- reset memo:
  `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
- original broad consolidation plan:
  `docs/plans/bayesfilter-student-dpf-baseline-consolidation-plan-2026-05-08.md`;
- original consolidation audit:
  `docs/plans/bayesfilter-student-dpf-baseline-consolidation-audit-2026-05-09.md`;
- first active execution plan:
  `docs/plans/bayesfilter-student-dpf-baseline-gap-closure-plan-2026-05-09.md`;
- first active execution audit:
  `docs/plans/bayesfilter-student-dpf-baseline-gap-closure-plan-audit-2026-05-10.md`;
- follow-on hypothesis plan:
  `docs/plans/bayesfilter-student-dpf-baseline-hypothesis-closure-plan-2026-05-10.md`;
- follow-on hypothesis audit:
  `docs/plans/bayesfilter-student-dpf-baseline-hypothesis-closure-plan-audit-2026-05-10.md`.

Program rule:

- execution plans may add phase-specific detail;
- the reset memo records current continuity state and phase results;
- this master program records durable scope, evidence, gates, and next program
  direction.

## Source Snapshot State

The student sources are stored as plain vendored snapshots under:

- `experiments/student_dpf_baselines/vendor/2026MLCOE/`;
- `experiments/student_dpf_baselines/vendor/advanced_particle_filter/`.

Nested `.git` metadata has been removed.  Snapshot provenance is recorded in:

- `experiments/student_dpf_baselines/sources.yml`;
- `experiments/student_dpf_baselines/PERMISSIONS.md`;
- `experiments/student_dpf_baselines/vendor/SNAPSHOT.md`.

Recorded upstream commits:

| Source | Snapshot commit | Use |
| --- | --- | --- |
| `2026MLCOE` | `020cfd7f2f848afa68432e95e6c6e747d3d2402d` | comparison-only |
| `advanced_particle_filter` | `d2a797c330e11befacbb736b5c86b8d03eb4a389` | comparison-only |

The students have granted permission to use and test their code internally for
this project.  Redistribution or public release status remains separate from
internal comparison use.

## Environment Caveats

The local audits observed:

- Python `3.13.13`;
- NumPy `2.1.3`;
- SciPy `1.17.1`;
- TensorFlow `2.20.0`;
- TensorFlow Probability `0.25.0`;
- CPU-only TensorFlow execution in recorded runs.

Known caveats:

- `advanced_particle_filter` documentation expects older TensorFlow and
  TensorFlow Probability versions;
- plotting-heavy or notebook-heavy workflows are not reliable first-pass gates;
- long-running kernel PFF tests must use explicit time bounds;
- future result reports must record command, environment, seed, particle count,
  source commit, runtime, and failure classification.

## Current Evidence Base

### Provenance and Snapshot

Status: complete for internal comparison.

Evidence:

- source snapshots are present under `experiments/student_dpf_baselines/vendor/`;
- upstream commit hashes are recorded;
- nested Git repositories were removed;
- permissions and snapshot records exist.

Interpretation:

- the stream is protected from student upstream code drift;
- future updates should be explicit new snapshots, not silent pulls.

### Original and Targeted Reproduction

Status: partially complete, sufficient for linear adapter work.

Evidence:

- `2026MLCOE` unit tests: 8 passed;
- `2026MLCOE` integration tests: 12 passed;
- `advanced_particle_filter/tests/test_basic.py`: 3 passed;
- `advanced_particle_filter/tests/test_filters.py`: 22 passed;
- `advanced_particle_filter/tests/test_kernel_pff.py`: partial `.F.` behavior
  and non-completion were later localized.

Result records:

- `experiments/student_dpf_baselines/reports/dependency-audit-2026-05-09.md`;
- `experiments/student_dpf_baselines/reports/advanced-particle-filter-reproduction-2026-05-10.md`;
- `experiments/student_dpf_baselines/reports/outputs/advanced_particle_filter_reproduction_2026-05-10.json`.

Interpretation:

- both snapshots are runnable enough for linear-Gaussian comparison;
- kernel PFF must not enter routine panels without a separate bounded gate.

### Adapter and Reference Spine

Status: complete for Kalman-path linear-Gaussian comparison.

Implemented:

- `experiments/student_dpf_baselines/adapters/common.py`;
- `experiments/student_dpf_baselines/adapters/mlcoe_adapter.py`;
- `experiments/student_dpf_baselines/adapters/advanced_particle_filter_adapter.py`;
- `experiments/student_dpf_baselines/fixtures/common_fixtures.py`;
- `experiments/student_dpf_baselines/fixtures/stress_fixtures.py`;
- `experiments/student_dpf_baselines/runners/run_reference_fixtures.py`;
- `experiments/student_dpf_baselines/runners/run_student_baseline_panel.py`;
- `experiments/student_dpf_baselines/runners/compare_student_outputs.py`;
- `experiments/student_dpf_baselines/runners/run_linear_stress_panel.py`;
- `experiments/student_dpf_baselines/runners/run_nonlinear_smoke.py`.

Interpretation:

- adapters can call both student snapshots without vendored-code edits;
- the common result contract can preserve missing metrics as null or structured
  failures instead of fabricating data.

### First Linear-Gaussian Panel

Status: complete.

Result:

- both student Kalman paths matched independent Kalman references to numerical
  precision on small fixtures;
- `2026MLCOE`: 12/12 runs ok;
- `advanced_particle_filter`: 12/12 runs ok;
- maximum cross-student filtered-mean RMSE was approximately `3.94e-16`;
- maximum cross-student log-likelihood absolute difference was approximately
  `8.88e-16`.

Result record:

- `experiments/student_dpf_baselines/reports/student-dpf-baseline-gap-closure-result-2026-05-10.md`.

Interpretation:

- the harness is useful for controlled linear comparison;
- this does not validate the student code as production-quality or generally
  correct.

### Linear Stress Panel

Status: complete for available diagnostics.

Result:

- `2026MLCOE`: 45/45 Kalman smoke runs ok, max Kalman log-likelihood error
  approximately `2.84e-14`;
- `advanced_particle_filter`: 45/45 Kalman smoke runs ok, max Kalman
  log-likelihood error approximately `7.11e-15`;
- advanced bootstrap-PF diagnostics degraded with lower observation noise and
  smaller particle counts;
- on `lgssm_cv_2d_low_noise`, median PF log-likelihood error was about `24.15`
  at 64 particles and `0.767` at 512 particles;
- min average ESS was about `10.17` at 64 particles on the low-noise fixture.

Result record:

- `experiments/student_dpf_baselines/reports/student-dpf-baseline-linear-stress-result-2026-05-10.md`.

Interpretation:

- Kalman paths remain reference-consistent;
- particle degeneracy pressure is visible in the advanced bootstrap-PF path;
- MLCOE particle diagnostics are still missing from the adapter layer.

### Kernel PFF Isolation

Status: classified as not routine-panel ready.

Result:

- `test_kernel_pff_lgssm`: timed out after 90 seconds;
- `test_kernel_pff_convergence`: failed because average iterations reached
  `100.0`, the maximum;
- `test_scalar_vs_matrix_kernel`: passed but took about 85.92 seconds.

Classification:

- `algorithm_test_sensitivity_and_long_runtime`.

Result record:

- `experiments/student_dpf_baselines/reports/advanced-particle-filter-kernel-pff-reproduction-2026-05-10.md`.

Interpretation:

- kernel PFF evidence is localized and reproducible;
- it should be handled by a dedicated debug/reproduction gate before any
  cross-student flow comparison.

### Nonlinear Smoke

Status: feasible as smoke only, not reference-consistent.

Result:

- `advanced_particle_filter` range-bearing Student-t smoke ran and executed
  EKF, UKF, and bootstrap-PF paths;
- `2026MLCOE` range-bearing smoke ran and executed EKF and UKF step paths;
- advanced nonlinear PF average ESS was about `9.44` for 128 particles;
- MLCOE EKF final estimate stayed at zero while UKF moved.

Result record:

- `experiments/student_dpf_baselines/reports/student-dpf-baseline-nonlinear-smoke-result-2026-05-10.md`.

Interpretation:

- nonlinear comparison is feasible only after adding independent references or
  explicit proxy metrics;
- method-specific anomalies must be treated as blockers or diagnostics, not
  smoothed over by cross-student comparison.

## Open Gaps

### Gap 1: MLCOE Particle and Flow Diagnostics

The current MLCOE adapter covers the Kalman smoke path.  It does not yet expose
BPF, GSMC, UPF, flow, DPF, resampling, ESS, or runtime diagnostics in the common
result contract.

Risk:

- linear stress comparisons currently compare advanced bootstrap-PF diagnostics
  against MLCOE Kalman-only evidence, not against MLCOE particle behavior.

Required closure:

- add one MLCOE particle adapter gate at a time;
- begin with BPF on existing linear stress fixtures;
- require no vendored-code edits;
- classify missing metrics as null or structured blockers.

### Gap 2: Nonlinear Reference Spine

The nonlinear smoke paths run, but there is no reference-backed nonlinear panel.

Risk:

- cross-student nonlinear agreement could reflect shared weakness or different
  likelihood assumptions rather than correctness.

Required closure:

- define fixed nonlinear fixtures with latent states and observations;
- define method-specific target labels;
- use EKF/UKF agreement bands, repeated-seed PF summaries, and latent-state
  RMSE as proxy metrics where exact references are unavailable;
- avoid comparing Student-t and Gaussian likelihoods without explicit target
  labels.

### Gap 3: Kernel PFF Debug Gate

Kernel PFF is reproducibly slow or failing under bounded tests.

Risk:

- adding kernel PFF to normal panels would create unstable CI/runtime behavior
  and unclear evidence.

Required closure:

- isolate reduced-size `MatrixKernelPFF` and `ScalarKernelPFF` experiments;
- record convergence iterations, tolerance, runtime, and failure state;
- do not patch vendored code in place;
- decide whether kernel PFF remains excluded, is wrapped as experimental-only,
  or needs a local patch record.

### Gap 4: HMC, DPF, Neural OT, and Differentiable Resampling

The vendored snapshots include HMC, DPF, neural/amortized OT, and differentiable
resampling code, but none of those workflows are validated in the current
student-baseline harness.

Risk:

- those directories can look like available evidence even though they have not
  passed reproduction, adapter, or reference gates.

Required closure:

- keep them out of result claims until each has a separate reproduction gate;
- require a small runnable command before adapter work;
- require explicit target semantics before comparing to BayesFilter concepts.

### Gap 5: Environment Reproducibility

The current results are local-environment evidence, not a locked environment.

Risk:

- TF/TFP behavior and long-running tests may change across environments.

Required closure:

- record environment in every report;
- optionally add a minimal environment note or constraints file for the
  student-baseline lane after the next adapter gate;
- avoid broad dependency changes unless a phase plan justifies them.

## Master Program Phases

### MP0: Governance Consolidation

Status: current phase.

Purpose:

- create this master program;
- make the student reset memo point to it;
- preserve the distinction between student experimental baseline work and DPF
  monograph writing.

Exit criteria:

- this file exists under `docs/plans/`;
- the reset memo identifies this as the active governing program;
- no monograph files are edited.

### MP1: MLCOE Particle Adapter Gate

Execution plan:

`docs/plans/bayesfilter-student-dpf-baseline-mp1-mlcoe-particle-adapter-plan-2026-05-10.md`.

Primary hypothesis:

MLCOE BPF can be wrapped without vendored-code edits and can expose enough
particle diagnostics to compare against the advanced bootstrap-PF diagnostics
on existing linear stress fixtures.

Implementation details:

1. Read MLCOE particle and model APIs under the vendored snapshot.
2. Add a narrow adapter method for one particle path, preferably BPF.
3. Reuse existing linear fixtures and result contract fields:
   - `particle_means`;
   - `particle_covariances`;
   - `ess_by_time`;
   - `resampling_count`;
   - `runtime_seconds`;
   - structured `diagnostics`.
4. Run only existing linear fixtures first.
5. Write a result report and JSON output under
   `experiments/student_dpf_baselines/reports/`.
6. Update the student reset memo with whether broader MLCOE particle adapter
   work is justified.

Primary criterion:

- at least one MLCOE particle path runs on one existing linear fixture through a
  quarantined adapter, or fails with a precise structured blocker.

Veto diagnostics:

- requires editing vendored MLCOE code in place;
- requires production `bayesfilter/` changes;
- result metrics cannot be mapped without fabricating unavailable fields;
- runtime is unbounded or output artifacts are too large for normal commit.

### MP2: Nonlinear Reference and Proxy-Metric Spine

Execution plan:

`docs/plans/bayesfilter-student-dpf-baseline-mp2-nonlinear-reference-spine-plan-2026-05-10.md`.

Primary hypothesis:

The current nonlinear smoke paths can become a meaningful comparison panel if
the fixture targets and metrics are explicit enough to avoid false
cross-student equivalence.

Implementation details:

1. Define one or two small nonlinear fixtures with fixed seeds, latent states,
   observations, process noise, and observation noise.
2. Declare target semantics for each implementation path:
   - EKF;
   - UKF;
   - bootstrap PF;
   - Student-t or Gaussian observation likelihood as applicable.
3. Add proxy metrics:
   - trajectory RMSE against latent state;
   - repeated-seed mean and dispersion for PF outputs;
   - ESS summary;
   - EKF/UKF consistency bands;
   - runtime and numerical-failure labels.
4. Record method-specific anomalies, such as MLCOE EKF staying at zero, as
   diagnostics.
5. Avoid direct likelihood comparison across different likelihood families.

Primary criterion:

- nonlinear panel results can be interpreted by declared target semantics, not
  only by cross-student agreement.

Veto diagnostics:

- fixture likelihoods differ but the report lacks target labels;
- outputs are compared without latent-state or proxy reference metrics;
- one implementation silently changes model assumptions in adapter code.

### MP3: Kernel PFF Debug Gate

Execution plan:

`docs/plans/bayesfilter-student-dpf-baseline-mp3-kernel-pff-debug-gate-plan-2026-05-11.md`.

Primary hypothesis:

The kernel PFF timeout/failure is caused by convergence tolerance,
max-iteration configuration, or test scale rather than import failure or
missing dependencies.

Implementation details:

1. Create reduced-size local runner commands around advanced
   `MatrixKernelPFF` and `ScalarKernelPFF`.
2. Use explicit iteration and time caps.
3. Record:
   - status;
   - runtime;
   - average and maximum iterations;
   - tolerance or convergence diagnostics where available;
   - numerical outputs needed to reproduce the failure classification.
4. Do not patch vendored code.
5. Decide whether kernel PFF remains excluded from routine panels or can enter
   a slow/experimental-only panel.

Primary criterion:

- the failure mode is narrowed enough to decide whether future comparison work
  is justified.

Veto diagnostics:

- debug requires broad dependency changes;
- debug requires patching vendored code;
- a single test takes too long for bounded local execution;
- diagnostics do not improve over the current classification.

### MP4: Flow and DPF Readiness Review

Execution plan:

`docs/plans/bayesfilter-student-dpf-baseline-mp4-flow-dpf-readiness-review-plan-2026-05-11.md`.

Primary hypothesis:

Flow and DPF comparison is useful only after both snapshots have a minimal
particle/flow adapter surface and after kernel PFF status is known.

Implementation details:

1. Inventory flow and DPF entry points in both vendored snapshots.
2. Classify each entry point as:
   - runnable now;
   - blocked by missing dependency;
   - blocked by missing assumption;
   - blocked by long runtime;
   - not comparable to the other snapshot.
3. Select at most one flow/DPF path for a first bounded comparison.
4. Require a reproduction command before adapter work.

Primary criterion:

- a concrete flow or DPF path is selected with a clear target, fixture, metric,
  and blocker plan.

Veto diagnostics:

- comparison relies on student naming similarity rather than matching target
  semantics;
- no independent or proxy metric is available;
- execution requires notebook conversion, plotting, or large artifacts as a
  first step.

### MP5: Controlled Baseline Extraction

Primary hypothesis:

After adapter and reference gates, selected student-baseline evidence can inform
a separate controlled baseline without copying student implementation code.

Implementation details:

1. Identify behaviors worth reproducing independently in
   `experiments/controlled_dpf_baseline/`.
2. Write clean-room controlled experiments using BayesFilter-owned code or
   independent minimal implementations.
3. Cite student-baseline evidence as comparison evidence only.
4. Keep production `bayesfilter/` changes behind a separate implementation
   plan.

Primary criterion:

- a controlled baseline reproduces the intended experiment without importing
  student code.

Veto diagnostics:

- any student implementation code is copied into production;
- controlled baseline behavior depends on unexplained student assumptions;
- controlled code is used to justify production API changes without a separate
  BayesFilter implementation plan.

## Result Labels

Use these labels consistently in reports and JSON summaries:

| Label | Meaning |
| --- | --- |
| `reproduced_original` | Original or README-described behavior ran successfully. |
| `reproduced_targeted` | A targeted test/example ran successfully, but not a full original report. |
| `reference_consistent` | Output agrees with an independent or declared reference within tolerance. |
| `cross_consistent` | Student outputs agree with each other on matched target semantics. |
| `smoke_only` | Code path runs, but no correctness or reference claim is made. |
| `blocked_missing_dependency` | Missing dependency prevents the run. |
| `blocked_environment_drift` | Version mismatch or runtime incompatibility prevents the run. |
| `blocked_missing_assumption` | The run requires undocumented assumptions or unavailable data. |
| `failed_algorithmic` | Code runs but returns invalid or failed numerical behavior. |
| `algorithm_test_sensitivity_and_long_runtime` | Behavior is reproducibly slow or test-sensitive under bounded commands. |
| `comparison_only` | Evidence can guide experiments but cannot certify production correctness. |

## Required Report Fields

Every future report should include:

- source snapshot commit;
- command and working directory;
- relevant environment versions;
- fixture name;
- seed;
- particle count where applicable;
- dtype where applicable;
- runtime;
- pass/fail/blocker status;
- reference metric or proxy metric;
- interpretation;
- next-phase justification or stop recommendation.

Every future JSON output should preserve:

- implementation name;
- fixture name;
- status;
- failure reason if any;
- available metrics;
- missing metrics as null;
- diagnostics object;
- provenance fields.

## Audit and Tidy Rules

Before committing any student-baseline phase:

1. run an import-boundary search over `bayesfilter` and normal `tests`;
2. compile or syntax-check edited experiment modules;
3. run `git diff --check` on edited files;
4. inspect generated output sizes;
5. update the student reset memo with result, interpretation, and next-phase
   justification;
6. stage only student-baseline files and student-baseline plans.

Minimum import-boundary search:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

The expected result is no production or normal-test imports from the student
baseline lane.

## Current Next Move

The next justified execution plan is the EDH/PFPF adapter spike selected by
MP4.

Active adapter-spike plan:
`docs/plans/bayesfilter-student-dpf-baseline-edh-pfpf-adapter-spike-plan-2026-05-11.md`.

The reason is concrete:

- MP1 closed the largest linear particle-diagnostic asymmetry by adding MLCOE
  BPF diagnostics;
- MP2 closed the immediate nonlinear reference/proxy-metric gap;
- MP3 narrowed the kernel PFF failure/timeout gap and classified kernel PFF as
  excluded from routine panels pending further debug.
- MP4 inventoried 28 flow/DPF-related surfaces and selected only one bounded
  comparison candidate: advanced `EDHParticleFilter` versus MLCOE `PFPF_EDH`.

The current decision is to run a small adapter-owned EDH/PFPF spike on the
existing nonlinear Gaussian range-bearing fixture.  Kernel PFF, stochastic
flow, DPF, dPFPF, neural OT, differentiable resampling, and HMC remain out of
routine comparison unless later reproduction gates change their classifications.

## Stop Rules

Stop and ask for direction if:

- a phase requires edits outside the owned lane;
- a phase requires modifying vendored student code in place;
- a phase requires copying student code into production;
- a phase requires broad dependency changes not authorized by a phase plan;
- a phase produces ambiguous failures that cannot be converted into structured
  blocker evidence;
- generated artifacts are too large or unsuitable for normal repository
  history;
- monograph-writing and student-baseline files cannot be cleanly staged
  separately.
