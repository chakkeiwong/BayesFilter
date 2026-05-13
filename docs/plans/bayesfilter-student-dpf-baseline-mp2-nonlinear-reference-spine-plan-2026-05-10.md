# Plan: MP2 nonlinear reference and proxy-metric spine

## Date

2026-05-10

## Status

Active next-phase plan for the quarantined student DPF experimental-baseline
stream.

This plan implements MP2 from:
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
- flow, DPF, HMC, kernel PFF, or neural OT claims.

## Goal

MP2 turns the prior nonlinear smoke result into a small interpretable nonlinear
comparison panel.  The target is not exact correctness certification.  The
target is a reference/proxy metric spine that can distinguish:

- runnable nonlinear filters;
- method-specific behavior;
- latent-state tracking error;
- EKF/UKF agreement or divergence;
- PF degeneracy across repeated seeds;
- incompatible likelihood semantics.

## Current Gaps

### Gap 1: Nonlinear smoke has no shared fixture

The prior nonlinear smoke used each student's own range-bearing setup.  That
proved feasibility, but it did not create comparable latent states or
observations.

Closure target:

- add an experiment-owned nonlinear Gaussian range-bearing fixture with fixed
  latent states, observations, process noise, observation noise, and seeds.

### Gap 2: Nonlinear metrics are not target-labeled

The advanced smoke used a Student-t range-bearing model; MLCOE used a Gaussian
style range-bearing model.  Direct likelihood comparison is invalid unless
target semantics are explicit.

Closure target:

- run a Gaussian range-bearing target for both implementations where feasible;
- label each method target as EKF, UKF, or BPF/bootstrap-PF;
- avoid cross-family likelihood comparison.

### Gap 3: MLCOE EKF zero-estimate behavior is unresolved

The smoke observed MLCOE EKF final estimate staying at zero while UKF moved.
This may be an initialization artifact, Jacobian singularity, model issue, or a
real method limitation.

Closure target:

- initialize MLCOE EKF/UKF away from the origin on the shared fixture;
- separately run an origin-initialized diagnostic;
- classify the zero-estimate behavior.

### Gap 4: PF stochasticity is not separated from systematic error

A single PF run is not enough for nonlinear interpretation.

Closure target:

- run repeated seeds for BPF/bootstrap-PF;
- summarize mean RMSE, dispersion, ESS, and runtime.

## Hypotheses

### N1: shared nonlinear fixture

A small Gaussian range-bearing fixture can be used by both student snapshots
without vendored-code edits.

Primary criterion:

- both implementations produce at least one non-blocked EKF or UKF path on the
  same latent states and observations.

### N2: MLCOE EKF zero behavior

The previous MLCOE EKF zero-estimate behavior is caused by origin
initialization and singular/near-singular range-bearing Jacobian geometry, not
by the entire EKF path being unusable.

Primary criterion:

- MLCOE EKF moves and produces finite RMSE under non-origin initialization;
- origin-initialized EKF is separately classified.

### N3: nonlinear PF degeneracy

Both student PF paths show lower ESS and larger latent-state RMSE on the
low-noise nonlinear fixture than on the moderate-noise fixture, with repeated
seeds exposing stochastic dispersion.

Primary criterion:

- repeated-seed PF summaries include finite RMSE and ESS for both
  implementations or structured blockers.

### N4: comparison-only reporting

The report can compare latent-state RMSE, EKF/UKF agreement, PF dispersion,
ESS, and runtime without treating student agreement as correctness.

Primary criterion:

- the report separates target labels, proxy metrics, and missing likelihood
  fields.

## Phase MP2.0: Preflight and Lane Guard

Actions:

1. Record current Git status.
2. Confirm student reset memo and MP2 plan are present.
3. Confirm no production import boundary:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

Exit gate:

- proceed only if edits can remain in the student-baseline lane.

## Phase MP2.1: Fixture and Method Design

Actions:

1. Add `experiments/student_dpf_baselines/fixtures/nonlinear_fixtures.py`.
2. Define two Gaussian range-bearing fixtures:
   - moderate noise;
   - low observation noise.
3. Store:
   - transition matrix;
   - process covariance;
   - observation covariance;
   - initial mean/covariance;
   - latent states;
   - observations.
4. Add shared observation and Jacobian helpers.

Exit gate:

- fixture arrays have consistent shapes and finite values.

## Phase MP2.2: Nonlinear Panel Runner

Actions:

1. Add `experiments/student_dpf_baselines/runners/run_nonlinear_reference_panel.py`.
2. For `advanced_particle_filter`, build a vendored `StateSpaceModel` from the
   shared fixture and run:
   - EKF;
   - UKF;
   - bootstrap-PF for repeated seeds.
3. For `2026MLCOE`, build an adapter-local TensorFlow model and run:
   - EKF;
   - UKF;
   - BPF for repeated seeds.
4. Record:
   - latent-state RMSE;
   - final-position RMSE;
   - EKF/UKF mean-trajectory RMSE;
   - PF average/min ESS;
   - PF repeated-seed dispersion;
   - runtime;
   - target labels and unavailable likelihood fields.

Exit gate:

- each implementation has at least one EKF/UKF result or a structured blocker;
- PF results are either finite or explicitly blocked.

## Phase MP2.3: Hypothesis Classification

Actions:

1. Summarize method and fixture results.
2. Classify N1-N4 as supported, partially supported, unsupported, or blocked.
3. Explicitly classify MLCOE origin-initialization behavior.
4. Avoid direct likelihood comparison across missing or different likelihood
   estimators.

Exit gate:

- result report gives interpretable method-specific proxy metrics.

## Phase MP2.4: Audit, Tidy, Reset Memo, Commit

Actions:

1. Run syntax checks for edited modules.
2. Run import-boundary search over production code and tests.
3. Run `git diff --check`.
4. Check generated artifact sizes.
5. Update the student reset memo with phase results and next justified phase.
6. Stage and commit only student-baseline files.

Exit gate:

- scoped commit excludes monograph-lane files and production code.

## Stop Rules

Stop and ask for direction if:

- shared nonlinear fixture cannot be represented in either implementation
  without vendored-code edits;
- both implementations fail all EKF/UKF paths;
- PF runs are too slow or numerically invalid across all seeds;
- the result requires direct likelihood comparison across incompatible or
  missing likelihood targets;
- any phase requires edits outside the student-baseline lane.
