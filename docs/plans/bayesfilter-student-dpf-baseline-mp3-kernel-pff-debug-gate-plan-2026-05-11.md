# Plan: MP3 kernel PFF debug gate

## Date

2026-05-11

## Status

Active next-phase plan for the quarantined student DPF experimental-baseline
stream.

This plan implements MP3 from:
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
- making kernel PFF routine-panel evidence without bounded diagnostics.

## Goal

MP3 narrows the already-classified `advanced_particle_filter` kernel PFF
failure/timeout.  The purpose is not to repair the student implementation.
The purpose is to determine whether the failure is mainly due to:

- test scale;
- convergence tolerance;
- `max_iterations` policy;
- kernel type;
- runtime cost inherent in the current implementation.

## Current Evidence

Prior focused reproduction classified kernel PFF as
`algorithm_test_sensitivity_and_long_runtime`:

- `test_kernel_pff_lgssm`: timed out after 90 seconds;
- `test_kernel_pff_convergence`: failed because average iterations reached
  `100.0`, the maximum;
- `test_scalar_vs_matrix_kernel`: passed but took about 85.92 seconds.

## Gaps to Close

### Gap 1: Scale sensitivity is not quantified

The prior test-level evidence does not show whether smaller fixtures converge
quickly or also hit iteration limits.

Closure target:

- run reduced 1D and 2D linear-Gaussian kernel PFF experiments with explicit
  particle, horizon, tolerance, and iteration caps.

### Gap 2: Tolerance and max-iteration effects are unclear

The failing test used `tolerance=1e-6` and `max_iterations=100`.  It is unclear
whether looser tolerances produce useful bounded diagnostics.

Closure target:

- compare strict and loose tolerances under the same small fixture.

### Gap 3: Scalar and matrix kernels have different costs

The prior scalar-vs-matrix test passed but was slow.  Runtime and convergence
differences need smaller controlled diagnostics.

Closure target:

- run both scalar and matrix kernels on reduced fixtures and summarize
  iteration counts, final flow magnitudes, RMSE, and runtime.

## Hypotheses

### K1: reduced fixtures are runnable

Kernel PFF can run on reduced 1D and small 2D linear-Gaussian fixtures within a
bounded local command.

Primary criterion:

- at least one scalar and one matrix kernel run completes with diagnostics.

### K2: convergence is tolerance-sensitive

Loose tolerance should reduce iteration counts and runtime relative to strict
tolerance on the same fixture.

Primary criterion:

- median iterations or runtime are lower under loose tolerance.

### K3: max-iteration hits are the main failure mode

When runs fail to converge, the failure mode should appear as `hit_max_iter`
rather than missing dependency or import failure.

Primary criterion:

- non-converged runs have finite diagnostics and `hit_max_iter=true`.

### K4: routine-panel readiness remains conditional

Kernel PFF should remain excluded from routine panels unless reduced cases show
consistent bounded convergence.

Primary criterion:

- the result report explicitly decides `routine_ready`,
  `slow_experimental_only`, or `excluded_pending_debug`.

## Phase MP3.0: Preflight and Lane Guard

Actions:

1. Record current Git status.
2. Confirm student reset memo and MP3 plan are present.
3. Confirm no production import boundary:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

Exit gate:

- proceed only if edits can remain in the student-baseline lane.

## Phase MP3.1: Reduced Diagnostic Runner

Actions:

1. Add `experiments/student_dpf_baselines/runners/run_kernel_pff_debug_gate.py`.
2. Build small linear-Gaussian fixtures locally inside the runner:
   - 1D autoregressive fixture;
   - 2D constant-velocity fixture.
3. Run bounded combinations:
   - kernel type: scalar and matrix;
   - tolerance: loose and strict;
   - particle counts: small and medium;
   - short horizons.
4. Record per-run:
   - status;
   - runtime;
   - kernel type;
   - particle count;
   - tolerance;
   - max iterations;
   - average/max iterations;
   - hit-max fraction;
   - final flow magnitude summary;
   - RMSE against latent states and Kalman.

Exit gate:

- at least one scalar and one matrix run completes or all failures are
  structured.

## Phase MP3.2: Classification and Report

Actions:

1. Summarize tolerance, scale, and kernel-type effects.
2. Classify K1-K4.
3. Decide whether kernel PFF is:
   - `routine_ready`;
   - `slow_experimental_only`;
   - `excluded_pending_debug`.
4. Write JSON outputs and a report under
   `experiments/student_dpf_baselines/reports/`.

Exit gate:

- classification narrows the prior failure beyond
  `algorithm_test_sensitivity_and_long_runtime`.

## Phase MP3.3: Audit, Tidy, Reset Memo, Commit

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

- diagnostics require vendored-code edits;
- diagnostics require production-code edits;
- reduced runs cannot complete within bounded local execution;
- failures do not improve classification beyond the prior test-level report;
- generated artifacts are too large for normal repository history.
