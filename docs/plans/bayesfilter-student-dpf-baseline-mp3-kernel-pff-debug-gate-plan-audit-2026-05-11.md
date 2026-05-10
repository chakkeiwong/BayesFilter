# Audit: MP3 kernel PFF debug gate plan

## Date

2026-05-11

## Scope

Independent developer-style audit of:

`docs/plans/bayesfilter-student-dpf-baseline-mp3-kernel-pff-debug-gate-plan-2026-05-11.md`.

This audit is limited to the quarantined student DPF experimental-baseline lane.

## Verdict

The plan is executable and correctly scoped.  It does not attempt to fix or
promote the vendored kernel PFF implementation.  It narrows the prior
`algorithm_test_sensitivity_and_long_runtime` classification by using reduced
fixtures, explicit tolerance settings, and diagnostics already exposed by the
student code.

Proceed with MP3.

## Required Tightenings

1. The runner should use reduced horizons and particle counts first.  Do not
   rerun the slow vendored tests as the primary diagnostic.
2. Diagnostics must distinguish "completed filter run" from "converged flow
   iterations."  A filter can complete while every time step hits
   `max_iterations`.
3. `hit_max_iter` must be derived from `diag.n_iterations >= max_iterations`
   and reported per run.
4. RMSE against Kalman is useful, but convergence diagnostics are the primary
   MP3 evidence.
5. Kernel PFF must remain excluded from routine student comparison panels unless
   every reduced diagnostic run is both bounded and consistently converged.
6. The report should not claim kernel PFF correctness from a reduced successful
   run.

## Audit Decision

Proceed to MP3.0 preflight and MP3.1 runner implementation.  Continue
automatically if reduced scalar and matrix kernel runs complete or fail with
structured diagnostics.
