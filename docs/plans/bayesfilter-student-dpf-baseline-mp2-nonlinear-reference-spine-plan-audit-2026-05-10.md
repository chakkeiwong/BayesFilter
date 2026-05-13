# Audit: MP2 nonlinear reference and proxy-metric spine plan

## Date

2026-05-10

## Scope

Independent developer-style audit of:

`docs/plans/bayesfilter-student-dpf-baseline-mp2-nonlinear-reference-spine-plan-2026-05-10.md`.

This audit is limited to the quarantined student DPF experimental-baseline lane.

## Verdict

The plan is executable and correctly follows from MP1.  It addresses the next
largest gap: nonlinear smoke is currently runnable but not interpretable as
reference-backed or proxy-metric evidence.

Proceed with MP2 under the listed stop rules.

## Required Tightenings

1. The shared fixture must use one declared likelihood family.  Use Gaussian
   range-bearing noise for MP2 so both student paths can be interpreted against
   the same target semantics.
2. Advanced `range_bearing.py` defaults to Student-t observation noise.  MP2
   should not reuse that factory for a Gaussian target unless the target label
   explicitly says Student-t.  Prefer an adapter-local advanced
   `StateSpaceModel` built from the shared fixture.
3. Angle residuals must be wrapped for bearing metrics.  State RMSE can be
   direct Euclidean RMSE, but observation residual diagnostics must handle
   angle periodicity.
4. MLCOE origin-initialized EKF should be a diagnostic run, not the main
   comparison target.  Main EKF/UKF runs should initialize from the fixture's
   non-origin initial mean.
5. PF likelihood comparisons should remain implementation-specific unless both
   paths use the same Gaussian observation target and expose comparable
   estimates.  Latent-state RMSE and ESS are the primary MP2 metrics.
6. If MLCOE BPF uses pre-resampling ESS and threshold-inferred resampling
   counts, reports must carry the same semantics established in MP1.
7. Generated reports should state that nonlinear proxy metrics do not certify
   production correctness.

## Audit Decision

Proceed to MP2.0 preflight and MP2.1 fixture design.  Continue automatically if
both implementations can run at least one EKF or UKF path on the shared fixture
without vendored-code edits.
