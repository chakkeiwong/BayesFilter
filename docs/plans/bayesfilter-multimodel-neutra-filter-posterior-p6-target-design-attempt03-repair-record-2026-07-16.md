# P6 SIR Target-Design Attempt 03 Repair Record

Date: 2026-07-16

Attempt root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-common/target-design/cpu-attempt-03`

Classification: `PLAN_AND_HARNESS_DIAGNOSTIC_MODE_MISMATCH`.

The attempt completed and is preserved, but its common decision is not the
final target-design decision. It exposed two prospective-contract defects:

1. The prior-predictive gate required all susceptible states to remain
   nonnegative even though the declared likelihood transition is unprojected
   additive Gaussian and therefore has support on all of `R^18`. All 4,096
   trajectories were finite and bounded; 80.3% remained nonnegative at every
   susceptible coordinate. Negativity is model-support telemetry, not evidence
   that the implementation failed to compute its declared target.
2. The runner compared eager analytic UKF scores with finite differences of
   separately XLA-compiled values. CPU-XLA value parity noise of about `1e-5`
   was amplified by division by `2h`, causing a false absolute FD gap of about
   `0.30` at `theta=(0,-1,0)`. A focused same-mode diagnostic found agreement
   between the manual score and raw TensorFlow autodiff within `1.5e-10`; an
   eager FD step ladder from `1e-3` through `5e-6` converged to the manual score
   within about `1e-6`.

The repair keeps each six-row stencil batched and evaluates analytic score and
FD values in the same eager mode. CPU/GPU XLA remains a separate deterministic
status and scale-normalized parity gate. It also gates prior-predictive validity
on finiteness and bounded magnitude while preserving negative-state telemetry.

No target, dataset, time order, prior, chart, filter, parameter, design point,
FD step, score tolerance, hardware class, or campaign budget changes. Attempt
03 cannot itself issue admission; a fresh complete attempt must pass the
repaired contract.
