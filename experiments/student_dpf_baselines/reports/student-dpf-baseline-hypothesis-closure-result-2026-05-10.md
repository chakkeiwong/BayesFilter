# Student DPF baseline hypothesis-closure result

## Date

2026-05-10

## Scope

This report summarizes the follow-on hypothesis-closure cycle for the
quarantined student DPF experimental-baseline stream.  It remains
comparison-only evidence.  No student code is promoted to production and no
BayesFilter correctness claim is made from student behavior.

## Inputs

Prior completed cycle:
`experiments/student_dpf_baselines/reports/student-dpf-baseline-gap-closure-result-2026-05-10.md`.

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-hypothesis-closure-plan-2026-05-10.md`.

Audit:
`docs/plans/bayesfilter-student-dpf-baseline-hypothesis-closure-plan-audit-2026-05-10.md`.

## Hypothesis Results

### H1: Linear stress

Status: supported for available diagnostics.

Evidence:

- `2026MLCOE`: 45/45 Kalman smoke runs ok; maximum Kalman log-likelihood error
  approximately `2.84e-14`.
- `advanced_particle_filter`: 45/45 Kalman smoke runs ok; maximum Kalman
  log-likelihood error approximately `7.11e-15`.
- Advanced bootstrap-PF diagnostics degraded as particle count decreased and
  observation noise tightened.
- On `lgssm_cv_2d_low_noise`, median PF log-likelihood error was about:
  - `24.15` at 64 particles;
  - `9.07` at 128 particles;
  - `0.767` at 512 particles.
- On the same low-noise fixture, min average ESS was about:
  - `10.17` at 64 particles;
  - `17.02` at 128 particles;
  - `80.04` at 512 particles.

Interpretation:

The Kalman paths remain reference-consistent under larger linear fixtures.
The advanced bootstrap-PF path shows the expected particle-degeneracy pressure
under lower observation noise and fewer particles.  MLCOE particle diagnostics
remain unavailable because the current adapter only wraps the Kalman path.

### H2: Kernel PFF isolation

Status: supported as classified failure/timeout.

Evidence:

- `test_kernel_pff_lgssm`: timed out after 90 seconds.
- `test_kernel_pff_convergence`: failed because average iterations were
  `100.0`, hitting the maximum iteration count.
- `test_scalar_vs_matrix_kernel`: passed but took 85.92 seconds.

Interpretation:

The kernel PFF issue is localized and reproducible.  It is best classified as
`algorithm_test_sensitivity_and_long_runtime`, not as a missing dependency.
Kernel PFF should remain excluded from routine cross-student panels until a
separate debug/reproduction plan narrows the algorithmic or test-design cause.

### H3: Nonlinear smoke

Status: supported as feasibility smoke only.

Evidence:

- `advanced_particle_filter` range-bearing Student-t smoke ran in about
  `0.565` seconds; EKF, UKF, and bootstrap-PF paths executed.
- `2026MLCOE` range-bearing smoke ran in about `2.895` seconds; EKF and UKF
  step paths executed.
- Advanced nonlinear PF average ESS was about `9.44` for 128 particles.
- MLCOE EKF final estimate stayed at zero while UKF moved, indicating
  method-specific nonlinear behavior that must be checked before comparison.

Interpretation:

Nonlinear adapter work is feasible, but current evidence is not a correctness
or reference-consistency result.  The next nonlinear phase must add independent
references or explicit proxy metrics and must treat method-specific anomalies
as first-class blockers.

## Remaining Gaps

1. MLCOE particle/flow diagnostics are not yet exposed through adapters.
2. Kernel PFF is not routine-panel ready.
3. Nonlinear fixtures lack independent references and method-specific
   tolerances.
4. HMC/DPF/neural OT code remains completely outside the validated baseline
   harness.

## Recommended Next Phases

### Next Phase A: MLCOE particle adapter gate

Hypothesis:
MLCOE BPF/GSMC/UPF paths can be wrapped without vendored-code edits and will
provide ESS/runtime diagnostics comparable to the advanced bootstrap-PF
diagnostics on linear stress fixtures.

Test:
Add one MLCOE particle adapter at a time.  Start with BPF on linear-Gaussian
fixtures.  Require structured failure records and no vendored-code patches.

### Next Phase B: nonlinear reference spine

Hypothesis:
The nonlinear range-bearing smoke paths can be compared against explicit proxy
references such as EKF/UKF agreement bands, repeated-seed PF summaries, and
trajectory RMSE against simulated latent states.

Test:
Create a small nonlinear panel with fixed seeds, latent states, observations,
and method-specific metrics.  Do not compare advanced Student-t likelihoods
directly to MLCOE Gaussian-style range-bearing likelihoods without explicit
target labels.

### Next Phase C: kernel PFF debug gate

Hypothesis:
The kernel PFF timeout/failure is caused by convergence tolerance, max-iteration
configuration, or test scale rather than import/dependency failure.

Test:
Run reduced-size local adapter experiments around `MatrixKernelPFF` and
`ScalarKernelPFF` with explicit iteration/time caps.  Record convergence
diagnostics but do not patch vendored code.

## Decision

The student-baseline harness is now useful for controlled linear Kalman and
advanced bootstrap-PF diagnostics.  It is not yet ready for full nonlinear,
flow, kernel PFF, HMC, DPF, or neural OT comparison.
