# Phase 3 Subplan: Stationary/Lyapunov Implementation

Date: 2026-07-08

## Phase Objective

Implement or adapt TensorFlow/XLA-compatible discrete lower-triangular LGSSM
construction and stationary covariance helpers required by the Phase 1/2
target.

## Entry Conditions Inherited From Previous Phase

- Phase 1 passed `PASS_PHASE1_MODEL_CONTRACT_LOWER_TRIANGULAR_V1`.
- Phase 2 passed `PASS_PHASE2_SYNTHETIC_FIXTURE_VALID`.
- The contract JSON and synthetic fixture exist under
  `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/`.

## Required Artifacts

- Source/test changes limited to lower-triangular LGSSM construction,
  stationary covariance solve, and focused tests.
- Phase 3 result:
  `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase3-stationary-implementation-result-2026-07-08.md`.

## Required Checks/Tests/Reviews

- Unit tests for Phase 1 raw-to-constrained transform, lower-triangular `A`,
  `H=I`, diagonal positive `Q/R`, and discrete Lyapunov residuals.
- Unit tests that load the Phase 2 fixture and reproduce `P_inf`/diagnostics.
- If derivatives are implemented in this phase, include finite-difference
  derivative tests; otherwise explicitly defer derivatives to Phase 4 and do
  not claim target score readiness.
- Compare against local continuous stationary utilities where relevant.
- Source scan forbidding runtime `GradientTape`, `jacobian`, `batch_jacobian`.
- CPU-hidden `py_compile`, focused pytest, `git diff --check`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter materialize the Phase 1/2 lower-triangular LGSSM model and stationary covariance without hidden nonstationarity or runtime autodiff? |
| Baseline/comparator | Phase 1 contract, Phase 2 fixture, Lyapunov equation residual, and existing local stationary utilities. |
| Primary criterion | Transform/shape/stationary residual tests pass under CPU-hidden local checks. |
| Veto diagnostics | Runtime autodiff, non-XLA-compatible operations in admitted route, nonfinite covariance, contract mismatch, or derivative mismatch if derivatives are implemented. |
| Explanatory diagnostics | Residual magnitudes and finite-difference tolerances. |
| Not concluded | Full target score correctness, XLA compile readiness, NeuTra usefulness, or HMC readiness. |
| Artifact | Source/tests/result. |

## Forbidden Claims/Actions

- Do not run training or HMC.
- Do not run GPU/CUDA commands.
- Do not run posterior/reference sampling.
- Do not claim score correctness unless a score path and derivative tests are
  actually implemented and checked.
- Do not use test-only finite differences in admitted runtime.

## Exact Next-Phase Handoff Conditions

Phase 4 may begin only if lower-triangular construction and stationary
covariance tests pass, Phase 2 fixture reproduction passes, and no runtime
autodiff path is admitted.

## Stop Conditions

Stop for unresolved derivative mismatch or XLA-incompatible stationary helpers.
