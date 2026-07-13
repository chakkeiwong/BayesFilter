# Phase 4 Subplan: Target Score And XLA Compile

Date: 2026-07-08

## Phase Objective

Build the exact Kalman likelihood/value-score target adapter for the Phase 1/2
synthetic multidimensional lower-triangular LGSSM and pass a `jit_compile=True`
finite compile gate.

## Entry Conditions Inherited From Previous Phase

- Phase 3 passed `PASS_PHASE3_STATIONARY_MATERIALIZATION`.
- Phase 2 data artifact is valid.
- Phase 1 contract JSON and Phase 2 data/manifest JSON are available.

## Required Artifacts

- Target adapter code/tests, limited to this testing-lane target unless a
  separate reviewed plan promotes public API.
- Compile diagnostic JSON under
  `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/`.
- Phase 4 result:
  `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase4-target-score-compile-result-2026-07-08.md`.

## Required Checks/Tests/Reviews

- Value/score finite checks on the Phase 2 observations.
- Analytical/manual first-order score vs finite-difference diagnostics. If a
  full first-order score path is not implemented, write a blocker and do not
  proceed to NeuTra/HMC phases.
- `jit_compile=True` compile timing/size proxy; no `jit_compile=false` run.
- Source scan for forbidden runtime autodiff.
- CPU-hidden focused pytest and `git diff --check`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the target adapter compute the declared lower-triangular LGSSM log posterior and score in an XLA-compatible path? |
| Baseline/comparator | Phase 1-3 contract/artifacts, finite-difference tests, exact Kalman likelihood. |
| Primary criterion | Finite value/score and compile diagnostic with `jit_compile=True`; score residuals within tolerance. |
| Veto diagnostics | Runtime `GradientTape`, `jacobian`, `batch_jacobian`, `jit_compile=false`, score mismatch, nonfinite values, target/signature mismatch. |
| Explanatory diagnostics | Compile time, graph/HLO size if available, score residuals. |
| Not concluded | HMC sampling validity, NeuTra usefulness, posterior convergence, or product/default readiness. |
| Artifact | Code/tests/compile JSON/result. |

## Forbidden Claims/Actions

- Do not train NeuTra.
- Do not run full HMC.
- Do not run any HMC sampling or tuning.
- Do not run GPU/CUDA commands.
- Do not use non-JIT fallback as evidence.
- Do not use finite differences in the admitted runtime score path.
- Do not claim posterior correctness from finite score checks or compile
  success.

## Exact Next-Phase Handoff Conditions

Phase 5 may begin only if the target adapter compile gate passes and target
signatures are stable.

## Stop Conditions

Stop if score correctness or XLA compile fails without a reviewed repair.
