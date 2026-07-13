# Phase 2 Subplan: Synthetic Data Fixture

Date: 2026-07-08

## Phase Objective

Generate and validate the fixed-truth 4D lower-triangular LGSSM synthetic
dataset specified by the Phase 1 contract, including stationary initial state,
hashes, raw truth, and moment diagnostics.

## Entry Conditions Inherited From Previous Phase

- Phase 1 passed `PASS_PHASE1_MODEL_CONTRACT_LOWER_TRIANGULAR_V1`.
- The machine-readable contract exists at
  `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_contract_v1.json`.

## Required Artifacts

- Synthetic data JSON and manifest under:
  `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/`.
- Required generated filenames:
  - `lower_triangular_lgssm_synthetic_data_v1_seed20260708.json`;
  - `lower_triangular_lgssm_synthetic_data_v1_manifest_seed20260708.json`.
- Phase 2 result:
  `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase2-synthetic-data-result-2026-07-08.md`.

## Required Checks/Tests/Reviews

- Verify dimension `4`, horizon `256`, parameter order, raw truth, and
  transforms match Phase 1.
- Verify `H = I_4` and diagonal positive `Q/R` match Phase 1.
- Verify eigenvalues/stationarity margin for the lower-triangular `A`.
- Verify `P_inf = A P_inf A' + Q` residual.
- Verify deterministic seed `20260708`, truth, dimensions, hashes, and
  observation statistics.
- Run moment/autocovariance sanity checks.
- `python -m json.tool` on generated JSON.
- `git diff --check` on Phase 2 docs/code if any.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the synthetic fixture instantiate the Phase 1 target with recoverability diagnostics and no hidden nonstationarity? |
| Baseline/comparator | Phase 1 contract JSON/result and stationary Lyapunov residual. |
| Primary criterion | Valid data artifact with fixed truth, stationary initial law, hashes, and moment sanity checks. |
| Veto diagnostics | Contract mismatch, nonstationary `A`, invalid covariance, missing seed/truth/hash, weak or degenerate signal, malformed JSON. |
| Explanatory diagnostics | Empirical lag covariances, moment estimates, signal/noise ratios. |
| Not concluded | HMC readiness or posterior correctness. |
| Artifact | Data JSON, manifest, result. |

## Forbidden Claims/Actions

- Do not train NeuTra.
- Do not run HMC.
- Do not run posterior/reference sampling.
- Do not use moment checks as posterior evidence.
- Do not change Phase 1 dimensions, parameter order, transforms, or prior
  family without a visible contract amendment.

## Exact Next-Phase Handoff Conditions

Phase 3 may begin only if the data artifact and manifest are valid, hashes are
recorded, the stationary residual/moment diagnostics do not veto, and the Phase
2 result records all nonclaims.

## Stop Conditions

Stop if the fixture is nonstationary, degenerate, or not recoverable enough for
a serious estimation test.
