# Phase 1 Subplan: Model Contract

Date: 2026-07-08

## Phase Objective

Write the formal model/parameter contract for the first multidimensional
lower-triangular LGSSM synthetic estimation target selected by Phase 0.

## Entry Conditions Inherited From Previous Phase

- Phase 0 selected `lower_triangular_first`.
- Source and local-code support gaps are recorded.

## Required Artifacts

- Phase 1 result/model contract:
  `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase1-model-contract-result-2026-07-08.md`.
- Optional machine-readable contract JSON under:
  `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/`.

## Required Checks/Tests/Reviews

- Check parameter names, dimensions, transforms, stationarity law, initial
  state law, nonclaims, and signatures.
- Review for similarity-transform ambiguity and hidden nonstationary paths.
- Check that the contract includes `H = I`, fixed coordinate order, diagonal
  positive `Q/R`, lower-triangular `A`, and stationary `P_inf`.
- `git diff --check` on Phase 1 docs.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the proposed model contract make stationarity and coordinate identification explicit enough for implementation? |
| Baseline/comparator | Phase 0 source/design result. |
| Primary criterion | Contract defines a lower-triangular `A`, `H=I`, diagonal `Q/R`, stationary `P_inf`, parameter names, bounds, seeds, and nonclaims. |
| Veto diagnostics | Missing `P_inf`, free `H`, dense unconstrained latent similarity, unordered/ambiguous coordinates, or unsupported identifiability claim. |
| Explanatory diagnostics | Parameter count, lower-triangular pattern, stationarity margins, planned truth values. |
| Not concluded | That data are recoverable or HMC will pass. |
| Artifact | Phase 1 result/contract. |

## Forbidden Claims/Actions

- Do not generate data yet.
- Do not implement or edit algorithmic code in this phase.
- Do not run runtime/model execution commands.
- Do not run HMC/training.
- Do not claim full global identifiability.

## Exact Next-Phase Handoff Conditions

Phase 2 may begin only if the contract specifies the exact synthetic truth
template, dimensions, parameter order, raw-coordinate transforms, prior family,
and stationary initialization formula.

## Stop Conditions

Stop if the model contract leaves latent-coordinate equivalence or stationarity
ambiguous.
