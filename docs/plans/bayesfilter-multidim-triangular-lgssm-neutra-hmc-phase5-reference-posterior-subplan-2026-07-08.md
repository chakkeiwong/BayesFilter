# Phase 5 Subplan: Reference Posterior

Date: 2026-07-08

## Phase Objective

Create a serious reference posterior comparator or write a blocker explaining
why a stronger reference route is needed before NeuTra-HMC promotion.

## Entry Conditions Inherited From Previous Phase

- Phase 4 target/score compile gate passed.

## Required Artifacts

- Reference posterior JSON/manifest or blocker.
- Phase 5 result:
  `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase5-reference-posterior-result-2026-07-08.md`.

## Required Checks/Tests/Reviews

- MAP/Hessian or baseline sampler diagnostics with predeclared limitations.
- Reference artifact hashes and nonclaims.
- Review of whether reference is strong enough for Phase 9/10.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | What comparator can support serious HMC estimation validation for this target? |
| Baseline/comparator | Exact target adapter plus synthetic truth; optional baseline HMC/Laplace/reference route. |
| Primary criterion | A reference route with explicit limitations, or a blocker before training/sampling promotion. |
| Veto diagnostics | Treating Laplace or short chains as exact truth, missing uncertainty limits, malformed artifact. |
| Explanatory diagnostics | MAP, Hessian, reference means/covs, truth z-scores. |
| Not concluded | HMC superiority or scientific validity. |
| Artifact | Reference JSON/result/blocker. |

## Forbidden Claims/Actions

- Do not change pass criteria after seeing reference diagnostics.
- Do not run long HMC without explicit approval.

## Exact Next-Phase Handoff Conditions

Phase 6 may begin only if a reviewed reference/comparator standard is adequate
for interpreting NeuTra-HMC evidence.

## Stop Conditions

Stop if no defensible comparator exists.
