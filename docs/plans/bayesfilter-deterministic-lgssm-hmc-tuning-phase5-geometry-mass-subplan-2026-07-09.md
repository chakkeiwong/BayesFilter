# Phase 5 Subplan: Geometry And Mass Driver

Date: 2026-07-09

## Phase Objective

Implement deterministic geometry and mass initialization in the Python driver
using BayesFilter's quadratic initializer and mass-matrix helpers.

## Entry Conditions Inherited From Previous Phase

- Phase 4 XLA value/score gate passes.
- Config fixes geometry settings and mass regularization caps.

## Required Artifacts

- Result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase5-geometry-mass-result-2026-07-09.md`
- Geometry JSON with arrays, diagnostics, and hash.
- Mass handoff JSON with covariance, factor, regularization report, and hash.

## Required Checks / Tests / Reviews

- `fit_low_rank_spd_quadratic_geometry` result accepted.
- `covariance_from_precision` returns finite SPD covariance.
- Reconstruction error within declared tolerance.
- Determinism check for geometry and mass artifact hashes.
- Focused pytest for driver geometry/mass path.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can existing BayesFilter tools deterministically produce an HMC mass initializer? |
| Baseline/comparator | Phase 4 compiled value/score and Phase 2 geometry config. |
| Primary pass criterion | Accepted geometry, SPD mass covariance, condition cap respected, stable hashes. |
| Veto diagnostics | Geometry rejected, nonfinite samples, non-SPD mass, condition cap exceeded, reconstruction failure. |
| Explanatory diagnostics | Center score norm, eigen summaries, regularization report. |
| Not concluded | Not a certified MAP covariance, HMC convergence, or posterior recovery claim. |

## Forbidden Claims / Actions

- Do not manually edit the mass matrix after seeing diagnostics.
- Do not promote geometry fit as posterior correctness.
- Do not run serious HMC.

## Exact Next-Phase Handoff Conditions

- Phase 6 can pass the mass artifact and adapter to staged kernel tuning.

## Stop Conditions

- Geometry/mass artifacts are unstable or rejected.
- The required mass conversion needs manual intervention.
