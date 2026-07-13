# Phase 3 Subplan: LGSSM Fixture Driver

Date: 2026-07-09

## Phase Objective

Implement or bind a deterministic fixture driver that creates the identifiable
multidimensional stationary LGSSM data set with `T=120` using prior means as
truth.

## Entry Conditions Inherited From Previous Phase

- Phase 2 config schema passes local/review gates.
- Config names fixed truth, priors, seeds, dimensions, and output paths.

## Required Artifacts

- Result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase3-lgssm-fixture-result-2026-07-09.md`
- Driver or driver section in:
  `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`
- Fixture JSON/NPZ artifact and hash under `docs/benchmarks/artifacts/`.

## Required Checks / Tests / Reviews

- Determinism check: same config and seeds produce identical fixture hash.
- Stationarity check for lower/block-lower-triangular transition.
- Initial state distribution recorded and finite.
- No HMC run.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the LGSSM fixture be generated deterministically from prior means? |
| Baseline/comparator | Phase 2 config and established stationary LGSSM implementation. |
| Primary pass criterion | Fixture hash stable across repeat generation and stationarity/shape checks pass. |
| Veto diagnostics | Nonstationary transition, missing initial distribution, nonfinite data, non-deterministic hash. |
| Explanatory diagnostics | Eigenvalues, stationary covariance summary, observation dimensions. |
| Not concluded | No posterior recovery, HMC tuning, or scientific adequacy claim. |

## Forbidden Claims / Actions

- Do not tune or sample posterior.
- Do not alter priors after observing generated data.
- Do not use DSGE target mechanics.

## Exact Next-Phase Handoff Conditions

- Phase 4 can compile/evaluate value and score on the frozen fixture.

## Stop Conditions

- Identifiability/stationarity contract is violated.
- Fixture cannot be reproduced bitwise or hash-stably for fixed config.
