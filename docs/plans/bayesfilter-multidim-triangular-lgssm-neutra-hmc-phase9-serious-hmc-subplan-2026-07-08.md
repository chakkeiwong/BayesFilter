# Phase 9 Subplan: Serious CPU HMC Estimation

Date: 2026-07-08

## Phase Objective

Run serious CPU-hidden multicore HMC estimation and report per-parameter
diagnostics for the multidimensional triangular LGSSM target.

## Entry Conditions Inherited From Previous Phase

- Phase 8 pilot passed.
- Serious HMC settings and approval are in place.

## Required Artifacts

- Serious HMC JSON/logs/sample summaries/private shards as needed.
- Phase 9 result.

## Required Checks/Tests/Reviews

- Per-parameter split R-hat, bulk ESS, tail ESS.
- Truth z-scores and credible intervals.
- Reference posterior residuals.
- Stationarity checks for posterior draws.
- CPU-hidden and `jit_compile=True` policy checks.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does serious NeuTra-HMC estimate the multidimensional LGSSM parameters under predeclared diagnostics? |
| Baseline/comparator | Phase 5 reference comparator, synthetic truth, Phase 8 pilot settings. |
| Primary criterion | Clean hard veto screen, max R-hat threshold, ESS threshold, finite/reference/truth diagnostics within predeclared limits. |
| Veto diagnostics | Nonstationary draws, divergences/errors above threshold, R-hat/ESS failure, reference mismatch, policy violation. |
| Explanatory diagnostics | Acceptance, runtime, posterior correlations, MCSE. |
| Not concluded | General LGSSM or product readiness. |
| Artifact | Serious HMC result JSON/MD. |

## Forbidden Claims/Actions

- Do not rank methods unless uncertainty analysis supports it.
- Do not change thresholds after seeing results.

## Exact Next-Phase Handoff Conditions

Phase 10 may begin only after Phase 9 writes a pass/block/insufficient-evidence
result.

## Stop Conditions

Stop for failed hard vetoes or missing diagnostics.
