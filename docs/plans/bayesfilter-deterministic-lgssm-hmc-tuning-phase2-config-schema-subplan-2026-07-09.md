# Phase 2 Subplan: Deterministic Config Schema

Date: 2026-07-09

## Phase Objective

Define a versioned JSON config schema that fully determines the LGSSM fixture,
geometry initializer, HMC kernel tuning, burn-in extension, retained-sampling
extension, pass/fail criteria, seeds, and artifact paths.

## Entry Conditions Inherited From Previous Phase

- Phase 1 inventory identifies the BayesFilter APIs to use.
- No unresolved tool-boundary blocker remains.

## Required Artifacts

- Result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase2-config-schema-result-2026-07-09.md`
- Config schema note.
- Initial config artifact:
  `docs/benchmarks/configs/multidim_lgssm_serious_hmc_tuning_2026_07_09.json`

## Required Checks / Tests / Reviews

- JSON parses.
- Schema has fixed seeds and deterministic artifact paths.
- Config rejects `use_xla=false`, missing CPU-hidden sample policy, and missing
  burn-in/sampling caps.
- Claude review of schema if materially changed after local check.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the entire tuning/recovery run be determined by config plus code? |
| Baseline/comparator | User requirement that tuning must be explicit Python with deterministic outcome. |
| Primary pass criterion | Config contains no manual post-result decision hook and all tuning knobs are predeclared. |
| Veto diagnostics | Missing seeds, missing caps, non-XLA fallback, unclear pass/fail thresholds, missing artifact hashes. |
| Explanatory diagnostics | Config hash and schema field table. |
| Not concluded | No target correctness or HMC success claim. |

## Forbidden Claims / Actions

- Do not run the config.
- Do not change final pass criteria after result observation.
- Do not encode free-form agent discretion into the schema.

## Exact Next-Phase Handoff Conditions

- Phase 3 can consume the config to generate the LGSSM fixture deterministically.

## Stop Conditions

- Config cannot encode required burn-in/sampling extension rules.
- The target model parameterization is not yet fixed enough for deterministic fixture generation.
