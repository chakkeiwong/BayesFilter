# Phase 6 Subplan: Kernel Tuning Driver

Date: 2026-07-09

## Phase Objective

Bind the deterministic driver to BayesFilter staged HMC kernel tuning with
`HMCKernelTuningConfig.serious`, `use_xla=True`, and CPU-hidden sample
generation.

## Entry Conditions Inherited From Previous Phase

- Phase 5 geometry and mass handoff passed.
- Runtime approval has been granted for HMC tuning commands expected to exceed
  small smoke-test duration.

## Required Artifacts

- Result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- Kernel tuning JSON with final kernel payload, repair history, selected mass,
  selected step, selected leapfrog count, XLA metadata, and hashes.

## Required Checks / Tests / Reviews

- Driver calls existing BayesFilter tuning APIs, not hand-coded tuning choices.
- `use_xla=True` and `chain_execution_mode="tf_function"` are enforced.
- Runtime metadata confirms JIT/XLA; false/missing metadata is a veto.
- Focused tests cover config validation and non-XLA rejection.
- Claude review before any serious runtime launch.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter select a fixed HMC kernel through deterministic staged tuning? |
| Baseline/comparator | Phase 5 mass artifact and existing `tune_hmc_kernel` serious policy. |
| Primary pass criterion | Tuning returns a final kernel payload with no hard vetoes and confirmed XLA. |
| Veto diagnostics | Non-XLA fallback, tuning hard veto, missing final kernel, manual kernel override. |
| Explanatory diagnostics | Acceptance trace, repair rounds, timing, selected trajectory. |
| Not concluded | No posterior convergence or recovery claim. |

## Forbidden Claims / Actions

- Do not manually select step size/leapfrog based on diagnostics.
- Do not run `jit_compile=false`.
- Do not treat accepted tuning as final HMC success.

## Exact Next-Phase Handoff Conditions

- Phase 7 can use the frozen kernel payload as the only sampling kernel input.

## Stop Conditions

- Runtime approval is missing.
- Kernel tuning cannot confirm XLA/JIT execution.
- Existing tuner cannot produce a final kernel under configured caps.
