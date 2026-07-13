# Phase 6R Subplan: Adapter Contract Repair For XLA HMC Kernel Tuning

Date: 2026-07-09

## Phase Objective

Repair the deterministic LGSSM posterior adapter so BayesFilter's existing
XLA-only HMC kernel tuner can call it through the compiled TFP full-chain route.

## Entry Conditions Inherited From Previous Phase

- Phase 5 geometry and mass artifacts passed.
- Phase 6 kernel tuning was approved and attempted with `use_xla=True`.
- Phase 6 did not produce a frozen kernel because the bootstrap HMC screen
  returned a hard veto before any Phase 7 burn-in or retained sampling.

## Required Artifacts

- Repair subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6r-adapter-contract-repair-subplan-2026-07-09.md`
- Phase 6 result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- Updated kernel tuning artifact:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`

## Required Checks / Tests / Reviews

- Focused tests must verify:
  - adapter value/score handles a batch of parameter vectors;
  - adapter target-status telemetry exposes the required HMC diagnostic fields;
  - `use_xla=True` and `chain_execution_mode="tf_function"` remain enforced;
  - no runtime `GradientTape` or `jit_compile=False` token is introduced.
- Re-run Phase 6 only through the deterministic driver:

```text
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning
```

- Validate JSON artifacts and run `git diff --check` on touched files.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can adapter-contract fixes let the existing BayesFilter XLA HMC tuner reach a valid kernel handoff without manual HMC mechanics? |
| Baseline/comparator | Original Phase 6 hard-veto artifact with no final kernel. |
| Primary pass criterion | Phase 6 artifact has `passed=true`, no hard vetoes, confirmed XLA, and a final kernel payload/hash. |
| Veto diagnostics | Non-XLA fallback, runtime `GradientTape`, manual step/leapfrog override, missing telemetry fields, nonfinite samples/target/log-accept, missing final kernel. |
| Explanatory diagnostics | Bootstrap acceptance, repair triggers, elapsed time, public tuner progress. |
| Not concluded | No posterior convergence, posterior recovery, sampler superiority, default readiness, production readiness, GPU readiness, or scientific claim. |
| Artifact preserving result | Phase 6 result note and `kernel_tuning.json`. |

## Forbidden Claims / Actions

- Do not run `jit_compile=False` or a non-XLA fallback.
- Do not manually choose or expose HMC step size, leapfrog count, budgets, or
  grids outside BayesFilter-owned tuning artifacts.
- Do not start Phase 7 burn-in or retained sampling unless Phase 6 passes and a
  separate approval is granted.

## Exact Next-Phase Handoff Conditions

Phase 7 remains blocked unless Phase 6 produces a final frozen kernel payload
with no hard vetoes and confirmed XLA metadata.

## Stop Conditions

- Focused tests fail.
- The repaired Phase 6 run still hard-vetoes before a frozen kernel is produced.
- XLA/JIT metadata is missing or false.
- The repair requires changing the statistical target or manually selecting HMC
  mechanics.

## Skeptical Audit

Pass. The repair target is an adapter/compiled-runner contract mismatch, not a
manual HMC tuning decision. The repair preserves the same LGSSM target, prior,
fixture, mass artifact, deterministic config, CPU-hidden sample-generation
policy, and `use_xla=True` route. Passing this phase would still establish only
a kernel-tuning handoff, not posterior convergence or parameter recovery.
