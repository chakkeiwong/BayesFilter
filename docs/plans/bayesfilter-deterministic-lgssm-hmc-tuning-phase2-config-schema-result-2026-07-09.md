# Phase 2 Result: Deterministic Config Schema

Date: 2026-07-09

Status: `PASSED`

## Scope

Phase 2 created a versioned deterministic JSON config skeleton for the serious
LGSSM HMC tuning/recovery program. It did not run the config, generate a new
fixture, compile XLA, tune HMC, sample chains, or make posterior claims.

## Artifact

- Config:
  `docs/benchmarks/configs/multidim_lgssm_serious_hmc_tuning_2026_07_09.json`

## Schema Contents

The config fixes:

- source contract path and target id;
- 4D lower-triangular LGSSM shape and raw parameter names;
- truth policy: prior mean in raw coordinates from the source contract;
- serious fixture horizon: `T=120`;
- simulation, geometry, and tuning seeds;
- CPU-hidden multicore sample generation policy;
- target-path `jit_compile=true` / `use_xla=true`;
- non-XLA runtime veto;
- no runtime `GradientTape`;
- quadratic geometry settings;
- mass conversion settings;
- serious kernel tuning policy;
- deterministic burn-in extension rules;
- deterministic retained-sampling extension rules;
- final R-hat/ESS/truth-distance gate;
- artifact paths and nonclaims.

## Local Checks

Local JSON/schema sanity checks passed:

- JSON parses.
- Required top-level sections exist.
- `truth_and_data.horizon == 120`.
- `execution_policy.jit_compile == true`.
- `execution_policy.use_xla == true`.
- `execution_policy.jit_compile_false_runtime_allowed == false`.
- `execution_policy.gpu_sample_generation_allowed == false`.
- `execution_policy.required_environment.CUDA_VISIBLE_DEVICES == "-1"`.
- Burn-in and sampling controllers include R-hat, ESS, extension sizes, and caps.
- Final recovery gate requires all parameters and `R_hat <= 1.01`.

Config hash:

```text
sha256:683e45cef9a46e14a3ee2de3e51d5fc19a0512feb43e376e30c2da19e1a2ccb0
```

## Evidence Contract Assessment

| Field | Assessment |
| --- | --- |
| Question | Can the entire tuning/recovery run be determined by config plus code? |
| Primary criterion | Met for schema/config skeleton: no manual post-result tuning hook is present. |
| Veto diagnostics | No missing seeds, caps, XLA veto, or final thresholds found in the config. |
| Explanatory diagnostics | Config hash and source contract path recorded. |
| Not concluded | No target correctness, XLA runtime success, HMC tuning success, convergence, or recovery claim. |

## Deterministic Burn-In And Sampling Policy

Burn-in:

- start with 4 chains and 2000 burn-in results per chain;
- check every 1000 per chain;
- extend by 1000 per chain until all R-hat and ESS checks pass or the cap of
  16000 per chain is reached;
- if the cap is reached without passing, fail closed.

Retained sampling:

- start with 4000 retained results per chain;
- check every 2000 per chain;
- extend by 2000 per chain until all R-hat and ESS checks pass or the cap of
  40000 per chain is reached;
- no manual chain exclusion or posthoc thinning is allowed.

Final recovery:

- every parameter must satisfy `R_hat <= 1.01`;
- every parameter must satisfy bulk ESS >= 1000 and tail ESS >= 400;
- every posterior mean must be within `3 * posterior_sd` of truth;
- all checks are all-parameter gates, not aggregate gates.

## Decision Table

| Field | Decision |
| --- | --- |
| Phase decision | `PASS_TO_PHASE3` |
| Primary criterion status | Deterministic config schema exists and passes local sanity checks |
| Veto diagnostic status | No Phase 2 veto triggered |
| Main uncertainty | Driver code still needs to implement/enforce the schema exactly |
| Next justified action | Implement or bind deterministic LGSSM fixture generation for `T=120` |
| What is not concluded | No HMC run, no posterior recovery, no convergence, no runtime feasibility, no scientific/default/product claim |
