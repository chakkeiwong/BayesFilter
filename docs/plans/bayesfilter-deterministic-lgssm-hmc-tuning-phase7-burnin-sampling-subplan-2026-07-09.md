# Phase 7 Subplan: Burn-In And Sampling Controller

Date: 2026-07-09

## Phase Objective

Implement deterministic burn-in and retained-sampling extension logic in Python,
using predeclared R-hat and ESS checks rather than agent judgment.

## Entry Conditions Inherited From Previous Phase

- Phase 6 final kernel payload exists.
- Phase 6AA refreshed `kernel_tuning.json` with `passed=true`, confirmed
  XLA/JIT, no hard vetoes, and final kernel hash
  `8ddf25a3b572893e19e814fad5ca5b6150718e36f760c159b47db1231d92ffff`.
- Phase 7 public handoff kernel hash is
  `391558a9b5f4cdc1b9dff9a5e9bceba668dedded7298c1d8c76daea42f42039a`.
- Config fixes minimums, check intervals, caps, chain count, ESS thresholds, and
  R-hat threshold.
- Explicit user approval is required before executing Phase 7 runtime.

## Required Artifacts

- Result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase7-burnin-sampling-result-2026-07-09.md`
- Sequential sampling controller JSON with per-check diagnostics and final
  retained sample artifact references.

## Required Checks / Tests / Reviews

- Unit test deterministic extension rules on synthetic diagnostic sequences.
- Runtime controller uses CPU-hidden multicore sample generation and XLA-only
  target path.
- Final stopping rule is coded before execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can burn-in and sample count be governed without agent tuning? |
| Baseline/comparator | User-specified criteria: `R_hat <= 1.01`, sufficient ESS, truth within `3 * posterior_sd`. |
| Primary pass criterion | Controller either stops with pass diagnostics or fails at configured cap with explicit reason. |
| Veto diagnostics | High R-hat at cap, ESS below floor at cap, nonfinite chain, divergent/failing telemetry, XLA fallback. |
| Explanatory diagnostics | Per-check R-hat/ESS trajectory, acceptance, runtime. |
| Not concluded | No posterior recovery until Phase 8 evaluates truth-vs-posterior criterion. |

## Forbidden Claims / Actions

- Do not start Phase 7 without explicit user approval.
- Do not manually extend burn-in after seeing diagnostics.
- Do not change thresholds after observing chains.
- Do not discard chains manually unless predeclared by config.

## Exact Next-Phase Handoff Conditions

- Phase 8 can compute final posterior recovery from retained sample artifact.

## Stop Conditions

- Controller requires manual choice to proceed.
- Retained samples cannot be produced under configured caps.
