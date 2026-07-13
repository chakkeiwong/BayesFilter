# Phase 8 Subplan: Serious Recovery Run

Date: 2026-07-09

## Phase Objective

Run the deterministic serious LGSSM HMC recovery test and evaluate the
predeclared pass/fail criteria.

## Entry Conditions Inherited From Previous Phase

- Phase 7 controller is implemented and tested.
- Explicit user approval is recorded for the serious runtime.
- All target-path runtime uses `jit_compile=True`.

## Required Artifacts

- Result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase8-serious-recovery-result-2026-07-09.md`
- Full JSON artifact with manifest, hashes, chains, diagnostics, per-parameter
  table, and pass/fail decision.
- Bounded log file under `docs/benchmarks/artifacts/`.

## Required Checks / Tests / Reviews

- Run deterministic driver from clean command line.
- Verify artifact schema and hashes.
- Verify all parameters meet `R_hat <= 1.01`.
- Verify ESS floors.
- Verify posterior mean within `3 * posterior_sd` of truth for every parameter.
- Claude review of result interpretation before any promotion language.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the serious HMC run recover prior-mean truth for the identifiable LGSSM fixture? |
| Baseline/comparator | Generated `T=120` fixture with truth equal to prior means. |
| Primary pass criterion | R-hat, ESS, and all per-parameter `abs(mean - truth) <= 3 * posterior_sd` pass. |
| Veto diagnostics | Any R-hat/ESS/recovery failure, nonfinite chain, XLA fallback, manual intervention, artifact mismatch. |
| Explanatory diagnostics | Acceptance, runtime, posterior covariance, MCSE, per-parameter z scores. |
| Not concluded | No sampler superiority, broad model adequacy, production readiness, or DSGE claim. |

## Forbidden Claims / Actions

- Do not modify pass criteria after results.
- Do not hide failed parameters behind aggregate statistics.
- Do not call descriptive runtime/acceptance a success criterion.

## Exact Next-Phase Handoff Conditions

- Phase 9 can write a closeout separating pass/fail, uncertainty, and next work.

## Stop Conditions

- Runtime approval missing.
- Required artifact incomplete.
- Any final pass criterion fails at configured caps.
