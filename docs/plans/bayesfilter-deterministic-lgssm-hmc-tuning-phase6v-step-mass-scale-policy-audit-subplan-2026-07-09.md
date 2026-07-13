# Phase 6V Subplan: Deterministic Step/Mass-Scale Policy Audit

Date: 2026-07-09

## Phase Objective

Audit and repair, if justified, BayesFilter's deterministic fixed-mass
step/mass-scale policy after Phase 6U showed unstable proposed HMC
transitions: proposed target log-probability and HMC log-acceptance correction
are nonfinite for every fixed-mass screen transition, while accepted/current
target log-probability and samples remain finite.

This is not a retained sampling phase and not a manual HMC tuning phase.

## Entry Conditions Inherited From Previous Phase

- Phase 5 geometry and mass artifacts passed.
- Phase 6R adapter repair passed focused tests.
- Phase 6S removed the XLA compile-abort blocker.
- Phase 6T proved the log-accept hard veto is true full-trace evidence.
- Phase 6U localized the mechanics blocker to unstable proposed HMC
  transitions:
  - accepted/current target log-probability finite;
  - samples finite;
  - proposed target log-probability nonfinite;
  - HMC log-acceptance correction nonfinite.
- Phase 7 burn-in and retained sampling remain blocked.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6v-step-mass-scale-policy-audit-subplan-2026-07-09.md`
- Refreshed Phase 6 result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- Updated code only if the audit identifies a deterministic policy or
  implementation mismatch in:
  `bayesfilter/inference/hmc_kernel_tuning.py`,
  `bayesfilter/inference/hmc_budget_ladder.py`,
  `bayesfilter/inference/hmc.py`, or directly related tests.
- Result artifacts from any rerun:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
  and private diagnostics.

## Required Checks / Tests / Reviews

- Inspect the deterministic formulas and code paths that set:
  - initial fixed-mass step scale;
  - joint `(L, epsilon)` candidate grid anchors;
  - repair-step direction and bounds;
  - latent fixed-mass transform orientation;
  - mass covariance/factor orientation used by the HMC latent adapter.
- If a fix is made, add focused tests proving the policy change is
  deterministic and not agent-selected.
- Run focused HMC tuning tests relevant to the touched code.
- Run deterministic driver tests:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py`.
- Run `git diff --check` on touched files.
- Scan touched runtime files for forbidden non-XLA fallback and runtime
  `GradientTape` tokens.
- Use a bounded one-path Claude read-only review of this subplan before code
  execution when Claude is available.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does BayesFilter's deterministic fixed-mass step/mass-scale policy create unstable proposed HMC transitions for the LGSSM Phase 6 target, and if so, what deterministic policy repair is justified? |
| Baseline/comparator | Phase 6U mechanics result: all latest screens have finite accepted/current target values but nonfinite proposed target and correction values. |
| Primary pass criterion | A reviewed deterministic policy/implementation audit either identifies no code/policy bug and writes a blocker result, or implements a deterministic repair with focused tests and a Phase 6 rerun that produces either a final kernel or a more specific structured repair blocker. |
| Veto diagnostics | Manual step-size/leapfrog/mass selection, non-XLA fallback, runtime `GradientTape`, target/prior/fixture changes, public leakage of private mechanics, invalid artifacts, or treating nonfinite proposals as acceptable. |
| Explanatory diagnostics | Candidate step/mass scale summaries, latent transform orientation checks, proposed-target finite counts, correction finite counts, acceptance, trajectory-window relation, repair bounds. |
| Not concluded | No posterior convergence, posterior recovery, sampler superiority, production/default readiness, GPU readiness, or scientific claim. |
| Artifact preserving result | Refreshed Phase 6 result plus private diagnostic hashes and any focused test evidence. |

## Forbidden Claims / Actions

- Do not start Phase 7 burn-in or retained sampling.
- Do not run `jit_compile=False` or a non-XLA fallback.
- Do not manually choose a smaller step size, different leapfrog count, or
  different mass matrix from observed diagnostics.
- Do not change the LGSSM target, prior, generated data fixture, or pass
  criteria.
- Do not demote nonfinite proposed transitions to a non-veto condition.

## Exact Next-Phase Handoff Conditions

If Phase 6V produces a deterministic policy repair, rerun Phase 6 kernel tuning
under the same XLA-only CPU-hidden contract and refresh the Phase 6 result.

If no deterministic repair is justified, write a blocker result preserving the
Phase 6U evidence and stop for human direction.

Phase 7 may start only after a refreshed Phase 6 result records a final kernel
payload/hash with `passed=true`, confirmed XLA/JIT, and no hard vetoes, followed
by separate explicit runtime approval.

## Stop Conditions

- The audit cannot distinguish policy behavior from target/model behavior with
  existing artifacts.
- The needed fix would require manual tuning rather than deterministic code.
- The needed fix would require non-XLA execution.
- Focused tests fail.
- A rerun aborts before writing a structured result.

## Skeptical Audit

Pass. Phase 6U localized the hard veto to proposed-transition instability.
This plan targets deterministic policy and implementation causes without
promoting proxy diagnostics, changing the target, or crossing into sampling.
