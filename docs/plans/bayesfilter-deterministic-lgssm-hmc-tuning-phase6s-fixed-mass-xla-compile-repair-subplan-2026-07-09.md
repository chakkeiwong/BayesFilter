# Phase 6S Subplan: Fixed-Mass Candidate Grid XLA Compile Repair

Date: 2026-07-09

## Phase Objective

Repair the Phase 6 fixed-mass joint `(L, epsilon)` candidate-grid execution so
it reuses XLA-compiled runner contracts across candidate leapfrog counts where
BayesFilter already supports dynamic `num_leapfrog_steps`.

## Entry Conditions Inherited From Previous Phase

- Phase 5 geometry and mass artifacts passed.
- Phase 6R adapter repair passed focused tests.
- Phase 6 rerun reached the fixed-mass candidate grid with `use_xla=True`.
- Phase 6 did not produce a final kernel because the process aborted during XLA
  CPU code generation before candidate-grid completion.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6s-fixed-mass-xla-compile-repair-subplan-2026-07-09.md`
- Phase 6 blocked result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- Updated code:
  `bayesfilter/inference/hmc_budget_ladder.py`
- Focused tests:
  `tests/test_hmc_budget_ladder.py`
- Rerun result if repair passes local checks:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`

## Required Checks / Tests / Reviews

- Focused tests for the fixed-mass budget ladder must show:
  - reusable runner cache can be shared across fixed-mass ladders;
  - `num_leapfrog_steps` can be passed dynamically through the existing
    `ReusableFullChainHMCRunner`;
  - static contract hashes omit `num_leapfrog_steps` only on the private dynamic
    route;
  - no `jit_compile=False`, runtime `GradientTape`, or manual HMC parameter
    selection is introduced.
- Deterministic driver tests must still pass.
- JSON validation must pass for the existing progress, geometry, and mass
  artifacts.
- If Claude review is available, use one bounded read-only review of this
  subplan and the Phase 6 result. If the review gate is rejected by policy,
  write a Codex substitute review and record that it is weaker.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can Phase 6 reduce fixed-mass candidate-grid XLA compile pressure without changing the target, mass, deterministic tuner, or XLA-only rule? |
| Baseline/comparator | Blocked Phase 6 run that aborted during fixed-mass candidate 2 XLA CPU code generation. |
| Primary pass criterion | Focused tests pass and the Phase 6 rerun either produces a final kernel artifact or reaches a structured BayesFilter hard-veto/budget-exhausted result instead of process abort. |
| Veto diagnostics | Non-XLA fallback, runtime `GradientTape`, manual step/leapfrog override, weakened log-accept hard vetoes, changed target/mass/fixture, invalid JSON artifacts, or process abort before structured result. |
| Explanatory diagnostics | Runner build/reuse counts, dynamic-leapfrog route metadata, candidate-grid progress, compile/runtime timing, hard veto categories. |
| Not concluded | No posterior convergence, posterior recovery, sampler superiority, production/default readiness, GPU readiness, or scientific claim. |
| Artifact preserving result | Phase 6 result refresh and `kernel_tuning_public/hmc_kernel_tuning_progress.json` / `kernel_tuning.json` if produced. |

## Forbidden Claims / Actions

- Do not run `jit_compile=False` or a non-XLA fallback.
- Do not manually choose step size, leapfrog count, candidate grids, burn-in
  budgets, or sample budgets outside BayesFilter-owned deterministic APIs.
- Do not weaken `screen_log_accept_nonfinite_or_missing` or any log-accept hard
  veto.
- Do not start Phase 7 burn-in or retained sampling.

## Exact Next-Phase Handoff Conditions

Phase 7 remains blocked unless a refreshed Phase 6 result records a final
kernel payload/hash with `passed=true`, confirmed XLA/JIT, and no hard vetoes.

If Phase 6S only converts the process abort into a structured hard veto or
budget-exhausted artifact, the next step is another Phase 6 repair subplan, not
Phase 7.

## Stop Conditions

- Focused tests fail.
- The repair requires changing the LGSSM target, prior, fixture, mass artifact,
  or deterministic tuning policy.
- The repair requires non-XLA execution.
- The rerun process aborts again before producing a structured result artifact.

## Skeptical Audit

Pass. The repair targets an engineering compile-pressure blocker in the fixed
candidate-grid execution route. It does not reinterpret acceptance or
log-accept diagnostics as posterior evidence, and it preserves the same target,
fixture, mass artifact, HMC tuning APIs, CPU-hidden execution policy, and
`use_xla=True` rule.
