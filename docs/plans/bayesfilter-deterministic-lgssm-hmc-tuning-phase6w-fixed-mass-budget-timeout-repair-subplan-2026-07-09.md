# Phase 6W Subplan: Fixed-Mass Final-Local Budget/Timeout Repair

Date: 2026-07-09

## Phase Objective

Repair the deterministic fixed-mass step budget/timeout controller after
Phase 6V removed the previous mechanics hard veto but stopped with
`fixed_mass_step_budget_incomplete_non_promoting`.

The repair target is the orchestration policy for completing the fixed-mass
joint `(L, epsilon)` final-local candidate work after selected-pair progress.
It is not manual HMC tuning, not retained sampling, and not a posterior
convergence or recovery phase.

## Entry Conditions Inherited From Previous Phase

- Phase 5 geometry and mass artifacts passed.
- Phase 6R adapter repair passed focused checks.
- Phase 6S removed the fixed-mass XLA compile-abort blocker.
- Phase 6T/6U proved the prior hard veto was real and localized it to
  proposed-transition mechanics.
- Phase 6V repaired the isolated nonfinite proposal-mechanics screen for the
  serious fixed-mass ladder:
  - latest fixed-mass log-accept ratio finite count: 500;
  - latest proposed target log-probability finite count: 500;
  - latest log-acceptance-correction finite count: 500;
  - latest samples finite: true.
- Latest Phase 6V public result:
  - `passed=false`;
  - `final_status=budget_exhausted`;
  - `diagnostic_role=fixed_mass_step_budget_incomplete_non_promoting`;
  - `xla_confirmed=true`;
  - `hard_vetoes=[]`;
  - `final_kernel_hash=null`.
- Phase 7 burn-in and retained sampling remain blocked.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6w-fixed-mass-budget-timeout-repair-subplan-2026-07-09.md`
- Refreshed Phase 6 result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- Updated ledger:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md`
- Updated runbook:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md`
- If code is changed, likely touched files are limited to:
  - `bayesfilter/inference/hmc_kernel_tuning.py`;
  - `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`;
  - focused HMC tuning tests.
- Phase 6 rerun artifacts:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
  and public/private diagnostics.

## Required Checks / Tests / Reviews

- Review this subplan with a bounded one-path Claude read-only review when
  Claude is available.
- Inspect the fixed-mass timeout path in:
  `run_hmc_fixed_mass_step_stage`,
  `_fixed_mass_step_next_candidate_soft_deadline_veto`,
  `_phase7_fixed_step_stage_config`,
  `run_hmc_tune_verify_repair_loop`, and the deterministic driver config.
- Prefer existing deterministic policy surfaces before adding a new mechanism:
  `HMCStagedTimeoutPolicy`,
  `HMCGeometryScaledBudgetTimingPolicy`, and existing attempt-budget policies.
- If code changes are made, add focused tests proving:
  - pre-candidate/no-progress public timeout remains a hard veto;
  - selected-pair progress budget-incomplete remains non-promoting;
  - the deterministic driver uses staged timeout policy for serious Phase 6;
  - no manual step/leapfrog/mass decision is introduced.
- Run focused tests for touched HMC tuning code.
- Run deterministic driver tests:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py`.
- Run `git diff --check` on touched files.
- Scan touched runtime files for forbidden runtime `GradientTape` and
  `jit_compile=False` tokens.
- Rerun only the Phase 6 kernel tuning stage after local checks pass:
  `CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the deterministic fixed-mass budget/timeout controller allow selected-pair-progress final-local work to complete, or produce a more specific non-promoting blocker, without manual HMC tuning? |
| Baseline/comparator | Phase 6V result: XLA confirmed, no hard vetoes, mechanics finite, but fixed-mass budget incomplete after selected-pair progress. |
| Primary pass criterion | A refreshed Phase 6 artifact has `passed=true`, confirmed XLA/JIT, no hard vetoes, and a final kernel payload/hash; or it preserves a more specific structured blocker without unsupported claims. |
| Veto diagnostics | Non-XLA fallback, runtime `GradientTape`, manual step/leapfrog/mass selection, target/prior/fixture changes, treating timeout as sampler success, invalid artifacts, or starting Phase 7 sampling. |
| Explanatory diagnostics | Stage timeout policy payload, fixed-mass public timeout closeout, attempt count, selected-pair progress flag, final-local grid status, repair triggers, compile/runtime timing. |
| Not concluded | No posterior convergence, posterior recovery, sampler superiority, production/default readiness, GPU readiness, DSGE readiness, or scientific claim. |
| Artifact preserving result | Refreshed Phase 6 result plus JSON artifacts and private/public diagnostic hashes. |

## Forbidden Claims / Actions

- Do not start Phase 7 burn-in or retained sampling.
- Do not run `jit_compile=False` or a non-XLA target-path fallback.
- Do not use runtime `GradientTape`.
- Do not manually choose step size, leapfrog count, mass matrix, budget, or
  timeout from observed diagnostics.
- Do not change the LGSSM target, prior, data fixture, or Phase 6 pass
  criterion.
- Do not convert budget-incomplete status into a pass.
- Do not claim posterior convergence, recovery, HMC readiness, or scientific
  validity from this phase.

## Exact Next-Phase Handoff Conditions

If Phase 6W produces `passed=true` with final kernel payload/hash, confirmed
XLA, and no hard vetoes, update the Phase 6 result and stop at the Phase 7
approval boundary.

If Phase 6W remains blocked, update the Phase 6 result with the new structured
blocker and stop or draft the next repair subplan.

Phase 7 may start only after:

1. Phase 6 produces final kernel payload/hash with `passed=true`;
2. XLA/JIT is confirmed;
3. hard vetoes are empty;
4. a separate explicit user approval authorizes Phase 7 runtime.

## Stop Conditions

- The proposed fix requires manual tuning rather than deterministic code.
- The fix requires non-XLA execution.
- The fix changes the target, prior, fixture, or pass criteria.
- Focused tests fail.
- The Phase 6 rerun aborts before writing a structured result artifact.
- Claude or substitute review identifies a material blocker that cannot be
  fixed inside this phase.

## Proposed Repair Direction

The first repair candidate is to use existing deterministic staged timeout
policy for serious Phase 6 instead of a one-call public timeout only. The
policy is already documented as machine-protection only, has stage-specific
budgets, and separates meaningful progress from timeout closeout. This should
give fixed-mass final-local work its own deterministic stage clock without
changing HMC mechanics or manually selecting a kernel.

If that is insufficient, the next allowed repair is a narrow outer-loop
continuation rule for the exact non-hard-veto condition
`fixed_mass_step_budget_incomplete_after_selected_pair_progress`, but only if
it carries a deterministic attempt-budget or staged-timeout enlargement and
does not fabricate a selected kernel payload.

## Skeptical Audit

Pass with constraints. The current artifact answers the old mechanics question:
the latest fixed-mass mechanics are finite and no hard veto remains. The new
blocker is a budget/timeout orchestration blocker after selected-pair progress.
This subplan targets that controller using existing deterministic policy
surfaces first. It does not promote proxy diagnostics, change the target, or
cross into retained sampling.
