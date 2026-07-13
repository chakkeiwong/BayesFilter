# Phase 6X Subplan: Fixed-Mass XLA Compile-Memory Reuse Repair

Date: 2026-07-09

## Phase Objective

Repair the Phase 6W XLA host compile-memory blocker without changing HMC
mechanics, target, prior, fixture, pass criteria, or the `jit_compile=true`
runtime contract.

The specific repair target is compile proliferation in the fixed-mass joint
`(L, epsilon)` stage. Phase 6W progressed beyond the earlier budget blocker,
completed fixed-mass grids, found selected-pair progress, and then crashed with
LLVM/XLA CPU compile-memory errors before a refreshed `kernel_tuning.json`
could be written. This phase should reduce or isolate redundant XLA compiled
contracts. It must not use a non-XLA fallback.

## Entry Conditions Inherited From Previous Phase

- Phase 5 geometry and mass artifacts passed.
- Phase 6R through Phase 6V repaired adapter, compile, telemetry, mechanics,
  and step/mass-scale blockers enough that the latest completed structured
  artifact had:
  - `xla_confirmed=true`;
  - `hard_vetoes=[]`;
  - `final_status=budget_exhausted`;
  - no final kernel payload/hash.
- Phase 6W changed serious Phase 6 to the deterministic staged timeout policy
  and passed focused local checks.
- Phase 6W rerun made meaningful progress but crashed with process code `139`
  before writing a refreshed result artifact.
- Latest progress artifact after the crash records:
  - `phase7_last_attempt_index=1`;
  - fixed-mass initial grid candidate count `7`;
  - hard-veto count `0`;
  - selected-pair progress `true`;
  - current stage `fixed_mass_ladder_repair_screen_call_start`.
- Crash stderr included LLVM/XLA CPU messages containing
  `Cannot allocate memory`, `releaseMappedMemory failed`, and defunct
  `JITDylib` / resource tracker messages.
- The stale `kernel_tuning.json` remains the prior Phase 6V artifact and must
  not be treated as a Phase 6W pass.
- Phase 7 burn-in and retained sampling remain blocked.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6x-fixed-mass-xla-compile-memory-reuse-repair-subplan-2026-07-09.md`
- Refreshed Phase 6 result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- Updated ledger:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md`
- Updated runbook:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md`
- Likely touched runtime files:
  - `bayesfilter/inference/hmc_kernel_tuning.py`
  - focused fixed-mass/kernel tuning tests
- Phase 6 rerun artifacts, if the serious stage is rerun:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
  and public/private diagnostics.
- The refreshed Phase 6 result must state whether the Phase 6X cache-reuse
  mechanism engaged in the serious rerun. At minimum, record the latest
  available route evidence for:
  - reusable runner contract count;
  - runner reuse flags;
  - whether dynamic leapfrog was enabled;
  - whether reuse crossed initial/edge/final-local grid-round boundaries;
  - latest completed stage and candidate/round progress.
- If the serious rerun aborts before writing `kernel_tuning.json`, the Phase 6
  markdown result remains the minimum structured blocker artifact. It must
  record the process exit code, bounded stderr signature or hash, last
  progress-artifact SHA-256, private-event SHA-256 when available, whether the
  cache-reuse mechanism was enabled, and whether the abort occurred before or
  after selected-pair progress.

## Required Checks / Tests / Reviews

- Review this subplan with a bounded one-path Claude read-only review.
- Inspect and patch only the fixed-mass joint grid compile-reuse path unless a
  directly adjacent compile-memory issue is proved.
- Add focused tests proving that the dynamic-leapfrog reusable runner cache is
  reused across initial, edge-repair, and final-local joint grid rounds when
  the static XLA contract is unchanged.
- Preserve existing tests proving:
  - no-progress public timeout remains a hard veto;
  - selected-pair progress timeout remains non-promoting;
  - the serious deterministic driver uses staged timeout policy;
  - no manual step/leapfrog/mass decision is introduced.
- Run:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py`
- Run `git diff --check` on touched files.
- Scan touched runtime files for forbidden runtime `GradientTape` and
  `jit_compile=False` tokens.
- Rerun the Phase 6 kernel tuning stage only after the focused checks pass:
  `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6x python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning`
- After the serious rerun, inspect the refreshed result or last progress and
  private diagnostics to confirm whether the shared-cache route actually
  engaged. A focused test of cache reuse is insufficient by itself for the
  Phase 6X result note.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the fixed-mass joint grid avoid redundant XLA host compilation across equivalent static contracts and either complete Phase 6 or produce a structured non-promoting blocker? |
| Baseline/comparator | Phase 6W: selected-pair progress reached, then process crashed with XLA/LLVM CPU compile-memory errors before result refresh. |
| Primary pass criterion | A refreshed Phase 6 artifact has `passed=true`, confirmed XLA/JIT, no hard vetoes, and a final kernel payload/hash; or it preserves a more specific structured blocker without unsupported claims. |
| Veto diagnostics | Non-XLA fallback, runtime `GradientTape`, manual step/leapfrog/mass selection, target/prior/fixture changes, invalid artifacts, process abort before refreshed result, or starting Phase 7 sampling. |
| Explanatory diagnostics | Reusable runner contract count, runner reuse flags, stage/round progress, compile/runtime timing, XLA compile-memory error text if still present. |
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
- Do not convert a crash, timeout, or compile-memory blocker into a pass.
- Do not claim posterior convergence, recovery, HMC readiness, or scientific
  validity from this phase.

## Exact Next-Phase Handoff Conditions

If Phase 6X produces `passed=true` with final kernel payload/hash, confirmed
XLA, and no hard vetoes, update the Phase 6 result and stop at the Phase 7
approval boundary.

If Phase 6X remains blocked, update the Phase 6 result with the new structured
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
- If that abort occurs, write the minimum structured blocker artifact in the
  Phase 6 result note before stopping; do not leave the crash represented only
  by terminal output.
- Claude or substitute review identifies a material blocker that cannot be
  fixed inside this phase.

## Proposed Repair Direction

Use one dynamic-leapfrog reusable runner cache for the entire fixed-mass joint
`(L, epsilon)` stage, not one fresh cache per grid round. The existing
budget-ladder reusable route already removes `seed`, `step_size`, and, for
dynamic leapfrog mode, `num_leapfrog_steps` from the static contract. The
initial, edge-repair, and final-local joint grid rounds therefore often need
the same two XLA contracts: one tune contract and one fixed-screen contract.

The first implementation should thread the shared cache from
`run_hmc_fixed_mass_step_stage` into every `_run_joint_l_epsilon_grid_round`
call for that stage. This reduces redundant XLA compilation while preserving
the same deterministic candidate generation, selection, acceptance screens,
mass artifact, and pass/fail rules.

## Skeptical Audit

Pass with constraints. The latest failure is not evidence against the LGSSM
target or HMC mechanics. It is a process abort with XLA/LLVM compile-memory
messages after selected-pair progress. A cache-scope repair directly targets
redundant compilation and does not promote proxy metrics. The plan still fails
closed if the process aborts again or writes no refreshed result.
