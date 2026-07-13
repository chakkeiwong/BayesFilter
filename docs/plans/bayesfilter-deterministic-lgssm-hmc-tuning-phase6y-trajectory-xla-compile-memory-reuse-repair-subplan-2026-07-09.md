# Phase 6Y Subplan: Trajectory Handoff XLA Compile-Memory Reuse Repair

Date: 2026-07-09

## Phase Objective

Repair the Phase 6 trajectory handoff compile-memory blocker by reusing the
already-built fixed-mass screen reusable-runner cache when the trajectory
selected-pair screen has the same static XLA contract. This is a compile-scope
repair only. It must not change the LGSSM target, prior, fixture, mass
artifact, deterministic candidate generation, acceptance screens, or pass/fail
criteria.

## Entry Conditions Inherited From Phase 6X

- Phase 6W reached selected-pair progress but aborted with XLA/LLVM CPU
  compile-memory errors before writing a refreshed result.
- Phase 6X moved the blocker past fixed-mass final-local selection:
  fixed-mass stage completed, selected pair existed, and the process entered
  `trajectory_candidate_call_start`.
- The Phase 6X rerun then aborted with exit code `134`; stderr included
  `LLVM compilation error: Cannot allocate memory`,
  `allocateMappedMemory failed with error: Cannot allocate memory`, and
  `LLVM ERROR: Unable to allocate section memory!`.
- `kernel_tuning.json` remains stale from Phase 6V and cannot support Phase 7.
- Phase 7 burn-in and retained sampling remain blocked.

## Required Artifacts

- This subplan.
- Updated runbook row for Phase 6Y.
- A refreshed Phase 6 result note recording Phase 6X closure and Phase 6Y
  outcome:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`.
- Updated execution ledger:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md`.
- Focused implementation/test changes, expected in:
  - `bayesfilter/inference/hmc_kernel_tuning.py`;
  - `tests/test_hmc_kernel_tuning_fixed_mass_step.py`.
- Phase 6 rerun artifacts, if the serious stage is rerun:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
  and public/private diagnostics.

If the serious rerun aborts before writing `kernel_tuning.json`, the Phase 6
markdown result is the minimum structured blocker artifact. It must record the
process exit code, bounded stderr signature or hash, last progress-artifact
SHA-256, private-event SHA-256 when available, whether trajectory cache handoff
was enabled, and the last progress stage.

## Required Checks / Tests / Reviews

- Review this subplan with a bounded one-path Claude read-only review before
  implementation.
- Add focused tests proving that:
  - the fixed-mass stage exposes private reusable-runner cache handoff only
    outside its public payload/artifact hash;
  - the frozen-step trajectory stage can reuse the fixed-mass screen runner
    when the static contract matches;
  - the frozen-step trajectory stage does not reuse the fixed-mass runner when
    the static contract differs, and instead builds a separate compatible
    runner without changing HMC semantics;
  - trajectory candidate route telemetry records handoff source and
    `runner_reused=true`;
  - no manual step/leapfrog/mass decision is introduced.
- Preserve existing fixed-mass cache reuse and timeout tests.
- Run:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py`
- Run:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py`
- Run `git diff --check` on touched files.
- Scan touched runtime files for forbidden runtime `GradientTape` and
  `jit_compile=False` tokens.
- Rerun the Phase 6 kernel tuning stage only after focused checks pass:
  `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6y python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning`
- The Phase 6Y result note and run manifest must explicitly record that this
  serious rerun intentionally hides GPU devices with `CUDA_VISIBLE_DEVICES=-1`
  because this phase is CPU-hidden HMC sample generation / compile validation,
  not NeuTra GPU training or GPU-readiness evidence.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the trajectory selected-pair screen avoid a redundant XLA host compile by reusing the compatible fixed-mass screen reusable-runner contract and either complete Phase 6 or produce a structured non-promoting blocker? |
| Baseline/comparator | Phase 6X: fixed-mass completed, selected pair existed, then process aborted at `trajectory_candidate_call_start` with XLA/LLVM CPU compile-memory errors. |
| Primary pass criterion | A refreshed Phase 6 artifact has `passed=true`, confirmed XLA/JIT, no hard vetoes, and a final kernel payload/hash; or it preserves a more specific structured blocker without unsupported claims. |
| Veto diagnostics | Non-XLA fallback, runtime `GradientTape`, manual step/leapfrog/mass selection, target/prior/fixture changes, invalid artifacts, process abort before refreshed result, public payload leakage of live runner objects, or starting Phase 7 sampling. |
| Explanatory diagnostics | Trajectory runner reuse flags, handoff source, static contract hash, compile/runtime timing, last progress stage, XLA compile-memory error text if still present. |
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
- Do not serialize live runner objects into public artifacts.
- Do not convert a crash, timeout, or compile-memory blocker into a pass.
- Do not claim posterior convergence, recovery, HMC readiness, or scientific
  validity from this phase.

## Exact Next-Phase Handoff Conditions

If Phase 6Y produces `passed=true` with final kernel payload/hash, confirmed
XLA, and no hard vetoes, update the Phase 6 result and stop at the Phase 7
approval boundary.

If Phase 6Y remains blocked, update the Phase 6 result with the new structured
blocker and stop or draft the next repair subplan.

Phase 7 may start only after:

1. Phase 6 produces final kernel payload/hash with `passed=true`;
2. XLA/JIT is confirmed;
3. hard vetoes are empty;
4. a separate explicit user approval authorizes Phase 7 runtime.

## Stop Conditions

- The proposed fix requires manual tuning rather than deterministic code.
- The fix requires non-XLA execution.
- The fix changes the target, prior, fixture, mass artifact, or pass criteria.
- Focused tests fail.
- The Phase 6 rerun aborts before writing a structured result artifact.
- If that abort occurs, write the minimum structured blocker artifact in the
  Phase 6 result note before stopping; do not leave the crash represented only
  by terminal output.
- Claude or substitute review identifies a material blocker that cannot be
  fixed inside this phase.

## Proposed Repair Direction

Thread the private fixed-mass reusable-runner cache and static contract payload
map from `run_hmc_fixed_mass_step_stage` into the immediately following
`run_hmc_frozen_step_trajectory_stage`. The live cache must stay private and
out of `payload()`/`artifact_hash()`.

The trajectory stage already calls the same reusable-runner helper with
`dynamic_num_leapfrog_steps=True`. For the serious selected-pair handoff, the
static screen contract should match the fixed-mass screen contract when
`num_results`, `num_burnin_steps`, trace policies, target scope, adapted mass
signature, target dimension, initial state shape, and dynamic leapfrog status
match. In that case, the trajectory stage should reuse the existing compiled
runner and record public-safe route telemetry showing the handoff source and
reuse flag.

## Skeptical Audit

Pass with constraints. The latest failure is a process abort at the trajectory
handoff compile boundary after the previous fixed-mass compile reuse repair
made meaningful progress. Reusing an equivalent static XLA contract directly
targets redundant compilation and does not promote proxy metrics. The plan
still fails closed if the process aborts again or writes no refreshed result.
