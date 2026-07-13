# Phase 6Z Subplan: Verification Chunk XLA Compile-Memory Repair

Date: 2026-07-09

## Phase Objective

Repair the Phase 6Y final-verification XLA CPU LLVM compile-memory blocker by
separating the sequential verification compile chunk size from the total
verification evidence budget. The repair should reduce the static XLA graph
compiled for each verification chunk while preserving the same total
verification cap, the same selected fixed kernel, the same CPU-hidden policy,
`jit_compile=true`, and all hard vetoes.

This is a compile-scope repair only. It is not Phase 7 burn-in/sampling and it
must not change the LGSSM target, prior, fixture, mass artifact, deterministic
candidate generation, acceptance screens, R-hat threshold, or final pass
criterion.

## Entry Conditions Inherited From Phase 6Y

- Phase 6X moved the blocker past fixed-mass final-local selection and into
  the trajectory handoff stage.
- Phase 6Y moved the blocker past the trajectory stage:
  `last_completed_stage=trajectory_complete`.
- The Phase 6Y trajectory screen passed with selected
  `num_leapfrog_steps=4`, `step_size=0.6272447304363297`, and
  `selected_trajectory_hash=46b34400e68b6d18bed98a496d90ddf0f1d463055ab65533c24545f6f4b79cf2`.
- The Phase 6Y process then aborted at `verification_start` with exit code
  `134` and XLA CPU LLVM allocation errors.
- `kernel_tuning.json` remains stale from Phase 6V and cannot support Phase 7.
- Phase 7 burn-in and retained sampling remain blocked.

## Required Artifacts

- This subplan.
- Updated runbook row for Phase 6Z.
- Refreshed Phase 6 result note:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`.
- Updated execution ledger:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md`.
- Focused implementation/test changes, expected in:
  - `bayesfilter/inference/hmc_kernel_tuning.py`;
  - `tests/test_hmc_kernel_tuning_outer_loop.py`;
  - `tests/test_hmc_kernel_tuning_public_api.py` if public config payload tests
    need updating.
- Phase 6 rerun artifacts, if the serious stage is rerun:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
  and public/private diagnostics.

If the serious rerun aborts before writing `kernel_tuning.json`, the Phase 6
markdown result is the minimum structured blocker artifact. It must record the
process exit code, bounded stderr signature or hash, last progress-artifact
SHA-256, private-event SHA-256 when available, the verification chunk cap, the
minimum retained count required before a pass, and the last progress stage.

## Required Checks / Tests / Reviews

- Review this subplan with a bounded one-path Claude read-only review before
  implementation. If the approval layer blocks Claude, record that and use a
  local Codex substitute review, explicitly marked weaker than Claude.
- Add focused tests proving that:
  - `HMCTuneVerifyRepairLoopConfig` validates an optional
    `verification_chunk_max_results`;
  - the sequential final verification uses
    `check_interval=min(verification_chunk_max_results, max_results)` when the
    cap is set;
  - the total `max_results` and R-hat threshold are unchanged;
  - a pass is blocked until at least
    `verification_min_retained_results_for_pass` retained samples have been
    accumulated, so reducing chunk size cannot silently weaken the evidence
    threshold;
  - public `HMCKernelTuningConfig` forwards the verification chunk policy into
    the private loop config without exposing manual step/leapfrog/mass tuning;
  - no `jit_compile=False` or runtime `GradientTape` path is introduced.
- Run:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_outer_loop.py`
- Run:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_hmc_kernel_tuning_outer_loop.py tests/test_deterministic_lgssm_hmc_tuning_driver.py`
- Run `git diff --check` on touched files.
- Scan touched runtime files for forbidden runtime `GradientTape` and
  `jit_compile=False` tokens.
- Rerun the Phase 6 kernel tuning stage only after focused checks pass:
  `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6z python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning`

The result note and manifest must explicitly record that this serious rerun
intentionally hides GPU devices with `CUDA_VISIBLE_DEVICES=-1` because this
phase is CPU-hidden HMC verification compile repair, not NeuTra GPU training or
GPU-readiness evidence.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can final fixed-kernel verification avoid the Phase 6Y XLA CPU LLVM compile-memory abort by using a smaller deterministic fixed-size verification chunk while preserving the same total verification budget and pass gate? |
| Baseline/comparator | Phase 6Y: trajectory completed and passed, then process aborted at `verification_start` with XLA/LLVM CPU compile-memory errors before a refreshed result artifact. |
| Primary pass criterion | A refreshed Phase 6 artifact has `passed=true`, confirmed XLA/JIT, no hard vetoes, and a final kernel payload/hash; or it preserves a more specific structured non-promoting blocker without unsupported claims. |
| Veto diagnostics | Non-XLA fallback, runtime `GradientTape`, manual step/leapfrog/mass selection, target/prior/fixture changes, invalid artifacts, process abort before refreshed result, pass before minimum retained verification count, or starting Phase 7 sampling. |
| Explanatory diagnostics | Verification chunk cap, total max retained verification count, minimum retained pass count, chunk count, compile trace count, runtime, R-hat summary, last progress stage, and XLA compile-memory error text if still present. |
| Not concluded | No posterior convergence, posterior recovery, sampler superiority, production/default readiness, GPU readiness, DSGE readiness, or scientific claim. |
| Artifact preserving result | Refreshed Phase 6 result plus JSON artifacts and private/public diagnostic hashes. |

## Forbidden Claims / Actions

- Do not start Phase 7 burn-in or retained sampling.
- Do not run `jit_compile=False` or a non-XLA target-path fallback.
- Do not use runtime `GradientTape`.
- Do not manually choose step size, leapfrog count, mass matrix, budget, or
  timeout from observed diagnostics.
- Do not change the LGSSM target, prior, data fixture, mass artifact, or Phase
  6 pass criterion.
- Do not lower the total verification cap or R-hat threshold in order to pass.
- Do not allow R-hat pass before the configured minimum retained verification
  count.
- Do not convert a crash, timeout, or compile-memory blocker into a pass.
- Do not claim posterior convergence, recovery, HMC readiness, or scientific
  validity from this phase.

## Exact Next-Phase Handoff Conditions

If Phase 6Z produces `passed=true` with final kernel payload/hash, confirmed
XLA, and no hard vetoes, update the Phase 6 result and stop at the Phase 7
approval boundary.

If Phase 6Z remains blocked, update the Phase 6 result with the new structured
blocker and stop or draft the next repair subplan.

Phase 7 may start only after:

1. Phase 6 produces final kernel payload/hash with `passed=true`;
2. XLA/JIT is confirmed;
3. hard vetoes are empty;
4. a separate explicit user approval authorizes Phase 7 runtime.

## Stop Conditions

- The proposed fix requires manual tuning rather than deterministic code.
- The fix requires non-XLA execution.
- The fix changes the target, prior, fixture, mass artifact, selected kernel,
  R-hat threshold, total verification cap, or pass criteria.
- Focused tests fail.
- The Phase 6 rerun aborts before writing a structured result artifact.
- If that abort occurs, write the minimum structured blocker artifact in the
  Phase 6 result note before stopping; do not leave the crash represented only
  by terminal output.
- Claude or substitute review identifies a material blocker that cannot be
  fixed inside this phase.

## Proposed Repair Direction

Add a private deterministic verification compile policy to the loop config:

- `verification_chunk_max_results`: optional positive integer limiting the
  static `FixedSizeHMCChunkConfig.max_results` used by sequential verification;
- `verification_min_retained_results_for_pass`: optional positive integer
  requiring at least this many retained verification samples before a
  sequential R-hat pass can promote the fixed kernel.

For serious XLA runs, set the public one-call kernel tuning config to forward a
bounded verification chunk cap such as `250` while preserving the Phase 6Y
attempt-1 total verification cap of `1000` and requiring at least `1000`
retained verification samples before a pass. This changes compile shape, not
the evidence gate. If R-hat drops below threshold before 1000 retained samples,
the verifier must continue until the minimum retained count is met or a hard
veto fires.

## Skeptical Audit

Pass with constraints. The Phase 6Y failure is a process abort at the final
verification compile boundary after the trajectory handoff completed. The
repair targets static XLA compile size without changing the scientific target,
selected kernel, total verification budget, or pass criteria. The main
misleading-pass risk is that smaller chunks could let sequential R-hat pass
early; the minimum retained pass count is required specifically to block that
failure mode.
