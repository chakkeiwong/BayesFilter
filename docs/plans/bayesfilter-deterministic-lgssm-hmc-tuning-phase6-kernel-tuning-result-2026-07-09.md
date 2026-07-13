# Phase 6 Result: Kernel Tuning Structured Hard Veto

Date: 2026-07-09

## Scope

This note records the latest BayesFilter deterministic LGSSM HMC
kernel-tuning gate after the Phase 6R adapter repair and the Phase 6S
fixed-mass XLA compile-pressure repair. It is a kernel-tuning handoff gate
only. It is not posterior convergence evidence, parameter-recovery evidence,
sampler superiority evidence, production readiness, default readiness, or a
scientific claim.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter select a frozen fixed HMC kernel through deterministic staged tuning with `use_xla=True`? |
| Baseline/comparator | Phase 5 geometry/mass artifacts, the Phase 6R adapter repair, and the Phase 6S XLA compile-pressure repair. |
| Primary pass criterion | `kernel_tuning.json` exists with `passed=true`, confirmed XLA/JIT, no hard vetoes, and a final kernel payload/hash. |
| Veto diagnostics | Non-XLA fallback, runtime `GradientTape`, missing final kernel, tuning hard veto, nonfinite samples/target/log-accept trace, invalid artifacts, or process abort before result artifact. |
| Explanatory diagnostics | Bootstrap status, windowed mass status, fixed-mass candidate-grid progress, compile/runtime timing, repair triggers. |
| Not concluded | Posterior convergence, posterior recovery, sampler validity, sampler ranking, production/default readiness, GPU readiness, or DSGE readiness. |
| Artifact preserving result | This result note plus `kernel_tuning.json`, `kernel_tuning_public/hmc_kernel_tuning_result.json`, `kernel_tuning_public/hmc_kernel_tuning_progress.json`, and private diagnostic event hashes. |

## Result

`BLOCKED_STRUCTURED_HARD_VETO`.

The Phase 6S repair succeeded at the engineering level it targeted: the run no
longer aborted during XLA CPU code generation. The deterministic driver exited
with process code `0` and wrote structured result artifacts.

The latest Phase 6 kernel gate still failed:

- `kernel_tuning.json` exists.
- `passed=false`.
- `final_status=hard_veto`.
- `xla_confirmed=true`.
- `hard_vetoes=["screen_log_accept_nonfinite_or_missing"]`.
- `final_kernel_payload=null`.
- `final_kernel_hash=null`.
- artifact hash:
  `sha256:733310aeeac8801779f33c8039e66eb73307e9dd9c2e7fe45d121d88835b73e2`.
- file SHA-256:
  `fb61514c4dde1ed36b0160ed63f2375c1d67c700db7e6340676d9aed42df696a`.

Therefore there is no valid final fixed HMC kernel handoff and Phase 7 remains
blocked.

## Artifact-Backed Progress

From the structured artifacts:

- `kernel_tuning_public/hmc_kernel_tuning_progress.json` reached
  `current_stage=result_written`.
- Bootstrap public summary:
  - `preflight_passed=true`
  - `hard_veto_present=false`
  - `jit_compile_metadata=true`
  - `final_status=repair_budget_exhausted`
  - `handoff_kernel_source=geometry_preflight_fallback`
- The first windowed mass stage completed and produced adapted mass signature:
  `eb30e03e82d353cc62ad5c33e0bd90c56cd92efa677e3aa2e7fe56949e8ac497`.
- A second repair-loop windowed mass stage completed and produced adapted mass
  signature:
  `6fc83e21c82e34ff0a2db5d160049000280f71da4d8ba7289a94646b193bc6d5`.
- The fixed-mass joint `(L, epsilon)` candidate grid completed two bounded
  attempts:
  - attempt 0: 7 of 7 candidates hard-vetoed;
  - attempt 1: 7 of 7 candidates hard-vetoed.
- All 14 candidate completions carried the same hard veto:
  `screen_log_accept_nonfinite_or_missing`.
- No candidate produced a selected step/leapfrog pair or final kernel payload.

The observed blocker is now a log-acceptance screen mechanics blocker, not an
XLA compile-abort blocker and not a missing-telemetry blocker.

## Phase 6T Root-Cause Closure

Phase 6T added fail-closed private diagnostic summaries for the last fixed-mass
ladder round and reran the deterministic kernel stage with
`CUDA_VISIBLE_DEVICES=-1`, `use_xla=true`, and no non-XLA fallback. The run
exited with code `0` after XLA compilation was confirmed.

Private event artifact:

```text
docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/private_diagnostics/hmc_tuning_events.jsonl
sha256: 14896aed6d0885d02b5270b531d863e208af49c0adce2324ff20337fb3b3e173
```

Private diagnostic classification:

- 14 instrumented fixed-mass candidate completions included
  `last_ladder_round_private_diagnostics`.
- All 14 used full trace evidence for `log_accept_ratio`.
- All 14 had `log_accept_ratio_finite=false`.
- The latest seven candidate screens each had
  `finite_count=0`, `nonfinite_count=500`, `total_count=500` for
  `log_accept_ratio`.
- The same latest seven screens each had
  `target_log_prob_finite=true`, `finite_count=500`, `nonfinite_count=0` for
  `target_log_prob`.
- `samples_all_finite=true` for those screens.

Conclusion: the Phase 6 hard veto is supported by full trace evidence. It is
not caused by missing log-accept telemetry propagation. Because accepted target
log-probability and samples are finite while the log-acceptance ratio is wholly
nonfinite, the next repair must inspect HMC proposal/kinetic-correction
mechanics rather than increasing burn-in or retained sampling budgets.

## Phase 6U Mechanics Closure

Phase 6U added private-only mechanics summaries for:

- accepted/current target log-probability;
- proposed target log-probability;
- HMC log-acceptance correction;
- summed `log_accept_ratio`.

The deterministic kernel stage was rerun with `CUDA_VISIBLE_DEVICES=-1`,
`use_xla=true`, and no non-XLA fallback. The run exited with code `0` after
XLA compilation was confirmed.

Updated private event artifact:

```text
docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/private_diagnostics/hmc_tuning_events.jsonl
sha256: f368dc2b1a3c84ce2f15ac3fdffca0dfafa108e32d07445a2cfd3e5df9004912
```

Private diagnostic classification:

- 28 total candidate completions in the accumulated private event stream now
  include `last_ladder_round_private_diagnostics`.
- The latest 14 include Phase 6U proposed-target and log-acceptance-correction
  summaries.
- For the latest seven fixed-mass screen candidates:
  - `log_accept_ratio`: 0 finite, 500 nonfinite;
  - accepted/current `target_log_prob`: 500 finite, 0 nonfinite;
  - `proposed_target_log_prob`: 0 finite, 500 nonfinite;
  - `log_acceptance_correction`: 0 finite, 500 nonfinite;
  - samples finite: true.

Conclusion: the hard veto is caused by unstable proposed HMC transitions under
the fixed-mass screen, not by current-state target evaluation failure and not
by missing telemetry. The next step is a deterministic step/mass-scale policy
audit and repair plan. It is not Phase 7 sampling.

## Checks

Post-run artifact validation:

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json
```

Result: passed.

Previously completed Phase 6S local checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py
```

Result after Phase 6T: `37 passed, 2 warnings`.

Result after Phase 6U: `38 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result after Phase 6T: `16 passed`.

Result after Phase 6U: `16 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_nonlinear_ssm_phase4_full_chain_hmc.py::test_phase4_tiny_full_chain_hmc_jit_returns_finite_samples_and_metadata
```

Result after Phase 6U: `1 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `11 passed, 2 warnings`.

Result after Phase 6U: `11 passed, 2 warnings`.

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_result.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json
```

Result after Phase 6T: passed.

Result after Phase 6U: passed.

## Decision Table

| Decision | Status |
| --- | --- |
| Phase 6 final kernel tuning gate | `BLOCKED_BUDGET_INCOMPLETE_NON_PROMOTING` |
| Primary criterion | Failed: no final kernel payload/hash was produced. |
| Veto diagnostics | No hard vetoes remain in the latest Phase 6V rerun; XLA was confirmed and no non-XLA fallback was used. |
| Main uncertainty | Whether the fixed-mass final-local candidate grid can complete under deterministic stage-budget/timeout policy without manual tuning. |
| Next justified action | Draft and review Phase 6W: deterministic fixed-mass final-local budget/timeout repair. |
| What is not concluded | This does not reject the LGSSM target, the prior, the fixture, BayesFilter HMC, NeuTra, or the research direction. |

## Phase 6V Step/Mass-Scale Policy Closure

Phase 6V added an opt-in private repair classification for the isolated
nonfinite-proposal mechanics screen observed in Phase 6U. The classification
remains non-promoting: it is enabled only for serious tuning, requires finite
current target values and finite samples, and converts only the isolated
`screen_log_accept_nonfinite_or_missing` screen into a private repair trigger.

The deterministic kernel stage was rerun with `CUDA_VISIBLE_DEVICES=-1`,
`use_xla=true`, and no non-XLA fallback. The run exited with code `0`.

Latest Phase 6V artifact status:

- `passed=false`
- `final_status=budget_exhausted`
- `diagnostic_role=fixed_mass_step_budget_incomplete_non_promoting`
- `xla_confirmed=true`
- `hard_vetoes=[]`
- `final_kernel_payload=null`
- `final_kernel_hash=null`
- `kernel_tuning.json` payload artifact hash:
  `sha256:8b3367c52619957080a6c28e51262cfed537b05c2d6d2d6b76a151d3da355484`
- `kernel_tuning.json` file SHA-256:
  `5bb6948c2bbc90038818bf98c4713d6d99983d593bd576c552810bbe622d482d`
- public result SHA-256:
  `3909d89b4f7cfbcb30899aee26f005cdc744c4b34ddf446826e3ab3268db4a28`
- public progress SHA-256:
  `510085bb385c32d7e35b410442571260722862df4243e635942b9e7aca77eb44`
- private event SHA-256:
  `1e818b8ef1d49ef8f5570f2ea6a4a3210af6e5836ccde29c68537eab346b5079`

Latest private mechanics evidence shows the old hard veto is repaired:

- `log_accept_ratio`: 500 finite, 0 nonfinite in the latest candidate screen;
- accepted/current `target_log_prob`: 500 finite, 0 nonfinite;
- proposed `target_log_prob`: 500 finite, 0 nonfinite;
- `log_acceptance_correction`: 500 finite, 0 nonfinite;
- samples finite: true.

The latest blocker is therefore not a mechanics hard veto. The blocker is
public timeout/budget closeout after selected-pair progress in the fixed-mass
step stage:

- active repair trigger:
  `fixed_mass_step_budget_incomplete_after_selected_pair_progress`;
- final attempt status:
  `phase5_fixed_mass_step_status:budget_exhausted`;
- no final-local handoff kernel was produced.

Phase 7 remains blocked.

Phase 6V checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: `56 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `11 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `67 passed, 2 warnings`.

```text
git diff --check -- bayesfilter/inference/hmc_budget_ladder.py bayesfilter/inference/hmc_kernel_tuning.py tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
```

Result: passed.

```text
rg -n "GradientTape|batch_jacobian|tape\.|jit_compile\s*=\s*False|jit_compile=False" bayesfilter/inference/hmc_budget_ladder.py bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
```

Result: no matches.

## Stop / Handoff

Phase 7 burn-in and retained sampling remain blocked. Phase 7 may start only
after Phase 6 produces a valid final kernel payload/hash with confirmed XLA and
no hard vetoes, followed by a separate explicit runtime approval.

## Phase 6W Budget/Timeout Repair Closure

Phase 6W changed the serious deterministic Phase 6 driver to use the existing
geometry-scaled staged timeout policy instead of the prior one-call public
timeout. Local focused checks passed before rerun:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `11 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: `17 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `67 passed, 2 warnings`.

```text
git diff --check -- bayesfilter/inference/hmc_budget_ladder.py bayesfilter/inference/hmc_kernel_tuning.py tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
```

Result: passed.

```text
rg -n "GradientTape|batch_jacobian|tape\.|jit_compile\s*=\s*False|jit_compile=False" bayesfilter/inference/hmc_budget_ladder.py bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
```

Result: no matches.

The serious Phase 6 rerun used:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6w python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning
```

It made meaningful progress beyond the previous budget-incomplete blocker:

- host XLA compilation was observed;
- attempt 0 completed fixed-mass 7 of 7 candidates with no hard veto and moved
  to attempt 1;
- attempt 1 completed windowed mass;
- attempt 1 completed fixed-mass initial grid 7 of 7 candidates;
- the latest progress artifact reports `selected_pair_exists=true`.

The process then crashed with exit code `139` before writing a refreshed
`kernel_tuning.json`. The stderr blocker contained LLVM/XLA CPU compile-memory
messages including `Cannot allocate memory`, `releaseMappedMemory failed`, and
defunct JIT/resource-tracker messages.

Latest artifact state after the crash:

- `kernel_tuning.json` is still the stale Phase 6V artifact:
  file SHA-256
  `5bb6948c2bbc90038818bf98c4713d6d99983d593bd576c552810bbe622d482d`;
- public progress SHA-256:
  `1da089a95ba23a60187271f6affde31c32deb6a913b050c3c3fd1088baa70bd2`;
- private events SHA-256:
  `2e46ad180748caf7ad64d4c796d2016746036642cdd1f5065e59dc691adbef74`.

Decision: `BLOCKED_XLA_CPU_LLVM_COMPILE_MEMORY`.

Phase 7 remains blocked because Phase 6 still has no refreshed passed artifact
and no final kernel payload/hash. The next justified action is Phase 6X:
reduce redundant fixed-mass joint-grid XLA compile contracts while preserving
`jit_compile=true`, deterministic tuning, and all pass/fail gates.

## Phase 6X Fixed-Mass Compile-Reuse Closure

Phase 6X changed the fixed-mass joint-grid implementation to reuse one
dynamic-leapfrog reusable-runner cache across initial, edge-repair, and
final-local grid rounds. This repair was scoped to XLA compile pressure only:
it did not change the LGSSM fixture, prior, target, mass artifact, candidate
generation, acceptance screens, `use_xla=true` requirement, or Phase 6 pass
criterion.

Local focused checks passed before the serious rerun:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: `18 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `68 passed, 2 warnings`.

```text
git diff --check -- bayesfilter/inference/hmc_kernel_tuning.py tests/test_hmc_kernel_tuning_fixed_mass_step.py docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md
```

Result: passed.

```text
rg -n "GradientTape|batch_jacobian|tape\.|jit_compile\s*=\s*False|jit_compile=False" bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
```

Result: no matches.

The serious Phase 6 rerun used:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6x python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning
```

The run made meaningful progress beyond the Phase 6W crash point:

- host XLA compilation was observed;
- attempt 0 completed the fixed-mass initial grid and moved to attempt 1;
- attempt 1 completed the fixed-mass initial grid;
- a selected pair existed;
- the final-local grid completed 3 of 3 candidates and selected a final-local
  pair;
- `last_completed_stage` became `fixed_mass_step_complete`;
- the run entered `trajectory_candidate_call_start`.

The process then aborted with exit code `134` before writing a refreshed
`kernel_tuning.json`. The stderr blocker included:

- `LLVM compilation error: Cannot allocate memory`;
- `allocateMappedMemory failed with error: Cannot allocate memory`;
- `LLVM ERROR: Unable to allocate section memory!`.

Latest artifact state after the Phase 6X abort:

- `kernel_tuning.json` remains stale from Phase 6V:
  `5bb6948c2bbc90038818bf98c4713d6d99983d593bd576c552810bbe622d482d`;
- public progress SHA-256:
  `c701cb887c7c7b8c452938d5e05caac7fe1eecdc2b07a2b90ebdef1332663563`;
- public result SHA-256:
  `3909d89b4f7cfbcb30899aee26f005cdc744c4b34ddf446826e3ab3268db4a28`;
- private event SHA-256:
  `e2473e79fbd6c99567c1d999cff006238df15d80aec07ece62cb1af1d6b198e6`.

Latest progress artifact values:

- `current_stage=trajectory_candidate_call_start`;
- `last_completed_stage=fixed_mass_step_complete`;
- `phase7_last_attempt_index=1`;
- latest private summary reports `round_kind=final_local`,
  `candidate_completed_count=3`, `candidate_pass_count=2`,
  `candidate_hard_veto_count=0`, and `selected_pair_exists=true`;
- the trajectory call start reports `route_category=reusable_runner`,
  `candidate_count=1`, `candidate_index=0`, `num_results=500`,
  and `num_burnin_steps=125`.

Decision: `BLOCKED_XLA_CPU_LLVM_COMPILE_MEMORY_TRAJECTORY_STAGE`.

Phase 7 remains blocked because Phase 6 still has no refreshed passed artifact
and no final kernel payload/hash. The next justified action is Phase 6Y:
reuse the compatible fixed-mass reusable-runner cache at the trajectory
selected-pair handoff boundary while preserving `jit_compile=true`,
deterministic tuning, and all pass/fail gates.

## Phase 6Y Trajectory Compile-Reuse Closure

Phase 6Y threaded a private fixed-mass reusable-runner cache handoff into the
frozen-step trajectory selected-pair stage. The handoff is intentionally private
and is excluded from public payloads and artifact hashes. This repair changed
only XLA compile-cache lifetime at a compatible static-contract boundary; it
did not change the LGSSM target, prior, fixture, mass artifact, candidate
generation, acceptance bands, or Phase 6 pass criterion.

Claude review for the Phase 6Y subplan was attempted but rejected by the
approval layer as an external data-exfiltration risk. No workaround was
attempted. A local Codex substitute review was used instead and is weaker than
Claude review. The first substitute review required a negative
static-contract-mismatch cache test and explicit CPU-hidden manifest wording;
the patched subplan then received local substitute `VERDICT: AGREE`.

Local focused checks passed before the serious rerun:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: `20 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `70 passed, 2 warnings`.

```text
git diff --check -- bayesfilter/inference/hmc_kernel_tuning.py tests/test_hmc_kernel_tuning_fixed_mass_step.py docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md
```

Result: passed.

```text
python -m py_compile bayesfilter/inference/hmc_kernel_tuning.py tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: passed.

```text
rg -n "GradientTape|batch_jacobian|tape\.|jit_compile\s*=\s*False|jit_compile=False" bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
```

Result: no matches.

The serious Phase 6Y rerun used:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6y python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning
```

This intentionally hid GPU devices with `CUDA_VISIBLE_DEVICES=-1`; it was an
HMC CPU-hidden sample-generation / XLA compile-validation run, not NeuTra GPU
training and not GPU-readiness evidence.

The rerun made meaningful progress beyond the Phase 6X trajectory crash point:

- host XLA compilation was observed;
- attempt 0 completed fixed-mass 7 of 7 candidates with no hard veto and moved
  to attempt 1;
- attempt 1 completed fixed-mass initial grid 7 of 7 candidates;
- attempt 1 final-local grid completed 3 of 3 candidates, with 2 viable;
- selected fixed pair:
  `num_leapfrog_steps=4`, `step_size=0.6272447304363297`,
  `screen_acceptance_rate=0.686`,
  `trajectory_length=2.5089789217453187`;
- frozen-step trajectory completed and passed:
  `classification=passed_screen`, `screen_acceptance_rate=0.726`,
  `hard_vetoes=[]`;
- trajectory round selected
  `selected_trajectory_hash=46b34400e68b6d18bed98a496d90ddf0f1d463055ab65533c24545f6f4b79cf2`;
- progress reached `current_stage=verification_start` and
  `last_completed_stage=trajectory_complete`.

The process then aborted with exit code `134` before writing a refreshed
`kernel_tuning.json`. The stderr signature included:

- `allocateMappedMemory failed with error: Cannot allocate memory`;
- `Failed to satisfy suballocation request`;
- `LLVM ERROR: Unable to allocate section memory!`;
- `LLVM compilation error: Cannot allocate memory`.

Latest artifact state after the Phase 6Y abort:

- `kernel_tuning.json` remains stale from Phase 6V:
  `5bb6948c2bbc90038818bf98c4713d6d99983d593bd576c552810bbe622d482d`;
- stale `kernel_tuning.json` artifact hash:
  `sha256:8b3367c52619957080a6c28e51262cfed537b05c2d6d2d6b76a151d3da355484`;
- public progress SHA-256:
  `8a97fb73ee83c52f854f60e3eb2f1216d7e9abbe26f565e25c98e7560d2cdbf6`;
- public result SHA-256:
  `3909d89b4f7cfbcb30899aee26f005cdc744c4b34ddf446826e3ab3268db4a28`;
- private event SHA-256:
  `782d8120e41fb300604070847437222a45392d4b2d2dd2288f04e127764a75ed`.

Decision:
`BLOCKED_XLA_CPU_LLVM_COMPILE_MEMORY_VERIFICATION_STAGE`.

Phase 7 burn-in and retained sampling remain blocked. Phase 6 still has no
refreshed `passed=true` artifact and no final kernel payload/hash. The next
justified action is Phase 6Z: reduce the final verification XLA compile
contract by using a deterministic fixed-size verification chunk cap while
preserving the total verification budget, minimum retained evidence threshold,
`jit_compile=true`, CPU-hidden policy, and all hard vetoes.

## Phase 6Z Close Record: Verification Chunk Compile-Memory Repair

Status: `BLOCKED`

Timestamp: `2026-07-09T17:25:53Z`

Implemented repair:

- added deterministic `verification_chunk_max_results` threading from
  `HMCKernelTuningConfig` to the Phase 7 final sequential R-hat verifier;
- added `verification_min_retained_results_for_pass` so a small first XLA chunk
  cannot promote the final kernel before the minimum retained evidence count;
- preserved the budget-policy-owned total verification `max_results`;
- kept `jit_compile=true` / XLA execution and CPU-hidden HMC sample generation;
- did not use `GradientTape` in the runtime path and did not run a
  `jit_compile=false` fallback.

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m py_compile bayesfilter/inference/hmc.py bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
```

Result: passed.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_fixed_size_chunk_runner.py
```

Result: `17 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest --import-mode=importlib -q tests/test_hmc_kernel_tuning_outer_loop.py
```

Result: `66 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest --import-mode=importlib -q tests/test_hmc_kernel_tuning_public_api.py
```

Result: `34 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `11 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest --import-mode=importlib -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_hmc_kernel_tuning_outer_loop.py tests/test_hmc_kernel_tuning_public_api.py tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `170 passed, 2 warnings`.

```text
git diff --check -- bayesfilter/inference/hmc.py bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py docs/benchmarks/configs/multidim_lgssm_serious_hmc_tuning_2026_07_09.json tests/test_hmc_fixed_size_chunk_runner.py tests/test_hmc_kernel_tuning_outer_loop.py tests/test_hmc_kernel_tuning_public_api.py tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: passed.

```text
rg -n "GradientTape|batch_jacobian|tape\.|jit_compile\s*=\s*False|jit_compile=False" bayesfilter/inference/hmc.py bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
```

Result: no matches.

Serious Phase 6Z rerun:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6z python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning
```

This intentionally hid GPU devices with `CUDA_VISIBLE_DEVICES=-1`; it was an
HMC CPU-hidden sample-generation / XLA compile-validation run, not NeuTra GPU
training and not GPU-readiness evidence.

Observed progress:

- host XLA compilation was observed;
- attempt 0 completed fixed-mass initial grid 7 of 7 candidates, selected an
  initial handoff, and moved to attempt 1;
- attempt 1 windowed mass completed and passed;
- attempt 1 fixed-mass initial grid completed 7 of 7 candidates;
- attempt 1 fixed-mass final-local grid completed 3 of 3 candidates and
  selected `num_leapfrog_steps=4`;
- attempt 1 frozen-step trajectory completed and passed:
  `classification=passed_screen`, `diagnostic_role=trajectory_handoff_promotion_only`;
- latest private event before abort:
  `frozen_step_trajectory_round_selected`, `final_status=passed`,
  `num_leapfrog_steps=4`,
  `event_hash=8d6f2afbc61cb5836cc0bbceb51cb6d2bb7a5023abef511802b3a94a5115574c`;
- public progress reached `current_stage=verification_start` for
  `attempt_index=1` and `budget=2000`.

The process aborted with exit code `134` before writing a refreshed
`kernel_tuning.json`. The stderr signature again included:

- `LLVM compilation error: Cannot allocate memory`;
- `releaseMappedMemory failed with error: Cannot allocate memory`;
- `allocateMappedMemory failed with error: Cannot allocate memory`;
- `Failed to satisfy suballocation request`;
- `LLVM ERROR: Unable to allocate section memory!`;
- `JITDylib <xla_jit_dylib_...> is defunct`.

Latest artifact state after the Phase 6Z abort:

- `kernel_tuning.json` remains stale from Phase 6V:
  `5bb6948c2bbc90038818bf98c4713d6d99983d593bd576c552810bbe622d482d`;
- stale `kernel_tuning.json` artifact hash:
  `sha256:8b3367c52619957080a6c28e51262cfed537b05c2d6d2d6b76a151d3da355484`;
- public result SHA-256:
  `3909d89b4f7cfbcb30899aee26f005cdc744c4b34ddf446826e3ab3268db4a28`;
- public progress SHA-256:
  `aa62e50b2c5f1e7a85f3e04ee1479dcadf04c7e763dc620a793add1a12343862`;
- private event SHA-256:
  `24321079dc3cc5302038665dbb6880c18db244a2298f9724109f47b667998cb7`.

Decision:
`BLOCKED_XLA_CPU_LLVM_COMPILE_MEMORY_VERIFICATION_STAGE`.

Phase 6Z did not close Phase 6. It moved the live run through the passed
trajectory handoff and into final verification, but the XLA CPU LLVM memory
failure persists at final verification compile time. Phase 7 burn-in and
retained sampling remain blocked: there is still no refreshed `passed=true`
kernel-tuning artifact and no final kernel payload/hash.

Next justified action:

- Do not start Phase 7.
- Draft a Phase 6AA subplan focused on final-verification compile-memory
  reduction that preserves `jit_compile=true`, total retained evidence,
  minimum retained pass gate, hard vetoes, and CPU-hidden HMC execution.
- Candidate repair directions to evaluate in that subplan include reducing
  final verification XLA static graph size further, lowering verification
  chain count only with an explicit evidence-preserving replacement, or
  changing final verification to a sequential single-chain/chain-loop XLA
  contract that still yields valid R-hat evidence across chains.

## Phase 6AA Final Closure: SVD Score Wiring Retry

Status: `PASSED_KERNEL_HANDOFF_PHASE7_APPROVAL_REQUIRED`

Phase 6AA demoted the active QR derivative route and wired the serious
multidimensional LGSSM target to the SVD/eigh graph-status score backend. It
then refreshed both the XLA value/score gate and the deterministic kernel
tuning artifact under `CUDA_VISIBLE_DEVICES=-1`, `jit_compile=true`, and no
runtime `GradientTape`.

Result:

- `xla_compile_gate.json`: `passed=true`, `jit_compile=true`,
  `finite_value=true`, `finite_score=true`, `target_status_valid=true`.
- `kernel_tuning.json`: `passed=true`, `final_status=passed`,
  `diagnostic_role=fresh_fixed_kernel_verification_passed`,
  `xla_confirmed=true`, `jit_compile=true`, `hard_vetoes=[]`,
  `vetoes=[]`.
- `jit_compile_false_runtime_executed=false`.
- `runtime_autodiff_tape_executed=false`.
- final kernel hash:
  `8ddf25a3b572893e19e814fad5ca5b6150718e36f760c159b47db1231d92ffff`.
- Phase 7 public handoff kernel hash:
  `391558a9b5f4cdc1b9dff9a5e9bceba668dedded7298c1d8c76daea42f42039a`.
- verification acceptance rate: `0.71325`.

Artifact hashes:

- XLA gate artifact hash:
  `sha256:8941b369f6280ebc3c124220a9bab21f6889228deb92121d63f2fefba3ea6842`.
- XLA gate file SHA-256:
  `8c54c60d7d51cf5ee3d04dfa32df036fc9616c0647e399813ca846e3812e0343`.
- Kernel artifact hash:
  `sha256:f8c94073b60a6458538537317e49ed683ad0c94b525cafc77cc4d01822badaa2`.
- Kernel file SHA-256:
  `ee9f9308d055cb2482b1fbc2661fc2bd7fa21d7128a51902f49f237c98bddefa`.
- Public result file SHA-256:
  `2fc7a40c022465625e7855cda679a86a6735f7657b46f25db706d37bfef17d23`.
- Public progress file SHA-256:
  `db03bb242661889981c5898e34cc46995fa15f1235c558331aee76bd731d30bc`.
- Private event log SHA-256:
  `2e7499befaca4450d2109f8b55247a840682ee91949741182920014cb65f60fc`.

Close artifact:

- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6aa-svd-score-wiring-retry-result-2026-07-10.md`

Decision:

Phase 6 is closed at the kernel-handoff level. Phase 7 burn-in and retained
sampling have not run and require explicit user approval. This closure does
not establish posterior convergence, posterior recovery, sampler superiority,
production/default readiness, GPU readiness, DSGE readiness, or a scientific
claim.
