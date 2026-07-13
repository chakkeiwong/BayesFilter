# Phase 6AA Result: SVD Score Wiring Retry

Date: 2026-07-10

## Scope

This note records the Phase 6AA retry of the deterministic multidimensional
LGSSM HMC kernel-tuning gate after demoting the active QR derivative route and
wiring the serious target to the SVD/eigh graph-status score backend.

This is a kernel handoff result only. It is not posterior convergence evidence,
posterior recovery evidence, sampler superiority evidence, production/default
readiness, GPU readiness, DSGE readiness, NeuTra training evidence, or a
scientific claim.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | After replacing the active QR derivative route with SVD/eigh graph-status scoring, can the XLA value/score gate and Phase 6 kernel-tuning gate refresh without the stale QR compile path? |
| Baseline/comparator | Phase 6Z reached final verification, then aborted with XLA CPU LLVM allocation errors before writing a refreshed `kernel_tuning.json`. |
| Primary criterion | Refreshed `xla_compile_gate.json` has `passed=true`, `jit_compile=true`, finite value/score, and valid SVD status; refreshed `kernel_tuning.json` has `passed=true`, confirmed XLA, no hard vetoes, and final kernel payload/hash. |
| Veto diagnostics | `jit_compile=false`, runtime `GradientTape`, active QR derivative route in the serious target/driver, invalid SVD status, nonfinite value/score, manual tuning, target/prior/fixture changes outside the SVD wiring repair, process abort without blocker artifact, or Phase 7 sampling. |
| Explanatory diagnostics | XLA HLO size, SVD status telemetry, kernel-tuning repair-trigger history, final fixed-kernel verification acceptance, public/private artifact hashes. |
| Not concluded | Posterior convergence, posterior recovery, sampler validity, sampler ranking, production/default readiness, GPU readiness, DSGE readiness, or scientific validity. |
| Artifact preserving result | This result note plus refreshed `xla_compile_gate.json`, `kernel_tuning.json`, public tuning result/progress JSON, and private event-log hash. |

## Result

`PASSED_KERNEL_HANDOFF_PHASE7_APPROVAL_REQUIRED`.

The refreshed XLA score gate passed:

- `passed=true`
- `jit_compile=true`
- `finite_value=true`
- `finite_score=true`
- `target_status_valid=true`
- SVD/eigh telemetry: `status_code=0`,
  `valid_pre_regularized_score=true`, `floor_count_value=0`,
  `min_innovation_eigenvalue=0.04087247539561765`,
  `innovation_condition_estimate=3.989020834527715`
- XLA HLO byte count: `379528`
- artifact hash:
  `sha256:8941b369f6280ebc3c124220a9bab21f6889228deb92121d63f2fefba3ea6842`
- file SHA-256:
  `8c54c60d7d51cf5ee3d04dfa32df036fc9616c0647e399813ca846e3812e0343`

The refreshed kernel-tuning gate passed:

- `passed=true`
- `final_status=passed`
- `diagnostic_role=fresh_fixed_kernel_verification_passed`
- `xla_confirmed=true`
- `jit_compile=true`
- `jit_compile_false_runtime_executed=false`
- `runtime_autodiff_tape_executed=false`
- `hard_vetoes=[]`
- `vetoes=[]`
- final kernel hash:
  `8ddf25a3b572893e19e814fad5ca5b6150718e36f760c159b47db1231d92ffff`
- Phase 7 public handoff kernel hash:
  `391558a9b5f4cdc1b9dff9a5e9bceba668dedded7298c1d8c76daea42f42039a`
- verification acceptance rate: `0.71325`
- selected step hash:
  `ec7db59e51465eee95658167e1f7596e21d9ab0efdac11f54c2d397aa270ab40`
- selected trajectory hash:
  `6eaf7a563353b278a71dcfbe2515fda6d46c47ab2e38996b6b61fab1bbbd13b3`
- artifact hash:
  `sha256:f8c94073b60a6458538537317e49ed683ad0c94b525cafc77cc4d01822badaa2`
- file SHA-256:
  `ee9f9308d055cb2482b1fbc2661fc2bd7fa21d7128a51902f49f237c98bddefa`

The public tuning result also passed:

- status: `passed`
- result hash:
  `334326571db5ffcbfb51e078ebed3b0e0954dddb566c53bade612de887053e03`
- file SHA-256:
  `2fc7a40c022465625e7855cda679a86a6735f7657b46f25db706d37bfef17d23`
- active repair triggers: `[]`
- historical repair triggers:
  `acceptance_outside_pass_band_inside_repair_band`,
  `screen_nonfinite_proposal_mechanics_step_repair`,
  `joint_l_epsilon_no_viable_pair`,
  `phase5_fixed_mass_step_status:repair_or_retry`,
  `screen_acceptance_below_repair_band`

The public progress artifact ended at `current_stage=result_written`,
`last_completed_stage=loop_complete`, and `last_completed_substage=loop_complete`.
Its file SHA-256 is
`db03bb242661889981c5898e34cc46995fa15f1235c558331aee76bd731d30bc`.

The private event log file SHA-256 is
`2e7499befaca4450d2109f8b55247a840682ee91949741182920014cb65f60fc`.

## Commands And Checks

Static and focused checks completed before the serious rerun:

```text
python -m py_compile bayesfilter/testing/multidim_triangular_lgssm_tf.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
git diff --check <Phase 6AA touched files>
rg -n "GradientTape|batch_jacobian|tape\.|jit_compile\s*=\s*False|jit_compile=False" <active runtime files>
```

Result: passed; forbidden-token scans had no matches in the active runtime
files checked.

Focused tests:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_multidim_triangular_lgssm_tf.py
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_svd_linear_gaussian_score_tf.py
```

Results: `9 passed`, `11 passed`, and `15 passed`.

Serious XLA score-gate command:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-svd-retry python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage xla_score
```

Result: exited successfully and wrote the refreshed XLA gate artifact.

Serious kernel-tuning command:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6-svd-retry python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning
```

Result: exited with code `0` and wrote the refreshed kernel-tuning artifacts.
GPU devices were intentionally hidden with `CUDA_VISIBLE_DEVICES=-1`; this was
CPU-hidden HMC sample-generation and XLA compile-validation work, not GPU
readiness evidence.

## Decision Table

| Decision | Status |
| --- | --- |
| Phase 6AA XLA score gate | `PASSED` |
| Phase 6AA kernel handoff gate | `PASSED` |
| Primary criterion | Satisfied: refreshed `kernel_tuning.json` has `passed=true`, confirmed XLA, no hard vetoes, and final kernel payload/hash. |
| Veto diagnostics | No veto fired in the refreshed artifacts. No `jit_compile=false` runtime path, no runtime `GradientTape`, no Phase 7 sampling. |
| Main uncertainty | The kernel handoff has not yet been used for Phase 7 burn-in or retained sampling, so posterior convergence and recovery remain untested. |
| Next justified action | Stop at the Phase 7 approval boundary; if approved, execute the deterministic burn-in/sampling controller from the runbook. |
| What is not concluded | This does not establish posterior convergence, truth recovery, sampler superiority, production/default readiness, GPU readiness, DSGE readiness, or a scientific claim. |

## Handoff

Phase 6 is now closed at the kernel-handoff level. Phase 7 remains unexecuted
and requires explicit user approval because it performs long burn-in and
retained sample generation.
