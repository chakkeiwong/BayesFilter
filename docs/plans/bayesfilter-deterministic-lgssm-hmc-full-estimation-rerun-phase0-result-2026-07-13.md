# Deterministic LGSSM Full Estimation Rerun Phase 0 Result

Date: 2026-07-13

Status: `PASS_PHASE0_TO_PREFLIGHT_AND_FRESH_FIXTURE`

## Decision

Phase 0 closed the integration blocker. A fresh end-to-end run no longer needs
the historical migration certificate, adoption record, fixed transition ID,
fixed serious execution ID, fixed artifact filenames, or historical academic
campaign wrapper.

The implementation adds an isolated V3 Phase 7 configuration path while
preserving the V1/V2 validators and tests. No HMC experiment or fixture was run
during Phase 0.

## Implemented Artifacts

- `docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json`
- `scripts/build_hmc_full_estimation_preflight_manifest.py`
- `scripts/build_hmc_full_estimation_phase7_config.py`
- `scripts/run_hmc_full_estimation_campaign.py`
- `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py` V3 support
- `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`
  explicit Phase 7 config, no-overwrite, and final-recovery support
- `tests/test_hmc_full_estimation_campaign.py`

## Closed Problems

| Problem | Resolution |
| --- | --- |
| Historical fixed root seed | V3 binds serious seed `(20260713, 701)` while tuning remains `(20260709, 501)`. |
| Historical fixed filenames | V3 uses eight explicit governed source references under the new run root. |
| Certificate/adoption dependency | V3 has no certificate, adoption, or approval artifact input. |
| Hard-coded academic identities | The fresh builder derives transition, serious execution, smoke execution, provenance, and complete tuning-payload identities from the new replay. |
| Caller config ignored | The driver now requires and forwards `--phase7-config`. |
| Missing recovery stage | Recovery independently verifies the NPZ, recomputes modern diagnostics, reports raw-draw mean MCSE, and evaluates all 18 truth-recovery rows. |
| Artifact overwrite risk | Fresh stages, builder, preflight, smoke, and campaign terminal files use no-overwrite checks. |
| Tuning diagnostic ambiguity | V3 requires the exact shared modern R-hat definition, both component maxima, their maximum, at least 1000 verifier draws, acceptance in `[0.65, 0.75]`, and no hard veto. |
| Tuning/serious boundary ambiguity | Kernel tuning emits `final_kernel_requires_serious_sampling_pass=true`, and V3 requires it. |

## Verification

Command:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  pytest -q tests/test_hmc_full_estimation_campaign.py \
  tests/test_deterministic_lgssm_hmc_tuning_driver.py \
  tests/test_deterministic_lgssm_hmc_phase7_tf.py \
  tests/test_hmc_convergence.py tests/test_hmc_fixed_size_chunk_runner.py
```

Result: `75 passed, 2 warnings`.

Additional checks:

- Python compilation: passed.
- forbidden `jit_compile=False`, `--no-jit`, and runtime `GradientTape` scan:
  passed.
- V1/V2 historical controller regression tests: passed.
- V3 sorted-JSON round trip, worker request, builder, recovery, no-overwrite,
  and campaign-root tests: passed.

Frozen source SHA-256 values at close:

| Path | SHA-256 |
| --- | --- |
| `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py` | `bdabced432c6f02f86cb96da3de355bdfe7e6f746049d1ca08065cbe59fbcd71` before the final boundary-field patch; the execution preflight records the authoritative post-patch hash. |
| `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py` | `7a0979297a1010c6607cc2706ad16efb7ebe46338902dd92cde2265d7128fcd3` before the final boundary-field/MCSE patch; the execution preflight records the authoritative post-patch hash. |
| `scripts/build_hmc_full_estimation_phase7_config.py` | `19c2ad38f16c71c822cdf49bd4279c51b058129e920706e365ae27d11ea4b293` before the exclusive-write patch; the execution preflight records the authoritative post-patch hash. |
| `scripts/build_hmc_full_estimation_preflight_manifest.py` | The execution preflight records its authoritative final hash. |
| `scripts/run_hmc_full_estimation_campaign.py` | The execution preflight records its authoritative final hash. |
| `docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json` | `cc42fbfa61fb46f5e81299944facd44bcfca4ff5443442df3ce09af9b4b3c582` |

Git commit baseline: `d269f5bbd8531b878d4f25897a357fbc8f172488`.

## Review Record

Claude health and one-path read probes succeeded. Claude's bounded review found
`FEASIBILITY_MISSING_HARDCODED_IDENTITY_REPAIR`, matching the local audit. The
plan was revised before implementation. A new post-revision Claude call was
rejected by the environment's external-disclosure boundary and was not retried.
Therefore the post-revision verdict is local, not falsely attributed to Claude.

Local skeptical review verdict: `PASS_AFTER_V3_INTEGRATION_AND_TESTS`.

## Evidence Classification

| Ledger | Status |
| --- | --- |
| Engineering correctness | Phase 0 focused checks pass. |
| Numerical/sampler validity | Not tested in Phase 0. |
| Scientific interpretation | No result; no experiment ran. |

## Handoff

The next phase must write and verify the immutable preflight manifest before
creating the fixture. After that, execute fixture, XLA target validation,
geometry/mass, corrected tuning, actual-target smoke, independent-seed serious
sampling, and recovery until a declared continuation veto fires or closeout is
complete.
