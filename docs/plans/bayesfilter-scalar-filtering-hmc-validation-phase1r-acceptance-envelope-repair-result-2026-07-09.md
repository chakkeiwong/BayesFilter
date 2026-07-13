# Phase 1R Result: Acceptance-Envelope Repair Screen

Date: 2026-07-09
Status: `PASSED_FINITE_ACCEPTANCE_SCREEN_WITH_BOUNDARIES`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 1R passes the longer same-kernel finite/acceptance repair screen | Passed: all three seeds had finite samples, finite target log probabilities, finite log-accept ratios, no runtime errors, and acceptance in `(0.05, 0.99)` | No Phase 1R vetoes; native divergence telemetry was unavailable, so the native positive-divergence veto was unassessable and no zero-divergence claim is made | This is still a short CPU-hidden fixed-kernel screen and does not test posterior/reference agreement | Refresh and review Phase 2 local reference-agreement subplan | No HMC readiness, convergence, posterior correctness, zero-divergence claim, sampler superiority, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered: increasing retained draws from 16 to 64 while keeping kernel, burn-in, trajectory, and seeds fixed removed the Phase 1 all-accepted seed. |
| Baseline/comparator | Phase 1 failing artifact. |
| Primary criterion | Passed. |
| Veto diagnostics | None. |
| Explanatory diagnostics | Acceptance rates `[0.921875, 0.734375, 0.578125]`; log-accept max abs `[2.588266759698262, 128.65675528526683, 372.85150587307623]`; max abs `u` `[7.3994829375628735, 7.858316919728933, 10.427120225910855]`. |
| Not concluded | No HMC convergence, posterior correctness, zero divergences, sampler superiority, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase1r_longer_same_kernel.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py`

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `f297b103303c64019302ed5d9b9aaf2c8f919b64` |
| Git dirty status | Dirty; artifact recorded current planned edits and unrelated dirty workspace files |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 720 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | CPU-hidden debug/reference exception with `CUDA_VISIBLE_DEVICES=-1`; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seeds | `(20260709, 6101)`, `(20260709, 6102)`, `(20260709, 6103)` |
| Wall time | `320.83368024101947` seconds |
| Plan/result paths | Master, Phase 1R subplan, and this result file recorded in JSON |

## Native-Divergence Telemetry

All three seeds reported native divergence as `not_exposed_by_kernel`.
Unavailable native divergence was not treated as zero divergences.  The artifact
records:

- `zero_divergence_claim_made: false`
- `unavailable_native_divergence_is_zero_divergence: false`
- `log_accept_threshold_used_as_native_divergence: false`

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Passed for the Phase 1R finite/acceptance screen. |
| Statistically supported ranking | None; no method comparison and no uncertainty interval. |
| Descriptive-only differences | Acceptance, log-accept tails, target-log-prob ranges, sample ranges, and runtime. |
| Default readiness | Not assessed. |
| Next evidence needed | Phase 2 local reference agreement before any posterior/reference interpretation. |

## Checks

| Check | Status |
| --- | --- |
| `git diff --check` before runtime | Passed |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase1.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase1r.py` | Passed: `8 passed` |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_common_inference_runtime_contracts.py::test_hmc_screen_does_not_use_log_accept_threshold_as_native_divergence tests/test_common_inference_runtime_contracts.py::test_hmc_screen_keeps_unavailable_diagnostics_from_passing_as_zero` | Passed: `2 passed, 2 warnings` |
| Phase 1R runtime command | Exited `0`; artifact decision passed |

## Final Nonclaims

- No HMC readiness.
- No HMC convergence.
- No posterior correctness.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
