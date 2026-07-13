# Phase 7 Result: Correctness And Benchmark Harness Smoke

Date: 2026-07-09

## Decision

`PHASE_7_SMOKE_GATE_PASSED_FULL_LADDER_DEFERRED`

The benchmark harness now exposes a batch-native analytical score arm,
compiled scalar analytical row-loop comparator, and compiled autodiff row-loop
comparator.  The CPU-hidden FP32 XLA smoke passed dtype, shape, finite-output,
and parity gates for one small row.  Full CPU/GPU ladders were not launched by
this result because the subplan required the smoke artifact first, then a
refreshed exact-grid command block and GPU approval/provenance before GPU
runtime.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the benchmark harness time the batch-native analytical score path with honest scalar analytical and autodiff comparators while preserving dtype/device/JIT provenance? |
| Baseline/comparator | Batch-native analytical score, scalar analytical row loop, and scalar-value autodiff row loop across independent parameter proposals. |
| Primary criterion | CPU-hidden FP32 XLA smoke writes complete JSON/Markdown artifacts and all applicable rows pass dtype, shape, finite-output, and parity checks. |
| Veto diagnostics | Correctness failure, dtype mismatch, missing device provenance, nonfinite timed output, unapproved GPU run, or missing compile/warm split. |
| Explanatory diagnostics | Compile+first-call time, warm-start time, repeated warm median, TF32 flag, device placement, and diagnostic-only batch-static autodiff probe. |
| Not concluded | Statistical speed ranking, GPU performance, production/default readiness, HMC readiness, posterior correctness, or scientific validity. |
| Artifacts | Benchmark JSON/Markdown/logs and this phase result. |

## Harness Changes

- Updated `scripts/benchmark_kalman_qr_parameter_count_scaling.py`.
- Added `--batch-size`, manifest recording, isolated-row propagation, and input
  shape recording for `[B, P]` parameter batches.
- Added timed methods:
  - `batch_native_analytical_qr_score`
  - `scalar_analytical_row_loop`
  - `autodiff_row_loop_qr_score`
- Added non-timed diagnostic:
  - `batched_static_autodiff_probe`

## Repair Note

The first smoke attempt used a batched-static value-gradient autodiff
comparator.  That route produced finite values but `nan` scores for the
synthetic lower-triangular benchmark fixture because TensorFlow returned an
unconnected gradient for the batch-static QR value path.  This was treated as a
harness/comparator blocker, not as evidence against the batch-native
analytical score.

The repaired harness uses a scalar QR value autodiff row-loop comparator inside
one compiled function.  The batch-static autodiff route remains diagnostic-only
and is recorded in the JSON artifact as not timed.

## Checks Run

| Check | Result |
| --- | --- |
| `python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py` | passed |
| `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_batched_analytical_score_tf.py tests/test_linear_qr_dtype_contracts.py` | `19 passed` |
| CPU-hidden FP32 XLA smoke command from the subplan, with `--batch-size 2` | passed |
| `git diff --check -- scripts docs/benchmarks docs/plans bayesfilter/linear tests` | passed |

## Smoke Artifact Summary

Artifacts:

- `docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.json`
- `docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.md`
- `docs/benchmarks/logs/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.log`
- `docs/benchmarks/logs/kalman_qr_phase7_cpu_hidden_tests_2026-07-09.log`

Structured result:

| Field | Value |
| --- | --- |
| Device | `/CPU:0` with `CUDA_VISIBLE_DEVICES=-1` |
| JIT | `True` |
| Requested dtype | `float32` |
| Batch size | `2` |
| Dimension | `(10, 10)` |
| Parameter count | `50` |
| Timesteps | `8` |
| Applicable rows | `1` |
| Parity passed | `True` |
| Batch/autodiff value max abs residual | `0.0` |
| Batch/autodiff score max abs residual | `4.6566128730773926e-09` |
| Batch/scalar analytical score max abs residual | `3.725290298461914e-09` |
| Batch-static autodiff diagnostic finite | `False` |

Warm-call timings are descriptive only:

| Method | Warm median seconds |
| --- | ---: |
| `batch_native_analytical_qr_score` | `0.049120351031888276` |
| `scalar_analytical_row_loop` | `0.0712403489742428` |
| `autodiff_row_loop_qr_score` | `0.008719386998564005` |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| `PHASE_7_SMOKE_GATE_PASSED_FULL_LADDER_DEFERRED` | `passed_for_cpu_hidden_smoke` | `no timed-output veto fired`; batch-static autodiff diagnostic remains nonfinite and not timed | One-row smoke, single repeat, CPU-hidden reference/debug run only | Refresh exact CPU/GPU ladder commands, obtain/confirm GPU approval/provenance, then run descriptive ladder artifacts | No speed ranking, GPU/default readiness, HMC readiness, posterior correctness, or scientific validity |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the timed smoke row. |
| Statistically supported ranking | Not assessed. |
| Descriptive-only differences | Compile+first, warm-start, warm medians, and ratios from one smoke row only. |
| Default-readiness | Not assessed. |
| Next evidence needed | Full CPU/GPU ladder with exact grids, logs, device provenance, and replicated uncertainty if making any ranking. |

## Handoff

Phase 8 closeout may summarize this as a passed smoke gate and deferred full
ladder.  If continuing Phase 7 instead of closing, the next subplan update must
state exact CPU/GPU grid commands and must preserve the GPU approval/provenance
boundary before launching GPU runtime.
