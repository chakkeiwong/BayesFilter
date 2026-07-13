# Kalman QR Core/Batch/Dtype Grid Subplan

Date: 2026-07-09

## Phase Objective

Run the requested Kalman QR score benchmark grid with `T=120`, dimensions
`(10,10)`, `(20,20)`, `(30,30)`, parameter counts `50` and `150`, batch sizes
`1`, `4`, and `16`, JIT enabled, and:

- CPU runs at requested CPU thread/core settings `1`, `4`, and `16`;
- GPU runs for `float32` and `float64`.

## Entry Conditions

- Phase 7 CPU-hidden batched-score smoke passed after harness repair.
- Trusted TensorFlow GPU probe using
  `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` reports two logical GPUs.
- Benchmark harness supports `--batch-size`, `--dtype`, `--jit-compile`, and
  `--cpu-threads`.

## Required Artifacts

CPU artifacts, one JSON/Markdown/log set per requested CPU thread count and
batch size:

- `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads{1,4,16}_batch{1,4,16}_xla_2026-07-09.{json,md}`

GPU artifacts:

- `docs/benchmarks/kalman_qr_core_batch_grid_gpu_float{32,64}_batch{1,4,16}_xla_2026-07-09.{json,md}`

Logs under `docs/benchmarks/logs/` with matching basenames, plus this
subplan and a result/blocker note.

## Required Checks

Preflight:

```bash
python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py
git diff --check -- scripts docs/benchmarks docs/plans bayesfilter/linear tests
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -c "import json, tensorflow as tf; print(json.dumps({'physical_gpus': [d.name for d in tf.config.list_physical_devices('GPU')], 'logical_gpus': [d.name for d in tf.config.list_logical_devices('GPU')]}))"
```

CPU one-row thread-control preflight:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 --parameter-counts 50 --timesteps 8 --repeats 1 --batch-size 1 --device cpu --jit-compile --dtype float32 --cpu-threads 1 --output-json docs/benchmarks/kalman_qr_core_batch_grid_preflight_cpu_threads1_float32_2026-07-09.json --output-md docs/benchmarks/kalman_qr_core_batch_grid_preflight_cpu_threads1_float32_2026-07-09.md
```

GPU one-row preflight:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 --parameter-counts 50 --timesteps 8 --repeats 1 --batch-size 1 --device gpu --jit-compile --dtype float32 --output-json docs/benchmarks/kalman_qr_core_batch_grid_preflight_gpu_float32_2026-07-09.json --output-md docs/benchmarks/kalman_qr_core_batch_grid_preflight_gpu_float32_2026-07-09.md
```

Full CPU commands repeat the 3x2 dimension/parameter grid once per thread count
and batch size because the harness accepts one `--batch-size` per invocation.
Full GPU commands repeat the 3x2 grid once per dtype and batch size, if GPU
preflight passes.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | How do descriptive JIT warm-call runtimes vary across dimensions, parameter count, batch size, CPU thread count, and GPU dtype for the refreshed batched-score harness? |
| Baseline/comparator | `batch_native_analytical_qr_score`, `scalar_analytical_row_loop`, and `autodiff_row_loop_qr_score`. |
| Primary criterion | Every launched row passes dtype, shape, finite-output, and parity checks; artifacts record batch size, CPU thread manifest or GPU provenance, JIT, dtype, TF32 flag, compile+first, warm-start, and warm summaries. |
| Veto diagnostics | Row parity failure, nonfinite timed output, dtype mismatch, missing thread/device provenance, TensorFlow GPU invisibility, timeout without row artifact, or unsupported ranking claim. |
| Explanatory diagnostics | Warm medians, compile+first time, warm-start time, first-minus-warm, row subprocess metadata, TF32 flag, CPU thread manifest, and device placement. |
| Not concluded | Statistical speed ranking, universal superiority, production/default readiness, HMC readiness, posterior correctness, or scientific validity. |
| Artifact | JSON/Markdown/log artifacts plus result/blocker note. |

## Forbidden Claims And Actions

- Do not claim a statistically supported ranking from `repeats=1`.
- Do not call `--cpu-threads` a physical core pin; it records TensorFlow/env
  thread limits.
- Do not use the default sandboxed interpreter for GPU rows if it cannot see
  TensorFlow logical GPUs.
- Do not treat CPU-hidden results as GPU evidence.
- Do not treat the diagnostic batch-static autodiff probe as a timed arm.

## Stop Conditions

- Stop if preflight cannot record requested CPU thread settings.
- Stop if trusted TensorFlow GPU visibility fails.
- Stop if a row fails parity or emits nonfinite timed outputs.
- Stop if artifacts cannot preserve thread/device/dtype/JIT provenance.
- Stop if runtime exceeds visible execution feasibility; write a partial
  result with completed artifacts and exact resume commands.

## Exact Handoff Conditions

Write a result note when all launched artifacts complete or a stop condition
fires.  The result must include a decision table, inference-status table,
artifact list, and explicit nonclaims.
