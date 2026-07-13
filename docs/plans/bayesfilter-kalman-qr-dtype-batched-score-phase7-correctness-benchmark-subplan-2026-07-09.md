# Phase 7 Subplan: Correctness And Benchmark Ladder

Date: 2026-07-09

## Phase Objective

Run governed CPU/GPU XLA correctness and descriptive timing for scalar vs
batch-native analytical score and autodiff row-loop references across dtype,
batch size, parameter count, and dimensions.  The batch-static value-gradient
autodiff route is diagnostic-only in this phase until it returns finite
gradients for the benchmark fixture.

## Entry Conditions Inherited From Previous Phase

- Batch-native analytical score passes CPU/XLA correctness tests.
- Benchmark harness can record requested/observed dtype.
- Phase 6 result records that the existing benchmark harness still needs a
  batch-native analytical score arm before any ladder can answer the Phase 7
  question.
- GPU runtime approval and trusted provenance are available before GPU runs.

## Required Artifacts

- Updated `scripts/benchmark_kalman_qr_parameter_count_scaling.py` with a
  batch-native analytical score arm, scalar analytical row-loop comparator,
  autodiff row-loop comparator for independent parameter proposals, and a
  non-timed batch-static autodiff diagnostic.
- CPU/XLA benchmark JSON/Markdown artifacts for FP32 and FP64.
- GPU/XLA benchmark JSON/Markdown artifacts for FP32 and FP64 if approved.
- Logs under `docs/benchmarks/logs/`.
- Phase 7 result and refreshed Phase 8 closeout subplan.

## Required Checks, Tests, And Reviews

Commands must be refreshed before execution with exact grids.  Minimum smoke:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_batched_analytical_score_tf.py tests/test_linear_qr_dtype_contracts.py
CUDA_VISIBLE_DEVICES=-1 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 --parameter-counts 50 --timesteps 8 --repeats 1 --batch-size 2 --device cpu --jit-compile --dtype float32 --output-json docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.md
```

GPU benchmark commands require trusted/approved GPU execution and artifacts
must record device, JIT, dtype, TF32 mode, and trust basis.

The first executable Phase 7 step is harness refresh plus CPU-hidden FP32 smoke
only.  Full CPU/GPU ladders require a later refreshed exact-grid command block
after the smoke artifact proves all three benchmark arms and artifact fields
are present.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | How do observed runtimes vary descriptively across dtype, device, batch size, parameter count, and dimension after correctness gates pass? |
| Baseline/comparator | Scalar analytical row loop, batch-native analytical score, and scalar-value autodiff row-loop reference.  Batch-static autodiff value-gradient is diagnostic-only unless it returns finite gradients. |
| Primary criterion | All benchmark rows pass correctness/dtype checks and write complete artifacts. |
| Veto diagnostics | Correctness failure, dtype mismatch, missing device provenance, nonfinite output, unapproved GPU run, or benchmark artifact missing compile/warm split. |
| Explanatory diagnostics | Compile+first-call, warm-start, repeated warm medians, device placement, and TF32 flag. |
| Not concluded | Statistical speed ranking or production readiness; timings are descriptive unless replicated with uncertainty. |
| Artifact | Phase 7 result and benchmark artifacts. |

## Forbidden Claims And Actions

- Do not rank methods statistically from one-repeat timing.
- Do not claim GPU production readiness from a smoke grid.
- Do not hide GPU for GPU artifacts or expose GPU for CPU-hidden artifacts.
- Do not run full CPU/GPU ladders until the batch-native harness smoke has
  passed and the exact grids are refreshed in this subplan or a result handoff.

## Exact Next-Phase Handoff Conditions

Advance to Phase 8 only if artifacts are complete, correctness and dtype gates
pass, and timing interpretation remains descriptive.

## Stop Conditions

Stop if benchmark artifacts cannot preserve dtype/device provenance or if GPU
runtime is unavailable/unapproved.
