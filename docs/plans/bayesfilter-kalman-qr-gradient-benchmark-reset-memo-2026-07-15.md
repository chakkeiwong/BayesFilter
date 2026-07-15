# Kalman QR Gradient Benchmark Reset Memo

Date: 2026-07-15

Status: `CLEAN_LANE_RESTART_READY`

Scope: LGSSM analytical/autodiff QR gradient XLA viability, GPU memory growth,
CPU batch/process scaling, and matched CPU/process/GPU benchmarking.

Containing commit: use the commit containing this memo. The shared worktree may
still show another active lane; do not treat unrelated dirty files as part of
this reset or revert them.

## Current Answer

The original memory and XLA viability blockers are repaired for the tested
benchmark path:

- TensorFlow GPU memory growth is required before logical-device
  initialization. The previous near-full-device reservation was an allocator
  policy bug, not live benchmark tensor demand.
- The complete GPU lattice used at most about `302.5 MiB` allocator peak; the
  final matched `(30,150,12,16)` analytical run used `131.1 MiB` peak.
- The analytical and autodiff native-batch QR paths compile and run with XLA
  under the retained CPU/GPU contracts. GPU XLA currently requires the
  benchmark-local `--xla_gpu_enable_triton_gemm=false` workaround.
- Native CPU `B=16` is numerically correct but exposes little CPU parallelism.
  `tf.map_fn` and `tf.vectorized_map` wrappers did not repair that execution
  shape; strict `vectorized_map` has no pfor rule for `Qr`.
- Sixteen pinned Python processes, each running XLA `B=1`, use the available
  physical cores effectively and are much faster for tested CPU throughput,
  at the cost of about `9.7 GiB` aggregate RSS and substantial cold startup.

## Retained Results

### Full analytical/autodiff lattice

- Result:
  `docs/plans/bayesfilter-kalman-qr-gradient-scaling-lattice-gpu0-result-2026-07-14.md`
- CPU evidence:
  `docs/benchmarks/kalman_qr_gradient_scaling_lattice_r4_2026-07-14/status.json`
- GPU evidence:
  `docs/benchmarks/kalman_qr_gradient_scaling_lattice_gpu0_r2_2026-07-14/status.json`
- Outcome: `15/15` schedules, `180/180` records, and `90/90`
  analytical/autodiff pairs passed. Lattice timings are descriptive.

### Controlled CPU architecture comparison

- Result:
  `docs/plans/bayesfilter-kalman-qr-cpu-throughput-comparison-result-2026-07-14.md`
- Evidence:
  `docs/benchmarks/kalman_qr_cpu_throughput_comparison_r3_2026-07-14/status.json`
- Outcome:
  `RECOMMEND_PERSISTENT_SHARDING_FOR_TESTED_CPU_THROUGHPUT`.
- At `(30,150,120)`, native CPU `B=16` was about `57.6 s` versus
  `2.49 s` for 16 pinned XLA workers in the controlled confirmation.

### Single-process formulation shootout

- Result:
  `docs/plans/bayesfilter-kalman-qr-cpu-xla-formulation-shootout-result-2026-07-15.md`
- Evidence:
  `docs/benchmarks/kalman_qr_cpu_xla_formulation_shootout_r2_2026-07-15/status.json`
- Outcome: `NO_SINGLE_PROCESS_FORMULATION_REPAIR_NOMINATED`.
- Interpretation: do not describe the process result as XLA versus
  multiprocessing. Every fast CPU worker also uses XLA.

### Literal and matched `(30,150,12,16)` runs

- Literal 16-process evidence:
  `docs/benchmarks/kalman_qr_cpu_16_process_b1_d30_p150_t12_2026-07-15/result.json`
- Matched result:
  `docs/plans/bayesfilter-kalman-qr-matched-cpu-process-gpu-comparison-result-2026-07-15.md`
- Matched evidence:
  `docs/benchmarks/kalman_qr_matched_cpu_process_gpu_2026-07-15/status.json`

Matched six-block medians:

| Arm | Median | Throughput | Main resource |
| --- | ---: | ---: | ---: |
| CPU native `B=16` XLA | `6.219 s` | `2.57` proposals/s | `0.885 GiB` RSS |
| 16 CPU processes, `B=1` XLA | `0.242 s` | `66.03` proposals/s | `9.74 GiB` RSS |
| GPU native `B=16` XLA, FP32/TF32 | `0.281 s` | `56.85` proposals/s | `131.1 MiB` GPU peak |

GPU versus 16 processes remains unresolved: the paired GPU/process geometric
ratio was `0.823` with bootstrap interval `[0.503,1.345]`. GPU observations
were variable (`0.080--0.368 s`) on a hot shared display device. Do not rank
GPU and processes from these six blocks.

## Retained Implementation

- `AGENTS.md`: repository-wide TensorFlow GPU memory-growth policy.
- `bayesfilter/runtime/device_policy.py`: pre-import memory-growth environment
  helpers.
- `scripts/benchmark_kalman_qr_parameter_count_scaling.py`: GPU memory-growth,
  allocator telemetry, and scoped GPU XLA workaround.
- `docs/benchmarks/run_kalman_qr_gradient_scaling_lattice_2026_07_14.py`:
  complete CPU/GPU lattice supervisor.
- `docs/benchmarks/run_kalman_qr_cpu_throughput_comparison_2026_07_14.py`:
  pinned CPU native/process comparison.
- `docs/benchmarks/run_kalman_qr_cpu_xla_formulation_shootout_2026_07_15.py`:
  native/map/vectorized/unrolled formulation discriminator.
- `docs/benchmarks/run_kalman_qr_matched_cpu_process_gpu_2026_07_15.py`:
  matched three-arm benchmark.

## Generated-File Policy

Authored source, tests, plans, result notes, terminal structured artifacts, and
raw JSON needed to substantiate retained claims are tracked. Reproducible logs,
progress journals, generated per-method Markdown/payload mirrors, failed
attempts, and superseded exploratory run roots are ignored in `.gitignore`.

Terminal retained roots are:

- `kalman_qr_cpu_throughput_comparison_r3_2026-07-14`;
- `kalman_qr_cpu_xla_formulation_shootout_r2_2026-07-15`;
- `kalman_qr_gradient_scaling_lattice_r4_2026-07-14`;
- `kalman_qr_gradient_scaling_lattice_gpu0_r2_2026-07-14`;
- `kalman_qr_cpu_16_process_b1_d30_p150_t12_2026-07-15`;
- `kalman_qr_matched_cpu_process_gpu_2026-07-15`.

## Remaining Gaps

1. Process sharding exists as a benchmark architecture, not a persistent pool
   integrated into the real gradient caller. Cold startup is still material.
2. The best worker count under the memory/throughput tradeoff has not been
   established; `k={2,4,8,16}` remains useful if 10 GiB RSS is undesirable.
3. GPU versus process selection is unresolved at the small `T=12` shape due to
   GPU variability and only six blocks.
4. No lower-level row-parallel XLA QR kernel/layout repair has been implemented
   for native CPU batching.
5. These benchmarks establish engineering correctness and tested throughput,
   not HMC, posterior, default, production, or scientific readiness.

## Recommended Restart

The next smallest consequential experiment is persistent-pool lifecycle timing
inside the actual LGSSM gradient caller:

1. Keep workers alive across repeated caller invocations.
2. Measure initialization, dispatch/serialization, steady execution, cleanup,
   aggregate RSS, affinity, and parity separately.
3. Compare `k={4,8,16}` against GPU native `B=16` at one small and one
   representative larger workload.
4. Use fresh paired blocks and preserve GPU thermal/clock/power telemetry if a
   GPU-versus-process decision is required.

Create one Tier-2 live plan before that comparison. Do not rerun more
`tf.map_fn`/`tf.vectorized_map` wrappers unless a new lower-level mechanism is
identified.

## Restart Checks

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_gradient_scaling_lattice.py \
  tests/test_kalman_qr_cpu_throughput_comparison.py \
  tests/test_kalman_qr_cpu_xla_formulation_shootout.py \
  tests/test_kalman_qr_matched_cpu_process_gpu.py
```

Any future GPU command must use the repository memory-growth policy and trusted
GPU execution. CPU-only checks must set `CUDA_VISIBLE_DEVICES=-1` before
TensorFlow import.

## Nonclaims

No universal XLA/compiler failure, universal GPU or process superiority,
equal-cost hardware ranking, larger-workload extrapolation, persistent-pool
readiness, HMC/posterior readiness, default change, production readiness, or
scientific validity.
