# Matched Kalman QR CPU Process GPU Comparison Result

Date: 2026-07-15

Status: `COMPLETE_MATCHED_WORK_GPU_VS_PROCESSES_UNRESOLVED`

Plan:
`docs/plans/bayesfilter-kalman-qr-matched-cpu-process-gpu-comparison-plan-2026-07-15.md`

Raw artifact:
`docs/benchmarks/kalman_qr_matched_cpu_process_gpu_2026-07-15/status.json`

## Result

The matched `(D,P,T,B)=(30,150,12,16)` analytical QR value-and-gradient run
completed all six fresh blocks. All three arms used TensorFlow float32 and XLA
JIT on the same deterministic fixture and parameter rows:

| Arm | Median steady time | Throughput | Median effective CPU cores | Median RSS | GPU allocator peak | Median cold-to-ready |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU native `B=16` XLA | 6.219222 s | 2.573 proposals/s | 4.38 | 0.885 GiB | N/A | 25.474 s |
| 16 CPU processes, each `B=1` XLA | 0.242301 s | 66.033 proposals/s | 15.76 | 9.740 GiB | N/A | 18.945 s |
| GPU native `B=16` XLA | 0.281459 s | 56.847 proposals/s | 1.02 host cores | 0.953 GiB | 131.085 MiB | 8.288 s |

The median row is descriptive. Pairing the fresh blocks gives:

| Ratio | Geometric mean | Paired bootstrap 95% interval | Exact sign-test p |
| --- | ---: | ---: | ---: |
| 16 CPU processes / CPU native batch | 0.03906 | [0.03870, 0.03940] | 0.03125 |
| GPU native batch / CPU native batch | 0.03216 | [0.01970, 0.05240] | 0.03125 |
| GPU native batch / 16 CPU processes | 0.82326 | [0.50345, 1.34514] | 1.0 |

The two candidate-versus-baseline sign tests are `0.0625` after Holm
correction across that two-test family. Their effect sizes are very large and
their paired bootstrap intervals exclude one, but there are only six blocks.
The defensible engineering conclusion is that native CPU `B=16` batching is a
severe throughput bottleneck for this exact shape. GPU versus 16 CPU processes
is statistically unresolved: its interval crosses one and its sign test gives
no directional evidence.

## Raw Blocks

| Block | Order | CPU native `B=16` | 16 CPU processes | GPU native `B=16` |
| ---: | --- | ---: | ---: | ---: |
| 0 | processes, GPU, CPU native | 6.233857 | 0.238800 | 0.368244 |
| 1 | CPU native, processes, GPU | 6.131676 | 0.238934 | 0.080182 |
| 2 | GPU, CPU native, processes | 6.214319 | 0.241307 | 0.333719 |
| 3 | GPU, CPU native, processes | 6.153073 | 0.243477 | 0.081318 |
| 4 | processes, GPU, CPU native | 6.224126 | 0.243296 | 0.229199 |
| 5 | CPU native, processes, GPU | 6.263926 | 0.248074 | 0.342986 |

GPU timing is visibly variable, but block position does not explain it: each of
the first, middle, and last positions contains both relatively fast and slow
observations. GPU 0 was a shared display device, admitted at 28--42%
utilization and 89--90 C before its arms. Thermal/DVFS/display state therefore
remains a hypothesis, not an established cause. This unexplained variability
is the main reason not to choose between GPU and processes from their overall
medians.

## Validity And Resource Checks

- `18/18` fresh arm runs passed and `90/90` measured calls completed.
- All 108 worker logs contain TensorFlow's `Compiled cluster using XLA!`
  evidence.
- Every arm covered rows `0..15` exactly once and produced both the scalar
  value and 150-element analytical score per row.
- All 18 cross-arm parity checks passed. Against CPU native batching, maximum
  process residuals were `1.526e-5` for values and `1.583e-8` for scores;
  maximum GPU residuals were `0.03694` for values and `1.135e-5` for scores,
  below the declared float32 direct-parity limits (`0.04042` and
  `2.027e-4` for these outputs).
- CPU workers were pinned to distinct physical CPUs `16..31`, CPU-only workers
  used `CUDA_VISIBLE_DEVICES=-1`, NUMA node-0 placement passed, and every
  unattributed CPU-time check remained below its declared threshold.
- GPU workers used physical GPU 0, logical `/GPU:0`, XLA flag
  `--xla_gpu_enable_triton_gemm=false`, and
  `TF_FORCE_GPU_ALLOW_GROWTH=true`. Peak TensorFlow GPU allocation was only
  `137,452,800` bytes (`131.085 MiB`), far below the 16 GiB stop.
- The timed child artifact did not serialize the TF32 toggle. A post-run
  same-environment replay probe, not timed evidence, reported TensorFlow 2.20
  `tf32_execution_enabled=true` and memory growth true on the one visible GPU.
- GPU 0 compute PIDs remained the two known display contexts plus the active
  benchmark process. After completion it returned to the two display contexts
  and about 1.24 GiB device use. No benchmark workers remained.
- Focused harness verification: `10 passed`.

## Decision Table

| Decision field | Result |
| --- | --- |
| Primary criterion | Passed: six complete, XLA, matched-work blocks for all arms |
| Numerical/device/resource vetoes | None fired |
| Main uncertainty | Six blocks; large GPU thermal/order/shared-display variability |
| Engineering decision | Avoid native CPU `B=16` for this workload when throughput matters; GPU and persistent CPU process sharding both remain viable |
| Next justified action | Benchmark the real persistent caller lifecycle, or repeat GPU timing under controlled temperature/exclusive conditions if GPU-vs-process selection matters |
| Not concluded | Universal backend superiority, equal-cost ranking, larger-`T` behavior, or default/scientific/production readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for all three arms |
| Statistically supported ranking | GPU versus 16 processes is not supported; both are clearly separated descriptively and by paired bootstrap from native CPU, but the small test family and Holm-adjusted sign tests limit formal strength |
| Descriptive-only differences | GPU median, GPU order/thermal pattern, cold readiness, memory, and effective-core differences |
| Default readiness | Not assessed and no default change is proposed |
| Next evidence needed | Persistent-pool end-to-end caller timing; controlled/exclusive or temperature-balanced GPU blocks if choosing one candidate |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `3d353253dc93a102722e00cbca8803a1b3fce7fa` with unrelated dirty worktree preserved |
| Command | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_kalman_qr_matched_cpu_process_gpu_2026_07_15.py` |
| Environment | Conda `tfgpu`; TensorFlow `2.20.0`; Linux `6.8.0-124-generic` |
| CPU/GPU | CPU workers GPU-hidden and pinned; trusted managed-session GPU 0, RTX 4080 SUPER, growth enabled |
| XLA | Required for every arm; GPU Triton GEMM workaround disabled as declared |
| Precision | float32; GPU TF32 retained as the repository default |
| Seed | `20260715` for block order and paired bootstrap |
| Wall time | `745.406 s` |
| Output | `docs/benchmarks/kalman_qr_matched_cpu_process_gpu_2026-07-15/` |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |

## Post-Run Red Team

The strongest alternative explanations for the GPU spread are thermal/DVFS or
shared-display state rather than changing algorithmic work: pre-arm
temperatures were already 89--90 C, but the available utilization and order
telemetry does not identify a cause. The current design has too few
observations and no controlled cold-GPU state or clock/power trace to estimate
those effects. A controlled run showing stable GPU timings on one side of the
process-arm timing would overturn the unresolved conclusion.

The weakest evidence is the small six-block uncertainty analysis. The strongest
evidence is the exact-work parity and the roughly 25x separation between native
CPU batching and the two viable parallel routes in every block. This result
rejects the current native CPU batch formulation for this tested throughput
use case; it does not reject XLA, because all fast CPU workers and the GPU arm
also use XLA.
