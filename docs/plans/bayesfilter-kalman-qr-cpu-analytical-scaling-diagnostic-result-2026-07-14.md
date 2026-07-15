# Kalman QR CPU Analytical Scaling Diagnostic Result

Date: 2026-07-14

Status: `DIAGNOSED_CPU_THREAD_SCALING_NOT_ESTABLISHED`

Plan: `docs/plans/bayesfilter-kalman-qr-cpu-analytical-scaling-diagnostic-plan-2026-07-14.md`

## Result

The analytical implementation is numerically valid and batch-native, but the
completed lattice is not valid evidence of physical CPU-core scaling. The
surprising timings arise from the implementation and CPU/XLA execution shape,
plus a benchmark thread contract that should not have been described as a core
count.

The analytical kernel propagates forward sensitivities with shapes such as
`[B,P,D,D]` through a serial `T=120` `tf.while_loop`. Each time step performs
multiple batched QR derivative, triangular-solve, covariance, and einsum
operations. The reverse-mode comparator propagates only the `[B,D,D]` primal
filter forward and differentiates the scalar sum backward. Consequently, the
analytical method's working set and small-matrix operation count grow directly
with both `B` and `P`; it is not equivalent to a cheap scalar filter followed
by free batch parallelism.

## Focused Counterfactual

The same XLA analytical function was run at `D=10`, `P=50`, `T=120`, float32,
with process affinity restricted to logical CPU IDs 0-15. GPU was deliberately hidden.
Two synchronized measured calls followed an untimed warm call.

| Intra-op | Inter-op | B=1 median (s) | B=16 median (s) | B16/B1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.05398 | 0.74146 | 13.73x |
| 16 | 16 | 0.07508 | 2.26381 | 30.15x |
| 16 | 1 | 0.08808 | 2.07037 | 23.51x |

All outputs were finite. The one-thread result closely reproduced the lattice
cell (`0.05358 s` at B=1 and `0.72778 s` at B=16). Separating inter-op from
intra-op recovered only a small fraction of the slowdown. The dominant effect
is therefore CPU/XLA intra-op overhead or poor scheduling/locality for the
kernel's many small batched linear-algebra operations, not merely setting both
TensorFlow pools to 16.

The July 9 scalar analytical reference provides an independent structural
check at `D=10,P=50,B=16`: the batch-native analytical path was about 1.6-1.8x
faster than the statically duplicated scalar-row path and agreed numerically.
This rules out a hidden Python/row loop as the explanation. A new
`D=20,P=150,B=16` scalar-row compilation was stopped after the prospective
five-minute diagnostic bound because static graph duplication made it
infeasible; it produced no timing artifact.

## Interpretation

- RAM capacity is not the relevant constraint. Cache capacity, memory
  bandwidth, NUMA locality, and thread coordination can dominate well before
  system RAM is exhausted. One `[16,150,30,30]` float32 derivative tensor is
  about 8.2 MiB, and the loop carries or creates many such tensors.
- `parallel_iterations=1` correctly serializes the Kalman time recursion. More
  TensorFlow threads can only parallelize kernels within a time step; they do
  not make the 120 dependent time steps concurrent.
- `inter_op=16` does not help a computation compiled into one XLA cluster.
- Matrices with `D<=30` are small. Spawning and coordinating workers for
  repeated small QR, triangular-solve, and einsum kernels can cost more than
  single-thread execution.
- The lattice also had no CPU affinity or NUMA binding and admitted foreign CPU
  workloads. Its CPU rows remain valid correctness/XLA evidence, but not a
  calibrated core-scaling experiment.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Downgrade the CPU thread table to descriptive TensorFlow-pool observations; retain correctness evidence | Diagnosis reproduced under fixed affinity and outputs remained finite; prior analytical/autodiff and scalar-reference parity passed | Large scalar-row counterfactual hit the diagnostic continuation bound and was excluded; no numerical veto fired | Exact XLA CPU kernel scheduling and the best affinity/thread policy for larger cells | If CPU throughput matters, benchmark independent batch sharding across pinned one-thread workers versus one XLA batched process, with inter-op fixed to 1 and clean-host admission | No claim that 1 physical core is universally faster than 16, no optimal-thread claim, and no method speed ranking |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | No numerical or XLA veto in the admitted focused runs. The infeasible large scalar reference was stopped and excluded. |
| Statistically supported ranking | None; two measured calls in one process are diagnostic only. |
| Descriptive-only differences | All thread and batch timings and ratios. |
| Default-readiness | No policy/default change is supported. |
| Next evidence needed | Independent-process paired replications on a clean host, explicit CPU affinity/NUMA binding, inter-op fixed at 1, and comparison with process-level sharding. |

## Run Manifest

| Field | Value |
| --- | --- |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; `CUDA_VISIBLE_DEVICES=-1`; `TF_FORCE_GPU_ALLOW_GROWTH=true` |
| Device | CPU-only diagnostic; GPU deliberately hidden |
| JIT | XLA enabled for every admitted call |
| Affinity | `taskset -c 0-15`; logical CPU IDs 0-15 |
| Problem | Deterministic fixture; `D=10`, `P=50`, `T=120`, `B={1,16}`, float32 |
| Repetition | One untimed warm call, two synchronized measured calls, one output/finite check call |
| Helper | `docs/benchmarks/run_kalman_qr_cpu_analytical_scaling_diagnostic_2026_07_14.py` |
| Historical scalar reference | `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads1_batch16_xla_2026-07-09.json` and corresponding thread-16 artifact |
| Random seed/data | N/A; deterministic synthetic fixture |

Command pattern:

```bash
taskset -c 0-15 env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  OMP_NUM_THREADS=<intra> TF_NUM_INTRAOP_THREADS=<intra> \
  TF_NUM_INTEROP_THREADS=<inter> PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_cpu_analytical_scaling_diagnostic_2026_07_14.py \
  --intra <intra> --inter <inter> --dimension 10 --parameter-count 50 \
  --timesteps 120
```

## Code Anchors

- Analytical builder: `scripts/benchmark_kalman_qr_parameter_count_scaling.py:1944`.
- Batched derivative shape contract: `bayesfilter/linear/kalman_qr_derivatives_tf.py:81`.
- Batched QR derivative broadcasts base QR factors across `P`:
  `bayesfilter/linear/kalman_qr_derivatives_tf.py:205`.
- `[B,P,...]` forward-sensitivity operations:
  `bayesfilter/linear/kalman_qr_derivatives_tf.py:2179`.
- Serial time recursion: `bayesfilter/linear/kalman_qr_derivatives_tf.py:2364`.
- Original thread configuration set both TensorFlow pools:
  `docs/benchmarks/run_kalman_qr_gradient_scaling_lattice_2026_07_14.py:519`.

## Post-Run Red Team

The focused run is small and has only two measured calls, so its exact ratios
must not be generalized. Its value is discriminatory: the slowdown survives
affinity control, and changing only inter-op from 16 to 1 does not repair it.
The strongest alternative explanation is still XLA CPU backend behavior
specific to this kernel and host. That is compatible with, rather than contrary
to, the diagnosis that the original table was not a physical-core scaling
experiment.
