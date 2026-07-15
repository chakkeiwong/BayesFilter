# Kalman QR CPU Throughput Comparison Result

Date: 2026-07-14

Plan: `docs/plans/bayesfilter-kalman-qr-cpu-throughput-comparison-plan-2026-07-14.md`

Artifact root:
`docs/benchmarks/kalman_qr_cpu_throughput_comparison_r3_2026-07-14/`

Status: `COMPLETE`

Decision: `RECOMMEND_PERSISTENT_SHARDING_FOR_TESTED_CPU_THROUGHPUT`

## Result

For 16 independent canonical LGSSM proposals on this AMD EPYC 7742 host,
16 persistent one-thread XLA `B=1` workers had lower steady-state makespan than
one 16-thread XLA `B=16` process on all three fixed confirmation workloads.
All arms used the same physical cores `16..31`, NUMA-node-0 memory binding,
inter-op 1, float32, CPU XLA, two warm calls, and five synchronized measured
rounds per fresh-process block.

| Workload `(D,P,T)` | Batch-native median | Sharded median | Paired geometric ratio, sharded/batch | Paired bootstrap 95% interval | Descriptive reciprocal |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(10,50,120)` | 1.983 s | 0.0819 s | 0.0410 | `[0.0403, 0.0415]` | 24.4x |
| `(30,50,120)` | 20.970 s | 0.526 s | 0.0257 | `[0.0248, 0.0272]` | 38.9x |
| `(30,150,120)` primary | 57.619 s | 2.486 s | 0.0491 | `[0.0435, 0.0559]` | 20.4x |

The primary interval upper bound is below `0.95`, and both transfer interval
upper bounds are below the `1.05` non-regression veto. Each workload used 12
fresh-process pairs and a balanced six/six arm-order schedule. All 12 paired
ratios favored sharding on each workload; the two-sided sign-test p-value is
`0.00048828125` per workload and `0.00146484375` for each after the predeclared
Holm correction across the three-workload family. The primary workload's
bootstrap interval is the promotion criterion; the transfer intervals are
non-regression vetoes. The reciprocal values are descriptive ways to read the
ratios, not universal speedup claims.

## Why The Old Table Was Misleading

The old comparison treated TensorFlow thread count as if it were a CPU
throughput architecture. The analytical QR path propagates `[B,P,D,D]`
sensitivities through 120 serial filtering steps containing many small QR,
solve, and contraction operations. On this CPU/XLA implementation, increasing
intra-op threads for one `B=16` graph did not expose useful coarse-grained
parallelism. It increased scheduling and locality overhead.

Independent proposals provide the coarse-grained parallelism the graph lacks.
One compiled `B=1` worker per physical core avoids the unfavorable batched CPU
XLA graph and scales proposal throughput across processes. The nomination
ladder at `(20,150,120)` was monotone for sharding: medians were 9.131, 4.560,
2.464, 1.244, and 0.659 seconds for `k=1,2,4,8,16`. Batch-native medians were
18.867, 11.264, 14.652, 21.059, and 19.223 seconds. These nomination values are
descriptive only; the recommendation comes from the fixed confirmation.

## Correctness And Validity

- Every accepted arm processed explicit proposal IDs `0..15` exactly once.
- Rowwise sharded/batch value and analytical-score parity passed in every
  accepted block at the declared float32 `rtol=atol=2e-4` contract.
- A separate analytical/autodiff reference passed once per confirmation
  workload.
- Every worker and all of its threads reported the expected exact affinity.
  Both architectures used the same CPU prefix from `16..31`; SMT siblings
  `144..159` were excluded.
- Every worker had valid NUMA placement with at least 95% of anonymous pages on
  node 0; observed fractions were above 99.9% in the audited blocks.
- Each accepted arm had five synchronized rounds, finite outputs, no admitted
  contaminated round, RSS below 32 GiB, and clean process-group teardown.
- The final source fingerprint is
  `7d430ce3eec6f22a0b5f17b45e06c50c2943019460b3fffe4c2ba9600ff64b1f`.

The independent raw-artifact audit found 36/36 accepted confirmation blocks,
12 per workload. Thirty-three passed on their first in-window attempt and three
used the one allowed contamination retry. Arm order was exactly six
batch-first and six sharded-first per workload.

## Cold Start And Memory

| Workload | Batch cold-to-ready median | Sharded cold-to-ready median | Batch peak RSS | Sharded peak RSS |
| --- | ---: | ---: | ---: | ---: |
| `(10,50,120)` | 9.90 s | 11.17 s | 0.67 GiB | 8.72 GiB |
| `(30,50,120)` | 69.76 s | 20.12 s | 0.81 GiB | 9.65 GiB |
| `(30,150,120)` | 177.77 s | 25.73 s | 0.93 GiB | 9.74 GiB |

Sharding is deliberately memory-heavy because it creates 16 TensorFlow
processes. It remained below the 32 GiB cap, but its roughly 8.7--9.7 GiB peak
RSS is a real deployment cost. On the two `D=30` workloads, sharded cold
readiness was also lower because sixteen small `B=1` compilations ran in
parallel while the single batched graph compiled serially. The small workload
had similar cold readiness. These results do not imply that sharding is the
right architecture under a tight memory cap or for one-shot latency on another
host.

## Serial Anchors

One fresh `serial_b1` anchor was run per workload after confirmation. These are
explanatory only and were not part of the fixed same-core decision.

| Workload | Steady-state median | Cold-to-ready | Peak RSS | Parity |
| --- | ---: | ---: | ---: | --- |
| `(10,50,120)` | 0.851 s | 12.75 s | 0.55 GiB | pass |
| `(30,50,120)` | 7.548 s | 39.43 s | 0.60 GiB | pass |
| `(30,150,120)` | 21.966 s | 83.71 s | 0.61 GiB | pass |

The anchors show that a `B=1` graph can itself be cheaper than `B=16` on CPU
XLA, while 16-worker sharding adds proposal-level parallelism. A single anchor
per workload does not establish a statistical ranking.

## Contamination Record

The strict CPU contamination gate repeatedly stopped the small transfer
workload while unrelated high-thread TensorFlow jobs from the other lane were
active. Six failed two-attempt windows are preserved under
`rejected-contamination/`; all passed parity and placement, while only
batch-native's longer measured rounds exceeded the foreign-CPU threshold.
None of those timings entered a summary.

Trusted read-only process inspection identified sequential unrelated `pytest`,
affine-training, and dense-training jobs with 900--1200 unrestricted threads.
No foreign process was killed, re-affined, or otherwise changed. Execution
resumed only after bounded aggregate occupancy monitors found two consecutive
clean intervals. A diagnostic also confirmed that `/proc/<pid>/stat` matched
`CLOCK_PROCESS_CPUTIME_ID` for multithread process CPU accounting; missing
TensorFlow thread time was not the cause. Pinning the supervisor to CPU 63 was
retained as isolation but did not itself resolve contention.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Use persistent 16-way `B=1` sharding for the tested CPU throughput cells when roughly 10 GiB RSS is acceptable | pass: primary ratio interval `[0.0435,0.0559]` is below `0.95` | pass: parity, affinity, NUMA, contamination, memory, cleanup, and both transfer non-regression gates | one host, three workloads, deterministic proposal cloud | implement an optional persistent CPU worker-pool path and benchmark its real caller integration | universal CPU optimality, GPU ranking, default-policy promotion, HMC or scientific validity |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | pass for all 36 accepted confirmation pairs; rejected contaminated attempts were excluded |
| Statistically supported ranking | supported for the predeclared primary `(30,150,120)` fixed `sharded,k=16` versus `batch_native,k=16` comparison; the transfer cells passed their predeclared non-regression vetoes, and all three direction-only sign tests remain significant after Holm correction |
| Descriptive-only differences | nomination core ladder, reciprocal speedups, serial anchors, cold times, RSS, and individual block/round differences |
| Default-readiness | not established; this is a narrow CPU throughput architecture recommendation, not a BayesFilter backend or GPU/XLA default change |
| Next evidence needed | integrated worker-pool overhead, lifecycle/failure handling, memory-constrained arms, and additional representative workloads or hosts before broader promotion |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit at closeout | `3d353253dc93a102722e00cbca8803a1b3fce7fa` with unrelated dirty work preserved |
| Branch | `main` |
| Source fingerprint | `7d430ce3eec6f22a0b5f17b45e06c50c2943019460b3fffe4c2ba9600ff64b1f` |
| Python / environment | Python 3.13.13, `tfgpu`, `/home/ubuntu/anaconda3/envs/tfgpu` |
| TensorFlow / TFP | 2.20.0 / 0.25.0 |
| CPU | dual-socket AMD EPYC 7742; target CPUs `16..31`, node 0; supervisor CPU 63 on resumed confirmation windows |
| GPU | intentionally hidden with `CUDA_VISIBLE_DEVICES=-1`; TensorFlow GPU list `[]` |
| JIT / dtype | XLA JIT enabled, float32 |
| Inter/intra-op | inter-op 1; batch intra-op `k`; sharded workers intra-op 1 |
| TF memory policy environment | `TF_FORCE_GPU_ALLOW_GROWTH=true` even though GPU was hidden |
| Seeds | deterministic fixture; order/bootstrap seed `20260714`; no stochastic model sample |
| Evidence run interval | 2026-07-14 11:00:05--18:19:49 UTC, excluding later serial anchors and resource-wait diagnostics |
| Primary command | `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONDONTWRITEBYTECODE=1 taskset -c 63 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_kalman_qr_cpu_throughput_comparison_2026_07_14.py --phase confirm --output-root docs/benchmarks/kalman_qr_cpu_throughput_comparison_r3_2026-07-14` |
| Raw artifacts | `docs/benchmarks/kalman_qr_cpu_throughput_comparison_r3_2026-07-14/` (37 MiB) |
| Structured decision | `docs/benchmarks/kalman_qr_cpu_throughput_comparison_r3_2026-07-14/status.json` |

## Post-Run Red Team

The strongest alternative explanation is that process isolation, cache state,
or this host's particular XLA/BLAS/runtime build drives the result rather than
an architecture property that transfers broadly. That does not overturn the
tested-host recommendation because paired fresh-process blocks, identical
physical cores, balanced order, and transfer workloads all agree. It does limit
generalization.

The weakest evidence is external validity: one CPU model, one TensorFlow/XLA
version, a deterministic 16-row fixture, and three related LGSSM shapes. An
integrated pool could add dispatch, serialization, crash recovery, and idle
memory costs absent from the kernel harness. A result showing those costs erase
the confirmed advantage, or a representative workload exceeding the memory
cap, would overturn or narrow the implementation recommendation.

No claim is made about GPU performance, cross-host superiority, posterior
correctness, HMC readiness, package/default readiness, or scientific validity.
