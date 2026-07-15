# Kalman QR CPU/XLA Formulation Shootout Result

Date: 2026-07-15

Plan:
`docs/plans/bayesfilter-kalman-qr-cpu-xla-formulation-shootout-plan-2026-07-15.md`

Admitted artifact root:
`docs/benchmarks/kalman_qr_cpu_xla_formulation_shootout_r2_2026-07-15/`

Status: `COMPLETE`

Decision: `NO_SINGLE_PROCESS_FORMULATION_REPAIR_NOMINATED`

## Result

The prior benchmark's numerical and work-equivalence contract remains credible,
but its causal interpretation needed qualification. The current native
`B=16` XLA graph is not the only conceivable single-process formulation, so a
new controlled experiment compared it against TensorFlow mapping and scalar-row
formulations on the same 16 proposals, CPUs `16..31`, NUMA node 0, float32,
intra-op 16, inter-op 1, and XLA JIT.

No tested single-process alternative repaired the native-batch result.

| Formulation | Canary status | First executable call | Median warm makespan | Candidate/native | Median average cores | Peak RSS | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Native `[B,...]` batch | pass | 4.226 s | 2.036 s | 1.000 | 1.43 | 0.67 GiB | Exact baseline; lowest observed clean single-process median |
| `tf.map_fn`, `parallel_iterations=1` | resource-contaminated | 27.430 s | 3.243 s | 1.593 | 1.00 | 1.66 GiB | Rejected; two of five rounds exceeded the foreign-CPU gate |
| `tf.map_fn`, `parallel_iterations=16` | pass | 27.299 s | 3.233 s | 1.588 | 1.00 | 1.67 GiB | Correct but slower; no nomination |
| Strict `tf.vectorized_map` | compile/trace failure | N/A | N/A | N/A | N/A | N/A | Rejected: no pfor vectorization exists for `Qr` |
| Fallback `tf.vectorized_map` | pass | 65.493 s | 3.994 s | 1.962 | 1.48 | 5.06 GiB | Correct but slower and memory-heavy; no nomination |
| Static 16-row scalar graph | 900 s child timeout | N/A | N/A | N/A | N/A | N/A | Rejected for compile infeasibility |
| 16 sequential calls to one compiled XLA `B=1` executable | pass, explanatory only | 2.986 s | 1.195 s | 0.587 | 1.00 | 0.63 GiB | Confirms a native-batch penalty; not a single `B=16` graph or nomination candidate |

All admitted parity-valid alternatives returned the same explicit proposal rows
`0..15`. Maximum score residuals versus native batch were at most
`1.1921e-7`, below the declared float32 contract. Every admitted clean record
passed exact process/thread affinity, NUMA placement, source identity, memory,
finite output, shape, synchronization, and cleanup checks.

The canary was nomination evidence only. The plan required a parity-valid
single-process candidate at or below `0.80` times native. None qualified, so
the eight-block paired confirmation and larger transfer cells were not run.

## What The Mapping APIs Did

Strict `tf.vectorized_map` failed before XLA compilation with:

```text
ValueError: No pfor vectorization defined for Qr
Consider enabling the fallback_to_while_loop option to pfor, which may run slower.
```

This directly answers the pfor hypothesis: TensorFlow 2.20 cannot vectorize the
scalar analytical route through its QR operations. Enabling fallback produced a
valid graph, but it was about 1.96 times native, used about 5.06 GiB RSS, and
expanded the optimized HLO to about 18.5 MB.

`tf.map_fn` with `parallel_iterations=1` and `16` produced identical GraphDef
sizes, identical optimized-HLO hashes, and identical HLO censuses on the real
cell. Both used approximately one CPU core and took about 3.23 seconds. Under
XLA for this function, `parallel_iterations=16` did not create row-parallel
execution.

| Formulation | GraphDef nodes | GraphDef bytes | Optimized HLO bytes | HLO while census | HLO dot census | HLO fusion census |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native batch | 884 | 203,191 | 994,291 | 10 | 193 | 495 |
| Either `map_fn` arm | 39 | 2,996,174 | 10,876,928 | 465 | 1,634 | 4,707 |
| Fallback `vectorized_map` | 21 | 10,176,329 | 18,520,500 | 19 | 4,361 | 13,224 |
| Compiled `B=1` executable | 884 | 203,191 | 700,811 | 14 | 170 | 452 |

The censuses are explanatory, not performance criteria. They show that the
mapping APIs did not lower to 16 independently concurrent scalar XLA
computations. They instead created loop/vectorization structures that were
larger and no faster than native batching.

## Corrected Interpretation

The evidence now supports the following statement:

> For this analytical QR score on TensorFlow 2.20's CPU XLA backend, the
> current native `B=16` tensor formulation had the lowest observed clean
> single-process formulation among native batching, `tf.map_fn`,
> `tf.vectorized_map`, and static row unrolling tested here. It still uses only
> about 1.43 cores and is slower than 16 sequential invocations of a compiled
> `B=1` executable. Explicit concurrent `B=1` XLA workers remain the only
> tested route that exposes proposal-level CPU parallelism.

This is not evidence that XLA is generally slower than multiprocessing. The
fast route uses XLA in every worker. Nor does the result prove that no
single-process repair exists. It rules out the most direct TensorFlow mapping
wrappers under the current implementation and stack.

The remaining plausible repair space is lower-level:

- restructure the analytical derivative algebra or layouts so the CPU backend
  sees larger parallel kernels rather than many small batched QR/solve calls;
- implement a row-parallel custom operation or XLA custom call;
- use one process with multiple independently compiled/executed function
  contexts if TensorFlow provides a reliable concurrent-execution mechanism;
- accept persistent multi-process XLA sharding when its approximately 10 GiB
  memory cost is acceptable.

## Invalid Harness Record

The first root,
`docs/benchmarks/kalman_qr_cpu_xla_formulation_shootout_r1_2026-07-15/`, is
preserved as invalid harness evidence. The supervisor compared a relative
output path against the absolute repository root and stopped after one child.
The same child showed that sub-millisecond smoke calls were too short for
`/proc/stat` CPU accounting. No `r1` timing was admitted.

The repair normalized artifact paths explicitly and measured the tiny smoke in
fixed 2,048-call windows while reporting normalized per-call time. A regression
test covers the path bug. The fresh `r2` root used the repaired source
fingerprint.

## Protocol Deviation

The plan declared a 300-second first-executable-call timeout nested inside a
900-second child-wall timeout. The implemented supervisor enforced only the
900-second child-wall cap. Consequently, static unrolling was terminated after
900 seconds rather than being rejected at 300 seconds.

This deviation does not admit or favor a candidate: static unrolling produced
no output or timing and would also have failed the stricter 300-second gate.
It does mean the artifact supports only `compile/first-call infeasible within
900 seconds`, not a precise compile duration or proof that it could not finish
after 900 seconds. No other child approached either cap.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not nominate `map_fn`, `vectorized_map`, or static row unrolling as a single-process CPU/XLA repair | Failed: no eligible candidate reached `candidate/native <= 0.80`; the fastest valid candidate was `map_parallel_16` at `1.588` | Strict vectorization had no QR pfor rule; static unrolling timed out; one sequential-map record was contaminated; clean alternatives passed parity/placement/source/memory | Whether a lower-level layout/kernel or concurrent-execution design can expose proposal-level parallelism without 16 TensorFlow processes | Retain persistent XLA `B=1` workers as the tested CPU-throughput route; only start a new repair for lower-level kernel/layout work, not another mapping-wrapper sweep | No universal XLA/compiler defect, no claim that every possible single-process formulation fails, no GPU/default/HMC/scientific conclusion |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Native baseline and clean alternatives passed numerical, identity, placement, source, memory, and cleanup gates. Strict pfor and static unrolling were candidate failures; contaminated `map_sequential` timing was excluded from nomination. |
| Statistically supported ranking | None among formulations. The canary has one fresh process per formulation and supports nomination/rejection only. No candidate reached the prospective nomination threshold, so confirmation was correctly skipped. |
| Descriptive-only differences | Exact canary timing ratios, first-call times, CPU utilization, RSS, GraphDef/HLO sizes, and the explanatory sequential-`B=1` result. |
| Default-readiness | Not established. No public API, backend, XLA, or device default changed. |
| Next evidence needed | A concrete lower-level formulation with tiny-smoke parity and structural evidence, followed by the same small-cell nomination and fresh-process paired confirmation contract. |

## Evidence Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | The repaired `r2` supervisor completed smoke and canary, preserved candidate-local failures, enforced affinity/NUMA/source/resource gates, and left no worker processes. |
| Numerical validity | Every clean executable formulation passed rowwise value/score parity against native batch at the declared float32 tolerance. |
| Performance | No single-process repair was nominated. Native batch had the lowest observed clean single-process median in this canary; sequential `B=1` calls were descriptively faster, and prior concurrent XLA `B=1` workers remain the confirmed CPU throughput architecture. |
| Scientific interpretation | No scientific question or posterior result was tested. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `3d353253dc93a102722e00cbca8803a1b3fce7fa` with unrelated dirty work preserved |
| Source fingerprint | `6daffd05f2fd888bf06b8d6eafbd48c75cb8698422cf10bd110bf71020524785` |
| Python / TensorFlow | Python 3.13.13 / TensorFlow 2.20.0 |
| Host | Linux 6.8.0-124-generic; dual-socket AMD EPYC 7742 |
| Device | CPU only; GPU intentionally hidden with `CUDA_VISIBLE_DEVICES=-1` |
| CPU placement | Physical CPUs `16..31`, node 0; SMT siblings `144..159` excluded |
| JIT / dtype | XLA JIT enabled / float32 |
| Thread settings | intra-op 16, inter-op 1, OMP 16 for the canary |
| Workload | Deterministic `(D,P,T,B)=(10,50,120,16)` with explicit proposal rows `0..15` |
| Timing | Two untimed warmups and five synchronized measured calls per completed child; one fresh process per formulation |
| Commands | `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_kalman_qr_cpu_xla_formulation_shootout_2026_07_15.py --phase smoke|canary --output-root docs/benchmarks/kalman_qr_cpu_xla_formulation_shootout_r2_2026-07-15` |
| Raw structured artifact | `docs/benchmarks/kalman_qr_cpu_xla_formulation_shootout_r2_2026-07-15/status.json` |
| Raw optimized HLO | Ignored `optimized_hlo.txt` sidecars under each completed child directory; hashes and censuses retained in JSON |

## Post-Run Red Team

The strongest alternative explanation is that TensorFlow's high-level mapping
APIs are simply the wrong counterfactual: a custom C++/Eigen thread-pool kernel,
different tensor layout, or explicit asynchronous function execution could
still recover single-process row parallelism. This result does not reject that
possibility.

The weakest evidence is the absence of fresh-process confirmation for exact
timing ranks. That is intentional: no candidate met the prospective 20%
nomination threshold, so running confirmation would spend resources without a
qualifying repair. The strongest evidence is structural and numerical:
strict pfor cannot transform QR, both `map_fn` settings produced identical HLO,
all viable alternatives passed rowwise parity, and none exposed more than about
1.48 average cores.

A future result would overturn this conclusion if a bounded single-process XLA
formulation passes parity and demonstrates a confirmed candidate/native
interval below `0.90` under the same physical-core and contamination contract.
It would not retroactively invalidate the present comparison; it would add a
new formulation that was not tested here.
