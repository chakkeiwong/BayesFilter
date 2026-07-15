# Kalman QR CPU Throughput Comparison Plan

Date: 2026-07-14

Risk tier: `TIER_2_MATERIAL_RESEARCH_ENGINEERING`

Status: `AUTHORIZED_AFTER_PHASE1_HARNESS_REPAIR_AUDIT`

Phase 0 close record, 2026-07-14: the harness, source-bound supervisor/worker
protocol, and focused tests are implemented. `28` focused and lattice regression
tests pass. The real CPU-hidden XLA smoke at `D=2,P=3,T=4`, `k={1,2}` passed
analytical/autodiff parity, exact canonical-row reassembly, affinity, NUMA,
five-round synchronization, contamination, RSS, and cleanup gates. Smoke
artifact:
`docs/benchmarks/kalman_qr_cpu_throughput_comparison_2026-07-14/`.
The first material canary root,
`docs/benchmarks/kalman_qr_cpu_throughput_comparison_r1_2026-07-14/`, is
preserved as failed environmental evidence. Both `k=16` attempts passed work,
parity, affinity, NUMA, RSS, and cleanup gates, but the `B=16` arm had at least
one round above the prospective unattributed-CPU threshold in each attempt.
Execution therefore stopped as required.

Phase 1 resource repair audit, 2026-07-14: a prospective 20-second `/proc/stat` survey
of the four contiguous 16-core windows on NUMA node 0 found CPUs `16..31` had
the lowest mean background occupancy (`0.0157`) and fewer cores with a
one-second sample at or above 10% (`6`) than `0..15` (`0.0168`, `10`). This is
an environmental placement repair, not a result-dependent timing choice: no
valid canary timing is promoted, the fixed set applies to every subsequent arm,
and all contamination thresholds remain unchanged.

The resulting `r2` canary exposed a separate material harness defect before
nomination: `batch_native` correctly used the repaired pool, but sharded worker
specifications still used ordinal CPU IDs `0..k-1`. The internal placement
validator validated that wrong worker specification, so its pass status cannot
support a comparison. The numerical/reference results remained valid, but all
`r2` comparative timings are rejected. The allocation now comes from one shared
`worker_specs` contract, with regression tests requiring both architectures to
use exactly the same prefix of `16..31`. The new source-bound root is
`docs/benchmarks/kalman_qr_cpu_throughput_comparison_r3_2026-07-14/`.

## Research Question

For 16 independent LGSSM parameter proposals, what is the appropriate CPU
execution architecture for the analytical QR score on this host?

The experiment separates three questions that the old thread table conflated:

1. Does one XLA `B=16` graph benefit from additional intra-op threads?
2. At the same physical-core budget, is one batched XLA process or a pool of
   pinned one-thread `B=1` workers faster in steady state?
3. Does any steady-state advantage survive after cold compilation/startup cost,
   memory use, parity, and uncertainty are reported separately?

This plan does not rerun the full CPU/GPU lattice. It is a focused CPU
throughput experiment for `batch_native_analytical_qr_score`.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Which CPU architecture processes the same canonical 16 proposals with the lowest steady-state wall-clock makespan under a fixed physical-core budget? |
| Naive baseline | One persistent XLA `B=1` analytical worker on CPU 16 processes explicit proposal rows `0..15` serially. |
| Current implementation | One XLA `B=16` batch-native analytical process. |
| Candidate mechanism | `k` persistent XLA `B=1` workers, each pinned to one distinct physical core, with rows `0..15` partitioned across workers. |
| Core budgets | `k={1,2,4,8,16}` using the first `k` OS CPU IDs from the fixed pool `16..31`, all distinct physical cores on NUMA node 0; no SMT siblings. |
| Expected failure mode | CPU/XLA overhead and cache/locality costs make intra-op threading ineffective, while process sharding trades lower per-worker kernel overhead for additional startup and memory. |
| Tuning criterion | On `D=20,P=150,T=120`, nominate the parity-valid `(architecture,k)` with the lowest median steady-state makespan across five fresh-process paired blocks. Tuning results are nomination evidence only. |
| Confirmation criterion | On fixed confirmation workloads, the nominated architecture must have a paired 95% bootstrap interval for its wall-time ratio versus the other architecture at the same core budget entirely below `0.95` on the primary large cell, and no confirmation workload interval upper bound above `1.05`. |
| Promotion veto | Any rowwise value/score parity failure, non-finite output, incorrect proposal identity, affinity/memory-placement failure, missing synchronization, source drift, or resource contamination invalidates the affected block. |
| Continuation veto | Broken baseline, invalid timer/barrier, incorrect work equivalence, two contaminated attempts for the same block, aggregate RSS above 32 GiB, or inability to enforce and verify the CPU/NUMA contract. |
| Repair trigger | A candidate timeout or poor timing rejects or repairs that candidate; it does not stop other valid arms. A harness/parity/topology failure stops execution and triggers a focused repair. |
| Explanatory diagnostics | Kernel time, supervisor makespan, CPU time, startup/compile-to-ready time, RSS, per-core utilization, frequency, and core efficiency. |
| Nonclaims | No universal CPU ranking, GPU conclusion, optimality beyond the tested host/workloads, one-shot-latency recommendation from steady-state evidence, HMC/default/production readiness, or scientific claim. |

## Exact Work Equivalence

Every timed arm must evaluate the same canonical parameter cloud rows
`0,1,...,15` from the same deterministic fixture and source fingerprint.

- `batch_native_k`: one `B=16` call receives rows `0..15` in canonical order.
- `sharded_k`: `k` persistent `B=1` workers receive a deterministic partition
  of rows `0..15`; each worker reuses one compiled `B=1` function.
- `serial_b1`: the `k=1` sharded arm; one worker evaluates all 16 explicit rows.
- The existing `_make_parameter_batch(fixture, 1)` must not be used for
  sharding because it always selects proposal row 7. The new harness must
  gather explicit row IDs from `_make_parameter_cloud`.
- Timed work ends after TensorFlow synchronization of every output. Full output
  transfer/materialization and parity checking happen outside the timed region.
- The parent-supervisor steady-state makespan from barrier release to the final
  completion acknowledgement is the primary timing boundary. Worker-internal
  kernel intervals are explanatory cross-checks.

Before any timing is admitted, the reassembled sharded values and scores must
match the `B=16` analytical output row by row under the existing float32 direct
parity contract (`rtol=atol=2e-4` for values and scores). The unchanged-source
analytical/autodiff parity from the completed lattice is inherited as supporting
correctness evidence; a new untimed autodiff parity check is run once per
distinct `D/P/T` confirmation workload.

## Arms

| Arm | CPU allocation | TensorFlow settings | Purpose |
| --- | --- | --- | --- |
| `serial_b1` | CPU 16 | intra-op 1, inter-op 1 | Naive serial baseline and process-sharding `k=1` anchor |
| `batch_native_k` | first `k` CPUs from `16..31` | intra-op `k`, inter-op 1 | Strong scaling of the existing `B=16` XLA graph |
| `sharded_k` | Worker `i` pinned to CPU `16+i` | each worker intra-op 1, inter-op 1 | Independent-proposal parallelism with the same `k`-core budget |

All arms use `CUDA_VISIBLE_DEVICES=-1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, float32, XLA JIT, and one NUMA node. No arm
uses SMT siblings `144..159`.

## Topology And Resource Contract

The host is a dual-socket AMD EPYC 7742 system. OS CPU IDs `0..63` are the
first hardware threads of 64 distinct physical cores on NUMA node 0; IDs
`128..191` are their SMT siblings.

The launch wrapper is:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/hwloc-bind --membind node:0 -- \
  taskset -c <exact-os-cpu-list> env \
  CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  OMP_NUM_THREADS=<intra> TF_NUM_INTRAOP_THREADS=<intra> \
  TF_NUM_INTEROP_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python <worker-command>
```

Required preflight:

- `lscpu -e` confirms CPUs `16..31` remain distinct cores on node 0 and CPUs
  `144..159` are their excluded SMT siblings.
- `taskset` reports exactly the requested OS CPU set for every worker and all
  `/proc/<pid>/task/*/status` entries inherit that set.
- `/proc/<pid>/numa_maps` confirms at least 95% of newly allocated anonymous
  pages are on node 0 after each worker reaches the ready barrier.
- Target CPUs are below 10% busy over a two-second prelaunch sample and
  one-minute host load is at most 16.
- The supervisor samples target-CPU `/proc/stat` counters immediately before
  barrier release and after final acknowledgement, and samples process
  `pid/stat` CPU time plus last-CPU fields during the block. Unattributed busy
  time on the target CPU set above `max(0.25 CPU-second, 2% of allocated
  core-seconds)` contaminates the block. A contaminated block is discarded and
  retried once.
- Record CPU governor, available frequency telemetry, temperatures when
  readable, total/available memory, load, and process census before and after
  every block.
- Aggregate descendant RSS is sampled during execution. Stop the arm at
  32 GiB and classify it as a resource failure, not a numerical failure.

The supervisor never changes system governor, turbo, affinity of foreign
processes, or scheduler priority. Balanced arm order and fresh-process blocks
mitigate drift without changing machine-wide settings.

## Timing And Uncertainty Contract

- Each independent block uses fresh worker processes and fresh XLA compilation.
- Compilation/startup is complete before the ready barrier and is recorded as
  `cold_time_to_ready`; it is excluded from steady-state throughput.
- Each worker performs two untimed warm calls after compilation.
- Each block contains five synchronized steady-state rounds. The block estimate
  is the median round makespan; rounds are not treated as independent samples.
- Phase 2 uses five fresh-process blocks for nomination only.
- Phase 3 uses 12 fresh-process paired blocks. Comparator/candidate order is a
  fixed balanced randomized schedule with seed `20260714`.
- Statistical unit: fresh-process paired block.
- Report paired log wall-time ratios, geometric-mean ratio, percentile 95%
  paired bootstrap interval with 10,000 resamples and seed `20260714`, and exact
  paired sign-test result. Bootstrap the paired log ratios, then exponentiate
  interval endpoints. Also show every raw block; do not rank arms from medians
  alone.
- Correct for the Phase 3 workload family with Holm adjustment when asserting
  more than the single predeclared primary large-cell comparison. The
  non-regression screen remains a veto, not evidence of superiority.

## Workload Ladder

### Phase 0: Harness Construction And Falsification

Create:

- `docs/benchmarks/run_kalman_qr_cpu_throughput_comparison_2026_07_14.py`
- `tests/test_kalman_qr_cpu_throughput_comparison.py`

Required tests:

- explicit proposal IDs are exactly `0..15` for every arm;
- partitions cover each ID exactly once for every `k`;
- fake workers prove barrier/makespan boundaries exclude compilation and output
  materialization;
- source fingerprints and output identities fail closed;
- affinity and NUMA validators reject SMT, wrong core, wrong node, missing
  telemetry, or foreign overlap;
- synthetic `/proc/stat` and process samples validate target-CPU contamination
  accounting and the declared tolerance;
- synthetic timing fixtures validate paired ratios, bootstrap intervals, Holm
  adjustment, and sign tests;
- interruption/timeout cleans up every descendant process;
- real CPU-hidden tiny shape smoke (`D=2,P=3,T=4`) passes XLA, parity, and
  reassembly for `k={1,2}`.

Do not proceed if any fake-timer, work-equivalence, cleanup, or real-parity test
fails.

### Phase 1: Topology And Small-Cell Canary

Workload: `D=10,P=50,T=120`, total proposals 16.

Run one fresh block for:

- `serial_b1`;
- `batch_native_k` and `sharded_k` at `k={1,4,16}`.

Pass conditions: exact work identity, finite outputs, rowwise parity, verified
affinity/NUMA placement, five synchronized rounds, clean post-run cleanup, and
no resource contamination. Timings are diagnostic only.

### Phase 2: Architecture And Core-Budget Nomination

Workload: `D=20,P=150,T=120`, total proposals 16.

Run five fresh-process paired blocks for:

- `batch_native_k`, `k={1,2,4,8,16}`;
- `sharded_k`, `k={1,2,4,8,16}`.

Nominate one `(architecture,k)` by the lowest parity-valid median block
makespan. Preserve the full Pareto table for steady-state makespan,
`cold_time_to_ready`, aggregate peak RSS, and core efficiency. This phase may
select a candidate for confirmation but cannot support a superiority claim.

If two candidates differ by less than 5% descriptively, nominate the lower-core
or lower-memory candidate; record the tie rather than claiming a win.

### Phase 3: Fixed-Candidate Confirmation

Freeze the nominated candidate and its same-core comparator before running:

- primary: `D=30,P=150,T=120`;
- transfer check: `D=10,P=50,T=120`;
- transfer check: `D=30,P=50,T=120`.

For each workload run 12 fresh-process paired blocks. Include `serial_b1` as a
descriptive anchor, but do not add it to the primary superiority test unless it
is the frozen same-core comparator.

Decision outcomes:

- `RECOMMEND_PERSISTENT_SHARDING_FOR_TESTED_CPU_THROUGHPUT` only if sharding is
  the fixed nominee and meets the predeclared interval and non-regression gates,
  all parity/resource gates pass, and aggregate memory is within the cap.
- `RECOMMEND_BATCH_NATIVE_FOR_TESTED_CPU_THROUGHPUT` only if batch-native is the
  fixed nominee and passes the same symmetric interval, non-regression, parity,
  and resource gates against sharding at the same core budget.
- `CPU_ARCHITECTURES_STATISTICALLY_UNRESOLVED` if both pass correctness but the
  paired uncertainty interval crosses the decision thresholds.
- `INVALID_EXPERIMENT` only for harness, work-equivalence, topology, timing,
  source, or evidence corruption. A slow candidate is a candidate rejection,
  not experiment invalidity.

### Phase 4: Closeout

Write:

- structured status and raw block JSON under
  `docs/benchmarks/kalman_qr_cpu_throughput_comparison_r3_2026-07-14/`;
- `docs/plans/bayesfilter-kalman-qr-cpu-throughput-comparison-result-2026-07-14.md`.

The result must include the decision table, inference-status table, raw paired
blocks, uncertainty results, cold/steady-state separation, memory and topology
evidence, run manifest, failed/contaminated attempts, post-run red team, and
explicit scope of any recommendation.

## Runtime Budgets

- Worker compile-to-ready timeout: 900 seconds.
- Individual synchronized steady-state round timeout: 300 seconds.
- Complete block timeout, including startup and cleanup: 1,800 seconds.
- Phase 1 canary wall budget: 45 minutes.
- Phase 2 nomination wall budget: 3 hours.
- Phase 3 confirmation wall budget: 4 hours.
- Whole program wall budget: 8 hours, excluding time spent waiting for an
  admissible clean CPU window.
- Resource-wait budget before a phase: 2 hours with 30-second polling.

On budget exhaustion, terminate descendants, preserve the last complete block,
write a structured timeout/resource result, and stop. Do not reduce the number
of paired blocks, warm rounds, core arms, workloads, or uncertainty checks
after observing partial timings.

## Skeptical Pre-Execution Audit

Status: `PASS_AFTER_PHASE0_HARNESS_AND_REAL_XLA_SMOKE`.

- Wrong baseline: repaired. The plan includes both the naive serial `B=1`
  baseline and the current batch-native implementation.
- Unfair work: repaired prospectively. Every arm processes explicit canonical
  rows `0..15` exactly once; repeating the existing `B=1` row-7 fixture is
  forbidden.
- Thread/core conflation: repaired. Core budgets use exact OS CPU affinity,
  inter-op is fixed to 1, SMT siblings are excluded, and NUMA placement is
  verified.
- Proxy promotion: avoided. Canary and tuning timings cannot support a ranking;
  only fixed-candidate paired confirmation may support the narrow host/workload
  recommendation.
- Cold versus warm conflation: avoided. Compile-to-ready and steady-state
  makespan are separate evidence classes.
- Hidden implementation cost: aggregate RSS and cold readiness are reported;
  steady-state improvement cannot be presented as one-shot improvement.
- Statistical weakness: repaired prospectively with fresh-process paired
  blocks, raw block disclosure, paired intervals, and a predeclared primary
  comparison.
- Environment mismatch: CPU is deliberately selected and GPU hidden; this is
  a CPU throughput question, not a change to the repository's GPU default.
- Resource risk: bounded to 16 physical cores, 16 workers, 32 GiB aggregate
  RSS, and explicit cleanup/timeouts. After the first canary exhausted its two
  contamination attempts on CPUs `0..15`, the repaired fixed CPU set was
  chosen prospectively from a host-occupancy survey rather than observed arm
  timings; no threshold or scientific gate was weakened.

Execution is authorized through the declared material phases because the Phase
0 harness gates have passed without weakening work equivalence, topology,
parity, timing, or uncertainty contracts. Each later phase remains conditional
on the preceding structured result and its continuation vetoes.

## Proposed Commands

Harness checks:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_cpu_throughput_comparison.py
```

Canary, nomination, and confirmation commands after the harness exists:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_cpu_throughput_comparison_2026_07_14.py \
  --phase canary \
  --output-root docs/benchmarks/kalman_qr_cpu_throughput_comparison_r3_2026-07-14

CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_cpu_throughput_comparison_2026_07_14.py \
  --phase nominate \
  --output-root docs/benchmarks/kalman_qr_cpu_throughput_comparison_r3_2026-07-14

CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_cpu_throughput_comparison_2026_07_14.py \
  --phase confirm \
  --output-root docs/benchmarks/kalman_qr_cpu_throughput_comparison_r3_2026-07-14
```

## Stop And Handoff

Stop immediately for invalid work equivalence, bad parity, broken timing,
source drift, topology/NUMA mismatch, cleanup failure, or aggregate memory cap.
Retry one contaminated block; after a second contamination, write a resource
blocker and coordinate a clean CPU window rather than weakening admission.

A candidate that is merely slow is rejected and the planned ladder continues.
At handoff, preserve enough structured evidence that another agent can rerun a
single block, reconstruct every paired comparison, and verify that all arms
processed the same 16 proposal rows under the same core budget.
