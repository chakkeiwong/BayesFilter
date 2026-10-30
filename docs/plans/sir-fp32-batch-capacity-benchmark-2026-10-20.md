# SIR FP32 batch capacity and performance benchmark

## Active checkpoint and question

2026-10-20. The owner requests CPU batches 4, 8, 16, 32 and GPU batches
through 64 and 128. Measure the existing SIR analytical likelihood-and-score
filter, N=4032, T=50, state dimension 18, observation dimension 9, parameter
dimension 3. This is the expensive target-evaluation component of planned
NeuTra training. It does not train a map or measure a complete optimizer update.
Implementation and execution of this bounded diagnostic are authorized by that
request. The main NeuTra integration plan remains separate.

Terminal checkpoint: all ten requested attempts finished on 2026-10-20.
GPU batches 4/8/16 and CPU batch 4 completed full T=50 evaluations. GPU 32
failed on a single 21.04 GiB allocation; the other arms reached the declared
time or host-memory guards before completing the intended repetitions.
See [measured results](sir-fp32-batch-capacity-results-2026-10-20.md) and
`artifacts/sir-fp32-batch-capacity-20261020-02/summary.json`. No experiment
remains running. Attempt 02 used 5,128.2 monotonic seconds within its
6,480-second allocation; the longer UTC span is recorded as a timing caveat.
No further execution is needed to close this bounded ladder. The separate
NeuTra integration plan must account for the measured capacity and score
discrepancy before a complete training-step check.

Initial hardware: Ryzen 9 9950X, 16 physical cores / 32 logical CPUs available;
46 GiB RAM visible to Linux. RTX 5080 and RTX 4080 SUPER each have about
16 GiB VRAM. Use the 4080 SUPER because it has about 15 GiB free at inspection;
other GPU activity is present and must be recorded as a timing limitation.

## Evidence contract and research intent

- Main question: which requested native batch sizes execute the current FP32
  SIR value/analytical-score kernel within available memory, and what are their
  compilation, first-call, and warmed execution costs?
- Exact comparator: the same kernel, observations, fixed particle design,
  four-point parameter bank, controls and FP32 arithmetic across CPU/GPU and
  batch sizes. CPU uses one process, 16 intra-op threads and one inter-op thread;
  GPU uses one process, four intra-op threads and one inter-op thread.
- Numerical policy: uniform float32, TF32 off, XLA on, fixed explicit input
  signature. No FP64 recurrence, pfor, scalar filter mapping or microbatching.
  Existing preparation/checksum code may use FP64 on CPU outside the timed
  kernel; inspect every floating dtype in the traced kernel before execution.
  FP64 literals immediately converted to FP32 may appear before XLA constant
  folding; no dynamic FP64 arithmetic or FP64 tensor input is eligible.
- Controls: reuse the current SIR selection's flow_steps=16, reset_steps=5,
  reset_epsilon=1, correction_steps=0, direct shared-Gaussian contraction,
  and selected mixture weights. These are frozen workload choices, not newly
  tuned FP32 settings. Changing precision or batch size does not inherit the
  selection's scientific admission. No claim-bearing LEDH run is performed.
- Primary engineering criterion: execute the requested native batch and
  preserve synchronized timing and peak memory. Full-T=50 throughput is
  eligible only if every row is valid and completes all 50 observations.
- Numerical veto on full-filter performance interpretation: early termination,
  invalid rows, non-finite results, or non-FP32 graph arithmetic. Preserve
  attempted observation counts and failure fields; never divide a truncated
  execution's time into a claimed full-filter throughput.
- Continuation veto for an arm: time cap, memory guard, resource exhaustion or
  process failure. A resource failure does not establish numerical invalidity.
  Each independent requested size can still receive a bounded attempt.
- Explanatory diagnostics: trace/node count, model callback expansion, host RSS,
  allocator current/peak, CPU utilization, concurrent GPU activity, repeat
  variability, and cross-device/batch value and score differences.
- Repair triggers: a harness or serialization failure permits one localized
  retry in a fresh directory within the total budget. Do not change the filter
  algorithm, guards or controls to manufacture successful timings.
- No conclusions about posterior accuracy, convergence, learned-map quality,
  CPU worker-pool performance, superiority under uncertainty, or default policy.

The three cheap engineering alternatives are native CPU B=4, native GPU B=4,
and sequential reuse of one small native batch to supply a larger logical
batch. The first two are measured directly; the last can be forecast from
measured small-batch throughput but must be labeled an estimate, not a native
large-batch measurement. Conditional situations are device, batch size and
completed versus numerically truncated trajectories; do not pool them.

## Resource bounds and execution

One child process at a time avoids CPU/GPU benchmark overlap from this harness.
CPU sizes: 4, 8, 16, 32. GPU sizes: 4, 8, 16, 32, 64, 128. Each child has a
600-second wall cap including import, trace, compilation and measurement;
total campaign cap is 6600 seconds. Run one first call and three synchronized
warmed calls if time permits. These repetitions support descriptive timings,
not a statistical ranking. Stop at the total cap and report unattempted arms.

Host watchdog: poll the owned child every 0.2 seconds, stop at 24 GiB RSS or
less than 6 GiB system available RAM. Disable core dumps. Record peaks and the
guard reason. This bounds attempts without intentionally exhausting the host;
a watchdog observation is not an exact minimum-memory requirement.

GPU exception to normal memory growth: use the repository's verified logical
device memory limit of 12288 MiB, with TF_FORCE_GPU_ALLOW_GROWTH=false before
import. A hard allocator ceiling is needed to bound large-batch attempts while
another GPU client is active; growth alone cannot guarantee that. This is a
documented local benchmark exception, not a new default. Record the helper's
full policy and TensorFlow allocator peaks. NVIDIA reservation is not live
tensor memory. An OOM at this cap establishes failure within a 12-GiB allocation,
not automatically failure on an otherwise empty 16-GiB card.

Artifact root: docs/plans/artifacts/sir-fp32-batch-capacity-20261020-01/.
Each child has its own directory, log, result and watchdog record. The manifest
records Git/source hashes, command/environment, data/design identity, seeds,
hardware, controls, precision, timings, memory, and this plan. CPU children
explicitly hide GPU devices before framework import. The supervisor is launched
with trusted GPU permissions because it starts GPU children and queries NVIDIA.

Command:

```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/benchmark_sir_fp32_batch_capacity.py run --output docs/plans/artifacts/sir-fp32-batch-capacity-20261020-01 --gpu GPU-68251639-fe82-8f81-3ccc-2953c32e805b
```

## Default and assumption audit

| Choice | Provenance and purpose | Failure mode / earliest diagnostic | Status |
|---|---|---|---|
| N=4032, T=50, FP32 | Owner request / active SIR case | FP32 can fail before T=50; inspect validity and completed steps | Requested workload |
| Frozen SIR controls | Current literature-prior campaign selection | Selection was mixed precision and B=2; cannot transfer numerical admission | Workload baseline only |
| Four nearby parameter points, repeated to B | Center and one positive displacement per parameter, q=0.05 | Cannot represent all learned-map tails; record exact points and nonclaim | Convenience diagnostic bank |
| 16 CPU threads | Physical core count | XLA thread use may differ or memory bandwidth saturate; record CPU seconds/wall | Untuned resource baseline |
| 4080 SUPER | More free VRAM at inspection | Concurrent clients perturb timing; snapshot activity | Available hardware baseline |
| Three warmed repeats | Bounded capacity question | No uncertainty-supported speed ranking | Descriptive only |
| Memory/time caps | Leave host/GPU resources for existing work | A capped failure need not prove physical impossibility | Safety boundaries |

Skeptical audit before execution: the major misleading success would be a fast
FP32 run that evaluated only a few observations. Therefore validity and attempted
steps are mandatory and full-horizon throughput is withheld for truncated runs.
Another is calling a compilation/RSS failure GPU OOM: phases and watchdog reasons
are separate. First-call cost is not pure compilation time; report it as
compilation plus first execution, with Python tracing measured separately.
Existing source expands small callbacks per row, so graph growth is measured
rather than claiming that batch scaling is already efficient. No posterior or
training-quality promotion is attempted. With these changes the plan passes.

## Results checkpoint

Attempt 01 was stopped before measurements: the dtype audit exposed the existing
FP64 inverse-CDF selection inside the nominally uniform FP32 route. That FP64
operation is intentional: FP32 floor(N*u) changes ancestor indices near integer
boundaries, so replacing it with FP32 would change the experiment. Prepare the
same integer indices once outside the hot recurrence instead. The optional
fixed_draw_design binding in the shared factory freezes beta and uniforms
together and exposes only four remaining inputs. The numerical recurrence and
default six-input route are shared. Also cast the immutable SIR initial-mean
constant during preparation rather than capturing an FP64 input in FP32 graphs.
These changes require exact fixed-index boundary and full small-SIR recurrence
parity checks, plus inspection of optimized XLA HLO for surviving FP64 types.

The bounded retry uses artifact suffix -02 and the remaining budget, recorded
in its exact command. Attempt 01 remains a preparation/dtype finding, not a
CPU/GPU capacity result. No TF32, mixed recurrence, changed sampling branches,
smaller particle count or changed validity rule is used to make the retry pass.

Preparation regression: 8 passed in 36.19 seconds, CPU-only. The exact-index
tests include the known N=4032 integer-boundary cases. The actual SIR B=2,
N=32, T=2 XLA recurrence matches dynamic selection bit-for-bit for value,
score, particles, particle tangents, validity and attempted steps. Optimized HLO
contains no f64 tensors. This is implementation parity, not a numerical-quality
or capacity claim for N=4032. Retry command uses --total-seconds 6480, reserving
120 seconds of the 6600-second experiment allocation for attempt 01.

Implementation review: CLI/import compilation passed. Every timed call reaches
the shared native batch kernel through four FP32 tensor inputs, with the fixed
design bound before tracing. CPU/GPU preparation
is outside timing; source closure is collected after tracing to include lazy
imports. Output synchronization, full-horizon eligibility, peak memory,
per-arm deadlines, owned-process termination and separate first/warmed costs
are explicit. No algorithm or scientific default is changed.

Watchdog checks and execution are complete. The result note separates
engineering capacity, numerical eligibility and scientific interpretation,
including a decision table, inference-status table and the next justified
integration work. Complete logs and failed attempts are preserved. A post-run
observability amendment records both realtime and monotonic durations in future
runs; four supervisor tests passed after that change. Executed source snapshots
remain unchanged and identify the code that produced this ladder.
