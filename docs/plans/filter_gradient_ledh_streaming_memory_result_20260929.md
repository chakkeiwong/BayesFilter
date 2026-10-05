# Seeded LEDH streaming repair and owner-memory evidence

The registered seeded value owner now generates process noise inside the shared
TensorFlow time recurrence after the original prediction-validity gate. It
carries three uint64 Philox words rather than a full [T,N,d] process array.
Annealed resampling seeds PCG64(resample_seed+time) once per executed step and
advances once per stage. The supplied-array owner remains the comparator,
using the same numerical recurrence; no copied filter was created. The default
remains XLA, and no runtime NumPy, Python numerical loop, cache, TF32 policy,
seed stream, reset rule or analytical score authority changed. Added numeric
RNG-state/draw-count diagnostics expose early termination.

Execution plan: `filter_gradient_ledh_streaming_memory_20260929.md`. Frozen
array-composed authority is commit c7c0b88c2. Its shared LM/reset dependency is
the already repaired current source. It is not the erroneous pre-repair eager
LM result. The earlier eager filter remains an independent diagnostic wrapper
with current repaired reset. All numerical qualification preserves the existing
healthy1e-6 gate, independent normals/uniform checks and rejected-case roles.
Unsupported canonical LEDH, HMC and scientific claims remain blocked.

CPU04688/GPU04689 each freeze3 cases with seed123/124 and changed observations,
including source bytes and optimized HLO. Composed/dual-trust are healthy;
annealed is rejected and diagnostic only. CPU04690/GPU04691 each pass9 checks:
streaming/buffered/supplied-independent/eager-current-reset comparison, first
and later invalid prediction, invalid initial state, time-varying callbacks,
large seed-word carry, changed operands and mutable callback refresh. All
shared floating records, including the reported condition diagnostics, are
exactly equal to the buffered owner in these runs. RNG state exactly matches
an independently observed stateful generator's actual draw count. The seeded
owner traces once and has no host callbacks. Optimized HLO loses every tested
[T,N,2] process buffer while legitimate O(T) diagnostic histories remain.

CPU costs are one fresh process per arm at N8,d2,T3,seed123,resample17,
15 retained-owner calls and two additional fresh owners. Callbacks freeze the
same independent constants before timing. No HLO export contaminates costs.
CPU is explicitly a reference; its TensorFlow allocator statistics are
unavailable and recorded as null. RSS includes compiler/context residency.

|CPU arm/run|Cold seconds|Warm median ms|RSS growth after warm MiB|RSS growth after release MiB|
|---|---:|---:|---:|---:|
|04692 buffered_xla|5.090|0.990|522.11|988.82|
|04693 streaming_graph|2.488|11.142|152.16|275.86|
|04694 streaming_xla|4.974|1.179|512.00|959.86|

All three Python owners are collected in each arm, yet native/compiler RSS
persists until process exit. Repeated use of one XLA owner added about0.2MiB
after cold compilation; two new owners added about448MiB in streaming XLA.
Every cost worker was reaped and its /proc entry absent. This supports explicit
owner reuse and bounded worker lifetimes for changing callback configurations;
it neither proves an unbounded leak nor justifies global cache clearing.
The small memory difference between buffered and streaming cannot be attributed
solely to the384-byte random buffer. Compiler/graph differences dominate it.

Capacity used four predeclared sizes with a fresh CPU process per arm/point.
All eight trajectories are healthy and all shared record fields exactly match.
Each primary cost sample precedes HLO export and comparator construction.
No graph or optimized-HLO full-horizon random buffer remains in streaming.
At T128,N64,d2 the buffered array is131072 bytes; the per-step draw is1024
bytes plus24 bytes of Philox state. Other filter workspaces are unchanged and
no whole-filter O(N*d) claim follows.

|CPU run/arm|T,N,d|Healthy|Warm median ms (3 calls)|RSS growth after warm MiB|
|---|---|---|---:|---:|
|04695 buffered|3,8,2|True|1.534|517.45|
|04696 streaming|3,8,2|True|1.571|509.57|
|04697 buffered|3,64,2|True|1.633|536.66|
|04698 streaming|3,64,2|True|1.709|524.54|
|04699 buffered|32,64,2|True|10.106|536.27|
|04700 streaming|32,64,2|True|11.294|526.05|
|04701 buffered|128,64,2|True|36.918|536.56|
|04702 streaming|128,64,2|True|43.913|526.12|

These timings are descriptive. At the largest point, streaming is43.913ms
versus36.918ms buffered, a possible performance regression requiring controlled
attribution; it is not accepted or hidden as a speedup. The small horizon ladder
qualifies this buffer-removal mechanism, not production capacity or a statistical
performance ranking. The shared LM arithmetic and science controls are fixed
fixture baselines, not scope-specific tuning or admission evidence.

GPU costs/capacity did not launch: the unshared preflight declined and preserved
`cost-preflight-declined-20260928T172518388312Z.json`. PID2260909 holds contexts
on non-display GPUs2/3. GPU0 is remote desktop despite display_active=false;
GPU1 has active display. No unrelated job was stopped. GPU numerical checks
record verified growth before initialization, physical UUID, TF32 enabled and
owner_designated_managed_session_visible_gpu_trusted. Shared GPU numerical
qualification cannot support uncontended timing or GPU allocator comparisons.

CPU regression04703 preserves42 passes and one raw HLO-text comparison failure.
Its printed difference concerns TensorFlow debug op_name suffixes. The old
harness did not save both failing exports, so that historical diff alone cannot
prove every difference was metadata. Localized harness repair1 archives both
full exports and compares all instructions/constants/shapes/operands with only
source/debug metadata removed. No runtime or numerical gate changed. Renewed
CPU04704 passes43 checks, including supplied-input, seeded, random stream and
analytical-score/independent derivative checks. GPU04705 also passes43. Current-source readback/policy04706 passes164 checks. The guard covers278 sources with1436 existing exact allowances and no new exception. Full unmodified HLO exports are archived and their semantic comparison is independently reproduced by the readback.

|Decision|Primary criterion|Veto status|Main uncertainty|Next action|Not concluded|
|---|---|---|---|---|---|
|Buffer-removal implementation passes bounded fixtures|Exact buffered records, original RNG scheduling, no O(TNd) graph buffer|No numerical veto in qualified fixtures|Default-scale/GPU capacity pending|Retain repair branch; finish regressions and costs|Canonical/scientific admission|
|Retained owner is the measured repeat-call route|One trace, stable small retained RSS increment|Fresh-owner native residency remains|Long-lived changing callbacks unqualified|Explicit reuse and process-lifetime containment|No unbounded leak proof|
|Performance promotion remains open|Descriptive CPU timings only|Possible long-horizon slowdown; shared GPU timing veto|Paired uncertainty and fusion attribution missing|Bounded performance follow-up, uncontended GPU costs|Speed superiority/default readiness|

|Inference status|Finding|
|---|---|
|Hard veto screen|Qualified bounded numerical cases pass; invalid cases reject; GPU cost launch withheld|
|Statistically supported ranking|None|
|Descriptive-only differences|CPU cold/warm/RSS tables above|
|Default readiness|Whole master remains open|
|Next evidence needed|Controlled CPU performance attribution and unshared GPU allocator/cost/capacity evidence|

Post-run skeptical review: the strongest alternative explanation for RSS changes
is compiler ownership/graph restructuring, not random tensor memory. HLO/source
supports removal of the actual buffer; RSS alone does not. The weakest evidence
is single-process cost replication and unavailable CPU allocator accounting.
Long-horizon timing needs follow-up. Passing a linear fixture does not qualify
nonlinear target behavior, analytic-score consumer migration, NeuTra training,
posterior correctness or HMC. Main remains unmerged. Other master gaps and F14
are tracked separately; this result is a bounded implementation checkpoint.

Evidence archive: `docs/plans/artifacts/filter-gradient-repair-20260917/ledh-streaming-04706-evidence.tar.gz`; SHA256
`a116e77cfe5471799913bed97bcc7bb64db349f4a8b1914c75bc5f05bb27f430`; 372 members reopened and verified,
11640410 bytes. The declined preflight JSON is separately committed alongside
the archive. All final runtime/test/runner snapshots are preserved; intermediate
harness revisions have manifest hashes and logs, not every revision's bytes.
Unit used19 workers,637.001795 CPU/411.837878 GPU seconds, within24-worker
and3600/2400-second allocation. Remaining global budget25.673643 CPU/25.052052
GPU hours. No numerical worker remains active at this checkpoint.
