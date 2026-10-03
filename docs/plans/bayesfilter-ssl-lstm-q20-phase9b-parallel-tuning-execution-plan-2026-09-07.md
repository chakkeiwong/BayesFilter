# SSL-LSTM q=20 Phase 9B Parallel Tuning Execution Plan

Date: 2026-09-07  
Updated: 2026-09-08  
Status: `GPU_VALIDATION_AUTHORIZED_4_GPU_HOURS`  
Parent: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md`  
Master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`

## September 8 skeptical audit and implementation scope

The initial implementation is not executable integration: no actual tuning
worker calls the coordinator, it pre-creates directories rejected by existing
runners, it accepts any existing file as success, and its wave status can say
completed after a worker fails. Timeout handling also mislabels a finished
sibling and measures every worker until the last worker exits. Fix these with
synthetic process tests before GPU work. No scoped `AGENTS.md` files add rules
under the affected directories.

The owner's display restriction is stronger than filling three worker slots:
use all eligible non-display devices, up to the independent task count, and
queue excess tasks. Use a display device only when no non-display device is
eligible, not merely because another slot was requested. Admission requires
free memory for the worker's declared peak plus 5 GiB headroom, followed by
runtime headroom checks. The inherited 4 GiB allocator screen is a conservative
worker forecast hypothesis, not a guarantee of total NVIDIA reservation.

The first integrated workload is a **tuning-only Phase 0 diagnostic**, with
independent factor and strict cold-scope workers. Each constructs its own fresh
chart and calls the existing public measured-grid tuner through `_tune_scope`.
The registry and `docs/reference/hmc-tuning-interface.md` were checked:
`tune_fixed_transport_hmc_kernel` is the eligible artifact-authority interface
for this frozen Jacobian-corrected transport. No candidate-selection or kernel
implementation changes are needed. HMC transitions within each chain remain
sequential; candidate-level parallel evaluation is possible in principle but
not implemented in this bounded repair. Parallelizing tuning alone does not
repair the measured 8,314-second factor sampling forecast.

This audit permits routine implementation and CPU subprocess tests only. The
baseline, q=20 target, four-chain bank, precision, candidate grid, heldout
partitioning, and scientific vetoes remain unchanged. Startup/import cost,
different GPU performance, and fresh random streams can confound timing; no
speedup or sampler ranking follows from the CPU tests. A funded paired GPU
comparison and the separate full-controller readiness repairs are still needed.

The new entrypoint must support a framework-free plan-only mode and refuse a
serious launch before GPU detection unless an existing funded ledger covers
both complete worker allocations. It may not create a budget, reset the old
ledger, reopen the closed Phase 9A campaign, or issue a P1 closeout. The parent
alone reserves and settles each child once using individual process lifetimes;
concurrent worker-seconds are summed, not replaced by wave wall time. All jobs
use fresh directories and record their complete seed roots and source hashes.

The implementation now provides this entrypoint:
`docs/benchmarks/run_ssl_lstm_q20_phase9b_parallel_tuning_2026_09_08.py`.
It launches factor and strict as independent Python subprocesses, each using
the existing fresh-chart builder and `_tune_scope`. It is not a parallel
replacement for the complete P1 sampler. The legacy Phase 9A, readiness, and
P1 entrypoints remain serial and are not reopened by this change. Result note:
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-implementation-result-2026-09-08.md`.

### Research intent and default audit

| Item | Decision and diagnostic role |
|---|---|
| Main question | Can the unchanged cold-scope tuning computation run in isolated concurrent processes with complete evidence and correct cost accounting? |
| Mechanism | Concurrent factor/strict workers; no shared TensorFlow state or altered candidate selection |
| Expected failure | Initialization races, GPU contention, missing receipts, or stale source/scope identity |
| Promotion criterion | All required tuning workers reconcile; actual concurrency and worker cost are recorded. This promotes only the optional tuning diagnostic, not P1 readiness. |
| Promotion veto | Missing measured pair/heldout result, invalid device/memory/trace evidence, incomplete worker, or infeasible allocation |
| Continuation veto | Changed target/math, corrupted artifacts, unsafe resources, exhausted allocation, or missing required diagnostics |
| Repair trigger | Local worker startup, timeout, serialization, or accounting defect; failed candidate tuning remains candidate evidence, not research-direction rejection |
| Explanation | Worker and wave timing, queue width, startup/compile cost, GPU/allocator observations |
| Must not conclude | No posterior, convergence, whitening, statistical speedup, factor ranking, or default readiness |

| Choice | Provenance and status | Justification | Failure mode and early check |
|---|---|---|---|
| Two jobs; scope 2, chart 0, beta 1, four-chain bank | Inherited P1 cold-scope baseline; not a new scientific default | Isolate the required factor/strict comparison | Does not cover six-scope mixing; assert job identity and forbid P1 closeout |
| Grid, architecture, update counts, selection replications, heldout counts | Unchanged `_FACTOR_TUNING_R2_PROFILE` and `_build_fresh_chart`; inherited diagnostic protocol | Separate execution repair from scientific retuning | Short training/tuning cannot establish chart quality or convergence; retain original checks and nonclaims |
| GPU, float64, TF32 enabled, XLA on, batch-native training | Inherited active q=20 protocol | Preserve target and numerical execution | Verify startup, per-worker placement and original tuner receipts |
| 40% load and 5 GiB headroom | Owner directive; 5 GiB = 5,120 MiB | Resource sharing | Recheck before import and during execution; reject unknown telemetry |
| 4 GiB worker peak estimate | Inherited Phase 9A allocator screen, conservative hypothesis | Require at least 9,216 MiB free at admission | TF allocator does not bound all NVIDIA reservations; check live free VRAM and reject allocator excess |
| At most two workers; one per UUID | Derived from two required arms, limited by eligible non-display GPUs | Do not duplicate research work to fill a third device | Record actual width when fewer devices require a queue |
| CPU threads per worker | CPU affinity divided by wave width; inter-op one is a convenience choice | Avoid nested host-thread oversubscription | Throughput effects not measured; record the worker environment |
| Fresh stateless root pairs | SHA-256 of output path, arm, and role; int32 range with room for tuner offsets | Scheduler order cannot change seed identity | Check root/offset collisions before launch; retain the tuner's seed ledger |
| 50 ms process polling, 5 s telemetry cadence, 1 s telemetry timeout | Convenience supervision settings, not scientific thresholds | Responsive cleanup without continuous NVIDIA polling | Observation is periodic, not an instantaneous hard memory cap |
| 30 s termination grace | Inherited coordinator convenience; included in worker wall cap | Allow compiled calls to exit before KILL | Charge actual lifetime including cleanup; test ignored TERM |
| Worker allocation | No default or new allocation invented; required CLI value and existing funded ledger | Prevent reviving the nearly exhausted campaign | Both worker caps must fit before GPU detection; never multiply spending authority by GPU count |

## Purpose

The current Phase 9A and 9B launch paths are serial at the scope, arm, and
candidate-evaluation levels. This plan changes the execution architecture for
the next funded campaign; it does not authorize a new GPU run and does not
change the target, precision, XLA policy, chain schedule, tuning grid, or
scientific promotion criteria.

The immediate research-engineering question is:

> Can independent scope/arm tuning work run in separate GPU-pinned processes
> without changing measured-grid semantics, seed disjointness, artifact
> provenance, or the Phase 9B evidence contract?

The exact serial baseline is the current Phase 9A runner and Phase 9B P1
runner, which use one TensorFlow process and execute scopes/arms in order. The
new candidate is bounded process-level concurrency with one worker per selected
GPU. A worker owns one complete tuning scope within an arm; it does not share TensorFlow,
XLA, mutable state, or output files with another worker.

## Evidence contract

| Item | Binding rule |
|---|---|
| Primary question | Whether process-level concurrency executes unchanged tuning without semantic drift; full-controller readiness remains a later check |
| Baseline | Current serial runner with the same profile, target signature, scope, seeds, candidate grid, dtype, TF32, XLA, and chain configuration |
| Promotion criterion | Every declared tuning worker completes; every declared candidate pair is measured; artifacts and source hashes reconcile; no health/resource/provenance veto fires; summed worker cost and wave wall time fit the new allocation |
| Hard vetoes | Duplicate GPU assignment, framework import before GPU pinning, missing memory-growth verification, shared output root, stale/mismatched source or tuning scope, missing candidate pair, non-finite trace, timeout without parent settlement, or incomplete artifact |
| Explanatory diagnostics | Per-worker startup/compile/steady timings, allocator telemetry, process CPU use, GPU utilization, queue/wave timing, and candidate-level elapsed time |
| Not concluded | Parallel speedup alone does not establish posterior correctness, convergence, factor superiority, HMC readiness, default readiness, or scientific validity |
| Preserving artifact | A coordinator manifest plus one immutable worker root per task, each with its own run manifest, logs, tuning result, handoff, and source/seed/device receipt |

All continuous timing differences are descriptive until a declared multi-seed
comparison with uncertainty evidence is available. A completed parallel wave
shows execution viability; it does not rank factor and strict methods.

## Execution design

1. **Process boundary.** Use independent OS subprocesses, not threads and not a
   TensorFlow runner pool. The coordinator launches at most one worker per
   selected physical GPU. Each worker receives one UUID in `CUDA_VISIBLE_DEVICES`
   and `BAYESFILTER_SELECTED_GPU_UUID` before importing TensorFlow.
2. **Device policy.** Select from a trusted inventory using
   `bayesfilter_non_display_first_load40_headroom5g_v1`. Eligible non-display
   devices are used up to the task count. Queue excess jobs rather than fill a
   spare slot with the display GPU. Display fallback is permitted only if no
   non-display device meets the load and worker-peak-plus-headroom checks.
   Never stop display or unrelated processes. Existing low-load processes may
   coexist if the owner's resource conditions remain satisfied; there is no
   added exclusivity requirement.
3. **Memory policy.** Every worker receives
   `TF_FORCE_GPU_ALLOW_GROWTH=true` and must verify TensorFlow memory growth on
   every visible physical device before logical-device creation. A worker that
   cannot verify this fails closed.
4. **Task granularity.** The first implementation parallelizes independent
   scopes or independent arms. It does not split candidate pairs inside one
   tuner yet: the measured-joint-grid contract requires all pairs to be
   reconciled before selection, and the existing TensorFlow tuner owns that
   aggregation. Candidate-level parallelism is a later optimization only if a
   separate evidence-preserving interface is reviewed.
5. **Isolation.** Every task has a fresh versioned output root, a disjoint seed
   namespace, one source/plan hash binding, and one GPU UUID. Workers never
   write a common JSON file. The parent never merges partial tuning evidence
   into a passing result.
6. **Coordination.** The parent owns task launch, wall timeout, TERM/KILL
   cleanup, budget reservation, settlement, and final reconciliation. A set of
   tasks larger than the selected GPU count is split into successive waves;
   GPU UUIDs are never reused concurrently. The parent writes sibling
   `*-supervision` roots, leaving each worker output absent until the worker
   creates it. It charges individual lifetimes, preserves completed siblings
   when another times out, and fails the wave when any required worker fails.
   Only the parent writes the campaign ledger.
7. **Scientific semantics.** Candidate selection remains inside the worker only
   after all declared pairs for its scope/arm are measured. The parent may
   combine only complete, hash-bound worker receipts. A failed worker is a
   failed attempt or repair trigger, not a reason to drop its candidates or
   relax the grid.

The repository implementation is `bayesfilter/runtime/parallel_tuning.py`.
It provides the bounded worker environment, one-GPU-per-process validation,
private artifact roots, parent timeout/termination, and result reconciliation.
The multi-GPU placement helper is in
`bayesfilter/runtime/display_gpu_policy.py`.

## Pre-mortem

- **False speedup from contention:** multiple workers fit but compete for memory
  bandwidth or XLA compilation. Record per-worker allocator/GPU telemetry and
  compare the complete schedule, not isolated best-case calls.
- **Semantic loss through sharding:** a parent selects from incomplete candidate
  rows. Keep candidate selection local and require every declared pair before a
  handoff is eligible.
- **Seed or artifact collision:** workers overwrite files or reuse a tuning
  stream. Validate unique task IDs, GPU UUIDs, output roots, and seed ledgers
  before launch.
- **Framework initialization drift:** a worker imports TensorFlow before device
  pinning or memory-growth setup. Fail closed in the worker and test the launch
  environment before any target call.
- **Budget illusion:** concurrent elapsed wall time is confused with total GPU
  consumption. Charge each worker's measured runtime and retain the campaign
  cap; do not use parallelism to increase the authorized budget silently.

## Required checks before a serious wave

- CPU tests for GPU ordering, duplicate-device rejection, worker environment,
  process timeout, private artifacts, and parent reconciliation.
- Static source audit that the active worker imports no framework before device
  assignment and that the current tuning route remains the measured-joint-grid
  route.
- A tiny trusted GPU smoke with one worker per available GPU, memory growth
  verification, and a fresh output root. This smoke is engineering evidence,
  not tuning or scientific evidence.
- A fresh funded campaign plan that declares the total wall/GPU-second budget,
  wave count, timeout, stop conditions, and exact worker commands.
- A parallel readiness diagnostic in which both factor and strict arms produce
  complete, trace-valid receipts before P1 is reopened.

## Commands and allocation boundary

This inspection command imports no accelerator framework, detects no GPU,
creates no experiment output, and changes no budget:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_phase9b_parallel_tuning_2026_09_08.py --plan-only
```

After a plain-language allocation decision and a bounded GPU validation plan,
the existing `CampaignBudgetLedger.create` API may initialize a **new** campaign
ledger with the reported source/plan identity and diagnostic claim boundary.
Ordinary hashes record reproducibility; they are not approval tokens. Preserve
the old 5,200-second campaign and historical debits unchanged. Both complete
worker caps must fit in the new ledger before launch. Neither the launcher nor
plan-only mode creates or increases a ledger. Conditional invocation:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_phase9b_parallel_tuning_2026_09_08.py \
  --output-dir <fresh-versioned-root> --ledger <funded-campaign-ledger> \
  --worker-seconds <approved-per-worker-wall-cap-including-grace>
```

GPU detection and framework execution require trusted tool permissions. The
historical 183.799655-second remainder is not a new allocation. No serious GPU
execution occurs as part of these CPU implementation checks. A tuning-only pass
cannot replace full trace/startup diagnostics, strict controller timing, or an
affordable six-chunk P1 forecast. A timing comparison needs predeclared paired
diagnostic seeds and uncertainty analysis; the new fresh campaign seed roots
alone do not create a paired speedup experiment.

## September 8 authorized execution

The user authorized **4 aggregate GPU-hours = 14,400 GPU-seconds** for this
Phase 0 tuning validation. This is a new campaign, not an increase or reset of
the historical 5,200-second readiness ledger. It does not authorize P1/P2.

| Stage | Per-worker cap including termination | Aggregate reservation | Provenance |
|---|---:|---:|---|
| Two-worker GPU/XLA startup smoke | 300 s | 600 s | Convenience cap for import, verified memory growth, and two tiny compiled matrix calls; no training or q=20 claims |
| Fresh factor/strict cold-scope tuning | 3,600 s | 7,200 s | Conservative convenience cap under the approved increase; prior factor setup/tuning was about 300 s, strict full tuning remains unmeasured |
| Local repair/retry reserve | Only within remaining ledger balance | 6,600 s initially | Derived remainder, not an obligation to spend |

Unused reservations are released. A retry must fit its complete reserved cap;
source-equivalent infrastructure repairs may reuse the campaign budget but
never reuse prior charts, tuning draws, or output roots. Any source repair is
recorded as a new source revision with preserved prior ledger/attempt evidence,
not a new allocation. Never change target, grid, precision, or scientific
criteria merely to fit the cap.

Skeptical audit: the 135-test CPU evidence does not demonstrate real GPU
placement, XLA initialization, or numerical behavior. Before q=20 work, use the
same parent process supervisor and startup memory path for a tiny GPU smoke.
The smoke compares two repeated fixed-signature XLA matrix computations with
an independent CPU TensorFlow reference and records HLO evidence, trace count,
allocator, UUID, and actual process lifetimes. Its 16-by-16 float64 matrices,
two calls, and absolute 1e-10 tolerance are convenience mechanics checks, not
target-specific numerical tolerances or performance evidence. It must not
emit tuning evidence or a P1 closeout. The initial job definitions distinguish
smoke and tuning modes so no smoke result can satisfy a tuning result check.

Use `--smoke-only --worker-seconds 300` first, then the unchanged tuning mode
with `--worker-seconds 3600`. Both use the new funded ledger and separate fresh
attempt roots. The 30-second termination grace is inside those caps. The
coordinator is supervised from process start; GPU work is never launched with
only Python checks between compiled calls. CPU/source changes observed since
the last test run will be captured by fresh hashes and tests; concurrent source
mutation during a worker remains a run-invalidating veto.

The routine trusted NVIDIA probe on September 8 found all three devices. It
does not decide display identity or placement; the repository's UUID/PCI and
display-aware inventory is authoritative immediately before each launch. The
owner's 40% load and peak-plus-5-GiB-headroom policy remains unchanged.

## Stop conditions

Stop before a serious wave if the live inventory provides no eligible device,
if the remaining budget cannot pay for all required workers,
if any source/plan hash changes, if a worker fails a memory/device/trace-health
check, or if a required artifact is missing. Do not infer scientific failure
from a resource or orchestration veto.
