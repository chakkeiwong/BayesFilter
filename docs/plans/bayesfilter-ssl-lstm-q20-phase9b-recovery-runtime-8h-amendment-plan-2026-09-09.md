# Phase 9B eight-hour runtime-cap amendment plan

Date: 2026-09-09 (Asia/Shanghai).  
Status: `EXECUTION_PLAN_FROZEN_AFTER_CPU_VALIDATION`; live status is in the result note.  
Governing master: `bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`.  
Prior terminal result: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-result-2026-09-09.md`.  
Prior plan: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-plan-2026-09-09.md`.

## Authorization and boundary

The owner approves a new active ceiling of **28,800 seconds per arm** (eight
hours) and a fresh **86,400 aggregate GPU-worker-second** allocation (24 GPU-
hours). This is a new campaign boundary. It does not enlarge, rewrite, resume,
or combine the completed 36,000-second campaign, the September 8 tuning ledger,
the older readiness ledger, or any historical result.

Fresh campaign root:

`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/`

At plan freeze, no GPU execution has started under this amendment. The fresh ledger is prepared
at that root, with zero consumed and zero reserved seconds. A launch must use a
new output subtree and new seed namespace; it must not reuse the prior runtime
or component-profile output directories.

The amendment changes resource policy only. It does not change the q=20 target,
charts, bridge, factor/strict methods, TensorFlow/TFP backend, float64 choice,
XLA/TF32 policy, four-chain controller, 500-transition chunk, tuning authority,
health vetoes, recovery semantics, or posterior evidence requirements.

## Skeptical audit and evidence contract

### September 9 execution audit and ordering repair

The audit found the prepared campaign was not launchable: fresh launch rejected its
already-created directory, while resume requires a missing campaign-start
record. Its zero-spend ledger also predates the execution integration and uses
a different claim-boundary label. Repair initialization to adopt only this
explicitly prepared, unused ledger, preserving its original contents and
recording the actual source/plan binding before any worker starts. A used
campaign is never reset. Source changes after initialization fail closed.

The eight-hour amendment resolves the measured four-hour affordability failure.
Making status reuse mandatory would now add numerical-kernel risk without
being necessary for execution. The existing tuning shortcut derives only two
validity flags, not the full sampled-state telemetry required here; substituting
it would be wrong relative to the health contract. Keep the unchanged TFP HMC
kernel and full status evaluation. Status reuse remains an optional future
optimization, not an entry prerequisite. Superseded status-reuse prerequisites
have been removed without weakening any numerical check.

The immediate sequence is prepared-ledger initialization, actual P1 command
integration with the checkpoint-aware parallel controller, focused CPU replay
tests across warmup and retained chunk boundaries, then canary and full timing.
Runtime diagnostics and P1 use distinct stateless seed families and separate
checkpoint streams. The first two calls are timing-only; no diagnostic draws
are passed into P1. The unchanged public fixed-transport tuner remains the
artifact authority. Fresh charts are tuned on the smaller-step grid derived
in the prior grid-repair note; this is a warm-start hypothesis, not a default
or guaranteed viable selection.

Run one canary reference wave with a 3,600-second ceiling per worker, derived
by rounding the prior roughly 1,200-second strict setup upward to allow fresh
chart variation. Interrupt and resume waves keep the inherited 400-second
per-worker bounds. The two full timing workers each have the owner-approved
28,800-second maximum. After passing canary and complete health/timing, reserve
the P1 worker forecasts rounded upward to the next hour (convenience safety
margin), never more than 28,800 seconds per worker or the remaining ledger.
This uses measured allocations rather than insisting on reserving every
worker's maximum. The maximum is a ceiling, not a required spend or reservation.

P1 retains the existing bounded four-chain schedule: 500-transition chunks,
2,000 warmup transitions per chain with the latest 1,000 checked at R-hat <=1.05,
then up to 1,000 retained transitions with R-hat <=1.01. Warmup failure stops that
arm before retained sampling. It is a candidate/convergence failure, not an
infrastructure error to rerun until it passes. All completed chunks, including
warmup, are archived and health-checked before the next call. The result labels
P1 mechanics separately from P2/posterior/default promotion.

Pre-mortem: a cap change can hide missing consumer integration; an empty ledger
can accidentally mint a second allocation; a status shortcut can discard required
health; checkpoint indexing can work for two warmup chunks but fail at the
retained-stage reset. Focused tests target each boundary before GPU execution.
The audit permits this revised sequence. The 128-test focused CPU suite passes,
including actual CPU/XLA controller replay before third/fourth/first-retained
chunk publication, cumulative archive continuation, interrupted initialization,
P1 entrypoint delegation, measured per-arm allocations and retained suppression
after failed warmup. Stage-transition tests deliberately force a diagnostic pass
to exercise the retained boundary; they establish no convergence. The suite
also covers actual SIGKILL checkpoint publication, budget accounting, placement
and route-policy enforcement. This does not certify GPU execution or any
posterior claim.

The four-hour cap was a resource veto, not a numerical failure: the prior strict
reserve-inclusive forecast was 16,202.12 seconds. Eight hours makes that prior
forecast fit, but it does not establish that a future repaired controller will
fit. The primary question is whether the unchanged full-status controller,
now integrated into the P1 command, executes an affordable, recoverable and
fully health-checked two-arm schedule with distinct diagnostic and P1 draws.

| Intent field | Binding definition |
|---|---|
| Comparator | Factor backend and independently tuned strict backend, each on chart 0/beta 1/q=20; uninterrupted versus interrupted/resumed identical same-GPU trajectory within each arm |
| Primary criterion | Exact canary samples and complete trace replay; both runtime arms pass full health; six-chunk forecasts plus reserve fit 28,800 seconds per arm and available aggregate budget; P1 records every chunk and its bounded sequential outcome |
| Promotion veto | Nonfinite state/value/score/log acceptance/energy; invalid required status; lost movement; changed target or tuning identity; rejected-state status substitution; checkpoint corruption; memory-growth/XLA/device violation; output collision; stale source; or budget overrun |
| Continuation veto | Source/target/data/method change, corrupted committed evidence, invalid memory/device/XLA execution, unavailable required health, exhausted aggregate budget or no eligible comparator device; candidate R-hat failure is not research-direction rejection |
| Repair trigger | A localized checkpoint, initialization, consumer-integration, timing, or resource failure while target, data, method, hardware class and total authorized budget remain unchanged |
| Explanatory only | Acceptance, finite extreme energy, component timings, allocator readings, elapsed wall time and backend timing differences without uncertainty analysis |
| Not concluded | Posterior correctness, broad convergence, whitening, mode discovery, sampler superiority, default readiness, universal interruption recovery or P2 completion; bounded P1 completion requires its actual terminal receipts |

### Default and assumption audit

| Choice and provenance | Justification and failure mode | Early diagnostic and status |
|---|---|---|
| q=20, chart 0, beta 1, four chains, float64; inherited P1 scope | Cold target-facing mechanics; one chart cannot establish ensemble or mode coverage | Signature/profile binding; representative-scope hypothesis, not ensemble promotion |
| Epsilon grid (0.0275, 0.01375, 0.006875, 0.0034375), leapfrogs (3,8); prior grid-repair diagnostic | Avoid prior large-step pathology; a fresh chart may still be unstable | Fresh target-specific public tuner and untouched chunk health; warm-start hypothesis only |
| Fresh chart/training protocol inherited from Phase 9A source profile, checkpointed per arm | Holds method fixed with exact seed/training provenance; inherited capacity/budget may limit mixing | Training/profile receipts and P1 diagnostics; no chart-quality/default claim |
| 500-transition chunks, 2000 warmup/latest1000, retained1000; inherited bounded P1 protocol | Four warmup and up to two retained chunks; caps may underresolve mixing | Warmup R-hat <=1.05 and retained <=1.01; screens only, failure motivates later planned repair |
| 8-hour ceiling and 24 aggregate GPU-hours; owner authorized | Prior strict forecast fits; chart/device variability remains | Fresh two-call timing, rounded reservations and measured settlements; ceilings, not predictions |
| Full sampled-state status recomputation; current controller | Avoid unproved shortcut; proposed-state scores/native divergences remain unavailable | Full finite/status/movement health; unchanged baseline, unavailable fields are not zero |
| 40% launch utilization, 5 GiB headroom; owner policy; estimated peak 4 GiB inherited | Protect sharing/display; estimate is not a live-memory bound | Placement inventory, verified memory growth and recurring headroom monitor; stop on lost headroom |
| One-hour upward rounding and 30-second grace; convenience/inherited operational margins | Bound worker lifetimes conservatively; not statistical timing guarantees | External timeout, next-chunk reserve and aggregate ledger; no timing ranking |

## Bounded execution sequence

1. Inspect the prior result and verify that its source closure and two older
   ledgers remain unchanged. Do not rerun successful prior work merely because
   the cap changed.
2. Adopt only the unused prepared ledger, archiving its original bytes and
   recording the current binding. Connect the durable checkpoint route to the
   actual P1 consumer without changing the shared numerical kernel.
3. Run focused CPU tests for value/score/status selection, seed/trajectory
   equivalence, stale handoff/corruption rejection, interruption replay,
   accounting, and current-source P1 integration. A failed repair rejects the
   repair, not the research direction.
4. Run a short two-arm GPU parity/recovery canary on eligible non-display GPUs,
   with display fallback only if no non-display GPU meets the 40% utilization
   and 5 GiB headroom policy. Verify memory growth before TensorFlow logical
   device initialization.
5. Run the minimum complete two-arm runtime measurement needed to recompute
   setup, first/repeated 500-transition calls, health, serialization,
   interruption reserve and cleanup grace. Use independent processes and a
   fresh output subtree. Reserve at most 57,600 seconds for two eight-hour
   workers, leaving 28,800 aggregate seconds for canaries and repair retries.
6. Issue a Phase 0 closeout only if the actual P1 consumer, current source
   hashes, checkpoint route, health receipts, and reserve-inclusive forecasts
   all pass. Then automatically execute P1 in separate processes with measured
   reservations and distinct seeds. Preserve candidate warmup/retained failures
   as terminal results; never rerun a convergence failure until it passes.
7. Verify all committed bundle/tensor checksums and record P1 outcome, remaining
   budget and next scientific repair. A graceful resource-limited partial result
   remains resumable; cumulative archives are count-versioned.

### Exact entrypoint and environment

Initialization-only is CPU-only and does not import TensorFlow:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_phase9b_p1_sequential_canary_2026_09_05.py \
  --campaign-root docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z \
  --initialize-only
```

Launch the same entrypoint with `--resume` instead of `--initialize-only` in a
trusted process or detached user service. Each worker pins one GPU UUID, sets
`TF_FORCE_GPU_ALLOW_GROWTH=true` before import, verifies growth before logical
device creation, enables TF32 and uses the unchanged single-trace XLA kernel.
Service/jobs preserve the exact command/environment. The actual P1 command
delegates to the recovery coordinator with P1 enabled; it does not consume the
old r14 audit or old ledger. `phase0-readiness.json` is current-source
computational evidence, not an approval token. Keep this plan/code frozen after
initialization; update master/reset/result notes for live status. No external
release or posterior/default promotion is authorized by this command.

## Resource and stop conditions

The prior descriptive reserve-inclusive estimates were factor 10,751.04 seconds
and strict 16,202.12 seconds. They fit the new per-arm cap, and their combined
26,953.15 seconds fit the new 86,400-second aggregate allocation. This is only
an entry feasibility check; the repaired route must be remeasured. The budget
is accounted as parent-measured worker lifetimes, including failed attempts,
setup, compilation, serialization and cleanup, not elapsed wall time multiplied
by the number of workers.

The campaign stops without silently changing scope if source identity, target,
data, bridge, hardware class, privacy boundary, scientific contract, memory
policy, checkpoint integrity or aggregate budget changes. A failed candidate or
localized infrastructure repair remains a charged repair trigger. Do not drop
the interruption reserve, reuse diagnostic draws as untouched P1 evidence,
terminate unrelated GPU processes, or claim that the larger cap establishes
scientific validity.

## Required artifacts

The fresh campaign must contain a source-bound campaign start record, budget
ledger, per-attempt launch/job/supervision receipts, GPU and memory provenance,
checkpoint bundles, complete health receipts, canary comparison, runtime
forecast, post-run verification, and terminal result note. The result must
report hard vetoes, descriptive-only metrics, uncertainty limits, remaining
budget, failed attempts, and the exact next action. The prior campaign remains
the evidence for the four-hour diagnostic and is not overwritten.
