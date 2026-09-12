# Phase 9B durable recovery and two-arm runtime measurement

Date: 2026-09-09 (Asia/Shanghai). Status: implementation and focused validation.
Parent: `bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`.

## Authorization and question

The owner raises the active per-arm runtime ceiling from 2,600 to **14,400
seconds**, and authorizes **36,000 aggregate GPU-worker seconds** for repair,
interruption canaries, full health checks, and a two-arm runtime estimate.
This is a new bounded allocation, not permission to reset prior spending or
combine the unused September 8 allocation with this one. Old ledgers and runs
remain unchanged. The allocation does not itself establish posterior readiness.

Can the unchanged q=20, float64, GPU/XLA, four-chain controller preserve its
committed work across abrupt interruption, produce complete health receipts,
and execute an affordable six-chunk schedule on each backend? Factor and strict
are separate processes on eligible non-display GPUs. Preserve the owner's
40% admission-utilization limit, 5 GiB free headroom, memory growth, and display
fallback rule. A 4 GiB estimated worker peak is inherited from the passing
parallel-tuning admission screen, not a hard allocator cap.

## Skeptical audit and evidence contract

The old diagnostic is insufficient: it archives only after two long calls and
does not persist full trace health. The shared controller's callback route also
has no durable replay. The older archived controller rejects partial orphan
files rather than recovering them. Do not treat any of these as interruption
clearance. Repair the active shared callback route and its diagnostic consumer.

Recovery means a process can restart after interruption at any instruction,
verify the scientific identity and committed checksums, restore completed
stages/chunks, and recompute the unfinished unit from its saved inputs and
stateless seed. It does **not** mean restoring a live CUDA instruction pointer.
Keep 500-transition sampling chunks unchanged; an interrupted compiled chunk
can lose up to that chunk's work. Partial writes remain preserved but are never
mistaken for committed data. Committed corruption is a hard error, not permission
to silently regenerate evidence. Chart construction and tuning are separately
committed setup stages; unfinished setup stages may be replayed, not promoted.

| Intent field | Predeclared interpretation |
|---|---|
| Comparator | Same source, chart, tuning handoff, initial state, seed, device, precision, and chunk shape, uninterrupted versus interrupted/resumed |
| Primary recovery criterion | Exact restored committed tensor bytes and identical resumed versus uninterrupted sample/trace tensors on the same GPU; no duplicated or missing chunks |
| Runtime criterion | Complete first and repeated 500-transition calls for both arms; descriptive six-chunk forecast including setup, health, serialization, and one extra steady chunk reserve fits 14,400 s per arm |
| Health veto | Nonfinite state, target, log acceptance or energy; missing/invalid target status; any chain without movement; corrupted checkpoint or changed identity |
| Continuation veto | Invalid source/target identity, resource-policy violation, corrupt committed evidence, or exhausted allocation; a numerical candidate failure stops that candidate rather than rejecting the research direction |
| Repair trigger | Infrastructure interruption, partial write, failed replay canary, or missing telemetry; repair and retry in a fresh attempt within this allocation |
| Explanatory only | Acceptance, finite extreme energy errors, single-stream runtime, allocator peaks; no backend ranking |
| Not concluded | Posterior convergence, model/reference agreement, mixing, chart quality, speedup superiority, P2 readiness, or full P1 sampling completion |

## Execution and numeric provenance

1. Implement immutable committed bundles with ordinary SHA-256 checksums and
   atomic publication. Store state, seed, trace, identity, timing, and per-call
   start/end events. Add shared-controller replay without changing its kernel,
   random-seed derivation, warmup/retained classification, or stopping rules.
2. CPU-only tests hide GPUs before import. Inject process death before commit,
   during serialization, and after commit; check corruption and identity
   rejection, deterministic replay, health vetoes, and accounting. Tiny Gaussian
   chains are explicit reference mechanics, not q=20 scientific evidence.
3. GPU canary uses two eight-transition chunks per arm (convenience-chosen short
   mechanics fixture). Compute an uninterrupted reference; kill a sibling
   execution during its second chunk; resume from disk in a fresh process and
   require exact sample/trace equality. This short fixture does not forecast P1.
4. Only after both canaries pass, measure two full 500-transition controller
   calls per arm, through the actual shared controller. Stop before its third
   call as a declared diagnostic boundary. The real P1 requirements remain four
   warmup chunks (2,000 transitions), then two retained chunks (1,000), with
   unchanged modern R-hat windows and thresholds. No shorter chain is promoted.
5. Persist complete sampled-state status, shared health, and energy diagnostics
   (`delta_H = -log_accept_ratio`). The inherited finite extreme-log-acceptance
   threshold is explanatory only, not a newly invented divergence veto. TFP's
   unavailable native divergence count must not be reported as zero.
6. Forecast each six-chunk schedule from measured setup + first call + five
   repeated calls + observed archiving/diagnostics + one repeated-call reserve.
   This is a descriptive estimate, not a confidence bound or a guaranteed
   completion time. Record both arms even if one fails affordability.

Budget envelope: canaries/setup at most 4,000 aggregate worker-seconds initially;
two runtime workers at most 14,400 seconds each (28,800 aggregate); at least
3,200 seconds initially unreserved for localized repair. Released reservations
remain available within 36,000 seconds. Parent sums actual child lifetimes,
including failed/killed attempts; wall-time overlap is not a budget discount.
Use a 30-second termination grace (inherited supervisor safety allowance),
included within every worker cap. Do not start a worker without a reservation.
After parent/process loss, reconcile surviving child identity before retry;
uncertain unmeasured use is charged conservatively from its reservation.

The exact versioned campaign root, source closure, command, git revision,
environment, seed map, GPU UUIDs, startup memory receipts, worker timing, ledger,
and final result path are recorded before launch. The launcher is
`docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py`;
artifacts live under `docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/`.
Result: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-result-2026-09-09.md`.

Pre-mortem: a between-calls pause is not a mid-call recovery test; a shorter
fixture is not a full runtime estimate; replay with different seeds is not
equivalence; a successful tuning artifact is not complete controller health;
raising a cap does not repair missing diagnostics. The tests and separate
canary/runtime stages directly distinguish these failure modes. Public frozen-
transport tuning remains the sole tuning authority; restored typed results
must pass its existing handoff verifier and current source-closure check.

## Prelaunch audit and commands

The skeptical implementation audit passes for this diagnostic contract. The
shared kernel and seed derivation are unchanged; replay is outside TensorFlow's
compiled call. Full health checks require complete sampled-state telemetry,
while unexposed proposed-state scores and native divergence counts are not
silently claimed. CPU tests: **135 passed, 191 dependency warnings**, 21.84 s;
compilation and `git diff --check` pass. This includes five actual SIGKILL
commit-boundary cases, shared-controller numerical replay, lost-parent ledger
reconciliation, invalid health traces, new caps, and route discovery. The old
P1 source-audit fixture now explicitly verifies rejection of its stale receipt.

Trusted inventory at 2026-09-08 19:21 UTC (September 9 in Asia/Shanghai) shows
GPUs 1 and 0 eligible under owner utilization/headroom rules; GPU 2 is display.
GPU 0's unrelated lightweight context is not an exclusivity veto. No unrelated
process is touched. The initial campaign root is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-10gpuh-20260908T192200Z`.

First run the bounded canary stage:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py \
--campaign-root docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-10gpuh-20260908T192200Z \
--canary-only
```

After inspecting both canary receipts, the same command with `--resume` and
without `--canary-only` performs the two-arm full runtime measurement. Each
child is wrapped in its own external `timeout`, so parent death cannot leave
an unbounded GPU worker. A restarted coordinator holds the campaign lock,
reconciles prior reservations, terminates only positively identified campaign
workers if needed, and charges uncertain elapsed use conservatively. Normal
worker settlements are actual observed lifetimes and are never charged twice.
The short canary has a fresh interrupted branch per fault-injection attempt;
ordinary runtime retries reuse committed chunks from their original stream.
