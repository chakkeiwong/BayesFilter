# Phase 9B eight-hour amendment execution result

Updated: 2026-09-09 16:04:50 UTC (September 10, 00:04:50 Asia/Shanghai).
Status: `M4_P0_GPU_RUNNING_P1_AUTO_AFTER_READINESS`.
Plan: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-plan-2026-09-09.md`.
Master: `bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`.

## Engineering repair and pre-run audit

The unused prepared ledger can now be adopted without minting another budget;
its original bytes are archived and interrupted initialization is idempotent.
The actual P1 CLI delegates to the parallel checkpoint coordinator. Chunk
archives use global checkpoint indices, actual stage seeds and previous states;
cumulative archives are count-versioned. P1 uses distinct seeds, measured
per-arm reservations, the full-status shared controller and unchanged TFP HMC.
Failed warmup prevents retained sampling and is recorded as candidate failure.
A partial resource stop is not misreported as completed P1 and remains resumable.

No numerical-kernel/status-reuse optimization is introduced. The audit rejected
making that optimization an entry prerequisite after the cap increased. The
finiteness-only tuning shortcut is not equivalent to complete sampled-state
health and is not substituted for it.

Focused validation: **128 passed**, 191 dependency deprecation warnings, 35.83s.
CPU-only command uses `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true`
and `/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q` on
`test_ssl_lstm_q20_phase9b_recovery_runtime.py`,
`test_ssl_lstm_q20_phase9b_p1_canary.py`, `test_ssl_lstm_q20_phase9b_readiness.py`,
`test_durable_tensor_checkpoint.py`, `test_campaign_budget_ledger.py`,
`test_parallel_tuning.py`, `test_display_gpu_policy.py`, and
`test_neutra_hmc_route_policy.py`, all under `tests/`. Compile checks pass.
Retained-stage fixture diagnostics are deliberately forced to exercise control
flow; those tests establish recovery mechanics, not convergence.

Trusted NVIDIA inventory at 23:47 +08:00 passes: GPU 1 idle, GPU 0 at 2% with
337 MiB used, GPU 2 at 36% with 768 MiB used. These are snapshots, not guaranteed
availability. Launch selection rechecks display identity, load and memory;
unrelated processes are not terminated. GPU framework growth is verified in
each worker before logical-device initialization.

## Campaign and live artifacts

Root:
`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/`.
Budget: 86,400 aggregate GPU-worker seconds, each worker capped at 28,800.
At the 16:04:50 UTC snapshot, 7,200 seconds are reserved for the reference wave,
zero seconds are settled and 79,200 seconds remain unreserved. Live elapsed
worker time has not yet settled; zero settled time does not mean zero usage.
Older campaigns and their budgets remain separate and unchanged.

The actual P1 entrypoint started at **16:00:10 UTC**, detached user service
`bayesfilter-q20-phase9b-24gpuh-20260909.service`, coordinator PID 704521.
Service invocation uses `Type=exec`, the repository working directory,
`KillMode=control-group`, `TimeoutStopSec=30`, `TF_FORCE_GPU_ALLOW_GROWTH=true`,
`CUDA_VISIBLE_DEVICES=-1` for the framework-free coordinator, and
`PYTHONUNBUFFERED=1`. Workers override visibility with their selected UUID.
The command is the frozen plan's P1 entrypoint with the same campaign root and
`--resume`. Console log: `supervisor-20260909T155926Z.log` under the root.

First attempt: `launches/reference-561614e446/`. Factor worker PID 704530 runs
on GPU 0 (`GPU-a1ea1946-07c0-8ed5-2ba1-d96f82c89cd3`), not the display device.
Its chart checkpoint is committed and nine fresh public-tuner chain calls have
completed. The launch inventory found GPU 1 at 64% utilization from another
process, above the owner's 40% threshold, so strict queues. The display GPU
remains untouched. This is process-parallel-capable execution but **one active
arm at this snapshot**, not two simultaneous workers. Placement is rechecked
before queued work starts; no unrelated process is killed.

Worker memory growth is verified before logical-device initialization. Source
and frozen-plan binding still match. A committed bundle checksum verifies;
the chart bundle stores its checkpoint as a typed Python payload (no separate
TensorFlow tensor files in that bundle). GPU sequential health, interruption
comparison and full timing are not yet complete. Snapshot:
`execution-snapshot-20260909T160450Z.json`. Older ledger SHA-256 values still
match the prior result: historical ten-hour `f7ad2484ca723cfb210a38a60ae3ea4f841cff46a321f6294285562e7c5207ce`,
older readiness `1961d07cf421b06abfb6ec4b62830b44e4052849505d72cd3cf0cea087160415`.

`campaign-start.json` freezes source hashes, plan, command, Python, Git state,
budget and claim boundary. `campaign_budget_ledger.json` records all reservations
and parent-measured settlements. Per-attempt jobs, supervision and worker
manifests preserve seeds, source/target/training identities, GPU, memory growth,
XLA/TF32, wall time and output paths. `phase0-readiness.json` is issued only after
fresh recovery, complete runtime health and affordable measured allocations.
`p1-result.json` requires actual bounded P1 terminal evidence; partial results
have separate versioned names. Canary and validated timing files remain frozen.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue approved fresh campaign | Engineering checks pass; GPU reference setup/tuning running | Factor startup growth passes; strict queued by GPU 1 load; no new sequential health result yet | Fresh charts may tune poorly or mix slowly | Finish canary/runtime; automatically run bounded P1 if ready | No recovery/health or posterior claim from startup alone |

## Inference status

| Evidence class | Current conclusion |
|---|---|
| Hard veto screen | CPU mechanics pass; fresh GPU candidate vetoes not yet evaluated |
| Statistically supported ranking | None |
| Descriptive-only differences | Historical timings motivate feasibility only, not factor/strict superiority |
| Default-readiness | Not established |
| Next evidence needed | Fresh GPU replay/health/forecast; bounded P1 results; later multi-seed uncertainty and downstream posterior/reference validation |

Post-run red-team pending. Main pre-run alternative explanation: an inherited
small chart or short bounded schedule can fail mixing even with a correct
controller. That rejects the present candidate/screen, not the entire transport
research direction. Source identity, corrupted evidence or unavailable required
health are genuine continuation vetoes.
