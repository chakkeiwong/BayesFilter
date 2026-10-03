# Phase 9B diagnostic-NaN checkpoint repair

Date: September 10, 2026 (Asia/Shanghai).
Governing execution plan: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-plan-2026-09-09.md`.
Master: `bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`.

## Question and skeptical audit

Can the existing campaign resume after a rejected candidate's undefined tail
ESS prevented publication of the strict tuning checkpoint? This is a repair
of serialization and continuation, not a new tuning protocol or diagnostic
definition. The exact comparator is the same typed tuning result before and
after disk recovery, plus the original, already committed factor checkpoint.
No sampler, target, data, chart, grid, seed, dtype, XLA setting, health threshold,
candidate choice or campaign allocation changes.

The original wrapper hashes raw Python floats with strict JSON. The public
tuner already serializes undefined diagnostic values as explicit nonfinite
tags. That mismatch caused the stopped worker. The first repair's new-only
tests passed but did not test the old committed checkpoint: tuple markers in
the new hash change even finite payload hashes. That draft must not launch.
Use the typed result's existing nonfinite-safe public artifact hash instead,
which equals the old strict-JSON hash for finite payloads. Preserve typed tuple
markers in stored fields. Add a disk-reload regression and verify the actual
factor checkpoint before accepting the repair. The migration draft remains
historical, and the final source binding is recorded only after these checks.

The audit also checks that the two arms still tune separately, short ESS and
R-hat values are not convergence or superiority evidence, failed diagnostics
are not sanitized to zero or removed, and reused checkpoints retain their
original source identities. The sequence is executable only after the checks
below pass. No new review-token or launch-approval machinery is required.

## Intent and evidence contract

| Role | Criterion |
|---|---|
| Primary engineering pass | Disk round-trip preserves candidate choice, tuple structure, NaN/positive and negative infinity, public payload hash, hard vetoes, and the old factor result. |
| Candidate promotion veto | Undefined selection efficiency remains a veto of that candidate. It is not a veto of another candidate independently selected and verified by the public tuner. |
| Continuation veto | Corrupt checkpoint, changed numerical source/target/data, missing full health, unsupported tuner, invalid GPU memory growth/XLA, exhausted budget, or unavailable eligible comparator GPU. |
| Repair trigger | Local serialization, recovery or process/resource failure with the scientific contract unchanged. |
| Explanatory only | Short-chain ESS, finite R-hat, observed execution times and allocation counts. |
| What passing cannot establish | Posterior correctness, convergence, sampler ranking, default readiness, or bitwise recovery from arbitrary live GPU instructions. An interrupted uncommitted work unit is recomputed. |

Numeric provenance: the 86,400 aggregate GPU-worker-second budget and 28,800-second
per-arm ceiling are owner-approved, not new allocations. The unchanged reference
wave cap is 3,600 seconds; interruption/resume caps are 400 seconds. The active
plan derives these operational bounds from prior observed setup/call times.
The eight-result selection window is an inherited mechanics-only hypothesis,
not a defensible posterior diagnostic window. Zero-variance tail indicators
are the cheap regression exposing its expected undefined ESS. Synthetic test
thresholds are deliberately permissive so the test isolates the nonfinite veto;
they do not change campaign thresholds.

## Execution and preservation

1. Complete the local serializer repair and tests. Use CPU-only diagnostic
   execution with `CUDA_VISIBLE_DEVICES=-1`, memory-growth environment set,
   and one intra/inter-op thread. This is not NeuTra training.
2. Reload the actual finite factor checkpoint and a typed fixture constructed
   from the stopped strict artifact. Preserve candidate vetoes and validate the
   public artifact hash. Verify all committed bundle/tensor checksums.
3. Record the final localized source migration, preserving the original
   `campaign-start.json`, plan, historical attempts, setup/stream identities and
   ledger accounting. A migration changes execution provenance, not old evidence.
4. Probe GPUs with trusted permissions. Prefer non-display devices with load
   at most 40% and at least the inherited 4 GiB worker estimate plus the
   owner-required 5 GiB free headroom. Preserve the same-GPU replay comparator;
   do not move a partially validated arm or terminate unrelated processes.
5. Resume the existing P1 entrypoint with `--campaign-root ... --resume`.
   Reuse factor results and both charts. Rerun strict tuning because no typed
   tuning checkpoint committed. Continue reference, actual SIGKILL/resume
   canaries, two 500-transition timings per arm, Phase 0 and bounded P1 only
   under the unchanged plan. No P2 auto-launch.

Campaign root:
`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/`.
Repair evidence goes in a fresh `serializer-repair-validation-*` subdirectory;
each GPU launch still receives a fresh versioned `launches/` subtree. Use
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; preserve exact commands, commit
plus source hashes, memory/device receipts, logs and test results there.
Before the repair, the campaign has spent 2,187.64961813399 GPU-worker seconds,
has no reservations, and has 84,212.35038186601 seconds left. CPU diagnostics
do not charge this GPU ledger. Failed GPU retries do.

Results and ongoing status are recorded in
`bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-result-2026-09-09.md`
and the master reset memo. The strongest alternative explanation remains poor
mixing in a tiny selection window; this repair neither fixes nor conceals it.

## Completed pre-run audit

The corrected repair passes **161 CPU tests**. The saved finite factor
checkpoint restores with its original public hash; the actual strict public
artifact survives disk round-trip with all six diagnostic NaN tags, candidate
0 selection and candidate 1's nonfinite-efficiency veto preserved. Original
checkpoint, campaign-start, plan and budget bytes remain unchanged. Function
comparison confirms `_setup`, `_run_controller`, health, canary comparison,
forecast, allocation and seed definitions are unchanged. The changed execution
functions only carry and validate migration provenance.

The final migration records the exact before/after source hashes and test
receipts. Its earlier, unlaunched hash-incompatible draft is archived separately.
The trusted placement preflight finds both original non-display GPUs eligible:
GPU 1 has 0% load and 32,212 MiB free; GPU 0 has 1% load and 31,893 MiB free.
These are launch-time observations, not availability guarantees. The coordinator
rechecks placement and runtime headroom. No additional compute allocation is
introduced. This audit permits the bounded resume under the existing plan.
