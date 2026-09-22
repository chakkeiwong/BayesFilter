# 72-core full-run repair/reset memo

Date: 2026-09-03  
Parent plan: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`

## Attempt-01 classification

The first full launch ended in the controller before any preparation child
could import TensorFlow.  `_run_prepare` opened a file below
`full/attempt-01/fresh_chart_preparation/` while that directory was still
reserved for the child.  The resulting `FileNotFoundError` is a harness
ordering defect.  It provides no GPU, chart, numerical, or scheduling
evidence.  The empty attempt directory is retained to preserve provenance.

## Repair

The controller now creates the parent directory and writes the preparation
stderr log as a sibling file (`fresh_chart_preparation.prepare.stderr.log`).
If a preparation child exits without a manifest, the controller writes a
typed failure receipt with its return code and log path.  The top-level
preparation path also records unexpected exceptions.  The full run now passes
one monotonic campaign deadline to preparation and each sequential barrier;
barriers cannot silently reset the 14,400-second cap.

These are localized infrastructure repairs.  They do not change the target
signature, strict backend, chart/checkpoint protocol, candidate grid, seed
namespace, counts, CPU affinity allocation, GPU preparation device, or
promotion/nonclaim rules.

## Verification and retry rule

Before retrying, run:

```text
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  docs/benchmarks/run_ssl_lstm_q20_72core_process_parallel_2026_09_03.py \
  bayesfilter/inference/process_topology.py \
  tests/test_q20_process_topology.py
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_q20_process_topology.py
git diff --check
```

The next full launch must allocate a fresh `full/attempt-02` directory.  A
preparation failure, worker failure, cap stop, or incomplete artifact set is
classified at P3-R and preserved; it is not silently retried in place.  No
posterior, convergence, whitening, mode-discovery, or CPU-default claim is
permitted from this campaign.

## Post-attempt-02 identity repair

Attempt-02 completed its barriers but every one of the 48 screen tasks failed
with `chart checkpoint bridge signature mismatch`.  GPU preparation hashed the
fixed covariance-weight sum as `3.0`; CPU workers hashed the same rule as
`2.9999999999999996` because TensorFlow reduced the values in a different
order.  The resulting common failure was not candidate-local evidence.  The
controller is now fail-closed when a screen barrier has no completed task.

The source-owned q=20 bridge facts use `math.fsum` over the declared fixed
weights, making the identity independent of device reduction order.  A focused
regression covers the exact sum and repeatability.  Since this changes the
identity guard, the next action is a fresh post-repair canary, followed only on
success by a new full attempt directory; attempt-02 remains quarantined.

## Canary attempt-03 worktree-concurrency failure

Attempt-03 passed through parity, screen, and selection startup, then failed at
the finalization barrier because a concurrent Git merge removed the untracked
launcher while the children were still being launched.  The preserved stderr
logs show `can't open file ...run_ssl_lstm_q20_72core_process_parallel_2026_09_03.py`;
the attempt has no summary and supplies no numerical, resource, or topology
evidence.  This is an infrastructure/worktree-concurrency failure, not a
candidate veto.

The worktree is now stable.  The launcher was restored with deterministic
bridge identity checking, safe preparation logging, child joins, a single
campaign deadline, and fail-closed screen coverage.  The bridge/topology
regression passed (`6 passed`).  The next action is a fresh post-repair canary
(`attempt-04` or the next unused directory), followed by a new full attempt
only if that canary completes.  No prior attempt is resumed or overwritten.

## Canary attempt-04 pass and full-run authorization

One direct launch again failed before controller initialization because the
absolute-path process did not yet place the repository root on `sys.path`; no
artifact or numerical work was produced.  After that localized repair, the
fresh `canary/attempt-04` run completed in `585.6644140318967` seconds.  It
passed topology/affinity, CUDA-hidden CPU-worker, XLA, durable coverage, and
serial/process parity checks.  The detailed receipt is
`docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-canary-attempt-04-result-2026-09-03.md`.

P3 full execution is now authorized under a new `full/attempt-03` directory,
with the existing `14,400` second full cap and `15,600` second total campaign
cap.  The canary remains mechanics-only; no posterior or Phase 9B conclusion
may be drawn from it.

## Full attempt-03 non-finite serialization failure

The full launch reached the screen barrier, where several candidate probes
returned non-finite diagnostics.  The worker correctly classified the probe,
but the controller's strict JSON writer raised `ValueError: Out of range float
values are not JSON compliant: nan` while writing that failure record.  The
attempt is preserved by the controller receipt at
`docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03/full/attempt-03/controller_failure.json`;
no numerical result from it is admissible.

The artifact boundary now tags non-finite values (`{"__nonfinite__":"nan"}`)
and keeps the explicit failure/status row.  A focused regression suite passed
(`7 passed`).  This repair does not alter the target, bridge, chart, grid,
seeds, topology, or caps, but it requires a fresh canary before the next full
attempt.  The campaign is therefore in `P2_REPAIR_CANARY_RETRY_ACTIVE`.

## Canary attempt-05 pass and renewed full authorization

The fresh canary after the serialization repair completed in
`587.6819816830102` seconds.  All 8/2/6 barrier workers were ready, exited
cleanly, and wrote durable records; serial/process parity passed.  The
machine-readable and reader-facing receipt is
`docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-canary-attempt-05-result-2026-09-03.md`.

The next action is a fresh full attempt under the unchanged contract and cap.
No partial result from full attempt-03 is reused, and the canary remains
mechanics-only.

## Full attempt-04 controller interruption (2026-09-03)

The fresh full attempt reached 46 of 48 screen records.  The remaining screen
workers were still in their declared fixed-HMC evaluations when the controlling
session was interrupted.  The traceback places the interruption at
`_wait_for_paths`; no full summary or complete worker set exists.  This is an
operator/session interruption, not evidence of a candidate failure, numerical
veto, memory exhaustion, or process-topology failure.  The partial attempt is
preserved under `full/attempt-04/` and is excluded from all comparisons.

The retry keeps the target, fresh-chart protocol, bridge identity, candidate
grid, seeds, hardware class, `8x4 + 2x8 + 6x4` topology, and 14,400-second full
cap unchanged.  It uses a new attempt directory and an uninterrupted
controller session.  The attempt consumes the existing campaign budget; no
additional scientific authorization is needed under the unchanged contract.

## Full attempt-05 cap-closeout defect and repair (2026-09-03)

The uninterrupted retry completed all 48 screen tasks and 26 of 32 selection
tasks (13 in each replication stream).  The global 14,400-second cap arrived
while both workers were evaluating the next long candidate.  The controller
terminated those workers, but the old closeout path interpreted their still-
unset return codes as a generic worker failure and emitted neither a barrier
timeout receipt nor `full_summary.json`.  This is a harness/artifact defect,
not a numerical, memory, or topology finding.  The 26 completed records remain
diagnostic timing evidence only; no complete selection or finalization result
is admissible.

The repair introduces a typed `ParallelCampaignDeadline` path.  At a cap it
terminates children, records readiness, return codes before/after termination,
started and durable task IDs, and missing coverage in
`barrier_timeout.json`; `_run_full` writes `full_summary.json` with
`CAP_STOP_INCOMPLETE` and explicit nonclaims.  The focused timeout and
topology/bridge tests pass (`9 passed`).  Since the artifact boundary changed,
the next action is a fresh canary under the unchanged target, seeds, topology,
hardware class, and caps.  A cap increase or candidate repartition requires a
separately reviewed plan and is not smuggled into this repair.
