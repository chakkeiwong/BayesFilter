# Funded state-space continuation

Terminal update, September 29: this attempt finished all ten C1 recoveries
(fit 78 reused; nine fresh completions), passed eight SSM GPU mechanics cells,
and failed all four public preflight cells with `target lacks full-chain XLA
qualification`. The service exited; pricing and main did not start. The
post-hoc beta-binomial summary is 256/256 complete. Original confirmation and
coverage failures remain unchanged. The ledger has 57.878 GPU hours before the
next launch and no unsettled charge from this attempt.

The [XLA repair](bayesfilter-hmc-ssm-xla-preflight-repair-2026-09-29.md) and
[new execution note](artifacts/hmc-ssm-xla-repair-2026-09-29/result.md) describe
the resumed work. The launch-time status below is historical.

The implementation is ready to execute and its 36-GPU-hour ceiling is funded.
The owner added 50 GPU hours, bringing the settled available balance to 58.736
GPU hours including the separate 1,200-second NeuTra reserve. The additive
ledger preserves every earlier charge exactly once. It is now the active ledger:
`artifacts/hmc-ssm-funded-2026-09-28/grant-ledger.json`.

At launch all three same-class GPUs had foreign compute processes and reported
approximately full utilization. The detached service
`bayesfilter-hmc-funded-ssm-20260928-r1.service` is running the capacity queue;
it has not launched another numerical worker. The queue begins work automatically
when trusted telemetry finds capacity. It can wait at most 13,919.996 seconds,
the original four-hour contention reserve less the prior 480.004-second wait.
Waiting is conservatively charged once, separately from numerical stage limits.
Later contention can still delay or stop a fit; this queue is not an exclusive
GPU reservation.

The first recovery completed C1 fit 78, timed out fit 79 under sustained foreign
GPU load, and deferred eight others. The new attempt reuses completed fit 78
without consulting its posterior outcome and attempts the remaining nine original
seeds once each under the same 80-minute recovery ceiling. Original C1 source,
data, seeds, tuning rules and posterior checks remain fixed. The old results
stay intact; the combined 247/256 completed baseline remains a post-hoc recovery
summary, with nine missing fits.

After recovery, state-space GPU mechanics, public-pipeline preflight and complete
fit pricing run before the 32-slot main matrix. The SSM numerical package and
checked references are byte-identical to the earlier prepared version. Only the
external queue and recovery launchers changed. The package identity is
`f2346887f86eefafe110ba7ff77604b000c5e903123e71144b91fa8a22d4b621`.
The queue binds one GPU before SSM preflight and retains that UUID for pricing
and main sampling.

The latest 40 launcher/planner checks passed, covering exact seed selection,
reuse of completed fits, corrupt evidence, unknown/busy capacity, interruption,
one-time charges, original deadlines, fixed SSM device and stopping after a
failed preflight. An original-source summary check preserved the full 256-slot
denominator and reused fit 78. CPU tests intentionally hid GPUs. These checks
establish scheduling behavior; actual GPU numerical qualification remains pending.
The additional CPU repair charge is a conservative 900 seconds from the existing
CPU balance, including implementation, tests, snapshots and reporting.

The original campaign clock remains September 28 at 05:03:52 Shanghai.
New work stops September 29 at 23:03:52, computation stops September 30 at
03:03:52, and the terminal report is due September 30 at 05:03:52. Funding does
not reset those limits. Earlier CPU preparation time is reported separately.
The 36-GPU-hour cap, 22-hour main allocation, numerical criteria and complete
32-slot denominator remain unchanged.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Queue the funded continuation | Engineering regressions passed; source unchanged | All GPUs busy at launch | When capacity becomes available | Automatic bounded admission then nine remaining retries | Recovery or SSM completion |
| Run SSM preflight and prices | Prepared targets/references; GPU qualification pending | No numerical preflight has run yet | GPU compatibility and complete-fit cost | Execute preflight and full pilots before main | All 32 fits converge or fit their per-case ceilings |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Previous contention timeouts retained; original coverage failures remain |
| Statistically supported ranking | None |
| Descriptive-only differences | Retry runtime and telemetry |
| Default readiness | Not established |
| Next evidence | GPU preflight, complete-fit prices, per-slot posterior checks |

The strongest risk is another competing job arriving after an idle observation.
Admission and extensions bound that failure but cannot guarantee machine access.
Neither funding nor completed engineering tests establishes posterior calibration.
The [active plan](bayesfilter-hmc-c1-recovery-and-ssm-launch-2026-09-28.md)
records the scientific criteria and stop conditions. Exact command, configuration,
source/dependency hashes, JUnit receipts and live status are under
`artifacts/hmc-ssm-funded-2026-09-28/`; current progress is
`runtime-r1/status.json` and `runtime-r1/queue-recovery-progress.json`.
