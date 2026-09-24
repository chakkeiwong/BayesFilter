# q20 continuation through September 25

Active operational continuation is now maintained in the
[master continuity and command-approval repair](bayesfilter-q20-master-operations-repair-2026-09-21.md).
Its supervisor replaces the waiting one-shot observer described here; the
numerical parent, deadline and total compute allowance are preserved.

The owner postponed the deadline to September 25, 2026, 18:00 Asia/Shanghai
(10:00 UTC), and requested continued execution. This replaces the September 21
tuning stop as the final calendar deadline. Keep the existing total GPU-hour
allowance and separate CPU allowance: changing a calendar deadline does not
silently add device-hours. At the initial observation about 62.3 GPU campaign
hours remain. The compute balance can therefore stop work before September 25.

The question remains whether plain NeuTra HMC, or the already permitted tempered
NeuTra ensemble, yields a stable estimate of the fixed q20/T30 four-dimensional
float64 UKF-approximate posterior. The independent integration reference is the
comparator. Preserve all training, target, numerical-health, convergence, ESS,
MCSE, start-group and reference criteria. Training is complete for the direct
cohort. The currently running tuner has no verified kernel yet, so posterior
sampling remains conditional on complete tuning and fresh verification. Identity
mass remains fixed in transport coordinates.

## Operational repair and evidence contract

The live master embeds an eight-hour tuning allocation in its supervisor and
controller. Editing the service timer alone would not extend it. Do not alter a
live checkpoint or restart unfinished native work merely to change that limit.
Prepare and start a bounded continuation supervisor now. It observes the
existing service, then takes over only after the original coordinator and owned
worker have stopped and the original ledger has settled a tuning budget pause.
If the original master completes the estimate, the continuation exits. A target
invalidity or unclassified failure does not authorize a budget-only restart.

Use a fresh campaign directory, retaining the old campaign verbatim. Copy its
settled accounting and completed-stage receipts, preserve exact source and
configuration identity, and resume the latest numerical tuning checkpoint.
Grant one additional tuning attempt under the owner's deadline extension.
Extend only the resumed controller's elapsed-time ceiling; preserve all spent
time and work counters, candidates, starts, seeds, numerical evidence, pending
repairs, finite work-unit caps and verification rules. Record the old/new time
ceilings and the continuation driver's source checksum. The repository-owned
numerical binding and existing typed controller remain responsible for evidence
and retained-member export. No caller-issued numerical identity is introduced.

The extended cumulative tuning allowance is derived at takeover:

`previous tuning spend + min(remaining campaign funds - protected downstream funds,
calendar seconds until September 25 at 18:00 - stop grace)`.

Protected downstream funds are the existing first posterior-assessment quote,
first independent-reference assessment, posterior analysis and remaining shared
repair hold. These are measured quotes with the existing factor-two engineering
reserve, not a convergence guarantee. At the current prices the protection is
about 11.00 hours. This prevents an extended tuner from consuming the whole
campaign. Later sampling uses the selected member's measured trajectory cost.
Stop for exhausted money/time/work units, scientific invalidity, or a completed
valid estimate; do not renew the allowance on retry. A failed candidate alone
retains the existing planned repair/fallback behavior.

The continuation driver is host-side administration outside the unchanged
490-file numerical snapshot. It invokes the same worker bootstrap, GPU memory
growth verification, public numerical checkpoint loader and retained export.
It must record both source sets. It uses the same tfgpu environment and at most
one GPU worker at a time; four chains remain batched. Waiting for the existing
service performs no numerical work and grants no extra GPU allowance.

## Audit and verification before launch

Skeptical audit found two material traps: a later systemd timer leaves the
eight-hour internal limit in place, and restarting with a changed full protocol
would invalidate saved training/tuning identities. The continuation above
addresses both while preserving the numerical program and evidence. Acceptance
near one is a step-repair trigger, not convergence. No new method, ranking,
mass adaptation or relaxed screen is introduced. A new deadline alone cannot
prove the cohort is affordable or posterior-ready.

Before launching the continuation, check budget conservation and deadline
clipping with simulated clocks, refusal of unsettled/corrupt predecessors,
retention of completed stage receipts, and refusal of inappropriate terminal
states. Run a tiny CPU-only known-target checkpoint continuation through actual
fresh verification and sampling, comparing its numerical result with an
uninterrupted run. This is an engineering reference test with GPUs hidden; it
cannot establish transport quality or GPU performance. Record the exact checks
and their results below. Do not modify the running source snapshot.

Artifacts: `docs/plans/artifacts/ssl-lstm-q20-deadline-continuation-2026-09-21/`.
Preserve the extension request, launch command/environment, exact driver hash,
test log, timestamped waiting/running/terminal status, parent ledger receipt,
new campaign ledger and resumed worker/checkpoint receipts. The existing master
estimation phases govern progression after tuning; stop at the first valid
estimate. The active implementation and previous progress remain documented in
the September 20 expanded-execution plan.

## Prelaunch verification

Nine focused tests passed against the preserved execution snapshot. The real
checkpoint test was then extended to exercise the actual child-process entry and
worker bootstrap and passed again. It preserved prior numerical observations,
configuration, candidates and work counters, obtained fresh verification, and
produced exactly the same retained draws as the uninterrupted known-target
reference. The CPU-only test durations reported by pytest were 27.42 seconds
for the initial suite and 36.71 seconds for the updated worker test. These are
engineering checks; no q20 scientific conclusion follows.

A read-only replay of this campaign's actual qualification, pricing, training
and tuning requests confirmed that their identities are preserved in the fresh
campaign directory. All 490 numerical source files still match. The driver is
recorded separately in `job.json` and `launch.json`; the active source snapshot
was not edited. This resolves the identified identity and timeout risks and
passes the skeptical audit for the bounded continuation launch.

The protected downstream amount at the current prices is 39,595.95 seconds
(11.00 hours). With no intervening unplanned cost, the extended tuning ceiling
is approximately 53.32 cumulative hours, including the original tuning attempt.
This is a maximum allocation derived from the available balance, not expected
tuning time. Exact allocation is recalculated from settled spending at takeover.
The finite 200-work-unit search budget and all scientific criteria still apply.

## Launch observation, September 21 at 01:45 Asia/Shanghai

The continuation is active as
`bayesfilter-q20-deadline-20260925-1800-r2.service`, coordinator PID 745803.
It is waiting for the original master, which remains active with PID 93403 and
is still tuning. Nine numerical tuning observations are recorded, with no
verified member yet. The latest observation requests a larger step. The
continuation has launched no duplicate numerical worker. About 61.01 campaign
hours remain at this observation, including the original attempt's elapsed time.

The service starts at 01:44:28 and has a 407,721-second runtime bound, followed
by at most five seconds of stop grace, ending before the requested September 25
18:00 deadline. The driver also checks the absolute deadline before numerical
work. An initial waiting observer was replaced before any numerical takeover
because this systemd cannot change `RuntimeMaxUSec` on an existing unit; its
receipt is preserved as `launch-initial.json`. The original numerical service
was not restarted. `launch.json`, `status.json` and `launch-observation.json`
record the current service and observation.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue supervised execution with the authorized deadline extension | Posterior estimate pending | No terminal invalidity reported; continuation remains restricted to a settled tuning budget pause | Whether a kernel can be verified and posterior checks pass within the remaining money and work units | Current tuning, then checkpoint continuation if needed, followed by sampling and reference assessment | Convergence, whitening and posterior validity are not yet established |
