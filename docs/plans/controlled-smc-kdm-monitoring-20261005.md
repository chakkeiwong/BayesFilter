# Persistent replication run monitoring

The owner requested monitoring, completion/error reporting, and repair of
repairable execution problems on 5 October 2026. This is an operational addition
to the authorized [replication campaign](controlled-smc-kdm-replication-first-execution-20261003.md),
not another numerical experiment or a change in replication criteria.

## Scope and evidence

Observe attempt145 and the existing failure-replay, forward-capture, and
forward-comparison queues once a minute until that scheduled batch terminates.
Record a heartbeat, process identities, completed case counts, finite-result
screen, queue states, terminal decisions, available host memory, and budget
balances. Preserve each state change in an append-only event log and submit
Windows desktop notifications for startup, phase completion, errors, repairs,
and terminal batch status. Notification submission does not establish that the
desktop displayed it or the owner read it. No automatic new chat message is
available through the tools exposed to this session.

The primary operational success criterion is continuing, observable execution
within existing attempt/time limits, followed by accounted-for terminal states.
A heartbeat is not a numerical correctness criterion. Finite logZ is only the
existing explanatory screen; it does not certify covariance validity or
replication. A completed diagnostic with failed forward comparisons must be
reported as requiring investigation, even when its executable returned PASS.

The known Lorenz failure at case3537 and dependent APF blockage are frozen
baseline outcomes. Report their final source result as a partial baseline only
when both neuroscience drivers finish all2000 cases without invalid rows. Any
additional source failure is an alert. Full CSMC/KDM replication remains
incomplete even when this scheduled batch finishes.

## Defaults, repair limits, and stop conditions

| Choice | Provenance and role | Risk and early check |
|---|---|---|
|60-second poll|Operational convenience; no scientific effect|May delay detection by a minute; verify two live heartbeats|
|20-minute source progress warning|Conservative operational hypothesis; recent cases take tens of seconds|A slow valid case can warn; warning never kills/restarts numerical work|
|Two consecutive observations before liveness/invalid-row alerts|Avoid process/file-write transitions|Delays alerts one additional poll; terminal decisions are immediate|
|Low host memory warning below max(512MiB,5% total)|Explanatory resource warning|Does not prove OOM or trigger numerical changes|
|At most two queue restart calls per phase|Localized infrastructure repair budget|Restart only a dead WAITING queue with no attempt, before its original deadline; dispatcher rechecks dependencies, locks and budget|
|No numerical relaunch, process killing, author fixes, or environment changes|Existing scientific scope and authorization|A dead running worker, stalled job, failed comparison, or failed numerical attempt needs investigation; alert and preserve evidence|

Queue recovery uses the existing dispatcher, preserving serialization and
unique numerical output directories. It does not extend scientific budgets,
attempt ceilings, or original queue deadlines. Do not recover EXECUTING,
FAILED, BLOCKED, or never-queued jobs automatically. The monitor never invokes
the shared orphan-recovery routine while workers/supervisors may still exist.

The monitor stops at all scheduled phase terminal states, an unrecoverable
halt of the scheduled batch, the latest existing queue deadline plus that phase's execution limit and120 seconds,
or an explicit stop. It writes and notifies any fatal monitor exception. Host
inspection imports no array framework; CUDA is deliberately hidden. A recovered
GPU queue uses the already reviewed trusted dispatcher and its GPU policy.

## Skeptical review before execution

The main risks are calling a known author failure a new infrastructure error,
calling diagnostic completion replication acceptance, duplicating a live job,
extending a deadline during queue repair, and promising chat notifications that
this session cannot send unattended. The rules above explicitly prevent those
outcomes. Focused synthetic tests must exercise the decision branches and the
repair call chain. A live notification submission and two advancing heartbeat
records verify wiring; neither is evidence of completed replication. No proxy
metric, new baseline, scientific threshold, or GPU allocation is introduced.

## Execution

Implementation: `scripts/monitor_controlled_smc_kdm.py`. Tests:
`tests/replication/test_run_monitor.py`. Exact entry point:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/monitor_controlled_smc_kdm.py start
```

Use trusted permissions because the daemon may restart a reviewed queue and
submit desktop notifications. A singleton lock prevents duplicate monitors.
Each launch creates a fresh `monitor-*` directory under the campaign root;
`monitor-latest.json` points to its manifest and live report. The manifest
records command, Git/source provenance, host-only environment, deadlines,
process identity, elapsed time, and artifacts. Results and execution checks
will be added here after validation.

## Validation and launch result

Twelve focused tests passed, the daemon launched, and two advancing live
heartbeats were verified. See the [execution result](controlled-smc-kdm-monitoring-result-20261005.md).
Desktop notification submission works, but global Windows notifications are
disabled; the owner preference is pending. Monitoring and durable reporting
remain active without changing that global setting.
