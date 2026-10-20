# Replication monitor execution result

The persistent host-only watchdog is running as PID1190326. Live report:
[monitor-20261004T192719Z-1791142039902644390/live-status.md](artifacts/controlled-smc-kdm-replication-20260928/monitor-20261004T192719Z-1791142039902644390/live-status.md).
It checks once a minute, preserves completion/error events, and can restart only
confirmed dead WAITING queues before any numerical attempt. Source-method changes,
numerical retries, and budget/deadline extensions are excluded.

At UTC 2026-10-04T19:32:26.767343+00:00, both source worker and supervisor and all three
continuation queues were alive. The neuroscience cSMC matrix is2000/2000 and BPF
is997/2000. No new health alert or invalid
result row was observed. Lorenz cSMC remains3537/4000 with the known source
covariance failure; APF remains blocked. Full replication is incomplete.

## Verification and review

Twelve focused tests passed: advancing/stalled execution, missing workers,
confirmed queue death, exclusion of attempted/EXECUTING jobs from recovery,
deadline preservation, actual dispatcher wiring, partial-write tolerance,
known-source failure classification, diagnostic PASS with failed comparison,
notification payload handling, and final-phase monitoring duration.
The live check confirms separate advancing heartbeats with the numerical job
still running. Source code, Git provenance, exact daemon command, environment,
deadline, and process identity are in the monitor manifest. No numerical job was
restarted or interrupted to install monitoring.

Review repaired two real operational risks before launch: a restart must refuse
a proposed deadline beyond the original queue deadline, and the monitor must
include the final phase's600-second GPU execution allowance after its waiting
deadline. Completion messages preserve failed comparison verdicts instead of
equating a diagnostic PASS with replication acceptance. Transient artifact-read
errors are retried and logged; three consecutive failures trigger an alert.

Evidence: [tests](artifacts/controlled-smc-kdm-replication-20260928/monitor-review-20261005-01/tests02.log),
[live verification](artifacts/controlled-smc-kdm-replication-20260928/monitor-review-20261005-01/live-verification02.json),
[launch](artifacts/controlled-smc-kdm-replication-20260928/monitor-review-20261005-01/launch.json).

## Notification limitation

Windows accepted the local notification API submission but returned
DisabledForUser. A read-only settings check found global ToastEnabled=0.
The owner has been asked whether to enable Windows notifications; no setting
has been changed. Events and status are recorded regardless. This daemon cannot
post a new chat message; API submission is not display/read confirmation.
Abrupt machine shutdown or termination of the daemon also stops monitoring.

The original GPU ledger has about32.2 hours left, but the active tranche has
only2998.58 seconds. The active
execution contract now distinguishes them. No compute grant or scientific
qualification criterion changed.
