# Replication supervision recovery, September 25, 2026

The existing 48-hour authorization covers this localized infrastructure repair.
At 09:21 HKT attempt 16 was progressing through full-training epoch511/800,
but supervisor1346425 and evaluation watcher1357219 were absent. Worker1346434
and its loader children remained alive in process group1346434. The reason for
the supervisors' exits is not recorded. Do not restart the training worker.

Question: can the existing run retain its original deadline, honest accounting,
and conditional evaluation despite losing its parent? Baseline is the unchanged
source-trial protocol and its immutable source-repair11. The original deadline
is 14:07:10 HKT, not twelve hours after recovery. No scientific settings, seeds,
data, checkpoint selection or promotion criteria change.

Use a detached master supervisor with file-backed input/output. Verify the
worker's PID, command, process start time and group before adopting or stopping
it. Refuse another live supervisor or watcher. Preserve the old watch record.
Charge training from its original start until observed exit (a conservative
polling upper bound); preserve every previous charge. Enforce the original
deadline, with timeout a completion veto. On disappearance, an absent exit
status must be reported as unknown, never fabricated as zero. Only complete,
consistent terminal protocol artifacts and matching checkpoint hashes can
recover artifact completion; a known nonzero exit, missing/invalid artifact,
timeout or process-identity mismatch blocks evaluation. This establishes
artifact completion only, not scientific replication or learning quality.

Then invoke the existing untouched 5,000-trajectory T=150 evaluation, using
the existing 900-second child limit and cumulative budget. No new GPU run
is launched during recovery itself. Tests run CPU-only with CUDA hidden.

Skeptical audit: PASS for this infrastructure repair. Training progress is an
explanatory diagnostic, not completion. Stale watcher/ledger status is the
failure under test. The repair must neither reset the deadline/budget nor accept
a short-trial checkpoint. Test these failure conditions and detached execution
before attaching. No new algorithmic backend, baseline or numerical default.

Evidence: new supervision record/log and archived watcher under campaign
iapf-kdm-replication-20260924-02; focused test log; this note and active checkpoint.
Keep the actual training source snapshot immutable. Future long master launches
should also offer detached supervision so a tool-session exit does not remove
the only timeout/accounting controller.

## Terminal outcome, 2026-09-25T13:42:32+08:00

Training finished before the recovery attachment. The first attachment refused
the absent worker without launching compute. Added an explicit terminal-record
collector; all 32 focused CPU tests passed. It validated matching phases/epochs,
finite losses and hashes, preserved unknown process exit, and launched the
existing held-out evaluation, which passed as attempt 17.
Recovery status EVALUATION_FINISHED; no current worker.
Actual charges and the decision/inference tables are in
[the trial result](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/source-trial-result-20260925.md).
The code accepts completed-before-attachment recovery only when the process
group is absent and terminal evidence is within the original deadline.
