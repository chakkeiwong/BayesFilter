# Active modern-HMC checkpoint — 14 October 2026, 23:56 China time

## Current question and execution

Finish the original fixed-HMC posterior cohort, audit all member outcomes and
settle the October 14 grant before funding the next comparator. NUTS excluded.
No live numerical source, tuning membership, policy, seed or threshold changes.

Repo /home/ubuntu/python/BayesFilter. Branch
`preserve/shared-main-before-fab-20260926`; HEAD
`139cd11fe7fea37e971f2197a544cb26ccd39c44`. Shared dirty worktree: preserve
unrelated edits. Current work is not committed or pushed.

C = docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01
A = C/comparator-continuation-01
S = docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/full-model-source-e68cdf1c4

Numerical service: bayesfilter-modern-fixed-resume-20261014-01.service.
Active/running on GPU 1; parent PID2855367, worker PID2855374. Started
14 October 17:45 China time. Supervisor cap169200 seconds; deadline about
16 October 16:45 China time, not an ETA. Verified GPU memory growth, TF/TFP/XLA.
Parent A/manifest.json; progress A/run/progress.json; A/worker.log.

Six of eight verified candidates delivered: members 1,3,4,5,6,7. Member2 reached
10000 retained draws/chain with inadequate second_5 precision/reference screen;
no hard veto or R-hat/ESS failure. Member7 passed after2000 warm-up/8000 retained.
Member8 (L16,epsilon0.5) has two committed1000-transition chunks and is running
chunk000002, started23:35 China time. Worker remains runnable; no recorded
exception. The first two chunks took801.445 and800.133 seconds. Longer elapsed
time in the next chunk is not a new inactivity veto. Do not change count caps
or estimator to rescue a failed member.

Latest artifact audit at23:53: all1738 source/input hashes,1464 tensor checks
and61 committed chunks pass; candidate inventory, seeds/policies and original
first-member result unchanged. C/closeout-preparation-01/interim-audit-01.json.
This is an interim snapshot, not terminal scientific or convergence evidence.

## Automatic closeout and exact next action

New script scripts/settle_modern_hmc_posterior_continuation.py validates terminal
service/report consistency, complete member dispositions, original inventory
and budget. It reuses A/progress-audit-02/audit.py for source/tensor verification.
41 CPU-only bookkeeping/observer regressions pass; no numerical algorithms
changed. Frozen tested observer source and receipts: C/closeout-preparation-01/.

Detached observer bayesfilter-modern-fixed-closeout-20261014-01.service is
active; C/closeout-observer-01/manifest.json records health every30 seconds.
It waits for the numerical service to exit, then writes
A/terminal-audit-01/checkpoint-audit.json and settlement.json. It cannot launch,
stop or modify numerical work. On dead service/invalid evidence it records
closeout_requires_inspection and releases no balance. Review any such failure
and retry read-only closeout with a fresh output directory.

Next: inspect both services and progress. At terminal, read the automatic audit
and settlement, verify all eight dispositions, write the terminal scientific
result and refresh the master. If the whole cohort is assessed and funds remain,
prepare/launch the complete randomized-HMC eight-root design next under the
unchanged source and science contract. Its priority is baseline-first, not
selection from observed outcomes. Resource-stopped fixed-HMC takes precedence.
No next comparator has launched or received a reservation.

Plans: docs/plans/bayesfilter-modern-hmc-posterior-resume-2026-10-14.md;
docs/plans/bayesfilter-modern-hmc-cohort-closeout-2026-10-14.md.

## Budget and established limits

October14 grant172800 aggregate worker-seconds:169200 held for this inclusive
numerical attempt,3600 held for local work. Known local charge661.336020 seconds
plus observer active work and short unmetered inspection overhead. Full local
reserve stays held. C/closeout-preparation-01/local-costs-01.json. Charge only
inclusive parent elapsed at terminal; do not add child elapsed or count the
observer's overlapping idle wait as a second numerical worker. No release yet.

Prior allocation is separate:1579.592954 unallocated seconds and its held3600
local reserve with2046.093070 known charges. Do not combine/reuse old balances
until an explicit reconciliation. Original fixed-HMC run stopped at its cap;
all1540 original files remain unchanged. Original terminal note:
docs/plans/bayesfilter-modern-hmc-fixed-terminal-2026-10-14.md.

The matched MAMS cohort delivered4/4 under the original lugsail policy. The
optional lugsail_max_batch_means study completed768 calibration records and
parity tests but has unresolved coverage and no default promotion; it must not
enter the frozen comparison. Those completed phases are in the master and
linked terminal notes; do not rerun them.

The master is unfinished: other matched method/model cells, independent seeds,
stopping/coverage calibration and N8 consumer/docs/release remain. No supported
method ranking or release/default readiness follows from this one-target/seed
cohort. N6/N7 are separately conditional development, not automatic blockers
for a future explicitly scoped N2–N5 optional release.
