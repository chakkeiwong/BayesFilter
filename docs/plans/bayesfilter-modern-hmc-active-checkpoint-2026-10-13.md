# Active modern-HMC checkpoint

## Active continuation — 30 October 2026

Remote main, including the new LEDH conditional-parent, SIR, KSC and
range-bearing changes, is merged in the isolated integration checkout
`/tmp/BayesFilter-modern-hmc-integration-20261030`. LEDH source paths match
remote byte-for-byte. The shared dirty workspace remains preserved. See the
[integration record](bayesfilter-modern-hmc-ledh-integration-2026-10-30.md).

C3 is complete: all seven members assessed, four pass all declared checks.
The [C4 raw-draw audit](bayesfilter-modern-hmc-c4-raw-precision-result-2026-10-30.md)
also completes: all seven members and all 31 quantities reproduce stored
calculations and decisions, with all 643 original source/input hashes intact.
The failed first reader confused active and model coordinates; its repair is
regression-tested and does not alter C3 outcomes. Batch-scale sensitivity and
finite-run/reference uncertainty remain open.

Next: execute the reviewed [fixed-count replication](bayesfilter-modern-hmc-c4-fixed-count-replication-2026-10-30.md), one fresh, fixed-10,000-draw cohort
for all seven members, preserving warmup/health checks and original outcomes.
Compare its terminal estimates with replayed early-stop prefixes; neither
reference values nor favorable candidates determine stopping. The existing
C3 numerical source is the comparator, while new LEDH integration remains
separate. No ranking, broad reliability, default or release claim follows.
C2 wider migration/P19, C4 tail calibration, C5 independent SIR, and C6/N8
release/book reconciliation remain open. Historical status snapshots below
do not reopen completed work or supply additional budget.

<!-- C3-AUTOMATIC-CONTINUATION -->
## C3 automatic continuation — 2026-10-29T21:04:02.386742+00:00

Current continuation status: **posterior_assessment_completed**.
Exact progress: N5/c3-posterior-20261029-01/continuation-01/status.json.
Posterior reservation: 98215s; actual pricing charge: 3394.607s.
Unreserved balance after reservation: 80861.960s; SIR remains separate.
Posterior inclusive charge: 18347.480s.
Remaining upper balance after settlement: 160729.480s, preparation still withheld.
Members passing all declared checks: 4 of 7.
C3 diagnostic success does not grant posterior/default/release promotion.
C2 migration, C4 reliability, C5 independent SIR and C6/N8 reconciliation remain open.
<!-- C3-AUTOMATIC-CONTINUATION -->

## C3 terminal review — 30 October 2026

The seven-member posterior assessment finished at **05:03:49 Asia/Shanghai**
with exit code 0. L=9,13,18,25 at epsilon=.4 passed all declared sequential
and independent-reference checks. L=3/.4 and L=5/.4 reached the retained cap
without sufficient precision; L=25/.2 passed internal checks but narrowly
failed the reference screen (.101648 versus .1). All seven retain tuning
membership. No numerical hard veto was recorded; all 643 source/input hashes
match. See the [terminal result and audit](bayesfilter-modern-hmc-c3-posterior-result-2026-10-30.md).

This C3 assessment is complete; the master is not. No posterior worker remains
active. Next is C4 estimator/tail reliability, including the internal-stopping
versus reference-validation discrepancy. C2 wider migration/P19, C5 independent
SIR and C6/N8 integration/book/release reconciliation remain open. No thresholds
were relaxed, candidates dropped, or comparative ranking established. The
settled remaining upper balance is 160729.47980995933s (44.6471h), preparation
still withheld; separate SIR funds remain 7363.064379416s.

## Historical C3 preparation — 29 October 2026, 23:02 Asia/Shanghai

The following is the preserved pre-launch snapshot; the terminal review above
supersedes its live-process, pricing and remaining-work statements.

**Independent reference work passed its declared checks; seven-member GPU
pricing is now running.** The reviewed [C3 continuation](bayesfilter-modern-hmc-c3-reference-continuation-2026-10-29.md)
and [reference result](bayesfilter-modern-hmc-c3-reference-result-2026-10-29.md)
control the next work. Do not rerun the completed 1,248-trial qualification.

C3a independently reproduced the covariance likelihood at 28 full-target
points (maximum value discrepancy 2.8422e-14); central-difference scores passed
the declared combined tolerances. C3b initially failed: the inherited
generating-parameter geometry produced concentrated weights and a .118814-SD
reference envelope. Independent multistart optimization, checked local
curvature, broader proposal components and fresh calibration/validation streams
repaired this reference-proposal problem without changing the target or
relaxing the .02-SD screen. All 31 quantities now pass, with maximum empirical
envelope .00218096 SD. This is an operational integration check, not proof
of global mode coverage or a calibrated simultaneous error bound.

The expanded reference/recovery suite passes 66 tests, and seven additional checks against the exact frozen numerical source pass. The GPU/XLA
quantity preflight passes at all 28 saved points and all 31 quantities, maximum
discrepancy 1.4211e-14. Actual cost pricing started at 23:01:16 on October 29
under detached supervisor 977947 and worker 977950, using the original bound
GPU0 with verified memory growth. At 23:15 the L=3/.4 and L=5/.4 price
blocks were complete and L=9/.4 was running; no error was recorded.
The automatic continuation observer is live. Its block above is the
current authority for subsequent launch and terminal status.

Active root: N5/c3-posterior-20261029-01/price-01. Its inclusive cap is 7200s.
The supervisor records progress every 60s and reassesses capacity every 1800s;
it does not require an idle GPU or stop on its own high utilization. Every
verified member gets a 1000-transition-per-chain price block. Pricing draws
are excluded from posterior estimates. Freeze the subsequent posterior
reservation from actual all-member costs, then use the shared sequential
controller, fresh seeds and durable chunks. Any missing price remains a gap;
do not select only the cheapest or apparently favorable member.

Reference attempts cost 300.54860805615317s in total; preflight supervisor
cost 18.11339380289428s, within the 3600s preparation reserve. After retaining
that preparation reserve and reserving 7200s for pricing, the unreserved upper
balance is 175271.56712023835s. Do not separately charge the preflight child
or add the released reference reservation as new credit. SIR funds remain
7363.064379416s, untouched. Preparation overhead remains to be settled.

C3 posterior assessment is still open. C2 wider migration/P19, C4 estimator
and nonlinear-tail reliability, C5 independent-fit SIR and C6/N8 examples,
integration and official-book/release reconciliation remain open. Conditional
N6/N7 research is not automatically a scoped-release prerequisite. All
candidates remain experimental; these checks establish no ranking, posterior
promotion, new default or release. NUTS remains excluded.

## Qualification closeout — 29 October 2026, historical snapshot

**The public GPU qualification is complete. No worker remains active.**
It finished at 11:51:16 Asia/Shanghai on 29 October 2026 with exit code 0.
Supervisor 524957 and worker 524960 are absent in trusted process inspection.
Do not restart this completed cohort. The
[terminal result](bayesfilter-modern-hmc-public-qualification-result-2026-10-29.md)
and [master](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md) control
the next work; the master as a whole is not complete.

Root: docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/public-qualification-20261029-01.
Terminal supervisor manifest, settlement, evidence/result.json, native-audit.json,
members.json and checkpoint agree: 1,248 distinct trials, 48 completed work
items, seven verified/exported/reloaded members, no missing returns or partial
chunks. L=3/5 at epsilon=.8 failed movement; the other seven pairs passed.
All seven reloads preserve endpoints with finite continuation. All 629 source
hashes match. Tuning R-hat is reporting-only; NUTS is excluded.

Frozen source: /tmp/BayesFilter-modern-hmc-c2-exact-source-20261028 at
f9291ee4437aa8c00dabbe331bffc6871b574c5c. Consumer: N5/public-gpu-20261029-01/consumer.
Keep evidence and source for reproducibility. launch.json and progress.json
contain earlier stage snapshots; terminal result/manifest/settlement supersede
those statuses. The former October30 deadline is inactive.

Inclusive charge: 38325.626110966085 seconds. Released reservation:
56106.373889033915 seconds. Remaining unreserved upper bound:
186372.1157282945 seconds, with engineering/preparation withholdings retained.
Separate SIR balance: 7363.064379416 seconds. Do not recharge child times.
Compact terminal-review.json and terminal-inventory.json preserve this closeout.

Next: resolve C3 posterior/reference criteria, uncertainty, fresh streams and
sequential-controller use before a posterior campaign. C2 wider migration/P19,
C4 reliability, C5 independent fits and C6/N8 reconciliation remain open.
Qualification is experimental and establishes no ranking or posterior/default/
release promotion. No additional campaign was launched during this closeout.

## Historical checkpoint — 14 October 2026, 23:56 China time

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
