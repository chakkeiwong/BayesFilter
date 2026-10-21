# Modern-HMC continuation: additional 72 hours, 16 October 2026

This extends the October 7 modern-HMC master program. The owner authorized
72 additional hours. Consistent with this campaign's accounting, this means
259,200 aggregate worker-seconds shared across CPU/GPU work, not 72 hours
for each device. The conversion is derived from the owner's grant. Independent
numerical workers consume additive occupied time; a supervisor and its
overlapping child are charged once by inclusive supervisor elapsed time.
Observer sleeping time is not numerical worker time.

## Current terminal update, 16 October 2026

The randomized cohort completed at 17:13 China time with all eight verified
members assessed, five deliveries, two precision failures and one additional
reference-only failure. The terminal audit passed. The
[terminal result](bayesfilter-modern-hmc-randomized-terminal-2026-10-16.md)
records all outcomes. Earlier live observations below are historical snapshots.
No member is resource-censored; no posterior continuation is needed.

`additional-24h-20261016-01/reconciliation-02.json` reconciles the old settled
balance and both grants without another numerical settlement. It carries
forward preparation cost and charges the terminal observer's 40.50036860210821
active seconds once, excluding sleep. The available combined pool is
414778.2416104381 seconds before SNAPER preparation. Existing reserves remain
separate. The next bounded execution is the
[frozen SNAPER phase](bayesfilter-modern-hmc-snaper-feasibility-2026-10-16.md),
with a 180000-second ceiling and unchanged scientific rules. The bookkeeping
repair passes 90 focused tests. This grant does not establish completion of
N5/N8 or any method ranking or release status.

## Further allocation, 16 October 2026, 08:36 China time

The owner has added **another 24 hours**, derived as 86,400 aggregate
worker-seconds. This supplements the earlier 72 hours; it neither replaces
that grant nor creates separate CPU and GPU allowances. The two new grants
total 345,600 seconds (96 hours). After the recorded 60.07246739137918-second
preparation charge, their combined unallocated balance is
345539.92753260862 seconds, before the running closeout observer's eventual
active-inspection charge. The previous numerical reservation and old reserves
remain separate. The new receipt and reconciliation are
`priced-comparison-01/additional-24h-20261016-01/grant.json` and
`priced-comparison-01/additional-24h-20261016-01/budget-snapshot.json`, beneath
the existing N5 generic-adaptation artifact root.

Skeptical audit of this extension: **passes**. The only changed assumption is
the owner-authorized total compute allowance. The target, baseline, frozen
source, data, geometry, seeds, method order, count limits and evidence contract
below are unchanged. The running worker's original ceiling remains 137,000
seconds; the grant authorizes subsequent bounded work, without a duplicate
launch or retrospective charge. Candidate precision failures remain terminal
under this design; extra wall-time funding does not relax their draw caps or
authorize selectively rerunning them. Infrastructure failure, invalid evidence
and exhausted funding remain stops; prices and candidate failures do not
become continuation vetoes. The next numerical allocation follows settlement
of this attempt and uses a fresh output directory.

At this inspection, tuning is **complete**: all eight roots were independently
verified and retained. Four posterior assessments have finished. Members 1
and 2 reached 10,000 retained draws per chain with failed precision checks and
no numerical-health veto; member 1 also failed reference agreement. Members
3 and 4 passed posterior and reference screens. Member 5 is running and 6–8
have not started. All three services remain active with zero restarts. The
terminal audit and settlement do not yet exist. These interim outcomes do
not support a method ranking or release claim. The incomplete-qualification
recovery limitation discussed below no longer applies to this particular
attempt because its tuning has completed; posterior continuation remains
available if the original time ceiling censors unfinished posterior work.

The execution order remains: complete and audit randomized HMC; resume only
resource-censored work if necessary; prepare the frozen SNAPER comparator;
then fund feasible model/independent-seed and N8 checks. The new allocation
does not itself guarantee completion of the whole master program.

## Evidence contract and current state

The immediate question remains whether the frozen randomized-HMC consumer
completes the full NL-PP-SGQF candidate lifecycle. Its exact baseline is the
immutable `full-model-source-e68cdf1c4` and `designs-02/randomized_hmc.json`
specified in the October 15 feasibility plan. The existing service remains
unchanged, with its original 137,000-second inclusive ceiling. This new grant
does not retrospectively enlarge that cap or launch a duplicate. At the
October 16 01:13 China-time inspection it was qualifying: eight primary roots,
two independently verified members, and zero service restarts. No posterior
outcome is available.

Acceptance and numerical health determine tuning membership. Retain every
qualified candidate, including subsequent posterior failures. Posterior
R-hat, ESS, lugsail MCSE and the 18-quantity reference screen retain their
frozen roles and thresholds; none may reject, rank, repair or delay a tuning
candidate. The feasibility criterion is complete, auditable disposition of
declared roots and descendants, with separate posterior/reference outcomes
for every verified member. Numerical or identity failure vetoes the affected
evidence. Shared source/target corruption, missing indispensable evidence or
exhausted budget stops continuation until repaired or refunded. Candidate
failure triggers its already-declared repair or the next planned phase, not
abandonment of unaffected peers. Resource censoring is not candidate rejection.
Prices and progress counts are explanatory only. This single-target/seed
phase cannot establish ranking, general validity, stopped-interval coverage,
a new default or release readiness. NUTS remains excluded.

## Accounting and execution sequence

The immutable grant receipt is
`artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/additional-72h-20261016-01/grant.json`.
It records 259,200 new seconds independently of the existing 137,000-second
active reservation, 361.1951094693504 old unallocated seconds, the already-held
3,600-second local reserve, and the separate historical 1,579.5929542018566
seconds. Those historical amounts are not newly granted or released. Local
preparation under this addendum is charged by measured occupied command time
to this new grant in a separate receipt. No advance numerical reservation
is made by this addendum.

1. Record the grant and audit the current run without sampling or changing
   its immutable source. Preserve its original cap and prior receipts.
2. Prepare and test terminal audit/accounting for the frozen full-model
   launcher. The existing observer writes a terminal observation; it does
   **not** perform settlement. The older posterior-continuation settlement
   utility uses a different artifact layout. Add bounded read-only closeout
   with source/input, candidate-denominator, checkpoint and accounting checks.
3. At terminal exit, charge the original inclusive supervisor elapsed exactly
   once, report completed/censored identities, and reconcile released balances
   before reserving a fresh continuation directory. If tuning is complete,
   the existing posterior-resume driver is the next route. If tuning remains
   incomplete, first implement/test recovery through the frozen public tuning
   checkpoint interface; that driver currently requires complete tuning.
   Never silently restart qualification or claim automatic recovery.
4. Complete the frozen randomized cohort where its original count limits and
   remaining funding permit. Preserve count-limited failures; do not selectively
   rerun them or change the precision estimator after seeing results. Refresh
   exact command, source, full denominator, measured price, cap and fresh output
   before each numerical continuation.
5. Next consider the already-planned frozen SNAPER comparator; then fund
   remaining model/independent-seed cells and N8 consumer/documentation checks
   from reconciled balances. Existing MAMS and fixed-HMC evidence is retained.
   N5 and N8 remain open; conditional N6/N7 development is not automatically
   required for a scoped optional release. Completion in this grant is not
   promised.

## Default audit and skeptical review

| Choice | Provenance/status | Risk and earliest check |
| --- | --- | --- |
| 72 shared hours | Owner grant; existing accounting convention | Avoid doubling across devices; explicit receipt and reconciliation |
| Existing 137,000-second cap | Already-reserved ceiling, not ETA | May censor healthy work; retain checkpoints and distinguish rejection |
| Roots, seeds, four chains, 18 quantities, lugsail | Frozen comparison baseline, not universal defaults | Identity audit and full denominator; no result-dependent changes |
| Sequential next-method order | Existing program priority | Single-seed evidence cannot establish ranking |
| 60-second polling | Existing observer convenience | Observer failure must not kill healthy numerical work |
| Continuation driver | Inspected code requires complete tuning | Check terminal tuning before choosing recovery |

Skeptical audit: **passes for recording the grant, continuing the current run
and developing/testing read-only closeout**. Wrong-layout settlement and
assumed support for incomplete tuning were material flaws; this sequence
repairs them before a dependent launch. Baseline, data, geometry, seeds and
posterior rules remain unchanged. No proxy becomes a promotion criterion.
Interim `partial_budget` checkpoints must not be mistaken for terminal results,
nor verification for posterior success. Require service exit and terminal
artifacts. Fixture tests establish accounting mechanics, not sampler validity.
Contention affects costs, not the scientific target. Invalid evidence and
exhausted funding are true stops; candidate failures and cost forecasts are not.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue current phase | Frozen lifecycle preserved | No observed service/identity veto | Completion time | Observe; audit and settle at exit | Convergence, ranking, release |
| Prepare closeout | Correct terminal inventory/accounting | Reject active or inconsistent evidence | Terminal outcome unavailable | Fixture tests and bounded observer | Numerical validity from fixtures |

| Inference status | Interpretation |
| --- | --- |
| Hard veto screen | No new terminal evidence; member vetoes remain binding |
| Statistically supported ranking | None |
| Descriptive-only differences | Counts, elapsed time, qualification progress |
| Default-readiness | Not established |
| Next evidence needed | Terminal audit, matched comparators, independent replications |

## Execution record, 16 October 2026, 01:33 China time

The grant is recorded and the existing numerical run remains active with zero
restarts. The latest observer snapshot has eight created roots, three verified
candidates and 361 numerical chunk files; qualification is still running and
posterior assessment has not started. These are progress counts, not posterior
or reliability results. The machine had 1.1 TiB available at this turn's disk
inspection. No numerical rerun, source modification or policy change occurred.

Implemented `scripts/settle_modern_hmc_frozen_run.py`, a standard-library-only
terminal evidence/accounting reader, and its focused regression tests. The
79-test combined suite passes, with exact command and JUnit/log evidence in
`additional-72h-20261016-01/closeout-tests-02.*`. The earlier 78-test suite also
passed; test counts are not additive coverage claims. The real historical
fixed-HMC compatibility check initially exposed an omitted-default serializer
field: `max_results_per_chain=10000` with no reason is intentionally absent
from frozen policy payloads. The checker now recognizes that exact omission
while rejecting changed or missing nondefault caps. A regression pins both
cases. The failed compatibility receipt remains preserved. The corrected
check passes all 197 source/input hashes and 456 saved tensors, reproducing
the historical one-assessed/seven-unfinished disposition without replaying
samples or charging historical numerical time again. This validates evidence
reading, not numerical/statistical recomputation.

The tested observer was copied into `additional-72h-20261016-01/closeout-source-01/`
and launched as `bayesfilter-modern-randomized-closeout-20261016-01.service`
at 01:32 China time. `closeout-launch.json` and `closeout-launch-receipt.json`
record the exact command, code/test hashes, environment and launch result.
It uses CPU-only Python with GPU visibility explicitly hidden and reads the
original numerical service. Its bounded wait is the remaining original
ceiling plus the already-used 1,000-second observer margin; this margin is
operational convenience, not extra numerical allowance. It writes status to
`randomized-feasibility-01/terminal-audit-01/observer.json`. After service exit
it attempts the terminal audit and writes a single settlement, or records
`closeout_requires_inspection` if evidence is inconsistent. It cannot launch,
stop or retry numerical work. The original progress observer remains active.

`preparation-accounting-01.json` charges 60.07246739137918 measured seconds for
both regression invocations and both historical compatibility checks. The new
grant therefore has 259139.92753260862 seconds unallocated before the closeout
observer's eventual active-inspection charge. Observer sleeping is excluded.
The older active reservation, reserves and separate balances are unchanged.

Post-execution review: the strongest remaining limitation is that real
randomized terminal artifacts do not yet exist. Synthetic incomplete-tuning
fixtures and real fixed-HMC interrupted artifacts validate the intended reader,
but cannot guarantee every possible future interruption is recoverable. An
audit failure requests inspection and preserves the healthy numerical run;
it cannot turn bookkeeping failure into sampler rejection. Qualification
recovery remains a conditional implementation step, not a claimed capability
of the existing posterior-resume driver. The next numerical allocation waits
for the current terminal audit and reconciled budget.

## SNAPER execution, 16 October 2026, 22:30 China time

The next frozen comparator is running after its actual-source GPU preflight
and 90 focused accounting/observer tests passed. The six-proposal design and
scientific rules are unchanged. Its 180000-second inclusive allocation leaves
234414.24996451102 seconds unallocated after measured preparation, with old
reserves still separate. Read-only progress and terminal settlement services
are active. Exact launch and budget records are in
`priced-comparison-01/snaper-feasibility-01/`; the SNAPER phase plan and active
checkpoint govern the next inspection. No posterior result or ranking is
available from this new attempt yet.
