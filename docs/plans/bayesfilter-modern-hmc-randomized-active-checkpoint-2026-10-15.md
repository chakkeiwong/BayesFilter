# Modern-HMC active execution checkpoint

Updated 20 October 2026 after SIR terminal closeout.
Master: `bayesfilter-modern-hmc-implementation-plan-2026-10-07.md`.
Current question: which additional matched model/independent-fit evidence,
precision calibration and actual consumer qualification are needed for a scoped
N2–N5 release? SIR execution is complete. The master is not complete.

## Contract and source

NUTS is excluded. Acceptance/health determine tuning membership. Posterior
R-hat, ESS, MCSE and reference failure cannot reject, rank, repair or delay
qualified tuning members. Preserve unsuccessful/censored rows; no selective
posterior retry or retrospective threshold/default change.

Shared branch: `preserve/shared-main-before-fab-20260926`; checkout HEAD
`139cd11fe7fea37e971f2197a544cb26ccd39c44`. Preserve unrelated dirty work.
Let `B=docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation`
and `S=B/sir-cohort-20261017-01`. Numerical authority is immutable
`B/full-model-source-e68cdf1c4`, commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`; orchestration is
`S/continuation-source-03`. Do not edit deployed sources or historical results.

## Terminal SIR evidence

Plan: `bayesfilter-modern-hmc-sir-evidence-phase-2026-10-17.md`.
Result: `bayesfilter-modern-hmc-sir-result-2026-10-20.md`.
Closeout plan: `bayesfilter-modern-hmc-sir-closeout-2026-10-20.md`.
Service `bayesfilter-modern-sir-continuation-20261018-04.service` is inactive;
controller `S/continuation-04/status.json` is `completed_bounded_cohort`.
Finished 20 October 2026, 01:42 China time. All methods are settled.
Historical child PIDs in the terminal JSON are not evidence of a live process.

| Method | Verified | Posterior checks passed | Delivered | Other outcome |
| --- | ---: | ---: | ---: | --- |
| Fixed | 8 | 7 | 6 | Member 1 reference failure; member 8 resource-censored |
| Randomized | 8 | 8 | 7 | Member 1 reference failure |
| SNAPER | 6 | 6 | 6 | None |
| MAMS | 2 | 2 | 2 | None |

All 24 requested roots were retained after verification. All 23 completed
posterior assessments passed their posterior checks; 21 passed reference.
The two reference errors are `.10389830934231056` and
`.10192058612266966`, above the frozen `.1` screen. No failed member was
selectively retried. Fixed evidence remains in `S/continuation-03/fixed_hmc`
and is carried through `continuation-04/recovered/` without replay.
These descriptive counts do not rank methods or establish release readiness.

The saved strict audits passed. `S/closeout-20261020-01/audit.json` verifies
5,530 unique source/evidence hashes plus decisions and accounting. GPU 1,
XLA/TF32 and memory growth before initialization are recorded for every method.
Closeout is deterministic integrity/accounting work, not independent TF
ESS/R-hat/MCSE recomputation. No new numerical attempt was launched.

## Budgets and closed diagnostics

Inclusive numerical cost: `197315.48680924764` seconds. Controller engineering:
`123.76881196722388` seconds, with original hold released. The outstanding
submission receipt `.04695963021367788` seconds is now charged exactly once;
closeout check charge is `2.3410479170270264` seconds (final receipt write
excluded). Authoritative SIR available balance:
**`22375.605841124954` seconds**, in `S/closeout-20261020-01/settlement.json`.
Original controller accounting is preserved; do not add its charges again.
Pricing's `6423.1472200709395` seconds is already included.

Separate diagnostic balance: **`131192.0584563613` seconds**. Do not merge
silently with SIR funds, the separate 3600-second hold or historical
1579.5929542018566-second balance. No new study is allocated.

SGQF cohorts are terminal; SNAPER delivered five of six members. Stationary
stress, antithetic ESS, saved nonlinear replay, interval follow-up and their
documentation are complete; do not rerun. The interval study has 768 records
from 384 independent latent units; 22/108 fixed and 3/108 conditional stopped
undercoverage flags among 231 adjusted screens. Dependent cells mostly did not
stop. No replacement/default validated; stationary starts give no burn-in
evidence. Audit did not independently replay full TF ESS/R-hat. Documentation
passed 15 checks, PDF build and changed-page inspection.

## Exact next actions

1. Review the twenty-cell inventory and measured all-member costs against the
   master's remaining N5 confirmation requirements. Write an explicitly priced
   next phase for matched model/independent-fit evidence with complete denominators
   and uncertainty analysis; keep funding pools separate. Do not rerun selected
   failed members or invent a sweep to spend the available balance.
2. Preserve the open precision/actual-stopping calibration question; no current
   study supports replacing the estimator or relaxing posterior/reference gates.
3. Actual MacroFinance/DSGE consumer qualification remains N8 work. Its completed
   engineering slice passed 171 tests, 15 doc checks and an 82-page excerpt;
   it did not establish actual consumer or release readiness.
4. N6/N7 remain conditional branches, separate from a scoped N2–N5 release.
   No additional scientific/default promotion is supported by this cohort.

Closeout skeptical audit passed: frozen baseline and criteria, complete
failure/censoring denominator, unchanged evidence and reconciled inclusive
charges. The terminal results reject two member deliveries, not the research
direction. Next phases require their own priced evidence scope.
