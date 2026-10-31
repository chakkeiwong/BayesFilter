# Budget assessment for continuing state-space validation

This assessment concerns the unfinished 32-fit state-space campaign. It is a
cost assessment, not a new launch or an extension of the expired schedule.

The settled ledger retains **42.114 GPU worker-hours** and the master records
**10.455 CPU reference/test worker-hours**. GPU accounting includes the enclosing
worker's startup, compilation, waiting and shutdown. CPU worker-hours are not
CPU core-hours. The main reservation is cleared. The previous stop followed
the original latest-start deadline and individual fit allocations; it did not
exhaust the additive compute grant.

## What can be priced from completed work

| Remaining workload | Observed-runtime calculation | GPU worker-hours |
| --- | --- | ---: |
| Four K2 main fits | 4 x 4,841.006 seconds, completed pilot | 5.379 |
| Four K7 main fits | 4 x 7,157.783 seconds, completed pilot | 7.953 |
| Last K4 main fit | 2,128.952 seconds, longest completed K4 main fit | 0.591 |
| Total for these nine fits | Sum of the above | 13.923 |

These are measured-cost scenarios, not lower bounds, upper bounds or reliable
forecasts for new seeds. They include no new contingency. K0 already showed
that a completed pilot multiplied by 1.5 can substantially underprice a main
search. Its pilot visited 16 candidates; its four main attempts reached
70--98 candidates and still had pending work at their time limits.

The remaining cost is not established for four K0 checkpoint continuations,
four main fits each of K1/K3/K5/K6, the unfinished development pilots, and
further repairs. K1/K3 each consumed approximately 5,325 seconds without
finishing. K5 has repaired preparation but no complete repaired pilot. K6
still has no successful preparation handoff or complete-fit price. Its missing
posterior oracle also cannot be remedied merely by running longer. The three
completed K4 main fits remain preserved; their interval-coverage records are
unavailable and cannot be counted as coverage successes.

## Recommended allocation decision

Do not request a new compute grant yet. First propose up to **six GPU
worker-hours and two CPU worker-hours from the existing balance** for saved-work
diagnosis, focused repairs and complete-fit repricing. These are proposed
spending caps, not measured requirements or a promise that every pilot will
finish. The six-hour cap inherits the previous bounded repair-stage scale;
two CPU hours are a provisional engineering reserve, four times the preceding
conservative half-hour repair/test charge. Together they preserve approximately
36.114 GPU hours and 8.455 CPU hours for subsequent work.

The first discriminating work is to explain K0's search expansion and validate
continuation with a shared cumulative allocation; distinguish search/compilation
cost from GPU co-residency; resolve K3's reference-workload mismatch; and obtain
full repaired K5/K6 prices where numerically valid. Preserve original numerical
criteria and completed outcomes. A genuine numerical repair gets a fresh
configuration and output directory; unchanged interrupted work may reuse its
validated checkpoint. No result should be rerun merely to improve its outcome.

Use complete costs and unfinished-work evidence to assign the remaining common
pool. Reaching the diagnostic cap without those prices means the completion
budget is still unknown. It does not authorize another full unpriced sweep.
Any requested additional grant should then be the estimated shortfall plus an
explicitly justified reserve. A new schedule must be stated before launch;
the original latest-start cutoff has passed.

## Skeptical assessment

The main error would be reporting the 13.923-hour subtotal as the price of
finishing all 29 unfinished main slots. It omits twenty slots or continuations,
uncompleted pilots, repairs and unmeasured runtime variation. It would also be
wrong to treat unused grant hours as proof that all models can pass, or to
extend the previous wall-time window silently. No reliable all-finished budget
is supported by the current evidence. The existing balance supports a bounded
next repair; its adequacy for full completion remains unestablished.

Sources: `artifacts/hmc-ssm-main-2026-09-29/r1/allocation.json`, terminal receipts
and `terminal-audit-2026-09-30.json`; the active
`artifacts/hmc-ssm-funded-2026-09-28/grant-ledger.json`; and
`artifacts/hmc-repair-master-2026-09-16/program-progress.json`. This assessment
uses saved artifacts only. No numerical experiment or new budget reservation
was made.
