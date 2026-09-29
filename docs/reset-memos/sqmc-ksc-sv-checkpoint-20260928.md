# KSC discrepancy checkpoint — complete, 2026-09-29

Checkout: /home/chakwong/BayesFilter-SQMC; branch: sqmc-development.
Starting HEAD: 023e106102c89ee9d4787df3a55fbf5f68b88ecf.
Owner's plan/review/execute request is complete. No numerical worker is active.

Result: docs/benchmarks/sqmc-ksc-discrepancy-results-20260929.md.
Plan: docs/plans/sqmc-ksc-discrepancy-analysis-20260929.md.
Addendum: docs/plans/sqmc-ksc-reset-mechanism-addendum-20260929.md.
Evidence: docs/plans/artifacts/sqmc-ksc-discrepancy-20260929/attempt-01/.
Tables, CSV, uncertainty, traces, plots and terminal audit: analysis-01/ there.
Master summary: docs/benchmarks/sqmc-master-program-final-summary-20260929.md.

All 376 saved evaluations valid; 48 branch-matched FD checks pass; eight CPU
checks pass; 20 terminal evidence checks pass. No failed GPU attempt or retry.
At N=4,032, eight designs on case 213006 leave mean gamma errors 1.56–1.84.
All four lose to the Gaussian heuristic there; full seven-mixture integration
remains the reference. No method ranking or accuracy/default promotion.

Shared reset/correction preserves mean and variance but changes kurtosis from
2.77–2.82 to 1.32–1.34 and changes exact next-observation predictions/scores.
This is a direct local effect, not a global error decomposition or exclusive
cause. FD validates finite-program derivatives, not target-score accuracy.
Two retrospective cases and UNTUNED changed scopes limit generalization.

Ledger: docs/plans/artifacts/sqmc-ksc-discrepancy-20260929/budget.json.
Prior 24462.190031s + this investigation 7081.021525s =
31543.211557/43,200s charged; 11656.788443s (3.237997h) remain.
Deadline: 2026-09-30T16:15:40.010888+00:00; max two infrastructure retries/unit.
Do not double-count the immutable prior ledger or its included reserve.

Next research question: test richer residual designs and distribution
preservation, separating reset from correction, with safeguards retained and
fresh tuning/data before any promotion. Do not rerun completed diagnostics.
Literature review/proposal: docs/plans/sqmc-ksc-reset-repair-literature-20260929.md.
2025 GenUT paper and author code checked; no new numerical work. Next: specify
stagewise audit and isolated residual-design/protection tests under the same budget.
No core/model/default/HMC/environment changes; no independent reviewer.
This follow-up does not merge or push; original integration already completed.
