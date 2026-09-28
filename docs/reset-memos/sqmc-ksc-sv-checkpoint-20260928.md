# KSC checkpoint — bounded plan complete, 2026-09-29

Checkout: /home/chakwong/BayesFilter-SQMC, branch sqmc-development.
Kalman/retention commit: 479a4616. The commit containing this checkpoint records
the completed KSC stage. No push or merge.
Plan: docs/plans/sqmc-ksc-sv-comparison-20260928.md.
Result and terminal review: docs/benchmarks/sqmc-ksc-sv-results-20260929.md.
Detailed tables: docs/plans/artifacts/sqmc-ksc-sv-20260928/final-evidence-01/report.md.
Budget: docs/plans/artifacts/sqmc-ksc-sv-20260928/budget.json.

Completed: 5 CPU tests, 4-route GPU FD/graph/XLA/N1008 checks, 16 tuned scopes,
128 valid final cells, 256 particle score coordinates, 96 safeguard checks.
No invalid candidate scopes, reference failures or infrastructure retries.
Mixture reference replaced the wrong Gaussian oracle; KSC tangent repaired.
Gaussian has lower score error in 6/8 T120 pairs for each route, but larger
likelihood error. Only exploratory T10 SQMC-versus-IID intervals exclude zero;
no overall ranking or default promotion. Retain all four for further research.

Conservative aggregate use 6.780795/12 GPU-hours,
remaining 5.219205h; includes 300s prior-hook reserve.
No workers running. Renewal deadline 2026-09-30T16:15:40.010888+00:00.

No numerical work remains in this bounded plan. Read the completed result and
terminal review for any user-directed follow-up. Repository commit checks run
with GPU devices hidden. Do not rerun completed research. Future studies: fresh T120 numerical tuning,
N2016/more pairs, fixed-dataset uncertainty and broader regimes; none promoted.
