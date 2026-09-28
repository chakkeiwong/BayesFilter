# KSC checkpoint — campaign complete, 2026-09-29

Checkout: /home/chakwong/BayesFilter-SQMC, branch sqmc-development.
Particle evidence commit b7ed96ec; full-mixture correction commit c2ae4eb0.
Requested numerical work and terminal review are complete. No HMC, package
changes or scientific/default promotion. Merge/push and branch synchronization
were separately authorized by the owner on 2026-09-29; Git history records them.
Master summary: docs/benchmarks/sqmc-master-program-final-summary-20260929.md.
Current result: docs/benchmarks/sqmc-ksc-full-mixture-corrected-results-20260929.md.
Tables: docs/plans/artifacts/sqmc-ksc-full-mixture-20260929/final-evidence-01/report.md.
Plan/derivation: docs/plans/sqmc-ksc-full-mixture-correction-20260929.md.
Active ledger: docs/plans/artifacts/sqmc-ksc-full-mixture-20260929/budget.json
(links immutable original ledger; do not double-count charges).

Complete: all seven observation components retained in Gaussian-sum Kalman
updates with checked quadrature projection. 8 CPU tests, GPU FD/exact/graph-XLA
checks, all 32 datasets x 4 resolutions pass. Original 128 particle evaluations
reused unchanged. Max reference score-coordinate discrepancy 9.77e-15; max
likelihood discrepancy 5.12e-13. Independent error/SD/SE/paired-interval audit
passed. No invalid reference cases, infrastructure failures or retries.
Old one-Gaussian main comparison superseded; historical evidence preserved.
No overall winner; retain all four. Only exploratory T10 SQMC-versus-IID
intervals exclude zero; no within-SQMC ordering. Single regime/8 pairs and
restricted controls remain limitations, no fixed-dataset uncertainty estimate.

Correction charge 51.327502s; aggregate
6.795053/12 GPU-hours; remaining 5.204947h.
Includes prior work and unchanged 300s old-hook reserve.
Deadline 2026-09-30T16:15:40.010888+00:00; two infrastructure retries/unit limit.
No workers remain. Numerical correction is complete. Repository commit checks
run with GPU devices hidden. Do not rerun completed research.
Next research action: await a selected follow-up; no numerical work remains in
this campaign. Candidates are fresh T120 tuning, N2016, more pairs, fixed-dataset
Monte Carlo uncertainty and broader regimes. Wider master-program completion
and HMC readiness are not established by this closeout.
