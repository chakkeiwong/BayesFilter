# Observation-aware TT active checkpoint, 2026-09-16

## Active question and authorization

Does standalone SGQF-initialized pair TT supply a useful conditional proposal
for exact-importance-corrected Zhao-Cui filtering? The owner requested a plan,
thorough review and execution of regression improvements. A09 is complete.
Follow [master](observation-aware-tt-repair-complete-program-20260913.md) and
[A09 result](observation-aware-tt-warm-improvement-20260916-result.md).
SGQF fit loss is explanatory, never a continuation veto. No per-step TT/SGQF
selector, raw-TT normalizer, total analytical derivative or HMC claim.

Checkout /home/chakwong/BayesFilter, branch surrogate-hmc. Preserve unrelated
dirty files. No commit/push requested. Review was a skeptical executor review;
no independent review claimed. No numerical jobs remain running.

## Checked findings

Confirmation-02 completed 24 sequences in 1287.928335 seconds, RTX 5080,
TF float64/XLA kernels, verified memory growth and unchanged source snapshots.
Both degree-4 nominees reduce scalar filtering MSE versus the degree-3 warm
baseline by 7.5% and 8.1% under exploratory paired bootstrap intervals. Their
mutual ranking is unsupported. d4 mean ESS is about 322--325/512, but d4
promotion fails: s10 fails guide SPD and reference precision; s04 has a guide
factor collapse to 2--4e-16 at t8 and every guide-dependent proposal fails
log-evidence agreement. Reference precision passes for s04.

All 54 same-target fitted H2 values are below their own initializer; d4 mean
H2 baseline/capacity/preservation .003260/.001532/.002087 (descriptive).
Conditioning reaches 5.24e10; KKT .0137 leaves optimizer accuracy open.
All 27 focused tests and the final artifact audit pass.

Calibration-01 was invalidated for A07 data reuse. Calibration-02 nomination
and confirmation-01 had time-shift algorithm stream reuse; the latter is
ineligible for independent-sequence intervals. Evidence is preserved.
Confirmation-02 repairs actual particle/reference seed call chains, uses fresh
partition 2, and keeps all nominees frozen. Earlier independence-based MCSE
claims require separate audit; do not silently reuse them.

## Budget and exact next action

Sole ledger: artifacts/observation-tt-continuation-24h-20260915-01/budget.json.
A09 began 04:05:18 UTC; numerical work finished before its six-hour ceiling.
Final administrative closeout slightly exceeded it, as recorded in the ledger.
The ledger
charges elapsed work and tool/approval waits once, conservatively, with a
two-minute final-response allowance. These waits are not GPU computation.
Retain the A08 unmetered reservation. Approximately 11.6 hours remain.

Next scientific priority: prepare and review an amendment for SGQF covariance
scale and tail-coverage validity, using the saved s04 collapse as diagnostic
evidence and fresh heldout data for claims. Determine a principled scale check
and evaluate any numerics-changing protection explicitly. Only then revisit
TT solver conditioning. No new experiment or amended method is authorized by
this checkpoint alone; use the governing campaign scope and reviewed amendment.

- Plan/review/report/audit: artifacts/observation-tt-warm-improvement-20260916-01/
- Numerical root: ../benchmarks/artifacts/observation_tt_warm_improvement_20260916/
- Final attempts: attempt-calibration-02 and attempt-confirmation-02
- Prior objective/LaTeX repair: observation-aware-tt-downstream-filter-objective-20260916-result.md
