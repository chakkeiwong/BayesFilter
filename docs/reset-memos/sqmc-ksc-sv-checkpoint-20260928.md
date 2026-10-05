# KSC reset repair checkpoint — completed, 2026-09-30

Checkout /home/chakwong/BayesFilter-SQMC; branch sqmc-development; base f72cbfe3.
Owner authorized plan/review/execute, then recovery and continuation.
Plan: docs/plans/sqmc-ksc-reset-repair-plan-20260929.md.
Final result: docs/benchmarks/sqmc-ksc-reset-repair-results-20260929.md.
Master summary: docs/benchmarks/sqmc-master-program-final-summary-20260929.md.
Evidence: docs/plans/artifacts/sqmc-ksc-reset-repair-20260929/attempt-01/.

All 38 GPU units completed successfully, including the balanced eight-design
extension. All 525 evaluations passed numerical validity. Recovery checks:
383 evidence checks, 67 independent saved-result checks, and 55 CPU regressions
passed. Numerical source closure matches the final GPU manifest. CPU GPUs hidden.

Outcome: limited repair nomination for the three SQMC T10 scopes only.
IID fails validation at both horizons; all T120 candidates fail validation and
the untouched likelihood-error guard on data 243002. All four original baselines
lose to Gaussian Kalman on that long-horizon dataset; repaired candidates pass
all heuristic screens. No overall method ranking, admission/default or HMC claim.
Every planned repair phase has run; candidate failure did not stop the campaign.

Saved session 01a0e3d0-2332-7ea3-9fc4-d5e4dcee78d0 hit provider HTTP 502 errors,
then failed pre-sampling remote compaction. Session preserved; underlying gateway
cause unverified. recovery-01/session-diagnosis.json records selected events.
Fresh-session bubblewrap startup failures were separate; trusted commands worked.

Charged 36253.132322/43200 GPU-seconds; 6946.867678 seconds remain.
Original deadline 2026-09-30T16:15:40.010888+00:00. Recovery used no GPU.
No active worker. No retuning, environment/provider setting changes, merge or push.
Terminal review: attempt-01/recovery-01/terminal-review.md.

Next: use the completed reports for user-directed follow-up. No execution remains
under this plan; unused allowance alone does not authorize another experiment.
Local commit packages implementation, tests, plan, evidence and completion notes.
