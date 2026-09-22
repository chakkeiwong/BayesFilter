# Active iAPF checkpoint

Question: how variable are fresh final likelihoods from complete adaptive R
reconstructions, conditional on the paper's five linear-Gaussian dimensions?
Stage: phase 16 RUNNING; the 100-replicate stage is complete and the
300-replicate stage is active as of 2026-09-22 at 12:00 UTC.
Detached supervisor PID 4021639, launched
2026-09-22 at 09:02 UTC. It advances through 100, 300, and 1,000 labels per
method/dimension without per-stage intervention. Candidate failures continue;
source/numerical/artifact invalidity, repeated infrastructure failure or budget
exhaustion stops with an explicit checkpoint.

Checkout: /home/chakwong/BayesFilter; surrogate-hmc;
Research source commit db72d33c; branch synchronization is recorded in
docs/plans/iapf-branch-synchronization-2026-09-22.md. Read the actual branch
HEAD when resuming. Preserve unrelated dirty changes.
Master: docs/plans/younis-kdm-score-master-program-2026-09-14.md.
Plan: docs/plans/iapf-adaptive-replication-ladder-2026-09-22.md.
Live state, log, budget and completed-stage summaries are under:
`docs/plans/artifacts/iapf-adaptive-replication-ladder-20260922-01/`.
Read `checkpoint.json` and `budget.json` there for current counts and budget;
`launch-confirmation.json` preserves the initial execution verification.
Two single-thread CPU workers, GPU intentionally hidden, phase cap 36 charged
worker hours inside the owner's existing campaign budget. No wall deadline or
new approval gate. Do not modify sources recorded in `preflight.json` while
running. Same-seed d5/d80 numerical replay passed, 180 statistics checks passed,
and three supervisor regressions passed, including candidate-cap continuation
and bounded infrastructure retries.

Stage 100: 3,000 records, no failed/capped learners and no observed heuristic
promotion veto in this sample. The completed note is `stage100/result.md` in
the live root. Pointwise bootstrap intervals are conditional on one data set
per dimension; no population ranking or paper-identity claim follows.
At 12:00 UTC, 111 batches were complete; about 39.093 CPU hours and 47.745 GPU
hours remained, with 0.500 CPU hours reserved. Refresh these from live state.

Phase 15 is COMPLETE: 400 evaluations, 729 checks, all 200 learners complete;
full Gaussian oracle/Kalman error 1.82e-12. Initial d80 score floor distortion
up to .188 disappears by fit 2 in all recorded runs. Score/later doubling
fails the descriptive d40 heuristic screen (RMSE .340 versus FA .317).
Ten replicates do not establish a ranking. Detailed result and derivation:
`docs/plans/artifacts/iapf-adaptive-score-reference-20260922-01/result.md` and
`gaussian-mechanism.md` in that same directory.

Next action: inspect the live checkpoint and completed stage summaries; the
supervisor itself continues the frozen ladder. If it records a true failure,
inspect its exact error/attempt, repair locally within the unchanged contract
and remaining budget, then resume without overwriting prior evidence. On full
completion, review conditional uncertainty, tails, failed learners, floor
weights and classical controls before drawing a scientific conclusion.
These are independent R reconstructions, not Eq15, original-author code, full
paper replication, model-parameter scores, nonlinear validity or a new default.
