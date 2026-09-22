# Active iAPF checkpoint

Question: how variable are fresh final likelihoods from complete adaptive R
reconstructions, conditional on the paper's five linear-Gaussian dimensions?
Stage: phase 16 RUNNING; detached supervisor PID 4021639, launched
2026-09-22 at 09:02 UTC. It advances through 100, 300, and 1,000 labels per
method/dimension without per-stage intervention. Candidate failures continue;
source/numerical/artifact invalidity, repeated infrastructure failure or budget
exhaustion stops with an explicit checkpoint.

Checkout: /home/chakwong/BayesFilter; surrogate-hmc;
HEAD 6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef. Preserve unrelated dirty changes.
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
