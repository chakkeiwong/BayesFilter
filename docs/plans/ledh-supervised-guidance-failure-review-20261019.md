# Review: why supervised guidance has not established score improvement

Owner request: commit the completed work, then thoroughly review the mathematical
argument, actual implementation and preserved pilot evidence. Work on
ledh-refactor-with-policy-fix. No new GPU campaign, OT changes or default changes.

## Evidence contract and skeptical review

Question: does the existing negative/ambiguous comparison reveal a mathematical
error, implementation mismatch, insufficient optimization, inadequate statistical
evidence, or genuine evidence against guidance? Baseline is the same full
unguided analytical LEDH executor, not an old diagnostic or historical result.

Inspect the conditional-risk identity, fixed-guide differentiation, actual
fitting/selection call chain, stored per-dataset/per-coordinate scores and
paired particle replicates. Preserve all old results. No new samples, label
regeneration, tuning or selection on test results. Arithmetic and exact
decompositions explain existing outcomes; they cannot establish a new ranking.

Promotion criterion remains the earlier independently replicated score-risk
comparison with uncertainty evidence and heuristic vetoes. This review cannot
promote any method. Inconsistent keys, source hashes or loss arithmetic invalidate
the affected evidence; poor candidate performance only rejects that candidate.
Finite-difference agreement validates a finite-program derivative, not its
accuracy against the model score.

Pre-run skeptical audit: the bank has only two validation datasets, the score
objective selects existing guide settings rather than training their coefficients,
and the kernel is an approximation to a fitted quadratic-policy message, not
original neural KDM. These are material limits, not automatically refutations.
Frozen calibration scaling and observation/particle randomness require explicit
checking. A successful analysis command alone cannot resolve causal attribution.

## Bounded execution

1. Commit the supervised proposal, implementation, necessary dependencies and
   evidence on the LEDH branch, preserving unrelated dirty changes.
2. Audit the theorem assumptions and endpoint-to-fitter/consumer call chain.
3. Recompute existing paired losses and decompose component, dataset and particle
   contributions; inspect guide-fit diagnostics and selection sensitivity.
4. Write a verdict distinguishing established findings from causal hypotheses,
   update the concise checkpoint and master-program interpretation, and commit
   the review separately.

Budget: at most 600 CPU seconds for artifact inspection and focused tests; zero
GPU seconds. Diagnostic analysis uses Python standard-library arithmetic over
saved JSON only, with GPU devices intentionally hidden for framework tests.
Artifacts: docs/plans/artifacts/ledh-supervised-guidance-20261019/failure-review/.
No external reviewer launch is required; unavailable independent review must
be disclosed, not silently represented as complete.
