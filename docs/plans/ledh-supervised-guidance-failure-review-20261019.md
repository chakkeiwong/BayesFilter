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
   Check the change of variables through the shared affine LEDH flow: a guide
   fitted as a function of the post-flow state need not retain that meaning
   when applied to the pre-flow draw. Use an exact scalar Gaussian example and
   the shared flow call chain; this is a structural diagnostic, not a new
   candidate campaign or a claim of causal dominance on SIR.
4. Write a verdict distinguishing established findings from causal hypotheses,
   update the concise checkpoint and master-program interpretation, and commit
   the review separately.

Budget: at most 600 CPU seconds for artifact inspection and focused tests; zero
GPU seconds. Diagnostic analysis uses Python standard-library arithmetic over
saved JSON only, with GPU devices intentionally hidden for framework tests.
Artifacts: docs/plans/artifacts/ledh-supervised-guidance-20261019/failure-review/.
No external reviewer launch is required; unavailable independent review must
be disclosed, not silently represented as complete.

## Completed review, 20 October

Step 1 committed as 4abbaf9f1; the review is a separate commit. The theorem is
correct under its assumptions. Fitting/selection does not yet optimize guide
coefficients for score risk; the physical-state guide is changed by the
pre-flow composition. Exact change-of-variables and Gaussian counterexample,
MathDevMCP scalar algebra, and an executable shared-consumer check support this
finding. Its causal contribution to SIR remains unmeasured. Arithmetic and ten
runtime hashes check out. No new candidate was fitted or promoted.

Report: artifacts/ledh-supervised-guidance-20261019/failure-review/review.md.
Commands, attempts and budget: failure-review/review-run.json. Both master
programs and the manuscript now identify the aligned-guide repair before a
larger supervised comparison. Independent whole-document review remains
unavailable; the local derivation and executable evidence are explicitly scoped.
