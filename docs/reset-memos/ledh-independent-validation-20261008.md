# Completed checkpoint: independent T=50 validation

Question: does the frozen covariance-guided mixture retain likelihood/score
agreement on independent observations and particle designs?

Campaign complete at numerical commit 2a7b37852 on sqmc-development: 25/25 jobs,
192/192 valid filter rows, 48 bootstrap runs, eight author fits and eight
quadratic-score calculations; no failures or missing reference results.
Time: 8.29 wall hours and 16.56 aggregate worker-hours, within 12/24-hour limits.
No research jobs remain running. No further experiment is required by this plan.

Plan: docs/plans/ledh-independent-validation-20261008.md.
Final result: docs/benchmarks/ledh-independent-validation-results-20261008.md.
Master summary: docs/benchmarks/ledh-covariance-proposal-master-summary-20261009.md.
Evidence root: docs/plans/artifacts/ledh-independent-validation-20261008-01/.
Structured final comparison: report-20261009-011148/comparison.json.
Terminal review: terminal-review.json. Frozen tuning, observations and source
hashes were rechecked; the old/new comparisons change complete algorithms.

Findings: mean SIR score errors versus Zhao–Cui are 44.2 and 40.7, versus
16212 and 14189 for the old filter. Conditional paired intervals favor KSC
value/score on both datasets, SIR likelihood on both, and SIR score on one.
LGSSM likelihood is descriptively worse and predator--prey scores are mixed;
no universal non-deterioration or broad ranking is supported. Nonlinear
references retain rank/support bias and Monte Carlo uncertainty. Keep the
candidate optional; no canonical, default, HMC or fourth-moment promotion.

Execution: frozen T50/N1008 settings; FP64 GPU/XLA with TF32 off, RTX 4080 SUPER
selected by UUID, verified memory growth. CPU reference jobs hide GPUs.
Focused harness checks: 36 passed (22.82 s). Final reporter checks: four passed
(0.08 s), under ledh-independent-reporting-checks-20261009-01/. These overlap
prior implementation checks and are not separate scientific replications.

Current stage: final findings integrated and both PDFs built and inspected.
Monograph 650 pages, standalone 25 pages; no unresolved citations/references.
Build and visual receipts: ledh-independent-validation-docs-20261009-final/.
Next exact action: commit compact evidence and documentation, then synchronize
Git branches as authorized below. Large author .mat/derived/tail payloads remain local under
scoped ignores. The owner authorized merge to main, fetch/merge origin/main,
push main and merge main back into sqmc-development. Verify clean worktrees
and identical refs. Main worktree: /tmp/bayesfilter-sqmc-main-integration-20260924.
Preserve unrelated worktrees. No new tuning or research direction is authorized
by a documentation or Git operation.
