# Active checkpoint: independent T=50 validation

Question: does the frozen covariance-guided mixture retain the matched-run
likelihood/score agreement on independent datasets and designs?

Branch sqmc-development; starting commit a43d0e3c3; clean at start.
Owner authorized execution, documentation, commit, main/origin integration and
return synchronization. Existing matched campaign is complete, not rerun.

Plan: docs/plans/ledh-independent-validation-20261008.md.
New output: docs/plans/artifacts/ledh-independent-validation-20261008-01/.
Budget: 24 aggregate worker-hours / 40 attempts / 12 wall hours, unused.
Stage: narrow shared-runner extension and bounded supervisor implemented.
Focused CPU-only checks passed: 36 tests in 22.82 seconds; logs and command in
docs/plans/artifacts/ledh-independent-validation-checks-20261008-01/.
No filter equations or tuning settings changed.

Checked findings: original matched settings and tuning files exist; all four
models use T50/N1008 FP64 GPU/XLA. Previous SIR rank20 fit took 11379 seconds;
rank40 took 101234 seconds and is outside this plan. Current score reference
uses author TT paths with current-model callbacks and local quadratic fitting;
it is numerical, with no exact-oracle claim.

Next exact action: commit the reviewed launch code and plan, prepare eight
checked independent datasets, then launch the bounded supervisor. Complete the
reporting-only aggregator while workers run; inspect all failures and references
before interpretation, completion documentation and branch synchronization.
