# V6 Variance-Reduction Plan Review

Date: 2026-08-15  
Reviewed plan: `bayesfilter-classifier-score-variance-reduction-v6-plan-2026-08-15.md`  
Verdict: `PASS_WITH_IMPLEMENTATION_GATES`

## Review Findings

The four-arm paired design is the smallest direct test of the two proposed
mechanisms while retaining all nine cells. Holding architecture and optimizer
controls fixed is appropriate for a causal ablation, but the result cannot be
used to assert that the 8192-path arm is optimally trained.

The main methodological risk is CRN sample dependence. Its class marginals
remain correct, but ordinary row-level uncertainty would be wrong. The plan
avoids that error by treating bundle replicate as the training-level cluster
and by preserving paired arm seeds. Validation/calibration/test data remain
independent and shared, so they do not confound the treatment.

The primary outcome is correctly defined as variation across independently
trained bundles at fixed paths. The prior V5 standard deviations across
different paths are not reused as estimator uncertainty.

## Required Gates

1. Test that `n2048` data are exact prefixes of `n8192` for both noise modes.
2. Test CRN plus/minus base-noise identity and independent-arm inequality.
2a. Test each pair has one plus and one minus row at one delta, and that
    minibatch construction never splits a pair.
3. On the Gaussian simulator, verify both CRN class marginals against direct
   simulation and verify no observation or label leakage.
4. Test that all arms share validation/calibration/test and audit hashes inside
   each bundle replicate.
5. Test paired bootstrap indexing and a synthetic known-variance reduction.
6. Test Gaussian exact-score evaluation uses observation paths, not estimator
   outputs.
7. Record every completed bundle before starting the next; aggregation must
   refuse missing or duplicate bundle/arm/cell rows.
8. Maximum-count GPU smoke must establish memory growth, XLA, finite fitting,
   and bounded memory before the full launch.
9. Result schemas must separate `bundle_variance`, `path_variance`, and
   `exact_score_error`.

Execution may begin after these gates pass. A failed variance-reduction
candidate remains a valid result and does not authorize post-hoc arm or metric
changes.
