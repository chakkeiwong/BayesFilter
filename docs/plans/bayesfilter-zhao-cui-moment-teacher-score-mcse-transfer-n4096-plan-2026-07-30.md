# Zhao-Cui Moment-Teacher Score/MCSE Transfer Diagnostic at N=4096

Date: 2026-07-30
Status: executed; systematic displacement found and 0.5-MCSE screen failed
Classification: canonical-LGSSM transfer diagnostic, not moment-teacher evidence

## Research Intent And Evidence Contract

- Main question: at `N=4096`, does TF32 create a systematic displacement in a
  complete stochastic final score relative to the identical FP32/XLA program
  with TF32 disabled, and how large is that displacement relative to MCSE?
- Exact route: canonical Contract E-Chol LGSSM, `T=2`, `N=4096`, FP32, XLA,
  active reset, 20 Sinkhorn steps, 2 terminal-balance steps, exact-divisor
  transport policy `K=2048` with a `2 x 2` block grid, and estimator seeds
  81800--81807.
- Candidate/comparator: TF32 enabled versus TF32 disabled. Both arms use the
  same dtype, source, data, prepared noise, residual design, particles, seeds,
  ridge, chunking, and numerical controls.
- Systematic-displacement criterion: for a score coordinate, require both an
  exact two-sided paired sign-test `p <= 0.05` and
  `abs(mean drift) > 2 * paired-difference MCSE` before calling the displacement
  systematic for this scope.
- Practical-magnitude screen: report whether every coordinate satisfies
  `abs(mean drift) / reference MCSE <= 0.5`. The earlier `0.1` negligible-error
  screen remains reported separately and is not silently replaced.
- Hard vetoes: either arm fails finiteness, chart, reset, marginal, replay,
  work-count, XLA graph, or exact-chunk validity; paired source, commit, scope,
  seed, or prepared-input identity differs; fewer than eight seeds complete; or
  artifacts are incomplete.
- Explanatory diagnostics: score drift and its paired MCSE, sign counts, exact
  sign-test p-value, reference score MCSE, per-seed maximum drift, value drift,
  runtime, and allocator peak.
- Nonclaims: FP32-no-TF32 is not an exact mathematical oracle; this tests
  systematic TF32 displacement relative to that comparator. The Zhao-Cui
  moment-teacher final score remains unimplemented and untested. No long-
  horizon, nonlinear, HMC, or default-readiness conclusion follows.

## Defaults And Assumptions

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| `N=4096`, `K=2048` | user scope and repository exact-divisor policy | directly answers the requested particle scale | different block grid may change TF32 behavior | artifact binds `2 x 2` grid | claim scope |
| Eight paired seeds | smallest sample giving two-sided all-one-sign p=0.0078125 | resolves strong sign consistency at bounded cost | limited power for mixed-sign small shifts | report exact p and paired MCSE | bounded diagnostic |
| Two batches of four | N=1024 peak scaled conservatively against current free VRAM | avoids unsafe 16-seed memory extrapolation | batch-dependent graph or preparation drift | same fixed batch shape and per-batch identity checks | resource choice |
| `T=2` | smallest complete canonical horizon | measures propagated final score at bounded cost | may understate horizon accumulation | explicit short-horizon nonclaim | convenience scope |
| Balance 2, Sinkhorn 20 | prior validated canonical controls | avoids tuning on the bias result | new seeds/count may fail validity | fail-closed route gates | inherited baseline |
| 0.5 reference MCSE | user-proposed practical threshold | tests materiality rather than numerical negligibility | a systematic offset does not average away | report 0.1 and 0.5 screens separately | hypothesis |

## Skeptical Audit

- Wrong baseline: avoided by using the identical FP32 program with only TF32
  execution changed. This isolates TF32 but does not establish exact bias.
- Proxy promotion: the canonical score is transfer evidence only; it cannot
  admit the unimplemented moment-teacher score.
- Unfair comparison: paired seeds and prepared tensor hashes are mandatory.
- Hidden scaling change: `N=4096` changes the required chunk grid to `2 x 2`;
  this is bound in scope rather than compared as if identical to `N=1024`.
- Missing stop condition: stop after four GPU batch nodes (two precision arms,
  two seed batches each) and one aggregation, with at most one localized retry
  for an infrastructure/serialization failure.
- Artifact adequacy: eight paired final scores directly support mean drift,
  reference MCSE, paired MCSE, and an exact sign test.

Audit verdict: pass for the transfer question. The output must say
`systematic displacement relative to FP32-no-TF32`, not exact mathematical bias.

## Compute Budget And Artifacts

- GPU budget: four batch executions, each four seeds at `T=2`, `N=4096`, plus
  one CPU aggregation; stop at ten minutes total.
- Fresh root:
  `docs/benchmarks/artifacts/zhao_cui_moment_teacher_score_mcse_transfer_20260730/n4096_attempt01/`.
- Preserve every node and do not overwrite the earlier `N=1024` campaign.

Result:
`docs/plans/bayesfilter-zhao-cui-moment-teacher-score-mcse-transfer-n4096-result-2026-07-30.md`.
