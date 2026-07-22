# Cubature/GenUT Exact-SV N=1000 Plan

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_ENGINEERING_ONLY_SV_SCIENTIFIC_CLAIMS_REVOKED`

> **Correction, 2026-07-22:** This was not an SV-DGP experiment.  Direct iid
> Normal transformed observations were used.  The value gate, tuning, score
> summaries, and `N=1000` SV claim are revoked and must not guide active work.
> Only engineering feasibility evidence remains.  See
> `bayesfilter-exact-sv-nondgp-fixture-demotion-correction-2026-07-22.md`.

## Question

Does increasing the loop-native exact transformed-SV candidate to `N=1000`
at `T=50` materially reduce the same-target dense-reference discrepancy and
score variance under a freshly tuned `N=1000` scope?

## Evidence Contract

| Field | Contract |
|---|---|
| Candidate | Repository-registered loop-native Cubature candidate |
| Scope | exact transformed-SV, `N=1000`, `T=50`, state dimension 1, two parameters, float32/TF32 GPU/XLA |
| Reference | Independent float64 dense exact transformed-SV target value; diagnostic only |
| Feasibility gate | One warm-start seed is finite, GPU-resident, memory growth verified, reset residuals `<1e-2`, and allocator peak recorded |
| Tuning | Fresh calibration seeds `3401,3402`; validation seeds `3411,3412`; controls selected before claim |
| Claim | Untouched seeds `3420..3435` at frozen controls |
| Accuracy gate | Paired 95% confidence interval for mean value error includes zero and its half-width is no larger than `0.25`; individual rows remain finite |
| Score evidence | Report mean, SD, MCSE, and 95% CI for both recursive-score coordinates; no score-zero or superiority claim without a declared reference |
| Comparator | Contract E remains `BLOCKED_SAME_TARGET_COMPARATOR` unless a fresh paired finite-program contract is available |
| Nonclaims | No default, leaderboard, HMC, exact-filtering, method-superiority, or NAWM claim |

## Skeptical Audit

1. `N=96` was a mechanics-scale diagnostic and cannot answer the large-particle
   bias question. `N=1000` is therefore justified as a new scope.
2. Controls tuned at `N=12` are warm starts only. They cannot support the
   `N=1000` claim; fresh scope-specific tuning is mandatory.
3. A maximum per-seed error threshold is too strict for a Monte Carlo estimator
   and was not scientifically calibrated. The claim gate uses a paired 95%
   confidence interval for the mean error, while maxima remain explanatory.
4. A single seed establishes feasibility only. It cannot establish bias or
   score precision.
5. The `N x N` transport has quadratic memory/time scaling. The feasibility
   row precedes tuning and records TensorFlow allocator peak/current bytes.
6. The candidate XLA closure remains TensorFlow-only; Python loops and host
   conversions are allowed only in campaign orchestration/reporting.
7. Dense-grid refinement is not an exact theorem. The reference remains a
   diagnostic comparator and no Contract E ranking is inferred.

Audit decision: `PASS_STAGED_EXECUTION`. Stop before tuning on nonfinite output,
device fallback, memory-policy failure, OOM, or reset residual failure.

## Default And Assumption Audit

| Choice | Provenance and status | Failure mode | Early diagnostic |
|---|---|---|---|
| `N=1000`, `T=50` | User-requested target scope | Quadratic transport may be infeasible | One-seed feasibility timing and allocator peak |
| `epsilon in {1,2}`, steps in `{4,8}`, ridge in `{1e-5,1e-4}` | Prior small-scope controls, warm-start grid only | Grid may omit a better `N=1000` setting | Disjoint validation error and residual table; no global-optimum claim |
| Calibration `3401,3402`, validation `3411,3412`, claim `3420..3435` | Fresh deterministic convenience partitions | Two tuning seeds may overfit or rank controls unstably | Preserve all candidate summaries; untouched 16-seed claim decides accuracy |
| Dense order 257, radius 8 | Existing independent exact-SV diagnostic | Quadrature truncation could be mistaken for particle bias | Float64 implementation and fixed same observation data; no exact-theorem claim |
| TF32/float32/XLA GPU | Repository default candidate backend | Precision or compilation behavior may dominate | Finite checks, placement, reset residuals, recursive/FD parity, allocator record |

## Budget And Stop Conditions

- Completed feasibility budget: one successful warm-start row, after two
  permission-bound attempts that did not produce scientific evidence.
- Claim campaign budget: eight control candidates, two calibration and two
  validation seeds per candidate, then 16 untouched claim seeds at the frozen
  selected controls. One localized harness repair/retry is allowed without
  changing the scientific contract.
- Stop on a nonfinite candidate, CPU fallback, failed memory-growth policy, OOM,
  reset residual `>=1e-2`, corrupt/missing artifact, or seed overlap.
- A failed value-accuracy gate rejects this candidate at this scope but does not
  invalidate the harness or the broader research direction.

## Feasibility Result

The trusted GPU/XLA feasibility run completed at `N=1000,T=50`, seed `3400`
with transferred warm-start controls `epsilon=2`, Sinkhorn steps `8`, and
ridge `1e-4`. It was engineering-valid, took `3.671` seconds for the measured
cell (`6.332` seconds total process wall time), and recorded a TensorFlow
allocator peak of `75,340,288` bytes. The dense-reference value error was
`0.2224197388`; this single observation is feasibility evidence only.

Artifact:

`docs/benchmarks/artifacts/cubature_genut_exact_sv_n1000_20260721/feasibility_attempt03/result.json`

## Final Outcome

Fresh tuning and the 16-seed untouched claim completed in
`tuned_claim_attempt02`. The mean dense-reference value error was `0.01211`
with a paired 95% confidence interval `[-0.06662, 0.09084]` and half-width
`0.07873`, so the predeclared value gate passed. No engineering veto fired.

Result note:

`docs/plans/bayesfilter-cubature-genut-exact-sv-n1000-result-2026-07-21.md`

## Phases

1. Run one `N=1000,T=50` feasibility seed using the prior controls solely as a
   warm start.
2. If feasible, tune `epsilon`, Sinkhorn steps, and ridge on disjoint
   calibration/validation seeds using dense value discrepancy, representative
   same-scalar FD parity, and reset validity.
3. Freeze controls and run 16 untouched seeds.
4. Compute value-error and score uncertainty summaries and record the decision.
5. Do not proceed to leaderboard assembly unless the target-accuracy and all
   later model/comparator gates pass.

## Artifacts

Fresh root:

`docs/benchmarks/artifacts/cubature_genut_exact_sv_n1000_20260721/`
