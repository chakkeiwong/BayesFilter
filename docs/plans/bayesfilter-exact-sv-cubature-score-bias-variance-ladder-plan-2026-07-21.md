# Exact-SV Cubature Score Bias/Variance Ladder Plan

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_ENGINEERING_ONLY_SV_SCIENTIFIC_CLAIMS_REVOKED`

> **Correction, 2026-07-22:** This ladder used direct iid Normal transformed
> observations rather than an SV-DGP sequence.  Treating its conditional
> discrepancies as SV score bias was a planning error.  All accuracy, bias,
> tuning, OPG, particle-scaling, variance-policy, and mechanism conclusions are
> revoked for SV.  Only engineering mechanics and resource evidence remain.
> See `bayesfilter-exact-sv-nondgp-fixture-demotion-correction-2026-07-22.md`.

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | On one fixed exact transformed-SV data set at `T=50`, does the scalar Cubature finite value/recursive-score estimator become more accurate as `N` increases, and does antithetic initial/process noise reduce score variance at `N=1000`? |
| Candidate | Experimental scalar Cubature residual design plus finite Sinkhorn transport and Contract E--Chol restoration; manual forward-sensitivity score of the same executed finite scalar |
| Exact comparator | Float64 sequential dense-grid filter for the exact log-chi-square observation model, with diagnostic GradientTape differentiation of the independent dense program |
| Particle ladder | `N in {250,500,1000,2000}`, all divisible by `2d=2` |
| Tuning | Fresh per-`N` standard tuning and separate fresh `N=1000` antithetic tuning; calibration seeds `3501,3502`, validation seeds `3511,3512` |
| Claim | Sixteen untouched, common seeds `3520..3535`; maximum-`N` stateless noise arrays are sliced to make the stochastic inputs nested across `N` |
| Expected failure mode | Pathwise score noise and reset-carried finite-cloud errors accumulate over 50 increments; a persistent `probit_gamma` mean error may remain even when value error is centered |
| Promotion criterion | At `N=2000`, the value-error 95% interval includes zero with half-width `<=0.25`, and the Bonferroni-two-score familywise intervals for mean score error both include zero |
| Promotion veto | Nonfinite output, CPU fallback, memory-policy failure, reset/Sinkhorn residual `>=1e-2`, recursive/FD calibration error `>0.05`, stale or cross-scope tuning, or failed dense-oracle refinement |
| Continuation veto | Dense value disagreement `>5e-5` or dense score disagreement `>2e-4` across the declared quadrature refinements; corrupt/missing artifacts; OOM; seed-partition overlap |
| Repair trigger | Local harness, serialization, placement, or manifest failure with unchanged target, seed partitions, criteria, and budget |
| Explanatory diagnostics | Individual-run SD/range, pointwise intervals, paired bootstrap change in absolute mean error from `N=250` to `N=2000`, per-time and cumulative score-error decomposition, runtime/memory, and antithetic paired-bootstrap SD ratios |
| Nonclaims | No Contract E-versus-Cubature ranking, GenUT result, default/leaderboard/HMC readiness, broad nonlinear validity, NAWM result, or global tuning optimum |

## Statistical Contract

1. The sampling unit is one complete independently seeded particle filter run.
   Particle-wise or time-wise terms within one run are not treated as
   independent replications.
2. Report individual-run SD separately from uncertainty in the 16-run mean.
3. Value uses a coordinate-wise two-sided `t(15)` 95% interval.
4. The two score coordinates use Bonferroni familywise 95% intervals with
   `t(15, 1-0.05/(2*2)) = 2.4898797034798923`. Pointwise intervals remain
   explanatory.
5. Multiplicity is controlled within one `N` score table, not globally across
   all particle counts and ablations.
6. A deterministic paired percentile bootstrap over the 16 common seeds
   reports `abs(mean error at N=2000) - abs(mean error at N=250)`. It supports
   a particle-scaling statement only when its 95% upper endpoint is below zero.
7. The antithetic arm is a different finite estimator. A paired percentile
   bootstrap reports its SD ratio relative to standard `N=1000`; it is a
   variance ablation, not evidence about the original single-cloud scalar.

## Dense Reference Gate

Evaluate the exact same observations and realized float32-origin theta at:

- order `257`, radius `8`;
- order `401`, radius `8`; and
- order `401`, radius `10`.

The campaign reference is order `401`, radius `10`. Stop before particle
interpretation unless every alternative differs by at most `5e-5` in value and
`2e-4` in each score coordinate. Verify that score increments sum to the total
dense score.

## Tuning Contract

For each standard particle count and the separate antithetic `N=1000` scope,
search:

```text
epsilon in {1, 2}
sinkhorn_steps in {4, 8}
ridge in {1e-5, 1e-4}
```

Use the existing lexicographic objective:

1. absolute validation mean dense-value error;
2. validation value RMSE;
3. calibration recursive-score versus same-scalar FD discrepancy;
4. validation reset/marginal residual;
5. work and ridge tie breakers.

Dense score is not used to select controls. This preserves the transferable
tuning objective and reserves score comparison for the untouched claim. The
selected arm is a frozen representative of this bounded grid, not a proven
global optimum. Each tuning artifact binds `N`, horizon, data, coupling mode,
backend, design, controls, route identity, and all three seed partitions and is
written before its claim rows execute.

## Engineering Repairs Before Execution

1. Correct generic Gaussian GenUT construction. Ebeigbe Gaussian GenUT equals
   Cubature only at `d=3`; at scalar `d=1` it is the weighted three-point rule.
   The executed ladder remains explicitly Cubature.
2. Make raw physical-score error the primary LGSSM report object; preserve
   relative errors as descriptive and label their coordinate system clearly.
3. Add isolated candidate tests for the Sinkhorn barycentric JVP and the
   composed transport-plus-Contract-E restore JVP. The reset kernel's existing
   forward-accumulator certificate is not duplicated.
4. Preserve candidate `score_increments` in claim artifacts and compare them to
   dense score increments only at the host diagnostic boundary.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| Fixed observations `[7050,17]` | Continuity with the completed `N=1000` exact-SV run; reviewed baseline | Favorable single-data cancellation | Explicit single-data nonclaim; future independent-data confirmation remains required |
| `theta=[0.25,-0.15]` | Existing exact-SV fixture; baseline | Coordinate-specific behavior may not generalize | Preserve physical `gamma,beta` and coordinate labels |
| Four `N` values | User-directed large-particle follow-up; hypothesis | Quadratic cost may exhaust memory/time | Staged increasing `N`, allocator/timing records, stop on OOM |
| Eight-control grid | Existing candidate grid; warm-start hypothesis per scope | Near-ties or omitted controls may hide better settings | Preserve all summaries and make no global-optimum claim |
| Two calibration/two validation seeds | Bounded feasibility choice | Unstable tuning selection | Untouched 16-seed claim; report tuning near-ties |
| Common random numbers by slicing | Variance-reduction design | Shape-specific RNG could defeat pairing | Generate once at `N=2000` then slice, and record policy |
| Antithetic `Z,-Z` averaging | Fable transfer-minded variance hypothesis | May change bias or fail in asymmetric nonlinear response | Separate tuning/scope and paired bias plus SD-ratio diagnostics |
| TF32/float32 GPU/XLA | Repository default candidate backend | Precision/branch effects | Dense float64 comparator, fixed-branch FD probes, placement and residual gates |

## Skeptical Plan Audit

- Wrong baseline: repaired. The primary comparator is exact log-chi-square
  dense filtering, not KSC Kalman, Gaussian closure, or a historical LEDH row.
- Proxy promotion: repaired. FD is derivative-consistency tuning evidence only;
  residuals and runtime are veto/explanatory diagnostics, not accuracy claims.
- Hidden scope transfer: repaired. Every `N` and antithetic coupling mode tunes
  separately; prior `N=1000` controls are not reused as claim evidence.
- Holdout reuse: repaired. Claim seeds `3520..3535` have not appeared in the
  earlier `3420..3435` claim.
- Multiplicity: bounded within each two-score table; no global familywise claim
  across all displayed tables.
- GenUT ambiguity: repaired. Scalar GenUT is not aliased to Cubature and is not
  executed in this campaign.
- Environment mismatch: candidate execution is GPU/XLA/TF32 with verified
  memory growth; dense quadrature is an explicit CPU/float64 diagnostic.
- Artifact adequacy: rows preserve total and per-time value/score, exact errors,
  controls, placement, residuals, allocator data, and timing; summaries can
  distinguish bias, single-run variance, and CI-of-mean precision.
- Misleading pass: the primary `N=2000` gate cannot establish broad nonlinear
  correctness or a default. One data set and a bounded control grid remain the
  strongest limitations.

Audit decision: `PASS_AFTER_REPAIRS`. Execute engineering repairs and focused
tests first. Then run the dense gate, four tuned standard scopes, and the tuned
antithetic scope. Stop only on a continuation veto; a candidate gate failure is
scientific evidence and does not invalidate later planned diagnostics.

## Compute And Artifact Budget

- Five tuning scopes: four standard `N` values plus antithetic `N=1000`.
- Per scope: eight controls, two calibration seeds, two validation seeds, then
  16 untouched claim seeds for the selected controls.
- One localized infrastructure retry is allowed per failed phase without
  changing the scientific contract.
- Fresh output root:
  `docs/benchmarks/artifacts/cubature_exact_sv_score_ladder_20260721/attempt01/`.
- Planned result note:
  `docs/plans/bayesfilter-exact-sv-cubature-score-bias-variance-ladder-result-2026-07-21.md`.

## Execution Outcome

The campaign completed in `128.963` seconds. All engineering and dense-oracle
gates passed. The `N=2000` value gate passed, but both familywise score-error
intervals excluded zero, so the candidate promotion gate failed. Antithetic
averaging materially reduced variance but did not repair mean error and
introduced a detectable positive value error.

Result note:

`docs/plans/bayesfilter-exact-sv-cubature-score-bias-variance-ladder-result-2026-07-21.md`
