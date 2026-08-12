# GenUT Austria SIR Antithetic-Ensemble Result

Date: 2026-08-03

Plan: `docs/plans/bayesfilter-genut-austria-antithetic-ensemble-plan-2026-08-03.md`

Claim artifact:
`docs/benchmarks/artifacts/genut_austria_antithetic_ensemble_20260803/attempt01/result.json`

Status: `ANTITHETIC_NOT_HELPFUL_PROMOTION_VETOED`

## Direct Answer

Complete-run antithetic reflection did **not** improve Austria SIR GenUT value
or score computation in this experiment. It is rejected for this route.

- Antithetic `Z/-Z` constituent correlation was nearly zero rather than
  strongly negative: `0.0014` for value and `-0.147, 0.109, -0.104` for the
  three score coordinates.
- Severe constituent score tails were unchanged: `18/128` antithetic versus
  `17/128` independent constituents had `max(abs(score)) > 1000`.
- At equal cost and `K=4`, every paired-valid point estimate had larger
  antithetic variance: ratios `3.95` for value and `1.58, 1.63, 2.49` for
  `(log_kappa, log_nu, log_observation_noise)` score.
- Numerical validity was poor in both arms: `8/128` antithetic and `13/128`
  independent constituents were invalid. Only `3/16` `K=4` replicates had
  both complete eight-pass estimators valid.
- The averaged recursive score failed the predeclared finite-difference gate
  for its own fixed-noise antithetic scalar. Maximum relative error was
  `1.78` against a `0.05` tolerance; one `0.0008` endpoint was nonfinite.

The variance comparisons are therefore descriptive negative evidence, not a
statistically supported ranking. The hard numerical/derivative veto alone is
enough to forbid promotion.

## Executed Scope

| Item | Executed value |
|---|---|
| Target | Frozen `austria_sir_T20`, `y1:y20`, observation hash `cd794ad6...0f07` |
| Parameter point | `(0,0,0)` |
| Particles | `N=1008` |
| Candidate | Mean of `K` complete `Z/-Z` GenUT pairs |
| Equal-cost baseline | Mean of `2K` mutually independent complete GenUT runs |
| Ensemble sizes | `K=1,2,4`; `K=4` primary |
| Replicates | `16` independent ensemble rows |
| Backend | TensorFlow 2.19.1, GPU/XLA, FP32 tensors, TF32 enabled |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| Memory | Verified memory growth; allocator peak `97,884,672` bytes |
| Claim wall time | `136.67 s` |

The historical July tuning identity was stale after later opt-in additions to
the shared higher-moment callable. The campaign therefore reran the original
eight-arm Austria grid on disjoint calibration/validation data with pairwise
and projected-cumulant controls explicitly frozen to zero. Six arms were
eligible. Current-source tuning selected:

```text
epsilon=8
sinkhorn_steps=8
balance_steps=8
ridge=1e-5
diagonal higher-moment steps=4
diagonal strength=0.2
diagonal floor=1e-5
pairwise steps/strength=0/0
projected-cumulant steps/strength=0/0
```

This differs from the stale July `16/16` Sinkhorn/balance setting. The tuning
artifact is
`docs/benchmarks/artifacts/genut_austria_antithetic_ensemble_20260803/tuning_attempt01/result.json`.

## Equal-Cost Results

The intervals for `K=1,2` are pointwise 95% percentile-bootstrap intervals.
The `K=4` intervals are the declared Bonferroni familywise 95% intervals.
Ratios above one favor the independent baseline.

| K | Coordinate | Variance ratio anti/independent | Log-ratio interval | Paired-valid rows |
|---:|---|---:|---:|---:|
| 1 | value | 1.404 | `[-1.052, 1.604]` | 11/16 |
| 1 | `log_kappa` | 21.759 | `[-1.253, 4.491]` | 11/16 |
| 1 | `log_nu` | 17.599 | `[-1.056, 5.161]` | 11/16 |
| 1 | `log_obs_noise` | 4.102 | `[-0.584, 5.055]` | 11/16 |
| 2 | value | 1.904 | `[-0.095, 2.161]` | 7/16 |
| 2 | `log_kappa` | 2.453 | `[-0.801, 4.200]` | 7/16 |
| 2 | `log_nu` | 1730.542 | `[0.914, 7.702]` | 7/16 |
| 2 | `log_obs_noise` | 18.469 | `[1.414, 5.494]` | 7/16 |
| 4 | value | 3.945 | `[0.000, 3.598]` | 3/16 |
| 4 | `log_kappa` | 1.582 | `[-0.196, 4.005]` | 3/16 |
| 4 | `log_nu` | 1.628 | `[-0.017, 2.936]` | 3/16 |
| 4 | `log_obs_noise` | 2.494 | `[-0.834, 2.349]` | 3/16 |

The `K=4` bootstrap has only three paired-valid rows. Its intervals are
unstable and one lower endpoint degenerates to zero because bootstrap
resamples can have zero sample variance. These intervals cannot support a
ranking. They are retained only to show that the observed direction was not a
hidden antithetic benefit.

Runtime was essentially equal, as expected for equal complete-pass counts.
At `K=4`, mean accumulated runtime was `1.435 s` antithetic and `1.423 s`
independent. Variance-times-runtime ratios were also unfavorable:
`3.65, 3.11, 2.58, 2.49` for value and the three scores.

## Value And Score Interpretation

### Value

Antithetic value variance was larger at every tested `K`. The valid-only
`K=4` mean was `-683.259` with MCSE `0.130`; the independent valid-only mean
was `-683.178` with MCSE `0.163`. These means use different valid subsets
(`10` versus `8` rows), so their small difference cannot be interpreted as a
bias comparison. The deterministic SGQF approximation is `-682.348`, but SGQF
is not an exact oracle and cannot rescue either estimator.

### Score

The intended claimed target is the derivative of each ensemble's own fixed-
noise finite GenUT scalar. The quantity actually computed is the average of
the constituent recursive forward-sensitivity scores. The exact-arithmetic
branchwise derivation says these should agree, but the executed FP32/TF32
Austria route failed the numerical same-scalar check badly:

```text
realized step 0.0004
finite difference: [315.361, 108.299, 329.781]
recursive score:   [-92.054, -84.318, 11.609]
relative errors:   [1.292, 1.779, 0.965]
```

At step `0.0008`, the `log_kappa` finite-difference endpoint was nonfinite and
the other relative errors were `0.645` and `0.767`. Thus the HMC-facing
same-scalar score claim is not validated at this arithmetic and seed. The
failure may reflect branch changes and extreme numerical conditioning rather
than a missing analytic chain-rule term, but that distinction does not make
the executed force reliable for HMC.

Score distance to SGQF is explanatory only. No exact or converged same-event-
order Austria score reference was available. The event-order-mismatched
`O(N^2)` teacher was correctly excluded.

## Numerical Validity And Replay

| Diagnostic | Antithetic | Independent |
|---|---:|---:|
| Valid constituents | 120/128 | 115/128 |
| Nonfinite/program-invalid | 2/128 | 7/128 |
| Other residual-gate failures | 6/128 | 6/128 |
| Severe score tails | 18/128 | 17/128 |
| Valid complete `Z/-Z` pairs | 56/64 | N/A |

The lower antithetic invalid count is descriptive. It was not a predeclared
promotion criterion, has no uncertainty analysis, and severe tails were not
reduced.

A post-run replay warning also appeared. The identical stateless root seed
`140000`, source hashes, controls, GPU class, XLA, FP32, and TF32 settings gave
different results in separate processes:

```text
smoke antithetic average score: [126.088, -176.967, -39.517]
claim antithetic average score: [-92.054, -84.318, 11.609]
```

The corresponding average values differed by `0.0378`; one positive-
constituent value differed by `0.1376`. This establishes cross-process output
non-reproducibility for this unstable route under the executed GPU settings.
It does not establish that antithetic coupling caused the instability.

## Decision

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject complete-run antithetic reflection for current Austria GenUT | Failed: no coordinate nominated; all observed `K=4` variance ratios above one | Failed same-scalar derivative gate; invalid constituents in both arms | Exact cause of branch/numerical instability and absence of exact Austria oracle | Do not spend more passes on antithetic averaging; repair the base Austria score/numerical route first | no proof of general antithetic failure, exact bias, or posterior/HMC behavior |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | Failed: derivative parity and constituent validity |
| Statistically supported ranking | None; paired-valid counts are too small, especially `K=4` |
| Descriptive-only differences | All variance ratios, means, SGQF gaps, tail counts, invalid counts, runtime, and pair correlations |
| Viable candidates | Neither arm is HMC-ready; antithetic is not viable as a variance repair |
| Default readiness | Not established; no default change |
| Next evidence needed | A base GenUT Austria route with deterministic replay, valid same-scalar derivative behavior, and a same-event-order accuracy reference before another variance-reduction study |

## Negative-Result Classification

- **Implementation/harness failures repaired:** comparator-row selection,
  stale tuning identity, and the initial finite-difference step construction.
- **Numerical/diagnostic failure:** the final recursive score fails the
  same-scalar finite-difference gate; both arms contain invalid rows.
- **Evidence against this candidate:** reflected complete-run innovations do
  not provide negative correlation, lower tails, or lower equal-cost variance.
- **What remains viable:** other repairs to the base Austria score route; this
  result does not reject all variance reduction or all GenUT research.

## Post-Run Red Team

The strongest alternative explanation for the unfavorable variance ratios is
selection on the small paired-valid subset. That explanation weakens any
ranking, but it does not create evidence that antithetic reflection helps:
pair correlations and constituent tail counts use much larger samples and are
still unfavorable or neutral. A result that would overturn this decision is a
fresh, numerically valid same-scalar Austria route with strong negative `Z/-Z`
correlation and replicated equal-cost variance reduction. The weakest evidence
is score accuracy because no exact Austria oracle was available.

## Verification

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/highdim/test_genut_austria_antithetic_ensemble.py
# 6 passed
```

The serious tuning and claim runs used trusted GPU/XLA execution, FP32 tensors,
TF32 enabled, and verified TensorFlow memory growth. Raw constituent and
ensemble rows are retained next to the claim artifact.
