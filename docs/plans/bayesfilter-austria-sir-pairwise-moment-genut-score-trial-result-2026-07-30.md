# Austria SIR Pairwise-Moment GenUT Score Trial Result

Date: 2026-07-30  
Plan: `docs/plans/bayesfilter-austria-sir-pairwise-moment-genut-score-trial-plan-2026-07-30.md`  
Artifact: `docs/benchmarks/artifacts/austria_sir_pairwise_moment_genut_score_20260730/attempt01/result.json`  
Status: `PAIRWISE_SCORE_VARIANCE_PROMOTION_FAIL`

## Outcome

The pairwise correction is implemented and works mechanically. It matches all
ordered off-diagonal standardized co-skewness moments `E[z_i^2 z_j]` and all
unordered off-diagonal co-kurtosis moments `E[z_i^2 z_j^2]` through a bounded
deterministic residual-gradient map. Every iteration restandardizes the cloud,
so the full mean and covariance matrix remain restored. The complete executed
map has a manual JVP and no runtime autodiff or finite differences.

On the active Austria SIR target (`d=18`, `J=9`, `T=20`, `N=1008`), tuning
selected:

```text
epsilon=8
sinkhorn_steps=16
balance_steps=16
ridge=1e-5
diagonal steps=4
diagonal strength=0.2
pairwise steps=4
pairwise strength=0.02
pairwise floor=1e-5
```

The selected candidate substantially reduced the recursive-score variance and
both diagonal and pairwise moment residuals. It nevertheless failed the full
predeclared promotion rule because its mean finite likelihood moved by `1.260`,
which is about `7.9` baseline Monte Carlo standard errors. The value shift is a
change in the finite approximation, not an implementation invalidity, but it
prevents a no-regression/promotion verdict under this plan.

## Value And Score Results

Sixteen common particle seeds `98201..98216` were used. Intervals are
Student-t 95% confidence intervals across seeds.

| Quantity | Diagonal-only mean [95% CI], SD | Pairwise mean [95% CI], SD | SGQF diagnostic |
| --- | ---: | ---: | ---: |
| Value | -683.3638 [-683.7031, -683.0245], 0.6367 | -682.1039 [-682.4049, -681.8030], 0.5647 | -682.3480 |
| `log_kappa_scale` score | -865.923 [-2696.642, 964.796], 3435.632 | -16.304 [-35.760, 3.152], 36.512 | 28.739 |
| `log_nu_scale` score | 170.885 [-507.150, 848.920], 1272.439 | -109.627 [-119.091, -100.162], 17.762 | -106.659 |
| `log_observation_noise_scale` score | 114.981 [-45.925, 275.888], 301.967 | 15.907 [4.912, 26.902], 20.635 | 9.431 |

The candidate reduced score SD by factors of approximately:

```text
log_kappa_scale:                    94.1x
log_nu_scale:                       71.6x
log_observation_noise_scale:        14.6x
```

Equivalently, sample variance fell by `99.989%`, `99.981%`, and `99.533%`.
The paired-seed bootstrap aggregate geometric variance ratio was `0.000468`
with 95% interval `[0.000082, 0.063166]`, entirely below one.

SGQF is not an exact oracle. Its `log_nu_scale` and observation-noise scores
fall inside the pairwise candidate intervals. Its `log_kappa_scale` score does
not: `28.739` is above the candidate upper bound `3.152`. The candidate
therefore has much better precision but does not establish three-coordinate
score agreement.

The SGQF value is closer to the pairwise mean than to the diagonal-only mean,
but it is just outside the candidate value interval. This is descriptive and
does not rescue the failed value-shift gate.

## Moment Diagnostics

| Diagnostic | Diagonal-only | Pairwise | Ratio |
| --- | ---: | ---: | ---: |
| Mean normalized pairwise residual objective | 0.3190 | 0.1446 | 0.453 |
| Mean normalized diagonal residual objective | 1.3675 | 0.8700 | 0.636 |

Thus the pairwise step did not trade away the diagonal objective; both improved
on the untouched target. All 16 candidate rows passed OT/reset, full
mean/covariance, finite-score, device, and score-increment-additivity gates.

## Tuning Result

Thirteen unique arms were tested: zero pairwise steps plus steps `{1,2,4}` and
strengths `{0.005,0.01,0.02,0.05}`. Two arms passed all validation selection
vetoes:

```text
steps=4, strength=0.01
steps=4, strength=0.02  <- selected for lower validation score variance
```

The strongest `steps=2,strength=0.05` arm was invalid. Several smaller arms
reduced the pairwise objective but increased one score-coordinate variance,
confirming that residual reduction alone is not a sufficient tuning objective.
Selection did not read the claim observations, claim seeds, or SGQF values.

## Engineering Evidence

| Item | Result |
| --- | --- |
| CPU-hidden mathematical and regression tests | `34 passed` |
| Trusted GPU/XLA smoke | `PASS_GPU_XLA_SMOKE` |
| Final candidate rows | `16/16` valid |
| Runtime backend | TensorFlow FP32, TF32 enabled, XLA enabled |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| Memory policy | verified memory growth before logical-device initialization |
| TensorFlow allocator peak | `70,905,344` bytes, about `67.6 MiB` |
| Full campaign wall time | `215.06 s` |

XLA emitted register-spill warnings for several fused pairwise contractions.
This is a performance diagnostic, not a correctness veto. No OOM, CPU
fallback, NumPy runtime path, or non-XLA claim path occurred.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain pairwise correction as an opt-in Austria variance-repair candidate | Every score SD lower and aggregate variance-ratio CI below one | Full promotion fails on value mean-shift gate | Whether the shifted finite value is less biased and whether `log_kappa` discrepancy is residual bias | Run a fresh value/score tradeoff ladder around steps 4, strengths 0.01-0.02 with an independent dense/teacher diagnostic if feasible | no default or HMC promotion |
| Reject stronger diagonal-only tuning as the immediate repair | Pairwise map directly addresses omitted cross moments and sharply reduces variance | No implementation veto | Higher-order terms beyond pairwise remain omitted | Keep pairwise path opt-in while testing tradeoff | no full-distribution theorem |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Pass for implementation and all executed rows |
| Pairwise variance reduction | Statistically supported by the declared limited paired-seed bootstrap diagnostic |
| Score accuracy ranking | Not established; no exact Austria score oracle |
| Value no-regression | Failed under the predeclared baseline-SE gate |
| Default readiness | Not established |
| Next evidence | Independent value/score tradeoff validation and `log_kappa` bias localization |

## Post-Run Red Team

The strongest alternative explanation is that the pairwise map suppresses an
unstable tangent mode while changing the finite carried distribution enough to
move the likelihood target. That can improve apparent precision without
reducing score bias. The `log_kappa` SGQF exclusion supports this caution. A
result that would overturn the current non-promotion verdict is an independent
same-target study showing stable candidate values and score agreement under a
stronger reference/teacher while retaining the variance reduction.

The weakest evidence is score accuracy, not score variance. The variance
reduction is clear for these common seeds; equality to the true observed-data
score remains unsupported.
