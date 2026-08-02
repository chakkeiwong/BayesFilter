# Pairwise-Moment GenUT Cross-Model Trial Result

Date: 2026-07-30  
Plan: `docs/plans/bayesfilter-pairwise-moment-genut-lgssm-ksc-predator-prey-trial-plan-2026-07-30.md`  
Artifact: `docs/benchmarks/artifacts/pairwise_moment_genut_cross_model_20260730/attempt01/result.json`  
Status: `COMPLETE_CROSS_MODEL_PAIRWISE_FEASIBILITY`

## Outcome

The strong Austria-SIR score-variance repair does not transfer uniformly.

- LGSSM `T=50`: all 12 nonzero pairwise arms reduced the pair-moment residual,
  but every arm increased at least one validation score-coordinate variance.
  No arm passed the oracle-free tuning veto, so the diagonal-only baseline was
  retained. Pairwise matching is not useful for this affine scope under the
  tested grid.
- KSC-SV `T=10`: the latent state is scalar. There are zero off-diagonal
  co-skewness and zero off-diagonal co-kurtosis constraints. The pairwise route
  is therefore an exact structural no-op, confirmed across all 16 seeds.
- Predator-prey `T=20`: tuning selected four pairwise steps at strength `0.05`.
  The pair residual fell substantially on validation, but the untouched claim
  showed only a small aggregate score-variance change whose bootstrap interval
  includes one. Two of six score-coordinate SDs increased slightly. The
  candidate remains feasible but is not a supported score improvement.

These results support a narrower interpretation of the Austria result:
pairwise moments repair a severe omitted-cross-moment instability when one is
present; they are not a universal variance reducer.

## What "LGSSM Failed" Means

The LGSSM row did **not** fail implementation, numerical validity, or the
exact Kalman reference. It failed only the candidate-promotion/tuning screen:
the selected arm was the zero-step baseline because every nonzero arm
increased at least one validation score-coordinate variance under the declared
screen. Consequently, the final 16-seed LGSSM candidate is identical to the
baseline; no nonzero pairwise arm was promoted to the claim run.

There are two separate reasons this outcome is expected and should not be read
as a theorem against pairwise matching:

1. The generic correction computes its pairwise targets from the finite,
   likelihood-weighted source cloud (`higher_moment_contract_e.py:521-531`).
   For an exact Gaussian posterior, the population standardized targets are
   `E[z_i^2 z_j]=0` and `E[z_i^2 z_j^2]=1`, but the empirical targets are noisy
   `O(N^{-1/2})` estimates and their parameter tangents are noisy as well.
   The correction therefore follows finite-cloud noise rather than a known
   Gaussian target.
2. The replicated cubature reset has axis points `+/-sqrt(3)e_i`, whose
   off-diagonal co-kurtosis is zero, not the Gaussian value one. Pairwise
   matching consequently introduces a nonlinear shape deformation at every
   reset. At `T=50`, that deformation is carried into later weights, OT maps,
   and score increments. Lower pair residual does not imply lower likelihood
   or Kalman-score error.

The tuning screen was intentionally oracle-free but statistically weak: it used
two particle seeds per validation trajectory and rejected an arm if any of five
score-variance ratios exceeded one. Even the smallest arm was rejected by tiny
validation increases (for example, ratios `1.015`, `1.003`, and `1.013`), while
the strongest arm had a clearly adverse `phi2` ratio of `2.017`. A post-run
diagnostic against the exact Kalman score, preserved in
`docs/benchmarks/artifacts/pairwise_moment_genut_cross_model_20260730/attempt01/lgssm_kalman_postrun_diagnostic.json`,
found aggregate validation score RMSE `1.698` for zero pairwise correction and
larger RMSE for every nonzero arm, but that oracle was not used to select
controls.

Thus the defensible conclusion is: **pairwise matching was not useful for this
finite affine scope under this empirical-target map and tested grid**. It is
not: “the LGSSM implementation is broken” or “pairwise matching can never help
Gaussian models.” A fair Gaussian-specific follow-up would use fixed targets
`(0,1)` for the off-diagonal moments or omit higher-moment correction entirely,
and would use more than two tuning seeds.

## Value And Score Summary

All stochastic entries use `N=1008` and 16 common particle seeds. Intervals are
Student-t 95% intervals across seeds.

### LGSSM `T=50`

No nonzero arm passed validation, so the selected candidate equals the
diagonal-only baseline.

Exact Kalman reference:

```text
value:  -136.075975
score:  (5.655446, -3.835057, 0.302362, -1.917176, 4.354276)
```

| Quantity | Selected GenUT mean [95% CI] | SD | Kalman reference |
| --- | ---: | ---: | ---: |
| Value | -136.333498 [-136.582850, -136.084146] | 0.467948 | -136.075975 |
| `phi1` score | 5.795283 [4.959296, 6.631270] | 1.568861 | 5.655446 |
| `phi2` score | -4.049541 [-4.383751, -3.715331] | 0.627197 | -3.835057 |
| `phi3` score | 0.239935 [0.085473, 0.394397] | 0.289872 | 0.302362 |
| `q_scale` score | -1.983681 [-3.002136, -0.965226] | 1.911291 | -1.917176 |
| `r_scale` score | 5.537545 [3.641101, 7.433989] | 3.558975 | 4.354276 |

All 12 nonzero arms lowered the validation pair residual from `0.8694`; the
strongest arm lowered it to `0.4494`. That same arm had validation
score-variance ratios

```text
(0.795, 2.017, 1.110, 1.075, 0.799),
```

so moment-residual improvement did not imply score improvement. In particular,
the second score coordinate more than doubled in variance. Retaining the
zero-step arm was the correct oracle-free selection result.

### KSC-SV `T=10`

The zero-step and nominal nonzero pairwise arms were exactly identical after
the scalar structural-no-op repair.

| Quantity | GenUT mean [95% CI] | SD | Dense transformed-mixture reference |
| --- | ---: | ---: | ---: |
| Value | -19.953950 [-19.979315, -19.928584] | 0.047602 | -19.956279 |
| `z_gamma` score | -0.694425 [-0.710061, -0.678789] | 0.029344 | -0.705673 |
| `log_beta` score | 0.607675 [0.555822, 0.659529] | 0.097311 | 0.635493 |

The dense value reference converged between Legendre orders 401 and 601 to
`1.03e-13`. Its diagnostic centered-FD score had maximum step gap `2.37e-10`
and order gap `0`. This finite difference is a reference diagnostic only; the
GenUT runtime score remains the manual recursive score.

For context, fixed Zhao-Cui was

```text
value: -19.956289
score: (-0.705672, 0.635489)
```

and is essentially equal to the dense transformed-mixture reference on this
short scalar target. Fixed SGQF was

```text
value: -19.950942
score: (-0.692475, 0.609578).
```

Pairwise terms cannot improve KSC-SV as currently represented because the
latent state is one-dimensional. Improving its GenUT approximation requires a
different scalar moment family or distributional representation, not
off-diagonal pair moments.

### Predator-Prey `T=20`

Tuning selected pairwise steps `4`, strength `0.05`. There is no exact score
oracle for this row.

| Quantity | Diagonal-only mean [95% CI], SD | Pairwise mean [95% CI], SD |
| --- | ---: | ---: |
| Value | -102.739536 [-102.895313, -102.583759], 0.292340 | -102.744951 [-102.908661, -102.581242], 0.307226 |
| `r` score | -27.775234 [-28.957720, -26.592749], 2.219120 | -27.805207 [-28.997438, -26.612975], 2.237409 |
| `K` score | 0.077647 [0.036739, 0.118554], 0.076770 | 0.072292 [0.031901, 0.112684], 0.075801 |
| `a` score | -0.087487 [-0.090428, -0.084546], 0.005520 | -0.087744 [-0.090504, -0.084985], 0.005178 |
| `s` score | 1.042272 [0.781685, 1.302859], 0.489032 | 1.040905 [0.778998, 1.302813], 0.491511 |
| `u` score | 18.367237 [17.633828, 19.100646], 1.376357 | 18.434329 [17.729820, 19.138839], 1.322123 |
| `v` score | -23.650981 [-24.566622, -22.735340], 1.718345 | -23.734762 [-24.614392, -22.855132], 1.650764 |

Every paired candidate-minus-baseline value/score interval included zero. The
aggregate geometric score-variance ratio was `0.9533`, with bootstrap 95%
interval `[0.8315, 1.1277]`. Thus:

- there is no statistically supported variance reduction;
- the value shift `-0.0054`, paired CI `[-0.0217, 0.0108]`, is compatible with
  zero;
- four score-coordinate SDs decreased slightly and two increased slightly;
- no score-accuracy conclusion is possible without an exact reference.

SGQF and Zhao-Cui remain same-target diagnostics, not truth oracles. Their
scores were respectively

```text
SGQF:      (-27.6411, 0.08411, -0.08414, 0.85570, 17.5256, -22.6350)
Zhao-Cui:  (-22.6764, 0.13828, -0.08342, 0.24589, 17.6053, -22.8159)
```

and do not establish which finite nonlinear score is more accurate.

## Engineering Evidence

| Item | Result |
| --- | --- |
| Focused CPU-hidden mathematical/regression tests | `35 passed` across the full focused suite after repair; scalar exact-no-op test included |
| Trusted GPU/XLA smoke | `PASS_GPU_XLA_SMOKE` |
| Final claim rows | `96/96` baseline/candidate rows valid across three models |
| Backend | TensorFlow FP32, TF32 enabled, XLA enabled |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| Memory policy | verified memory growth before logical-device initialization |
| TensorFlow allocator peak | `134,545,408` bytes, approximately `128.3 MiB` |
| Final campaign wall time | `464.80 s` |

The first smoke attempt exposed an XLA GEMM autotuner layout failure for a
`[1008,5,3] x [3,3]` LGSSM tangent projection. The failed attempt is preserved.
The contraction was rewritten as the algebraically identical broadcasted
elementwise reduction. Independent forward-autodiff parity and the full focused
suite passed before the successful smoke and final campaign. This was an
engineering repair, not a change to the finite mathematical map.

XLA emitted register-spill warnings. They are performance diagnostics; no OOM,
CPU fallback, nonfinite row, covariance failure, or score-additivity failure
occurred.

## Decision

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain diagonal-only LGSSM | no nonzero arm passed validation score-variance veto | no implementation veto | whether another correction family addresses remaining Kalman error | do not spend more pairwise tuning budget on this affine scope | no rejection of GenUT generally |
| Treat KSC pairwise route as structural null | exact equality across 16 seeds | none | scalar higher-moment approximation remains imperfect | investigate scalar distributional matching if KSC needs repair | no pairwise KSC improvement possible in current state chart |
| Keep predator-prey pairwise arm diagnostic/opt-in | finite and value-stable, but aggregate variance CI includes one | promotion criterion not met | no exact nonlinear score oracle | no promotion; revisit only with stronger reference or a distinct failure signal | no score-accuracy or superiority claim |
| Retain Austria-specific pairwise hypothesis | prior Austria variance reduction was strong | not re-tested here | Austria value shift and `log_kappa` bias remain | pursue Austria-specific tradeoff/reference work | no universal pairwise default |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Passed after one preserved XLA infrastructure repair |
| Statistically supported ranking | None across nonlinear methods |
| Statistically supported pairwise variance reduction | None for these three rows; KSC is exact equality |
| Descriptive-only difference | Small predator-prey variance/value/score changes |
| Default readiness | Not established; pairwise remains opt-in |
| Next evidence needed | Austria-specific value/score reference work or a new model exhibiting a cross-moment score instability |

## Post-Run Red Team

The strongest alternative explanation is that two tuning seeds were too noisy
to nominate useful LGSSM or predator-prey controls. The untouched 16-seed
predator-prey result nevertheless did not support a reduction, while the exact
LGSSM oracle and consistent validation tradeoff give no reason to override the
selection veto. A larger tuning campaign could find a different tradeoff but is
not justified by this feasibility result.

The weakest evidence is predator-prey score accuracy because no exact oracle
exists. The strongest evidence is structural: pairwise correction is exactly
irrelevant at `d=1`, and the LGSSM nonselection follows the predeclared
oracle-free validation rule rather than post-hoc oracle tuning.
