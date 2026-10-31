# Expanded SQMC Kalman comparison

Program: FP64 GPU/XLA reference comparison, TF32 disabled, with exact-scope calibration of flow substeps 2 versus 8. Full-model repair scopes additionally calibrate transport smoothing epsilon on fresh partitions before complete-score tuning. Other numerical protections remain inherited hypotheses. This report does not describe the FP32/TF32 production configuration. Tuning artifacts and manifests for every unit are linked by path in audit.json. A finite candidate is not certified accurate.

Recorded 17/32 route/scope units and 68/128 final cells; 68 final cells are numerically valid. Recorded 740/8480 individual scores, including explicitly invalid raw coordinates. Guard-returned zero scores are sentinels, not actual gradients; errors for those values are unavailable. Engineering audit findings: 0. Missing units: 15.

Two data sets, each crossed with two filter designs, permit descriptive comparisons only. Filter repetitions on the same data are not independent data sets. No confidence interval or statistically supported method ranking is inferred. Previously inspected pilot pairs remain repeated evidence, not new untouched observations.

## Actual scores and likelihoods

[All score coordinates and absolute errors](scores.csv); [all actual log likelihoods and errors](likelihoods.csv). Separate score tables follow:

- [p44_d3_T10](p44_d3_T10.md)
- [p44_d3_T120](p44_d3_T120.md)
- [full_d3_T2](full_d3_T2.md)
- [full_d3_T10](full_d3_T10.md)
- [full_d3_T120](full_d3_T120.md)
- [full_d10_T2](full_d10_T2.md)
- [full_d10_T10](full_d10_T10.md)
- [full_d10_T120](full_d10_T120.md)

## Model conditioning

The transition norm and process-covariance eigenvalues are evaluated at the fixed data-generating parameter. Stable, moderately conditioned test models do not test near-singular or highly persistent regimes.

| Scope | Maximum absolute row sum of A | Smallest Q eigenvalue | Largest Q eigenvalue | Q condition number |
|---|---:|---:|---:|---:|
| p44_d3_T10 | 0.1347052643 | 0.162 | 0.234 | 1.444444444 |
| p44_d3_T120 | 0.1347052643 | 0.162 | 0.234 | 1.444444444 |
| full_d3_T2 | 0.77 | 0.1491374205 | 0.2690940835 | 1.804336448 |
| full_d3_T10 | 0.77 | 0.1491374205 | 0.2690940835 | 1.804336448 |
| full_d3_T120 | 0.77 | 0.1491374205 | 0.2690940835 | 1.804336448 |

## Descriptive errors

Invalid candidates are excluded from these summaries and remain visible in the raw tables. Component RMS accounts for parameter count, but both RMS and vector L2 mix coordinate scales. Compare these summaries within the same model and use the per-coordinate tables for substantive interpretation; normalization alone does not make different models comparable.

| Scope | Route | Valid / 4 | Flow steps | Epsilon | Mean absolute log-likelihood error | Mean score L2 error | Component RMS error | Largest coordinate error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| p44_d3_T10 | IID | 4 | 8 | 0.4 | 0.02514580143 | 0.07721764279 | 0.04381378772 | 0.1424514441 |
| p44_d3_T10 | Inverse CDF | 4 | 8 | 0.4 | 0.001373578049 | 0.00525145779 | 0.002817353169 | 0.007160435378 |
| p44_d3_T10 | Permutation | 4 | 8 | 0.4 | 0.0007301136034 | 0.008887364958 | 0.006136691332 | 0.02284495504 |
| p44_d3_T10 | Permutation cap .97 | 4 | 8 | 0.4 | 0.0005813891265 | 0.008477977838 | 0.005650423274 | 0.02002246499 |
| p44_d3_T120 | IID | 4 | 8 | 0.4 | 0.05127180932 | 0.1680867583 | 0.09533653787 | 0.316274939 |
| p44_d3_T120 | Inverse CDF | 4 | 8 | 0.4 | 0.003037618466 | 0.01260383555 | 0.008472584543 | 0.03137670047 |
| p44_d3_T120 | Permutation | 4 | 8 | 0.4 | 0.00123169498 | 0.01551373717 | 0.00916150836 | 0.0264850286 |
| p44_d3_T120 | Permutation cap .97 | 4 | 8 | 0.4 | 0.0008892455173 | 0.01013234975 | 0.006164830015 | 0.01843986946 |
| full_d3_T2 | IID | 4 | 8 | 25.6 | 0.04196447985 | 0.4275246975 | 0.1079277532 | 0.2964874949 |
| full_d3_T2 | Inverse CDF | 4 | 8 | 102.4 | 0.02656446975 | 0.3233992342 | 0.07852366138 | 0.1681460948 |
| full_d3_T2 | Permutation | 4 | 8 | 102.4 | 0.026033923 | 0.3080040729 | 0.07549901344 | 0.196156238 |
| full_d3_T2 | Permutation cap .97 | 4 | 8 | 102.4 | 0.02607088497 | 0.3077937698 | 0.0754603468 | 0.1970320253 |
| full_d3_T10 | IID | 4 | 8 | 25.6 | 0.05488912253 | 0.4577227487 | 0.1169825734 | 0.422951879 |
| full_d3_T10 | Inverse CDF | 4 | 8 | 102.4 | 0.02332795383 | 0.3305180725 | 0.08065731182 | 0.1754449289 |
| full_d3_T10 | Permutation | 4 | 8 | 102.4 | 0.01925429378 | 0.3024790841 | 0.07663854051 | 0.2404790352 |
| full_d3_T10 | Permutation cap .97 | 4 | 8 | 102.4 | 0.02380483799 | 0.2935411305 | 0.07375882532 | 0.2172599879 |
| full_d3_T120 | IID | 4 | 8 | 25.6 | 0.2351393059 | 1.606207124 | 0.4061025349 | 0.9441341526 |

## Conditional heuristic checks

[Per-data-set conditional comparisons](conditional_heuristics.csv) and [parameter-block errors](block_errors.csv) preserve the breakdown. Counts are observed losses among valid final cells against zero score, first-observation-only Kalman score, and matched IID particle estimates. An observed loss vetoes promotion in that situation; absence of a loss is not evidence of superiority. Both cheap score baselines are compared with the full-horizon Kalman score. Per-data-seed errors are preserved in summaries.json.

| Scope | Route | Zero-score losses | First-only losses | IID losses |
|---|---|---:|---:|---:|
| p44_d3_T10 | IID | 0 / 4 | 0 / 4 | 0 / 4 |
| p44_d3_T10 | Inverse CDF | 0 / 4 | 0 / 4 | 0 / 4 |
| p44_d3_T10 | Permutation | 0 / 4 | 0 / 4 | 0 / 4 |
| p44_d3_T10 | Permutation cap .97 | 0 / 4 | 0 / 4 | 0 / 4 |
| p44_d3_T120 | IID | 0 / 4 | 0 / 4 | 0 / 4 |
| p44_d3_T120 | Inverse CDF | 0 / 4 | 0 / 4 | 0 / 4 |
| p44_d3_T120 | Permutation | 0 / 4 | 0 / 4 | 0 / 4 |
| p44_d3_T120 | Permutation cap .97 | 0 / 4 | 0 / 4 | 0 / 4 |
| full_d3_T2 | IID | 0 / 4 | 0 / 4 | 0 / 4 |
| full_d3_T2 | Inverse CDF | 0 / 4 | 0 / 4 | 1 / 4 |
| full_d3_T2 | Permutation | 0 / 4 | 0 / 4 | 2 / 4 |
| full_d3_T2 | Permutation cap .97 | 0 / 4 | 0 / 4 | 2 / 4 |
| full_d3_T10 | IID | 0 / 4 | 0 / 4 | 0 / 4 |
| full_d3_T10 | Inverse CDF | 0 / 4 | 0 / 4 | 2 / 4 |
| full_d3_T10 | Permutation | 0 / 4 | 0 / 4 | 1 / 4 |
| full_d3_T10 | Permutation cap .97 | 0 / 4 | 0 / 4 | 1 / 4 |
| full_d3_T120 | IID | 0 / 4 | 0 / 4 | 0 / 4 |

## Interpretation and remaining uncertainty

The analytical recursion differentiates the finite particle program. Kalman differentiates the exact marginal log likelihood of the matched predict-first Gaussian model. These quantities differ at finite particle count and numerical resolution. Graph/XLA parity and finite differences check implementation behavior; they do not establish equality with the Kalman score.

The models have full observations, fixed baseline parameters and Gaussian noise. The full model evaluates derivatives in all A entries and lower-Cholesky Q coordinates, including covariance determinant and inverse-covariance terms. These runs do not evaluate nonlinear targets, maximum-likelihood points, partial observations, TF32 accuracy or HMC. The two-setting flow calibration and bounded smoothing ladder do not establish optimality or numerical protection adequacy. Each route calibrates separately; where selected epsilon differs, a route contrast includes that tuning difference and cannot isolate ancestry ordering. Q scores use the listed lower-Cholesky coordinates (log diagonal and unconstrained off-diagonal entries), not derivatives with respect to raw symmetric Q entries.

## Decision record

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve descriptive comparison | 68/128 final cells recorded; no universal accuracy threshold | See invalid cells and heuristic losses | Two independent data sets; narrow calibration | Diagnose specific failures; predeclare further replication if ranking is wanted | Superiority, default readiness, HMC or production admission |

| Inference status | Finding |
|---|---|
| Hard veto screen | Invalid candidates and any provenance/oracle failures are explicitly separated |
| Statistically supported ranking | None |
| Descriptive-only differences | All likelihood, score-error, runtime and heuristic comparisons |
| Default readiness | Not assessed; reference-precision variant with limited tuning |
| Next evidence needed | Target-specific numerical calibration and a predeclared multi-dataset uncertainty analysis |

