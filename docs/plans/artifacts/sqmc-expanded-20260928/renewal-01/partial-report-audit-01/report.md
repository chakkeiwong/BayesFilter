# Expanded SQMC Kalman comparison

Program: FP64 GPU/XLA reference comparison, TF32 disabled, with exact-scope calibration of flow substeps 2 versus 8. Other numerical protections remain inherited hypotheses. This report does not describe the FP32/TF32 production configuration. Tuning artifacts and manifests for every unit are linked by path in audit.json. A finite candidate is not certified accurate.

Recorded 4/32 route/scope units and 16/128 final cells; 16 final cells are numerically valid. Recorded 64/8480 individual scores, including explicitly invalid raw coordinates. Engineering audit findings: 0. Missing units: 28.

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

## Descriptive errors

Invalid candidates are excluded from these summaries and remain visible in the raw tables. Component RMS accounts for parameter count; vector L2 values should not be compared across dimensions as if their scales were equal.

| Scope | Route | Valid / 4 | Flow steps | Mean absolute log-likelihood error | Mean score L2 error | Component RMS error | Largest coordinate error |
|---|---|---:|---:|---:|---:|---:|---:|
| p44_d3_T10 | IID | 4 | 8 | 0.02514580143 | 0.07721764279 | 0.04381378772 | 0.1424514441 |
| p44_d3_T10 | Inverse CDF | 4 | 8 | 0.001373578049 | 0.00525145779 | 0.002817353169 | 0.007160435378 |
| p44_d3_T10 | Permutation | 4 | 8 | 0.0007301136034 | 0.008887364958 | 0.006136691332 | 0.02284495504 |
| p44_d3_T10 | Permutation cap .97 | 4 | 8 | 0.0005813891265 | 0.008477977838 | 0.005650423274 | 0.02002246499 |

## Conditional heuristic checks

[Per-data-set conditional comparisons](conditional_heuristics.csv) and [parameter-block errors](block_errors.csv) preserve the breakdown. Counts are observed losses among valid final cells against zero score, first-observation-only Kalman score, and matched IID particle estimates. An observed loss vetoes promotion in that situation; absence of a loss is not evidence of superiority. Both cheap score baselines are compared with the full-horizon Kalman score. Per-data-seed errors are preserved in summaries.json.

| Scope | Route | Zero-score losses | First-only losses | IID losses |
|---|---|---:|---:|---:|
| p44_d3_T10 | IID | 0 / 4 | 0 / 4 | 0 / 4 |
| p44_d3_T10 | Inverse CDF | 0 / 4 | 0 / 4 | 0 / 4 |
| p44_d3_T10 | Permutation | 0 / 4 | 0 / 4 | 0 / 4 |
| p44_d3_T10 | Permutation cap .97 | 0 / 4 | 0 / 4 | 0 / 4 |

## Interpretation and remaining uncertainty

The analytical recursion differentiates the finite particle program. Kalman differentiates the exact marginal log likelihood of the matched predict-first Gaussian model. These quantities differ at finite particle count and numerical resolution. Graph/XLA parity and finite differences check implementation behavior; they do not establish equality with the Kalman score.

The models have full observations, fixed baseline parameters and Gaussian noise. The full model evaluates derivatives in all A entries and lower-Cholesky Q coordinates, including covariance determinant and inverse-covariance terms. These runs do not evaluate nonlinear targets, maximum-likelihood points, partial observations, TF32 accuracy or HMC. The two-setting calibration does not establish optimality or numerical protection adequacy.

## Decision record

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve descriptive comparison | 16/128 final cells recorded; no universal accuracy threshold | See invalid cells and heuristic losses | Two independent data sets; narrow calibration | Diagnose specific failures; predeclare further replication if ranking is wanted | Superiority, default readiness, HMC or production admission |

| Inference status | Finding |
|---|---|
| Hard veto screen | Invalid candidates and any provenance/oracle failures are explicitly separated |
| Statistically supported ranking | None |
| Descriptive-only differences | All likelihood, score-error, runtime and heuristic comparisons |
| Default readiness | Not assessed; reference-precision variant with limited tuning |
| Next evidence needed | Target-specific numerical calibration and a predeclared multi-dataset uncertainty analysis |

