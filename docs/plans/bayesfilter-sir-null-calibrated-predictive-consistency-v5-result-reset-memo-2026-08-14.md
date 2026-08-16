# V5/V5.1 Null-Calibrated Predictive Consistency Result And Reset Memo

Date: 2026-08-14  
Plan: `docs/plans/bayesfilter-sir-null-calibrated-predictive-consistency-v5-plan-2026-08-14.md`  
Review: `docs/plans/bayesfilter-sir-null-calibrated-predictive-consistency-v5-plan-review-2026-08-14.md`  
Repair: `docs/plans/bayesfilter-sir-null-calibrated-predictive-consistency-v5p1-repair-2026-08-14.md`

## Decision

The new joint same-parameter predictive-consistency test passed for both the
Gaussian harness and SIR. This remedies the unprincipled V4 nine-cell
all-pass/marginal-threshold construction for the stated false-rejection
question.

It does **not** verify that the reported SIR values are the true observed-data
score. The Gaussian exact diagnostic still showed large pathwise score RMSE,
and the SIR null distribution is broad enough to accept numerically large fixed
outputs. The result is therefore predictive-consistency evidence only.

## Executed Artifacts

Gaussian full:
`docs/benchmarks/artifacts/sir_null_calibrated_predictive_consistency_20260814/gaussian_full_attempt01/`

SIR full:
`docs/benchmarks/artifacts/sir_null_calibrated_predictive_consistency_20260814/sir_full_attempt02/`

Persistent execution status:
`docs/benchmarks/artifacts/sir_null_calibrated_predictive_consistency_20260814/sir_full_attempt02_tmux_status/`

The SIR persistent worker exited `0`. It recorded start/finish timestamps,
worker log, PID, tmux session identifier, and exit code. The result checksum
matches the terminal manifest.

## Primary Results

| Result | Gaussian | SIR |
|---|---:|---:|
| Joint covariance rank | 9 | 9 |
| Calibration threshold | 4.1812 | 4.8028 |
| Audit failures / 500 | 29 | 15 |
| Observed audit failure rate | 5.8% | 3.0% |
| One-sided 95% lower failure bound | 4.18% | 1.86% |
| Null claim falsified | no | no |
| Fixed path accepted | yes | yes |
| Fixed-path distance | 1.6171 | 4.7483 |
| Zero-mean max-t diagnostic | passed | passed |

The calibration threshold is the 191st order statistic among 200 independent
calibration scores, providing the predeclared 95% split-conformal marginal
coverage construction. The independent audit is a falsification check and was
not used to retune the threshold.

## Fixed SIR Output

| Cell | Estimate |
|---|---:|
| `T20_j0` | 71.476 |
| `T20_j1` | -111.363 |
| `T20_j2` | 7.544 |
| `T40_j0` | -35.077 |
| `T40_j1` | -105.539 |
| `T40_j2` | -1.024 |
| `T50_j0` | 71.595 |
| `T50_j1` | -108.109 |
| `T50_j2` | -14.606 |

These values are inside the joint simulated same-parameter region. That is
not evidence that their absolute magnitudes or signs are the correct SIR
observed-data scores. The fitted null covariance has very large singular
values, headed by `28652.4` and `11029.6`; broad estimator variability makes
acceptance possible.

## Score-Interpretability Diagnostics

SIR did not pass all score-interpretability diagnostics:

- `T20/T40/T50`, coordinates `j0` and `j1`, exceeded the inherited maximum-
  delta AUC saturation screen;
- the three `j1` heads also exceeded the inherited per-delta ECE screen.

V5.1 correctly classified those inherited heuristic thresholds as diagnostics
rather than predictive-coverage vetoes. They remain important warnings against
interpreting the accepted vector as an accurate likelihood score.

The Gaussian exact-score RMSE by cell was approximately
`[5.02, 5.56, 5.24, 7.20, 7.81, 8.06, 8.32, 8.97, 9.23]`. Thus the Gaussian
run validated the coverage harness but did not validate pathwise score accuracy.

## Inference Status

| Question | Status |
|---|---|
| Was V4's nine-independent-cell threshold principled? | no; replaced |
| Does the joint test control same-parameter predictive rejection? | supported by the split-conformal construction; audit did not falsify it |
| Is the fixed SIR vector typical under repeated estimator outputs at `theta`? | yes, under the frozen V5.1 bundle |
| Is the vector the true SIR observed-data score? | unsupported |
| Are the large SIR magnitudes scientifically trustworthy? | unsupported and concerning |
| Does this validate a filter, ranking, HMC, or default route? | no |

## Post-Run Red Team

The strongest alternative explanation for the pass is not estimator accuracy
but estimator dispersion: the null region is broad because the classifier
score output is highly variable. A constant estimator could also achieve good
coverage, which is why the signal and zero-mean diagnostics are recorded but
cannot repair the missing pathwise truth.

Evidence that would overturn the cautious interpretation would be a new
filter-independent pathwise score identity or a simulator-based procedure that
is proved to identify the score rather than only its repeated-output
distribution. A UKF/SVD score comparison may be useful diagnostically, but it
would test agreement with that filter approximation, not filter-independent
truth.

## Reset

- Preserve V4, V5, V5.1, failed SIR attempt 01, and completed attempt 02.
- Use `scripts/run_tmux_with_status.sh` for future long GPU commands; the
  `nohup`-only wrapper did not survive the managed command host teardown.
- Do not use the V5 SIR vector as an oracle, promotion target, HMC score, or
  algorithm-ranking reference.
- Any next scientific step must explicitly target **pathwise score accuracy**,
  not another self-consistency or false-rejection calibration.
- Do not retune the conformal threshold or inherited AUC/ECE diagnostics on the
  completed audit data.

## Nonclaims

No filter, particle method, smoothing, Fisher identity, latent-state posterior,
or analytical complete-data score was used. The run does not establish exact
SIR scores, likelihood-ratio accuracy, filter correctness, statistical
superiority, HMC readiness, or default readiness.
