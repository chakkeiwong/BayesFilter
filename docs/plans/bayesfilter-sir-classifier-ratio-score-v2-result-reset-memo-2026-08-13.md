# Classifier-Ratio Score V2 Result And Clean Reset Memo

Date: 2026-08-13  
Status: `TERMINAL_EXACT_ORACLE_FAILURE__SIR_NOT_RUN`

## Outcome

The requested filter-independent test was executed through its mandatory exact
Gaussian calibration gate. The valid V2 run failed that gate, so the SIR
campaign at `T=20`, `T=40`, and `T=50` was not launched. This is the correct
stop condition under the reviewed plan; no SIR score reference is available.

The method used throughout was exactly the requested identity:

```text
Y+ ~ simulator(theta + epsilon * e_j)
Y- ~ simulator(theta - epsilon * e_j)
score estimate = calibrated classifier logit(y_obs) / (2 * epsilon)
```

Only observation paths entered the classifier. No filter, particle method,
resampling, smoothing, latent-state posterior, Fisher identity, complete-data
score, likelihood evaluator, or prior simulation-score artifact was used. The
runner source and runtime dependency audits passed.

## V2 Changes And Audit

V1 selected one architecture per horizon across incompatible location and
log-scale ratios. V2 selected among zero-initialized linear, zero-initialized
centered-quadratic logistic, and centered-quadratic MLP heads separately for
each stage/horizon/coordinate using simulated selection data only. The density
ratio target, observation paths, splits, epsilon ladder, calibration, support
gate, extrapolation, and exact-oracle tolerance were unchanged.

The first V2 launch was invalid because the linear head still used Glorot
initialization. It was preserved as
`v2_exact_full_attempt01`. After source and parametrized tests were repaired,
the valid launch was:

`docs/benchmarks/artifacts/sir_classifier_ratio_score_20260813/v2_exact_full_attempt02_zero_convex/result.json`

## Valid V2 Evidence

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, RTX 5080
through the trusted GPU wrapper, TensorFlow memory growth verified before
logical-device initialization, XLA enabled, TF32 disabled. Focused pre-run
suite: `15 passed`.

The valid run used 108 final heads: 3 horizons x 3 coordinates x 4 epsilons x
3 replicates. 43 heads passed all head-level gates. Overlapping failure counts:

| Gate | Failed heads |
|---|---:|
| Platt slope in `[0.5,2.0]` | 42 |
| ECE at most `0.03` | 30 |
| Test signal | 17 |
| AUC range | 7 |

Exact Gaussian cell status:

| Cell | Selected head | Admitted epsilons | Result |
|---|---|---:|---|
| `T20,j0` | linear | 0 | failed |
| `T20,j1` | linear | 0 | failed |
| `T20,j2` | centered-quadratic MLP | 3 | passed cell tolerance; extrapolated `-8.8916 +/- 3.2710` versus exact `-14.1977` |
| `T40,j0` | linear | 0 | failed |
| `T40,j1` | linear | 1 | failed |
| `T40,j2` | centered-quadratic MLP | 1 | failed |
| `T50,j0` | linear | 0 | failed |
| `T50,j1` | linear | 1 | failed |
| `T50,j2` | centered-quadratic MLP | 2 | failed |

The exact-oracle contract requires all nine cells to pass. Therefore the final
classification is `exact_oracle_failed`, and `SIR_NOT_RUN` is a hard
continuation veto. The one passing cell is descriptive diagnostic evidence,
not a generic admission of the procedure.

## Decision And Inference Status

| Decision | Primary criterion | Veto | Next action | Nonclaim |
|---|---|---|---|---|
| Do not execute SIR | all nine exact cells required; 8 failed | exact-oracle continuation veto | stop and review a new protocol before any launch | no SIR score value |
| Preserve V2 as negative evidence | valid source/runtime and artifact checks | head-level and cell-level gates | keep artifacts immutable | ratio estimation is impossible |
| No algorithm ranking | no algorithm outputs were evaluated | absent comparison evidence | none | no superiority or best method claim |

Hard-veto evidence is the failed exact gate. Descriptive evidence is the
head-level pass count and observed control choice. There is no statistically
supported ranking, no default-readiness evidence, and no HMC claim.

## Red-Team Interpretation

The strongest explanation is that the current finite-sample classifier and
calibration protocol is inadequate for weak location ratios and for logits
whose held-out calibration slope is outside the fixed `[0.5,2.0]` gate. This
does not refute the balanced-classifier identity. It establishes that the
current protocol cannot serve as an independent score reference across the
requested SIR scopes.

What would overturn the stop: a newly written and reviewed protocol that still
uses only balanced observation-path classification and `logit/(2*epsilon)`,
passes all nine untouched Gaussian cells, and then passes the SIR-specific
gates. The next protocol must predeclare any calibration or classifier-form
change and cannot tune on `y_obs`, weaken gates, or substitute a filter.

## Clean Restart State

Resume from:

- V2 plan: `docs/plans/bayesfilter-sir-classifier-ratio-score-v2-plan-2026-08-13.md`;
- V2 review: `docs/plans/bayesfilter-sir-classifier-ratio-score-v2-plan-review-2026-08-13.md`;
- this memo;
- valid result and manifest under
  `docs/benchmarks/artifacts/sir_classifier_ratio_score_20260813/v2_exact_full_attempt02_zero_convex/`;
- implementation under `bayesfilter/independent_score/`;
- runner `docs/benchmarks/run_sir_classifier_ratio_score_20260813.py`;
- wrapper `scripts/run_sir_classifier_ratio_score_gpu.sh`;
- tests `tests/highdim/test_classifier_ratio_score_tf.py`.

Do not read or resume the rejected Fisher/path-importance route:
`bayesfilter/highdim/simulation_score_tf.py`,
`docs/benchmarks/run_sir_simulation_score_20260813.py`, or
`docs/benchmarks/artifacts/sir_simulation_score_20260813/`. Those artifacts
remain historical negative evidence only.

No SIR output directory exists because the oracle veto correctly prevented the
launch.
