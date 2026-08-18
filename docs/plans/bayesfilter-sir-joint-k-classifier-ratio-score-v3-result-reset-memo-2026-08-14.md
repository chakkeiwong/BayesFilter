# Joint-k Classifier-Ratio V3 Result And Clean Reset Memo

Date: 2026-08-14  
Status: `TERMINAL_EXACT_ORACLE_FAILURE__SIR_NOT_RUN`

## Outcome

The proposed joint-k method was implemented and executed through the mandatory
exact Gaussian oracle. The valid full run failed its predeclared admission
gate, so SIR `T=20`, `T=40`, and `T=50` were not launched.

The estimator was an odd shared conditional logit:

```text
delta in {0.01, 0.02, 0.03, 0.04}
z(y, delta) = c1(y) r + c3(y) r^3 + c5(y) r^5
r = delta / 0.04
score = calibrated c1(y_obs) / (2 * 0.04)
```

Only simulated observation paths and the declared positive delta magnitude
entered the classifier. No filter, particles, resampling, smoothing,
latent-state posterior, Fisher identity, complete-data score, likelihood
evaluator, or prior simulation-score artifact was used. Source and fresh-
process runtime dependency audits passed.

## Plan, Review, And Artifacts

- plan: `docs/plans/bayesfilter-sir-joint-k-classifier-ratio-score-v3-plan-2026-08-14.md`;
- review: `docs/plans/bayesfilter-sir-joint-k-classifier-ratio-score-v3-plan-review-2026-08-14.md`;
- full result: `docs/benchmarks/artifacts/sir_joint_k_classifier_ratio_score_20260814/exact_full_attempt01/result.json`;
- repaired smoke: `docs/benchmarks/artifacts/sir_joint_k_classifier_ratio_score_20260814/exact_smoke_attempt02/`.

## Execution Evidence

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, RTX 5080 via
trusted GPU wrapper, TensorFlow memory growth configured before logical-device
initialization, XLA enabled, TF32 disabled. The combined focused suite passed
`21` tests before execution. The first smoke stopped before fitting because
the reduced profile used the full-run batch size; this was repaired, tested,
and rerun successfully. That smoke is diagnostic only.

The full run produced 27 final heads: 3 horizons x 3 coordinates x 3
replicates, with every head trained jointly over all four deltas. Only 2 of 27
heads passed all head-level diagnostics. The only recorded failure was:

| Gate | Failed heads |
|---|---:|
| Per-delta ECE `<=0.04` | 25 |

AUC increased with delta in the inspected rows, fitted temperatures were
positive and near one, and optimizer-completion gates passed. Thus this is a
valid calibration/methodology failure, not a dependency or optimizer
truncation failure.

## Exact-Oracle Cell Status

The exact contract requires three admitted replicates for every cell. No cell
met that requirement:

| Cell | Exact score | Admitted replicates | Status |
|---|---:|---:|---|
| `T20,j0` | 18.8999 | 1 | failed |
| `T20,j1` | 0.4061 | 1 | failed |
| `T20,j2` | -14.1977 | 0 | failed |
| `T40,j0` | 13.7273 | 0 | failed |
| `T40,j1` | -1.6084 | 0 | failed |
| `T40,j2` | -14.4768 | 0 | failed |
| `T50,j0` | 5.4009 | 0 | failed |
| `T50,j1` | -2.2795 | 0 | failed |
| `T50,j2` | -9.6477 | 0 | failed |

The individual score values retained in row artifacts are not admitted score
references. No SIR estimate exists.

## Decision And Inference Status

| Item | Status |
|---|---|
| Hard veto | exact-oracle cell admission failed |
| Joint-k admission | not admitted |
| SIR execution | not run |
| Statistical ranking | none |
| Descriptive evidence | shared odd model learned monotone AUC-vs-delta behavior, but per-delta ECE was inadequate |
| Default/HMC readiness | none |
| Next evidence | a newly reviewed calibration or ratio-estimation method must first pass all nine Gaussian cells |

This does not refute the conditional classifier identity. It shows that this
specific shared odd model plus temperature calibration and per-delta ECE gate
did not produce an admitted exact score reference.

## Red-Team Interpretation

The strongest alternative explanation is that per-delta ECE with ten bins is
too noisy or too strict for the available 2,048 test paths per delta. Another
possibility is that one global temperature cannot calibrate all delta levels
simultaneously. The experiment was designed to veto rather than distinguish
those hypotheses; relaxing ECE after the result would be methodological drift.

The monotone AUC pattern is useful explanatory evidence but cannot promote a
score reference. The exact gate remains failed.

## Clean Restart State

Resume only from the V3 plan, review, this memo, the full result directory, the
joint-k implementation, runner, wrapper, and tests listed above. Do not resume
V1/V2 or the rejected Fisher/path-importance route as though they were the
joint-k method. No SIR output directory exists because the exact gate blocked
the launch.
