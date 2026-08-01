# Tuned Cubature/GenUT LGSSM Result

Date: 2026-07-21

## Artifacts

- Plan: `docs/plans/bayesfilter-lgssm-cubature-genut-tuning-comparison-plan-2026-07-21.md`
- Tuning: `docs/benchmarks/artifacts/lgssm_cubature_genut_tuning_20260721_attempt2/`
- Frozen claim: `docs/benchmarks/artifacts/lgssm_cubature_genut_tuned_claim_20260721_attempt2/`
- Comparison: `docs/benchmarks/artifacts/lgssm_cubature_genut_tuned_comparison_20260721.json`

## Tuning Outcome

**Score-route correction (2026-07-21):** this historical `v1` tuning run used
`tf.GradientTape` for the candidate and Kalman scores. Its one-direction FD
check was disabled during tuning, so it is not evidence for the corrected
recursive score. Future claim-bearing runs must use the new `v3` compact
forward-sensitivity score with representative-point FD audit documented in
`bayesfilter-lgssm-cubature-genut-fd-representative-tuning-plan-2026-07-21.md`.

The tuner evaluated all 24 declared control tuples on disjoint calibration and
validation seeds. All 24 were engineering-valid. The selected grid tuple was:

```text
epsilon        = 4.0
sinkhorn_steps = 8
ridge          = 1e-6
```

Selection used the maximum absolute validation mean relative error over all
six coordinates, with squared-mean and low-iteration tie-breaks. The selected
validation objective was `0.8178379392`; this large value means tuning did not
make the score coordinates close to the Kalman target on the four validation
seeds. It did establish a reproducible, claim-independent control choice.

The tuning wall time was approximately `680.2 s`. The claim used untouched
seeds `82320..82335` and did not retune.

## Frozen Claim

The claim was `hard_valid=true` for both Cubature and GenUT. Both screens were
`inconclusive`, and both methods were identical for the Gaussian moment choice.
Peak TensorFlow allocator usage was `3,361,828,352` bytes (about `3.13 GiB`).

### T=50 Six-Coordinate Metric

Values are mean relative error with the same simultaneous 95% interval and
critical value `3.036283222821165` used by the prior Contract E runs.

| Method | Value | phi1 | phi2 | phi3 | q_scale | r_scale | Screen |
|---|---|---|---|---|---|---|---|
| Cubature = GenUT | `-0.069% [-0.327%,+0.190%]` | `+2.953% [-11.304%,+17.211%]` | `-0.417% [-8.212%,+7.378%]` | `-33.546% [-133.222%,+66.130%]` | `+17.076% [-85.727%,+119.879%]` | `+16.175% [-38.542%,+70.892%]` | `inconclusive` |

## Comparison With Previous Runs

| Arm | Value mean relative error | `q_scale` mean relative error | Screen |
|---|---:|---:|---|
| Contract E `N=5000` | `+0.148%` | `-9.912%` | `screen_fail` |
| Contract E `N=10000` | `+0.174%` | `-15.899%` | `screen_fail` |
| Tuned Cubature/GenUT `N=1008` | `-0.069%` | `+17.076%` | `inconclusive` |

The tuned Cubature/GenUT value mean is descriptively closer to zero than both
prior runs. The `q_scale` bias changes sign but is not reduced to an acceptable
range; its interval is extremely wide. The `phi3` and `r_scale` intervals also
remain too wide for a correctness or superiority conclusion.

## Interpretation

The result supports these historical claims:

- the tuning mechanism ran on the finite value program with an autodiff score;
- controls are selected without claim-seed leakage;
- the selected tuple is bound to the runner source hash and exact scope;
- the frozen claim is finite, replayable, and reset/marginal-valid; and
- the tuned value mean is descriptively closer to the Kalman target.

It does not support same-path finite-difference score tuning.

It does not support:

- global optimality, since only the declared 24-point grid was searched;
- method superiority, because the particle count, random streams, controls,
  and execution mode differ from prior Contract E arms;
- exact filtering likelihood or score correctness;
- a `1/N` convergence law; or
- nonlinear-model or NAWM validity.

The prior comparison helper labels the cross-run controls as unmatched and the
seed streams as unpaired. Thus the correct scientific verdict is:

> Tuning improved the descriptive value mean, but did not resolve the score
> discrepancy. The tuned Cubature/GenUT candidate remains a feasible diagnostic
> with an inconclusive Kalman screen, not an established improvement.

## Audit And Verification

- `python -m py_compile` passed for the runner, tuner, claim driver, and
  comparison helper.
- Focused test suite: `5 passed`.
- Tuning artifact contains 24 candidate rows, calibration/validation/claim
  partitions, selected controls, objective, and source hashes.
- Claim artifact binds the tuning artifact hash and runner source hash.
- No historical artifact was overwritten.

## Next Evidence

For a stronger method comparison, run a matched-control study at the same
particle count and execution policy, or tune both routes under a common
predeclared target-specific budget. Use paired per-seed streams only if the
random construction is deliberately shared; reused integer seed labels alone
are not paired evidence.
