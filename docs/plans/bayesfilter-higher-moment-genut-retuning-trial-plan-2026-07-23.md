# Higher-Moment GenUT Retuning Trial Plan

Date: 2026-07-23
Status: `EXECUTED_OPT_IN_DIAGNOSTIC`

## Research Intent

Determine whether the existing particle-filter route

`transition -> likelihood weighting -> entropic OT -> Contract E reset ->
higher-moment correction -> equal-weight continuation`

can obtain materially smaller selected diagonal skewness/kurtosis residuals by
tuning its existing higher-moment controls. This trial does not introduce a
new filtering algorithm and does not assume a likelihood or score oracle.

## Evidence Contract

| Field | Contract |
|---|---|
| Candidate | Existing `higher_moment_shape_jvp` correction inside `finite_value_score` |
| Baseline | Same route with correction steps `0` and the prior fixed OT/Contract E path |
| Primary tuning objective | Mean squared normalized post-correction diagonal skewness and kurtosis residual over calibration/selection trajectories and independent particle clouds |
| Hard vetoes | Nonfinite value/score, invalid OT/reset, covariance failure, score-increment sum failure, nonpositive row mass, or non-GPU/XLA execution |
| Secondary vetoes | Excessive correction displacement or replicate value/score variance relative to the baseline; thresholds are recorded in the result artifact |
| Claim evidence | Untouched fixed observations and 16 particle seeds for LGSSM and fresh transformed SV; predator-prey remains descriptive |
| Oracle use | Dense/Kalman oracles are post-run diagnostics only and are not read by tuning |
| Nonclaims | No unbiasedness, exact higher-moment projection, exact score, superiority, default/HMC/leaderboard promotion, or NAWM conclusion |

## Scope And Controls

The trial keeps OT controls fixed at the prior candidate values
`epsilon=2`, `sinkhorn_steps=8`, `balance_steps=8`, and `ridge=1e-5` while
tuning the higher-moment family. The bounded grid is:

```text
correction_steps: 0, 1, 2, 4
strength:         0.02, 0.05, 0.10, 0.20
floor:            1e-5
```

This is a feasibility trial, not a global search. It uses two calibration and
two selection trajectories per scope, with two independent particle clouds per
trajectory. Controls are selected on the selection partition, frozen, and
then evaluated on untouched claim seeds. A follow-up OT retuning stage is not
authorized by this plan.

## Objective And Diagnostics

For each row, define

\[
R_3 = d^{-1}\sum_j r_{3j}^2,\qquad
R_4 = d^{-1}\sum_j r_{4j}^2,
\]

where `r3` and `r4` are the standardized diagonal moment residuals emitted by
the executed finite program. The selection objective is `R3 + R4`, averaged
over rows. Ties use lower value/score replicate variance, then lower strength,
then fewer correction steps.

The trial records the maximum residual, mean residual objective, minimum row
mass, covariance-gap eigenvalue, score-increment sum residual, and value/score
replicate variance. The moment residual is explanatory and nominates a better
carried distribution; it is not evidence that the likelihood or score is more
accurate.

## Skeptical Audit Before Execution

1. The baseline must report its actual residual, not a synthetic zero. The
   zero-step diagnostic is repaired before tuning.
2. Selection data are not claim data. The fixed claim observation and claim
   particle seeds are not read until controls are frozen.
3. A lower residual can be caused by an aggressive cloud deformation. The
   secondary variance/displacement and numerical-health diagnostics therefore
   veto unsafe candidates; no candidate is promoted on residual alone.
4. The score remains the recursive total derivative of the same finite value
   program. No autodiff or finite differences enter runtime or selection.
5. This trial does not compare different model targets or alter canonical
   Contract E. Prior artifacts remain untouched and a new versioned output
   root is required.

Audit decision: `PASS_WITH_LIMITED_FEASIBILITY_SCOPE`.

## Execution And Artifacts

Implementation changes:

- actual zero-step moment residual diagnostics;
- separate retuning runner and result schema;
- unit test for baseline residual reporting.

Expected artifacts:

- `docs/benchmarks/artifacts/higher_moment_genut_retuning_20260723/attempt01/`;
- this plan;
- a terminal result note;
- a reboot reset memo containing controls, artifacts, verdict, and nonclaims.

The campaign is bounded to one GPU launch, FP32/TF32/XLA, `N>1000`, and the
existing 16 claim seeds. Stop on any hard veto or missing required artifact.

## Execution Record

- Focused CPU-hidden regression suite: `24 passed`.
- Attempt 01 stopped before scientific execution because of a wrapper import
  error; `attempt01/failure.json` preserves the classification and repair.
- Attempt 02 completed numerical work but failed identity finalization because
  the wrapper supplied an unsupported design-family label. It is not accepted
  as scientific evidence; `attempt02/failure.json` preserves the failure.
- Attempt 03 completed in `519.52` seconds with `hard_valid=true`, FP32, TF32,
  XLA, GPU memory growth, and the declared scope. Its artifact is
  `docs/benchmarks/artifacts/higher_moment_genut_retuning_20260723/attempt03/result.json`.

Post-run decision: the primary moment-residual diagnostic improved on every
claim scope, but all selections hit the strongest grid boundary. Likelihood and
score improvement remain unsupported, and LGSSM `T=50` paired per-seed value
absolute error regressed. The candidate is not promoted.
