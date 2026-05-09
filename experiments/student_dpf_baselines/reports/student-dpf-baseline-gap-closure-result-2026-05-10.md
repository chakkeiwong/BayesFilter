# Student DPF baseline gap-closure result

## Date

2026-05-10

## Scope

This report covers the first controlled comparison panel for the
quarantined student DPF baseline stream.  It is comparison evidence
only and does not promote student code into production.

## Panel

- fixtures: lgssm_1d_short, lgssm_cv_2d_short
- seeds: [0, 1, 2]
- particle counts: [128, 512]

## Reference Agreement

| Implementation | Runs | OK | Failed | Max log-likelihood error | Max mean RMSE | Max covariance RMSE | Median runtime seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE | 12 | 12 | 0 | 0 | 4.89777e-16 | 3.18852e-17 | 0.0561829 |
| advanced_particle_filter | 12 | 12 | 0 | 8.88178e-16 | 4.49725e-16 | 2.85161e-17 | 0.00362914 |

## Cross-Student Agreement

- comparable groups: 12
- comparison key: fixture_name, seed; advanced_particle_filter particle count retained as a separate run dimension because the MLCOE Kalman smoke adapter has no particle count
- max filtered-mean RMSE between students: 3.93503e-16
- max log-likelihood absolute difference between students: 8.88178e-16

## Interpretation

Both student snapshots can be called through quarantined adapters on
small linear-Gaussian fixtures.  For the Kalman path, both match the
independent NumPy Kalman reference to numerical precision.  This
supports using the harness for baseline comparison, but it does not
validate student implementations as production code.

The first panel intentionally excludes nonlinear, kernel PFF, HMC,
and differentiable-resampling workflows.  Those require separate
targeted reproduction gates.

## Next hypotheses

1. The two student implementations will remain reference-consistent
   on larger linear-Gaussian panels, but particle-filter ESS/runtime
   behavior will diverge as observation noise decreases.
2. Kernel PFF behavior in `advanced_particle_filter` is unstable or
   test-sensitive in the current environment, as indicated by the
   G1 partial failure; it should be reproduced in isolation before
   being compared to `2026MLCOE` flow filters.
3. Nonlinear fixtures will separate implementation assumptions more
   strongly than linear fixtures, especially around Jacobian shape,
   covariance regularization, and resampling thresholds.
