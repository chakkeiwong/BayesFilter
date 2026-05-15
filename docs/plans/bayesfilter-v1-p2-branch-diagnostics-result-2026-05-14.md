# BayesFilter V1 P2 Branch Diagnostics Result

## Date

2026-05-14

## Scope

This result executes P2/R3 from:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-p2-branch-diagnostics-plan-2026-05-14.md
```

P2 remains inside the BayesFilter V1 lane.  It modifies only testing
diagnostics and tests so branch summaries expose the diagnostics required by
the master.  No production filter behavior, Hessian code, HMC code, GPU/XLA
code, MacroFinance code, DSGE code, Chapter 18b text, or structural-lane plan
was modified.

## Phase Plan Audit

Pass with one scoped tightening.

The existing branch-summary helper already counted ok rows, active floors,
weak spectral gaps, nonfinite rows, support residuals, deterministic
residuals, rank, and point count.  The P2 plan also required representative
failure labels and structural-null residuals.  This was a real diagnostic gap,
not a mathematical or product-decision gap, so it was closed inside
`bayesfilter/testing/nonlinear_diagnostics_tf.py`.

Added summary fields:
- `max_structural_null_covariance_residual`;
- `max_fixed_null_derivative_residual`;
- `max_structural_null_count`;
- `failure_labels`.

Focused tests were tightened to ensure active-floor, weak-gap, and structural
fixed-support diagnostics remain visible.

## Validation

Focused branch diagnostic tests:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `18 passed, 2 warnings`.

The warnings are TensorFlow Probability deprecation warnings from
`distutils.version`, not BayesFilter failures.

## Branch Grid

The P2 CPU score branch grid used:

Model B:

```text
(rho, sigma, beta) in
(0.62, 0.20, 0.70),
(0.66, 0.23, 0.75),
(0.70, 0.25, 0.80),
(0.74, 0.27, 0.85),
(0.78, 0.30, 0.90)
```

Model C smooth-phase control and default structural fixed-support:

```text
(process_sigma, observation_sigma, initial_variance) in
(0.85, 0.90, 0.16),
(0.95, 1.00, 0.18),
(1.00, 1.02, 0.20),
(1.05, 1.08, 0.22),
(1.15, 1.15, 0.25)
```

Default Model C was run only with `allow_fixed_null_support=True`.

The branch grid was intentionally CPU-only with `CUDA_VISIBLE_DEVICES=-1`.
TensorFlow still printed CUDA plugin initialization/cuInit messages during
startup in this environment; those messages are sandbox/framework startup
noise and are not GPU benchmark evidence.

## Summary Rows

| Model | Backend | Fixed support | OK | Active floor | Weak gap | Nonfinite | Other blocked | Failure labels | Max structural null count | Max structural null covariance residual | Max fixed-null derivative residual | Min placement gap | Max point count |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Model B | SVD cubature | no | 5/5 | 0 | 0 | 0 | 0 | none | 0 | 0.0 | 0.0 | 3.317e-03 | 6 |
| Model B | SVD-UKF | no | 5/5 | 0 | 0 | 0 | 0 | none | 0 | 0.0 | 0.0 | 3.317e-03 | 7 |
| Model B | SVD-CUT4 | no | 5/5 | 0 | 0 | 0 | 0 | none | 0 | 0.0 | 0.0 | 4.408e-03 | 14 |
| Model C smooth phase | SVD cubature | no | 5/5 | 0 | 0 | 0 | 0 | none | 0 | 0.0 | 0.0 | 1.100e-01 | 6 |
| Model C smooth phase | SVD-UKF | no | 5/5 | 0 | 0 | 0 | 0 | none | 0 | 0.0 | 0.0 | 1.100e-01 | 7 |
| Model C smooth phase | SVD-CUT4 | no | 5/5 | 0 | 0 | 0 | 0 | none | 0 | 0.0 | 0.0 | 1.100e-01 | 14 |
| Model C default | SVD cubature | yes | 5/5 | 0 | 0 | 0 | 0 | none | 1 | 4.930e-32 | 1.420e-31 | 1.600e-01 | 6 |
| Model C default | SVD-UKF | yes | 5/5 | 0 | 0 | 0 | 0 | none | 1 | 1.062e-31 | 2.272e-16 | 1.600e-01 | 7 |
| Model C default | SVD-CUT4 | yes | 5/5 | 0 | 0 | 0 | 0 | none | 1 | 0.0 | 0.0 | 1.600e-01 | 14 |

All rows had:
- max support residual `0.0`;
- max deterministic residual `0.0`;
- no active floors;
- no weak spectral gaps;
- no nonfinite scores.

## Interpretation

H-P2.1 is supported.  Model B has a practical score branch box where SVD
cubature, SVD-UKF, and SVD-CUT4 all return finite values/scores with no active
floors or weak active spectral gaps.

H-P2.2 is supported on this tested scope.  Default Model C has a passing
structural fixed-support branch box for all three backends.  It remains
important that default Model C be run only with
`allow_fixed_null_support=True`; the collapsed smooth branch is still a
blocker from P1.

H-P2.3 is supported.  Branch summaries now carry explicit failure labels, and
the focused tests check active-floor and weak-gap labels directly.

## Gate Decision

P2 passes.

Primary gate:
- at least one practical Model B box passed for all selected backends;
- default Model C has a passing structural branch box;
- failure labels are visible and not converted into success.

Continuation:
- P3 is justified.  The next phase should refresh nonlinear benchmark
  artifacts with branch metadata.  Model B is the first smooth benchmark/HMC
  candidate.  Default Model C can be benchmarked at the structural
  fixed-support scope, but HMC remains gated until P4.
