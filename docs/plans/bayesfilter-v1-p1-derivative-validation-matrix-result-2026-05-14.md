# BayesFilter V1 P1 Derivative-Validation Matrix Result

## Date

2026-05-14

## Scope

This result executes P1/R2 from:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-p1-derivative-validation-matrix-plan-2026-05-14.md
```

P1 is an evidence-consolidation phase.  No production code, nonlinear Hessian
code, GPU/XLA code, HMC code, MacroFinance code, DSGE code, Chapter 18b text,
or structural-lane plan was modified.

## Phase Plan Audit

Pass.

The P1 plan stays inside the V1 lane and asks for a derivative-status matrix,
not new mathematics.  The only tightening applied during execution is a label
choice: Model A value evidence refers to the fixed affine Gaussian structural
oracle, while Model A score evidence refers to the parameterized smooth affine
structural score fixture used in the analytic score tests.  Both are affine
Gaussian structural laws, but the distinction avoids overclaiming a score for
the fixed no-parameter oracle object.

No veto diagnostic fired:
- default Model C score certification is tied to
  `allow_fixed_null_support=True`;
- nonlinear Hessians remain deferred;
- testing-only autodiff is not promoted to production;
- no production NumPy dependency was added;
- no out-of-lane file was touched.

## Validation Run

Focused nonlinear value, score, and branch diagnostics:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `41 passed, 2 warnings`.

The warnings are TensorFlow Probability deprecation warnings from
`distutils.version`, not BayesFilter failures.

## Derivative-Validation Matrix

Status vocabulary:
- `certified`: covered by the focused tests at the stated scope;
- `blocked`: intentionally blocked by a branch condition;
- `deferred`: deliberately not implemented in V1 at this phase;
- `diagnostic`: useful evidence that is not a public readiness claim.

| Model row | Backend | Value status | Score status | Score branch | Derivative provider | Reference target | Branch/null diagnostics | Compiled/eager | Hessian status | Public claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model A fixed affine Gaussian oracle | SVD cubature | certified by exact Kalman parity in `tests/test_nonlinear_sigma_point_values_tf.py` | certified on parameterized affine score fixture | smooth simple spectrum, no active floor | `smooth_affine_test_derivatives` in `tests/test_nonlinear_sigma_point_scores_tf.py` | centered finite difference of implemented cubature value filter | deterministic and support residuals zero in value test; smooth score branch diagnostics present | graph parity certified for cubature score | `deferred`; result Hessian is `None` | certified for affine value and affine score fixture |
| Model A fixed affine Gaussian oracle | SVD-UKF | certified by exact Kalman parity in `tests/test_nonlinear_sigma_point_values_tf.py` | certified on parameterized affine score fixture | smooth simple spectrum, no active floor | `smooth_affine_test_derivatives` in `tests/test_nonlinear_sigma_point_scores_tf.py` | centered finite difference of implemented UKF value filter | deterministic and support residuals zero in value test; smooth score branch diagnostics present | not separately certified in P1 | `deferred`; result Hessian is `None` | certified for affine value and affine score fixture |
| Model A fixed affine Gaussian oracle | SVD-CUT4 | certified by exact Kalman parity in `tests/test_nonlinear_sigma_point_values_tf.py` | certified on parameterized affine score fixture | smooth simple spectrum, no active floor | `smooth_affine_test_derivatives` in `tests/test_nonlinear_sigma_point_scores_tf.py` | centered finite difference of implemented CUT4 value filter and testing-only autodiff oracle score | deterministic and support residuals zero in value test; active-floor and weak-gap blockers tested | not separately certified in P1 | production Hessian `deferred`; testing-only oracle Hessian exists in `bayesfilter.testing.tf_svd_cut_autodiff_oracle` | certified score; oracle remains testing-only |
| Model B nonlinear accumulation | SVD cubature | certified finite value and diagnostics in `tests/test_nonlinear_sigma_point_values_tf.py` | certified | smooth simple spectrum, no active floor | `make_nonlinear_accumulation_first_derivatives_tf` | centered finite difference of implemented cubature value filter | deterministic residual zero; branch summary covers small smooth box | not separately certified in P1 | `deferred`; result Hessian is `None` | certified at local finite-difference scope |
| Model B nonlinear accumulation | SVD-UKF | certified finite value and diagnostics in `tests/test_nonlinear_sigma_point_values_tf.py` | certified | smooth simple spectrum, no active floor | `make_nonlinear_accumulation_first_derivatives_tf` | centered finite difference of implemented UKF value filter | deterministic residual zero; branch summary covers small smooth box | not separately certified in P1 | `deferred`; result Hessian is `None` | certified at local finite-difference scope |
| Model B nonlinear accumulation | SVD-CUT4 | certified finite value and diagnostics in `tests/test_nonlinear_sigma_point_values_tf.py` | certified | smooth simple spectrum, no active floor | `make_nonlinear_accumulation_first_derivatives_tf` | centered finite difference of implemented CUT4 value filter | deterministic residual zero; branch summary covers small smooth box | not separately certified in P1 | `deferred`; result Hessian is `None` | certified at local finite-difference scope |
| Model C smooth-phase control | SVD cubature | certified finite value and diagnostics for Model C; smooth-phase score fixture uses positive phase variance | certified on smooth-phase control | smooth simple spectrum, no active floor | `make_univariate_nonlinear_growth_first_derivatives_tf` | centered finite difference of implemented cubature value filter with positive phase variance | deterministic residual zero; branch summary covers small smooth-phase box | not separately certified in P1 | `deferred`; result Hessian is `None` | certified only for smooth-phase control, not default law |
| Model C smooth-phase control | SVD-UKF | certified finite value and diagnostics for Model C; smooth-phase score fixture uses positive phase variance | certified on smooth-phase control | smooth simple spectrum, no active floor | `make_univariate_nonlinear_growth_first_derivatives_tf` | centered finite difference of implemented UKF value filter with positive phase variance | deterministic residual zero; branch summary covers small smooth-phase box | not separately certified in P1 | `deferred`; result Hessian is `None` | certified only for smooth-phase control, not default law |
| Model C smooth-phase control | SVD-CUT4 | certified finite value and diagnostics for Model C; smooth-phase score fixture uses positive phase variance | certified on smooth-phase control | smooth simple spectrum, no active floor | `make_univariate_nonlinear_growth_first_derivatives_tf` | centered finite difference of implemented CUT4 value filter with positive phase variance | deterministic residual zero; branch summary covers small smooth-phase box | not separately certified in P1 | `deferred`; result Hessian is `None` | certified only for smooth-phase control, not default law |
| Model C default zero phase variance | SVD cubature | certified finite value and diagnostics in `tests/test_nonlinear_sigma_point_values_tf.py` | certified only with `allow_fixed_null_support=True`; collapsed smooth branch remains blocked | structural fixed support, no active floor | `make_univariate_nonlinear_growth_first_derivatives_tf` | centered finite difference of implemented default structural value filter | structural null count 1; structural-null covariance residual zero; fixed-null derivative residual zero; deterministic residual zero | not separately certified in P1 | `deferred`; result Hessian is `None` | certified at local structural fixed-support score scope |
| Model C default zero phase variance | SVD-UKF | certified finite value and diagnostics in `tests/test_nonlinear_sigma_point_values_tf.py` | certified only with `allow_fixed_null_support=True`; collapsed smooth branch remains blocked | structural fixed support, no active floor | `make_univariate_nonlinear_growth_first_derivatives_tf` | centered finite difference of implemented default structural value filter | structural null count 1; structural-null covariance residual zero; fixed-null derivative residual zero; deterministic residual zero | not separately certified in P1 | `deferred`; result Hessian is `None` | certified at local structural fixed-support score scope |
| Model C default zero phase variance | SVD-CUT4 | certified finite value and diagnostics in `tests/test_nonlinear_sigma_point_values_tf.py` | certified only with `allow_fixed_null_support=True`; collapsed smooth branch remains blocked | structural fixed support, no active floor | `make_univariate_nonlinear_growth_first_derivatives_tf` | centered finite difference of implemented default structural value filter | structural null count 1; structural-null covariance residual zero; fixed-null derivative residual zero; deterministic residual zero; positive placement-floor blocker tested | not separately certified in P1 | `deferred`; result Hessian is `None` | certified at local structural fixed-support score scope |

## Interpretation

H-P1.1 is supported.  Every Model A-C/backend cell has value evidence:
Model A has exact affine Kalman parity, and Models B-C have finite implemented
filter values plus deterministic/support diagnostics.

H-P1.2 is supported.  Every Model A-C/backend cell has either score
certification or an explicit branch statement.  Default Model C is score-ready
only through the Chapter 18 structural fixed-support branch; the old collapsed
smooth branch remains correctly blocked by the active-floor gate.

H-P1.3 is supported.  Hessian status is explicit for every nonlinear
sigma-point row.  Production nonlinear Hessians remain deferred.  The only
nonlinear Hessian-like object is the SVD-CUT4 testing oracle in
`bayesfilter.testing`, and it is not a production API claim.

H-P1.4 is supported.  Default Model C certification requires
`allow_fixed_null_support=True` and reports structural null diagnostics.

## Gate Decision

P1 passes.

Primary gate:
- no matrix cell has unknown value or score status;
- every certified score cell cites finite-difference, exact affine, or
  documented oracle evidence;
- every Hessian cell has explicit non-promotional status;
- default Model C structural rows use the structural fixed-support branch and
  report null diagnostics.

Continuation:
- P2 is justified.  The next phase should run wider nonlinear score branch
  diagnostics, using Model B as the main smooth score-ready target and default
  Model C through `allow_fixed_null_support=True`.
