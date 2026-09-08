# Direct Rectangular and Block-QR SR-UKF Execution Result

Date: 2026-08-17  
Plan: `bayesfilter_direct_rectangular_block_qr_srukf_execution_plan_2026_08_17.md`  
Review: `bayesfilter_direct_rectangular_block_qr_srukf_plan_review_2026_08_17.md`  
Status: `IMPLEMENTED_WITH_EXPLICIT_RECTANGULAR_SCORE_BOUNDARY`

## Implemented

The default full-rank `direct_factor_srukf` recursion now uses direct QR at
both factorization points:

1. Prediction factors the propagated residual stack `A_x` directly.
2. The measurement update factors the joint stack

   ```text
   A = [[A_y, G_r],
        [A_x, 0 ]].
   ```

   If `A.T = Q R` and

   ```text
   R = [[R_yy, R_yx],
        [0,     R_xx]],
   G = R.T = [[L_y, 0], [L_xy, L_f]],
   ```

   then the implementation uses

   ```text
   S = L_y L_y.T
   P_xy = L_xy L_y.T
   K = L_xy L_y^{-1}
   G_filtered = L_f.
   ```

   The gain orientation is important: the right solve is with `L_y`, not
   `L_y.T`. The identity is verified against a dense reference in the block
   QR tests.

The runtime calls `batched_stack_qr_lower(...,
compute_covariance_diagnostics=False)`. In that mode it reports direct stack
and QR derivative residuals and does not form `A A.T`, a Schur covariance, or a
covariance derivative. Covariance reconstruction remains available only for
standalone reference diagnostics.

The old downdate keys remain as compatibility aliases in filter diagnostics;
their status is explicitly `deprecated_conditional_pivot_alias`. The
nonnegative DZ5 route does not call the lower-rank downdate kernel.

Rectangular and singular-support primitives are implemented independently in
`bayesfilter/linear/rectangular_factor_tf.py`:

- fixed-pivot rectangular QR and its fixed-chart derivative;
- direct-stack SVD rank/value diagnostics;
- affine-support Gaussian likelihood with pseudodeterminant and off-support
  `-inf`; and
- a value-only direct-support conditional update returning a rectangular
  conditional factor.

The value-only temporal adapter is `bayesfilter/nonlinear/rectangular_srukf_tf.py`.
It uses a fixed zero-padded factor width and direct-stack SVD/support
conditional updates. It is intentionally separate from the default score
route and reports `value_only_rank_discovery` branch metadata.

The SVD is applied to the stack, never to a materialized covariance. Repeated
singular values, cutoff crossings, rank changes, and support changes remain
value-only branches and are not admitted as analytical score branches.

The default backend metadata is now `direct_qr_block_conditional`. The route
guard includes the block kernel and continues to reject SVD/eigen/Cholesky and
covariance-to-factor routes in admitted source files.

## Test command

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_stack_qr_tf.py \
  tests/test_lower_rank_downdate_tf.py \
  tests/test_block_qr_conditional_tf.py \
  tests/test_rectangular_factor_tf.py \
  tests/test_rectangular_srukf_tf.py \
  tests/test_factor_srukf_tf.py \
  tests/test_factor_srukf_model_parity.py \
  tests/test_factor_srukf_route_guard.py \
  tests/test_srukf_backend_policy.py
```

Result:

```text
31 passed, 1 warning in 12.18s
```

The warning is the pre-existing HDF5 build/runtime mismatch from the local
TensorFlow environment. `python -m compileall` passes for all changed Python
modules, and `git diff --check` passes.

## Covered cases

- full-rank block reconstruction and dense gain/conditional Schur identity;
- block factor, gain, and conditional-factor centered finite differences;
- eager execution and the existing SR-UKF XLA parity test;
- batch behavior;
- relative QR-pivot floor and fail-closed behavior;
- exact rank-one and rank-zero direct-stack SVD behavior;
- repeated singular values and rank-cutoff branch changes;
- fixed-pivot rectangular QR reconstruction and derivative finite difference;
- rectangular chart residual failure;
- malformed/duplicate permutations;
- singular innovation on support and off support;
- NaN/Inf and invalid-rank/shape/column inputs; and
- route/default metadata and forbidden-route checks.

## Boundaries and residual work

`TFFactorSRUKFModel` still requires square factors, so the default score route
is the full-rank fixed-pivot QR route. The separate
`TFRectangularSRUKFModel` adapter supports value-only temporal singular
filtering with fixed zero-padded widths and direct-stack SVD/support updates.
It is not a score route: rank/support branch identity and singular-vector
smoothness are explicit blockers. No claim is made here for singular temporal
SR-UKF scores, HMC readiness, or broad model-suite scientific equivalence.

The historical principal-root/SVD sigma-point implementations remain callable
only by explicit historical selectors and are not fallback routes.
