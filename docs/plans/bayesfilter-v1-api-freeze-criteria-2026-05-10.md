# BayesFilter v1 API Freeze Criteria

## Date

2026-05-10

## Purpose

This note defines the public v1 filtering API surface that external projects
may certify against before any client switch-over.  It does not freeze every
internal helper.  It gives MacroFinance and DSGE a stable target without making
either project depend on BayesFilter yet.

## Stable v1 API Candidates

### Linear Gaussian Value Filters

Stable candidates:

```text
TFLinearGaussianStateSpace
tf_linear_gaussian_log_likelihood
tf_kalman_log_likelihood
tf_masked_kalman_log_likelihood
```

QR/square-root value candidates:

```text
tf_qr_linear_gaussian_log_likelihood
tf_qr_sqrt_kalman_log_likelihood
tf_qr_sqrt_masked_kalman_log_likelihood
```

Required contracts:

- `tf.float64` support;
- dense and two-dimensional observation conventions;
- explicit mask convention for masked paths;
- no production NumPy dependency;
- stable `TFFilterValueResult` metadata and diagnostics.

### Linear QR Score/Hessian Filters

Stable candidates:

```text
TFLinearGaussianStateSpaceDerivatives
tf_qr_linear_gaussian_score_hessian
tf_qr_sqrt_kalman_score_hessian
tf_qr_sqrt_masked_kalman_score_hessian
```

Required contracts:

- parameter-major derivative tensors;
- first- and second-order derivative fields;
- Hessian symmetry within documented tolerance;
- no HMC readiness claim from derivative existence alone.

### Linear SVD/eigen Value Filters

Stable value-only candidates:

```text
tf_svd_linear_gaussian_log_likelihood
tf_svd_kalman_log_likelihood
tf_svd_masked_kalman_log_likelihood
```

Required contracts:

- value-only differentiability status;
- implemented-law diagnostics for floors and PSD projection residuals;
- no SVD/eigen derivative export in v1 unless a later need assessment passes.

### Structural TensorFlow Protocols

Stable candidates:

```text
TFStructuralStateSpace
structural_block_metadata
structural_filter_diagnostics
structural_filter_metadata
pointwise_deterministic_residuals
affine_structural_to_linear_gaussian_tf
make_affine_structural_tf
```

Required contracts:

- explicit stochastic/endogenous/deterministic-completion block metadata;
- deterministic-completion residual diagnostics;
- no DSGE economics in BayesFilter production modules.

### Nonlinear Sigma-point And CUT Value Filters

Stable value candidates:

```text
tf_svd_sigma_point_log_likelihood
tf_svd_sigma_point_log_likelihood_with_rule
tf_svd_sigma_point_filter
tf_svd_sigma_point_placement
tf_unit_sigma_point_rule
tf_cut4g_sigma_point_rule
tf_svd_cut4_log_likelihood
tf_svd_cut4_filter
```

Required contracts:

- declared point-count metadata;
- rank/support diagnostics;
- deterministic residual checks for structural models;
- no GPU performance claim from CPU graph parity alone.

### Smooth-branch SVD-CUT Derivatives

Stable candidate only under branch diagnostics:

```text
tf_svd_cut4_score_hessian
```

Required contracts:

- derivative target is the implemented regularized law;
- active floors and weak spectral gaps fail closed;
- HMC readiness requires target-specific branch-frequency and sampler gates.

## Internal Or Testing-only Surface

Testing/reference helpers remain outside the v1 production API:

```text
bayesfilter.testing.*
solve_kalman_score_hessian
tf_solve_differentiated_kalman_reference
tf_covariance_differentiated_kalman_reference
```

Factor helper APIs are useful but should be treated as internal unless a later
API review promotes them:

```text
bayesfilter.linear.qr_factor_tf.*
bayesfilter.linear.svd_factor_tf.*
```

## Freeze Gates

A symbol can be marked v1-stable only if:

1. it has a BayesFilter-local test;
2. it has shape/dtype behavior covered by tests;
3. diagnostics identify regularization, mask convention, or approximation law;
4. production code does not import NumPy;
5. it does not import MacroFinance or DSGE;
6. claims are limited to tested behavior.

## Explicit Non-goals For v1 Freeze

- no MacroFinance production switch-over;
- no DSGE production switch-over;
- no SGU production filtering claim;
- no GPU/XLA performance claim without escalated benchmark artifacts;
- no HMC readiness claim without exact target-model evidence;
- no linear SVD/eigen derivative API without a proven client need.
