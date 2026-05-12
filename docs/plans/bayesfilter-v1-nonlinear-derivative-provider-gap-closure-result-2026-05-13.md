# BayesFilter V1 Nonlinear Derivative-Provider Gap Closure Result

## Date

2026-05-13

## Scope

This result closes the derivative-provider gap pass for the BayesFilter V1
nonlinear filtering lane.  It stays out of MacroFinance, DSGE, Chapter 18b,
structural SVD/SGU plans, and the shared monograph reset memo.

## Plan And Audit

Plan:

```text
docs/plans/bayesfilter-v1-nonlinear-derivative-provider-gap-closure-plan-2026-05-13.md
```

Audit:

```text
docs/plans/bayesfilter-v1-nonlinear-derivative-provider-gap-closure-plan-audit-2026-05-13.md
```

The audit approved scoped execution with one key tightening: default Model C
has a deterministic phase coordinate with zero variance, so the current smooth
simple-spectrum/no-active-floor SVD score branch must not silently claim score
readiness for that default law.

## Implementation

Added testing-lane derivative providers:

- `make_nonlinear_accumulation_first_derivatives_tf`;
- `make_univariate_nonlinear_growth_first_derivatives_tf`.

Updated testing fixtures:

- Model B and Model C parameter arguments now accept TensorFlow scalar tensors
  as well as Python scalars;
- Model C accepts optional `initial_phase_variance`, defaulting to zero so the
  existing value benchmark is unchanged;
- derivative providers are exported from `bayesfilter.testing`.

Updated tests:

- Model B analytic scores are checked against centered finite differences of
  the implemented value filters for SVD cubature, SVD-UKF, and SVD-CUT4;
- Model C analytic scores are checked against centered finite differences on a
  nondegenerate phase-state testing variant for the same three backends;
- default zero-phase-variance Model C is explicitly tested as an active-floor
  blocker for the current smooth SVD score branch;
- score branch summaries now cover affine, Model B, and smooth-phase Model C,
  while separately counting default Model C as blocked.

## Mathematical Interpretation

H-D1 supported:
Model B received an explicit derivative provider for
\(\theta=(\rho,\sigma,\beta)\) without changing the production structural
callback contract.

H-D2 supported:
Model B analytic scores match finite differences of the implemented SVD
cubature, SVD-UKF, and SVD-CUT4 likelihoods on the selected smooth branch.

H-D3 partially supported:
Model C has an explicit derivative provider and passes score parity on a
nondegenerate phase-state testing variant.

H-D4 supported:
The default Model C value fixture with zero phase variance remains blocked by
the active-floor gate under the current smooth score branch.  This preserves
the distinction between the deterministic model law and numerical
regularization.

H-D5 supported:
Hessian, GPU/XLA, and nonlinear HMC work remain gated.  They are not justified
until score-ready parameter boxes and any needed fixed-null derivative branch
are selected and tested.

## Validation

Baseline before implementation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  -p no:cacheprovider
```

Result:
- `22 passed, 2 warnings`.

Focused score validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  -p no:cacheprovider
```

Result:
- `13 passed, 2 warnings`.

Focused branch validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `12 passed, 2 warnings`.

Final validation is recorded in the lane reset memo after the full hygiene pass.

Focused nonlinear/V1 validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `62 passed, 2 skipped, 2 warnings`.

Full default CPU validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Result:
- `193 passed, 5 skipped, 2 warnings`.

Hygiene:
- `py_compile` passed for touched Python modules/tests;
- `git diff --check` passed;
- `docs/source_map.yml` parsed with `yaml.safe_load`;
- scan for NumPy imports in production nonlinear code and the touched
  nonlinear testing helpers found no matches.

## Remaining Gaps

G-D1:
Default Model C still needs a fixed-null-direction SVD derivative policy before
it can become score-ready without adding artificial phase variance.

G-D2:
The score-ready nonlinear parameter boxes are still small.  Wider branch
diagnostics are needed before nonlinear HMC planning.

G-D3:
Hessian implementation remains deferred.

G-D4:
GPU/XLA point-axis scaling remains deferred and should be run only after score
branch diagnostics define stable target boxes.

G-D5:
Models B-C still do not have exact full nonlinear likelihood references; dense
one-step projections remain approximation diagnostics.

## Suggested Next Phase

The next phase should design and test the fixed-null-direction derivative
policy for structurally deterministic coordinates.  The main hypothesis is
that if the covariance null space is declared by model metadata and remains
fixed, derivative propagation can avoid differentiating arbitrary null-space
eigenvectors while preserving the implemented support law.
