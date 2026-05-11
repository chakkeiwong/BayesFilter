# BayesFilter V1 SVD Filter Analytic Gradient Audit Result

## Date

2026-05-12

## Scope

Executed the second nonlinear-filtering subplan in the BayesFilter V1 lane:

```text
docs/plans/bayesfilter-v1-svd-filter-analytic-gradient-audit-implementation-plan-2026-05-12.md
```

Protected files remained out of scope: Chapter 18b, structural reset memos,
the shared monograph reset memo, MacroFinance, and DSGE.

## Phase G0: Lane Recovery And Baseline

Baseline command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  -p no:cacheprovider
```

Result: `20 passed, 2 warnings`.

Interpretation: the existing value filters and old SVD-CUT derivative-oracle
tests were clean before implementation.  The warnings were TensorFlow
Probability `distutils` deprecation warnings.

## Phase G1: Derivation-To-Code Audit

MathDevMCP label lookup confirmed the relevant Chapter 18 labels in
`docs/chapters/ch18_svd_sigma_point.tex`:

| Label | Code target | Shape convention |
| --- | --- | --- |
| `eq:bf-svd-sp-point-first`, `eq:bf-svd-cut-point-first` | `_checked_smooth_eigh_factor_first_derivatives`, point placement in `svd_sigma_point_derivatives_tf.py` | parameters `p`, augmented dimension `d=n+q`, points `J`: `d_factor[p,d,d]`, points `J,d`, point derivatives `p,J,d` |
| `eq:bf-svd-sp-map-first`, `eq:bf-svd-cut-map-first` | `TFStructuralFirstDerivatives` map derivative callables | transition state Jacobian `J,n,n`, innovation Jacobian `J,n,q`, partial parameter derivative `p,J,n`; observation Jacobian `J,m,n`, partial parameter derivative `p,J,m` |
| `eq:bf-svd-sp-cov-first`, `eq:bf-svd-cut-cov-first` | `_weighted_covariance_first_derivatives` | centered points `J,k`, centered derivatives `p,J,k`, covariance derivatives `p,k,k` |
| `eq:bf-svd-sp-score`, `eq:bf-svd-cut-score` | solve-form likelihood score inside `_smooth_sigma_point_score_with_rule` | innovation `m`, covariance `m,m`, covariance derivative `p,m,m`, score `p` |
| `eq:bf-svd-factor-reconstruction-first` | factor derivative residual diagnostic | reconstructs the implemented covariance branch, with residual reported as `factor_derivative_reconstruction_residual` |

Audit finding: the previous production contract could not honestly implement
analytic scores for arbitrary callback models because `TFStructuralStateSpace`
stores only transition and observation functions, not their parameter and state
derivatives.  The plan was tightened to require an explicit
`TFStructuralFirstDerivatives` provider and to forbid hidden production
`GradientTape` use.

## Phase G2: Move Raw Tape Derivatives To Testing Oracle

Moved the old raw `GradientTape` SVD-CUT4 score/Hessian path to:

```text
bayesfilter/testing/tf_svd_cut_autodiff_oracle.py
```

The testing oracle keeps branch checks for active floors, weak spectral gaps,
and nonfinite values.  Production top-level and `bayesfilter.nonlinear`
exports no longer advertise `tf_svd_cut4_score_hessian`.

Validation command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result: `10 passed, 2 skipped, 2 warnings`.

## Phase G3-G5: Analytic Score Core, Integration, And Tests

Added:

```text
bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py
tests/test_nonlinear_sigma_point_scores_tf.py
```

Production score functions:

```text
tf_svd_cubature_score
tf_svd_ukf_score
tf_svd_cut4_score
tf_svd_sigma_point_score_with_rule
```

The functions return `TFFilterDerivativeResult` with `hessian=None` and
metadata status:

```text
analytic_score_smooth_branch_hessian_deferred
```

Implementation status:
- TF-only production path;
- explicit first-order structural derivative provider;
- branch-local eigenderivative formulas for simple spectra and inactive hard
  floors;
- solve-form likelihood score for the implemented covariance branch;
- factor derivative reconstruction residual in diagnostics;
- SVD cubature, SVD-UKF, and SVD-CUT4 share one score core and differ only by
  their fixed sigma-point rule.

Focused score validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  -p no:cacheprovider
```

Result: `6 passed, 2 warnings`.

Checks covered:
- finite-difference score parity for SVD cubature and SVD-UKF;
- finite-difference and testing-oracle score parity for SVD-CUT4;
- active-floor and weak-gap blockers;
- eager/graph parity for the cubature score;
- Hessian explicitly absent.

## Phase G6: Hessian Gate

Hessian remains deferred.  The current score implementation deliberately
requires only first derivatives of the model law.  A Hessian implementation
would require:

- second derivatives of initial moments and covariances;
- second derivatives of transition and observation maps;
- second derivatives of the eigensystem/factor branch;
- memory tests for `O(J p^2 n)` and `O(J p^2 m)` point derivative storage;
- a named downstream consumer such as Newton/Laplace or Riemannian HMC.

For ordinary HMC, the score is the required derivative object, so the Hessian
does not block the next nonlinear-filtering phase.

## Hypotheses

H-G1 supported: Chapter 18 score equations map cleanly to a shared fixed-rule
derivative core once the structural derivative provider is explicit.

H-G2 supported: raw autodiff remains useful as a testing oracle and no longer
needs to be a production export.

H-G3 provisionally supported on the smooth affine fixture: analytic scores pass
finite-difference and oracle checks.  The hypothesis still needs testing on
nonlinear Models B-C after analytic derivative providers for those fixtures are
added.

H-G4 supported: Hessian deferral is justified until score stability, nonlinear
derivative providers, and memory costs are measured.

## Remaining Gaps

1. Add analytic derivative providers for nonlinear Models B-C.
2. Add score tests on Models B-C against finite differences of the same
   implemented SVD sigma-point likelihood.
3. Add branch-frequency score diagnostics over parameter grids.
4. Add extended CPU and optional escalated GPU/XLA score benchmarks only after
   Models B-C score tests are stable.

## Final Validation

Focused nonlinear/V1 suite:

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
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result: `43 passed, 2 skipped, 2 warnings`.

Other checks:

```text
python -m py_compile ... : passed
git diff --check: passed
docs/source_map.yml YAML parse: passed
production NumPy scan in bayesfilter/nonlinear: no NumPy imports
```

Post-audit adjustment:
- `bayesfilter/nonlinear/svd_cut_derivatives_tf.py` is retained only as a
  migration guard that raises a clear error;
- the raw `GradientTape` implementation itself lives only in
  `bayesfilter.testing.tf_svd_cut_autodiff_oracle`.
