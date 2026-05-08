# BayesFilter Cross-Repo Filter Consolidation Map

Date: 2026-05-08

## Purpose

Consolidate filtering code from BayesFilter, MacroFinance, and the DSGE codebase
before additional implementation.  BayesFilter should become the shared
TensorFlow/TensorFlow Probability filtering library.  MacroFinance and DSGE
should keep model construction, economics, posterior wrappers, and
application-specific diagnostics.

Production BayesFilter filtering modules should use TensorFlow/TFP only.  NumPy
code currently committed in BayesFilter is legacy/reference until TF
replacements and compatibility gates allow removal.

## Donor Ownership By Filter Family

| Filter family | Primary donor | BayesFilter target | Notes |
|---|---|---|---|
| Dense linear Kalman value | MacroFinance `inference/hmc.py::tf_kalman_log_likelihood` | `bayesfilter/linear/kalman_tf.py` | Port dense Cholesky TF recursion; remove MacroFinance posterior concerns. |
| Backend dispatch | MacroFinance `inference/hmc.py::tf_lgssm_log_likelihood_backend` | `bayesfilter/linear/kalman_tf.py` | Dispatch `tf_cholesky`, `tf_masked_cholesky`, later `tf_svd`, `tf_qr`. |
| Static masked linear Kalman | MacroFinance `filters/tf_masked_kalman.py` | `bayesfilter/linear/kalman_tf.py` | Preserve dummy-row static-shape convention; all-missing row contributes zero likelihood. |
| Linear SVD value | MacroFinance `filters/tf_svd_kalman.py` | `bayesfilter/linear/kalman_svd_tf.py` | Use TF eig/SVD default; report implemented/floored covariance. |
| Linear analytic score/Hessian | MacroFinance `filters/tf_qr_sqrt_differentiated_kalman.py` | `bayesfilter/linear/kalman_qr_tf.py` plus derivative helper modules | QR/square-root is the production derivative route.  MacroFinance covariance-form and solve-form TF derivatives remain `bayesfilter.testing` references for parity and debugging. |
| Linear derivative references | MacroFinance `filters/tf_differentiated_kalman.py`, `filters/tf_solve_differentiated_kalman.py` | `bayesfilter/testing/*_differentiated_kalman_reference.py` | Already ported as non-production references; keep eager trace conversion here only. |
| Cubature/UKF quadrature | DSGE `src/dsge_hmc/filters/_quadrature.py` | `bayesfilter/nonlinear/quadrature_tf.py` | Port `CubatureRule` and `UnscentedRule`; add CUT4-G later. |
| SVD sigma-point primitives | DSGE `src/dsge_hmc/filters/_svd_core.py` | `bayesfilter/nonlinear/svd_factor_tf.py` | Add eigensolver interface; default `tf.linalg.eigh`, optional robust plugin. |
| Generic nonlinear SSM SVD filter | DSGE `src/dsge_hmc/filters/_svd_filters.py` | `bayesfilter/nonlinear/sigma_point_tf.py` | Port generic `ssm=` path, not DSGE economics. |
| Structural DSGE partition gates | DSGE `src/dsge_hmc/models/structural_metadata.py` | `bayesfilter/adapters/dsge.py`, `bayesfilter/nonlinear/protocols_tf.py` | Keep deterministic completion and mixed full-state blockers. |
| CUT4-G / SVD-CUT | DSGE `src/dsge_hmc/filters/CUTSRUKF.py` plus monograph derivations | `bayesfilter/nonlinear/svd_cut_tf.py` | Experimental donor only; implement moment tests before production claims. |
| HMC/performance gates | MacroFinance tests and DSGE extended SVD tests | `tests/` and `docs/benchmarks/` | Run after local value/gradient gates; GPU probes require escalation. |

## BayesFilter Legacy Surfaces

These modules remain legacy/reference while TF replacements are added:

- `bayesfilter/filters/kalman.py`
- `bayesfilter/filters/sigma_points.py`
- `bayesfilter/filters/particles.py`
- `bayesfilter/linear/types.py`
- `bayesfilter/linear/kalman_derivatives_numpy.py`
- `bayesfilter/results.py`
- `bayesfilter/backends.py`

No new production code should extend these NumPy paths.  New TF production
front doors should live in `*_tf.py` modules and pass source checks forbidding
NumPy imports and `.numpy()` calls.

## Donor Conflicts And Resolutions

| Conflict | Resolution |
|---|---|
| MacroFinance TF modules import NumPy for constants and eager trace conversion. | Replace with `math` or TF constants in BayesFilter.  Put eager trace serialization in explicit debug helpers only. |
| DSGE SVD core imports custom `robust_eigh` unconditionally. | Define a BayesFilter eigensolver protocol.  Default to `tf.linalg.eigh`; robust-eigh is optional plugin metadata. |
| MacroFinance and DSGE both have SVD value ideas. | MacroFinance owns linear SVD value; DSGE owns nonlinear sigma-point SVD primitives. |
| DSGE `_svd_filters.py` includes client debug capture hooks. | Keep failure-code/stat diagnostics; do not port pyfunc capture hooks into production paths. |
| CUTSRUKF is experimental/deprecated in DSGE tree. | Use only as CUT4-G and square-root pattern donor.  Require moment and shape gates before value filter integration. |
| Existing BayesFilter public imports expose NumPy. | Keep compatibility until TF replacements pass, then deprecate/remove public NumPy exports in a final phase. |

## Test Transplant Map

| Gate | Donor tests | BayesFilter target tests |
|---|---|---|
| Dense TF linear value | MacroFinance `tests/test_tf_kalman.py` | `tests/test_linear_kalman_tf.py` |
| Static mask convention | MacroFinance `tests/test_tf_masked_kalman.py` | `tests/test_linear_kalman_tf.py` |
| Linear SVD value | MacroFinance `tests/test_tf_svd_kalman.py` | `tests/test_linear_kalman_svd_tf.py` |
| Linear analytic derivatives | MacroFinance `tests/test_tf_qr_sqrt_differentiated_kalman.py`, `tests/test_filter_backend_parity.py`, `tests/test_one_country_analytic_backend_parity.py` | `tests/test_linear_kalman_qr_derivatives_tf.py` |
| QR derivative identities | MacroFinance `tests/test_tf_qr_derivative_identities.py` | `tests/test_linear_qr_factor_tf.py` |
| Generic SVD LGSSM | DSGE `tests/contracts/test_svd_lgssm_reference.py` | `tests/test_nonlinear_svd_sigma_tf.py` |
| Generic nonlinear SSM | DSGE `tests/contracts/test_svd_nonlinear_ssm_reference.py`, `test_svd_generic_nonlinear_ssm.py` | `tests/test_nonlinear_svd_sigma_tf.py` |
| XLA static shape | DSGE `tests/contracts/test_svd_generic_ssm_xla.py`, MacroFinance masked XLA smokes | `tests/test_xla_static_shape_tf.py` |
| DSGE structural partition | DSGE `tests/contracts/test_structural_dsge_partition.py` | `tests/test_dsge_adapter_gate.py` plus TF structural tests |
| CUT4-G moments | DSGE `tests/CUTSRUKF_test.py` | `tests/test_cut4g_quadrature_tf.py` |

Test-side NumPy is allowed temporarily for fixtures and assertions.  Production
BayesFilter TF modules must not import NumPy or call `.numpy()`.

## Execution Order

1. Add TF result, diagnostic, and tensor contract modules.
2. Port dense and masked TF linear value backend from MacroFinance.
3. Add TF source hygiene checks.
4. Port QR/square-root linear factor identities, value/masked value, and analytic derivatives.
5. Port linear SVD value backend.
6. Port quadrature and generic SVD sigma-point primitives from DSGE.
7. Port structural DSGE partition and deterministic completion gates.
8. Implement CUT4-G value gates.
9. Add derivative/Hessian work for SVD-CUT in subphases.
10. Migrate MacroFinance and DSGE client imports in separate cross-repo phases.

## Immediate Boundary

The current BayesFilter-only automatic execution is justified through:

1. consolidation map and baseline gates;
2. TF result/diagnostic/tensor contracts;
3. dense and masked TF linear value backend.

The next route is now explicit: QR/square-root linear derivatives from
MacroFinance are the production Phase 3 target.  Stop before SVD nonlinear,
CUT, GPU, or cross-repo migration phases unless the previous gates pass and the
next donor route is made explicit.
