# BayesFilter V1 Structural Sigma-point Score Result For Default Model C

## Date

2026-05-13

## Governing Master Program

This result closes R1 in:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

## Goal

Use the Chapter 18b structural sigma-point law to resolve the default Model C
zero-phase-variance score blocker without adding an artificial phase nugget or
inventing a separate fixed-null SVD theory.

## Result

Implemented an opt-in structural fixed-support score branch for SVD cubature,
SVD-UKF, and SVD-CUT4.

The branch:
- keeps the sigma-point variable as the pre-transition structural variable
  \((x_{t-1},\varepsilon_t)\);
- preserves the declared sigma-rule dimension;
- keeps structural null factor columns and their derivatives at zero;
- blocks positive covariance in the structural null direction;
- blocks moving-null derivative blocks;
- blocks positive placement floors and weak active spectral gaps;
- reports structural null, deterministic residual, and derivative-branch
  diagnostics.

## Files Changed

```text
bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py
bayesfilter/testing/nonlinear_diagnostics_tf.py
tests/test_nonlinear_sigma_point_scores_tf.py
tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py
docs/chapters/ch18_svd_sigma_point.tex
docs/chapters/ch28_nonlinear_ssm_validation.tex
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-plan-2026-05-13.md
docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-plan-audit-2026-05-13.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
docs/source_map.yml
```

## Validation

Focused score/branch validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `36 passed, 2 warnings`.

Focused V1 regression:

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
- `73 passed, 2 skipped, 2 warnings`.

Full default CPU validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Result:
- `204 passed, 5 skipped, 2 warnings`.

## Audit Notes

MathDevMCP label/code comparison tools failed internally on the new local
label during S3, so this result does not claim tool-certified code-document
equivalence.  The audit evidence is:
- MathDevMCP lookup of the Chapter 18b source proposition and likelihood
  equations during derivation;
- MathDevMCP algebra check of the innovation-score sign convention;
- MathDevMCP code/document search for the implemented diagnostic terms;
- manual code-document audit;
- targeted finite-difference and branch tests.

## Interpretation

H-S1 is supported on the tested scope.  Default Model C is now score-ready as a
BayesFilter V1 testing target under the structural fixed-support branch.  The
old smooth no-active-floor branch still blocks default Model C, which remains
the correct behavior when the structural fixed-support contract is not
requested.

This does not certify nonlinear Hessians, GPU/XLA scaling, nonlinear HMC, or
external MacroFinance/DSGE switch-over.

## Remaining Hypotheses

H-R2:
Model B and default Model C have practical parameter boxes where the selected
score branch remains finite, with no active floors, weak active gaps, moving
null blocks, or nonfinite scores.

Test:
run wider CPU branch grids for Model B and default Model C across SVD cubature,
SVD-UKF, and SVD-CUT4, recording ok fraction, structural-null diagnostics,
deterministic residuals, finite score status, and active-gap summaries.

H-R3:
Model B remains the first nonlinear HMC candidate because it is smooth and
does not depend on a structural null branch.  Default Model C can become the
second candidate only after H-R2 shows a stable branch box.

Test:
write a target-specific nonlinear HMC readiness plan only after H-R2 passes.

H-R4:
Nonlinear Hessians are still not a V1 requirement unless a concrete Newton,
Laplace, Riemannian HMC, or observed-information consumer is named.

Test:
keep Hessian work deferred until a consumer and branch gate are explicit.
