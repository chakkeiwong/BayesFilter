# BayesFilter V1 Nonlinear Implementation Audit Result

## Date

2026-05-12

## Scope

This audit reviewed the BayesFilter V1 nonlinear filtering implementation lane
after completion of the nonlinear model-suite, analytic-gradient, and remaining
master-program subplans.  The audit stayed inside the BayesFilter V1 lane and
did not edit MacroFinance, DSGE, Chapter 18b, structural SVD/SGU plans, or the
shared monograph reset memo.

Audited code paths:

- `bayesfilter/nonlinear/sigma_points_tf.py`
- `bayesfilter/nonlinear/svd_cut_tf.py`
- `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py`
- `bayesfilter/testing/nonlinear_models_tf.py`
- `bayesfilter/testing/nonlinear_diagnostics_tf.py`
- nonlinear sigma-point value, score, diagnostic, and benchmark tests.

## Audit Finding

No major implementation correctness issue was found in the audited nonlinear
filtering lane.  The value filters, analytic smooth-branch score fixture, and
testing diagnostics are consistent with the current V1 claims:

- Model A exact linear-Gaussian parity is tested against the Kalman reference.
- Models B-C are tested as nonlinear value-filter examples with dense one-step
  Gaussian projection diagnostics, not as exact full-likelihood certificates.
- Analytic score certification remains limited to the smooth affine fixture
  until Models B-C receive explicit derivative providers.
- SVD-CUT4 production value code no longer relies on raw `GradientTape`
  derivatives; raw autodiff is retained only as a testing oracle.

The audit found one small diagnostic exposure gap rather than a numerical
formula bug.  The raw filters already computed placement/innovation floor
counts and PSD-projection residuals, but the high-level `diagnostics.extra`
dictionary did not expose those fields for all value backends.  The patch adds
those fields to the SVD cubature, SVD-UKF, and SVD-CUT4 wrapper diagnostics and
adds focused assertions that the shared nonlinear diagnostic snapshot sees
them.

## Verification

Default CPU test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Result:

- `185 passed, 5 skipped, 2 warnings`.

Extended CPU branch diagnostics:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
  BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py -p no:cacheprovider
```

Result:

- `2 passed, 2 warnings`.

Focused nonlinear regression after the diagnostic exposure fix:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_structural_svd_sigma_point_tf.py \
  -p no:cacheprovider
```

Result:

- `24 passed, 2 warnings`.

Hygiene checks:

```bash
python -m py_compile \
  bayesfilter/nonlinear/sigma_points_tf.py \
  bayesfilter/nonlinear/svd_cut_tf.py \
  bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py \
  bayesfilter/testing/nonlinear_diagnostics_tf.py \
  bayesfilter/testing/nonlinear_models_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py

git diff --check

python -c "import yaml; yaml.safe_load(open('docs/source_map.yml')); print('source_map ok')"

python -m json.tool \
  docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-12.json \
  >/tmp/bayesfilter_v1_nonlinear_filter_benchmark_audit_json.txt
```

Result:

- all checks passed.

## Residual Risks

The audit does not close the following already-known gaps:

- Models B-C still need explicit derivative providers before analytic score or
  HMC readiness claims are valid for those nonlinear models.
- The current nonlinear Hessian plan remains deferred; V1 has score-first
  evidence, not full Hessian certification for SVD sigma-point filters.
- GPU/XLA scaling was not rerun in this audit because CUDA-visible commands
  require explicit escalated execution under the repository policy.
- Dense one-step projection references for Models B-C are approximation
  diagnostics, not exact nonlinear likelihood or posterior references.

## Next Justified Work

The next implementation pass should target derivative providers for Models B-C,
then rerun finite-difference score tests and branch diagnostics on smooth
parameter boxes.  GPU/XLA and nonlinear HMC should remain gated until those
score diagnostics pass.
