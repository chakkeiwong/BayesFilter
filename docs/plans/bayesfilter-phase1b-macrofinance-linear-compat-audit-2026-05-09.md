# Audit: Phase 1B MacroFinance linear compatibility

## Date

2026-05-09

## Purpose

Record the Phase 1B compatibility decision for using BayesFilter's TF/TFP
linear QR value and derivative backends as the replacement path for
MacroFinance static linear Gaussian filtering fixtures.

## Donor Scope

MacroFinance was inspected read-only under:

```text
/home/chakwong/MacroFinance
```

Relevant donor files:

```text
/home/chakwong/MacroFinance/tests/test_generic_lgssm_autodiff_validation.py
/home/chakwong/MacroFinance/tests/test_tf_qr_sqrt_differentiated_kalman.py
/home/chakwong/MacroFinance/tests/test_masked_qr_sqrt_differentiated_kalman.py
/home/chakwong/MacroFinance/filters/tf_qr_sqrt_differentiated_kalman.py
/home/chakwong/MacroFinance/filters/masked_qr_sqrt_differentiated_kalman.py
```

BayesFilter did not edit MacroFinance and did not add MacroFinance as a
production dependency.  MacroFinance imports are test-only and optional.

## Compatibility Fixture

The first compatibility fixture is the MacroFinance seeded static LGSSM family:

```text
state_dim = 2
seed = 902
params = SMALL_DIM_PARAMETER_POINTS[1]
```

The fixture has:

- static transition matrix and covariance;
- static observation matrix and covariance;
- two parameters:
  - transition persistence scale;
  - log measurement noise;
- parameter-major first derivatives;
- parameter-major second derivatives;
- dense observations;
- a sparse mask with one partially missing period and one all-missing period.

## Tests Added

BayesFilter added:

```text
tests/test_macrofinance_linear_compat_tf.py
```

The tests compare:

- BayesFilter dense QR value against MacroFinance TF direct-QR value;
- BayesFilter masked QR value against MacroFinance TF masked direct-QR value;
- BayesFilter dense QR score/Hessian against MacroFinance TF direct-QR
  differentiated backend;
- BayesFilter masked QR score/Hessian against MacroFinance masked QR sparse
  oracle;
- tensor ranks for the donor fixture, confirming the first target path is
  time-invariant rather than time-varying.

## Result

Focused Phase 1B test:

```text
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_macrofinance_linear_compat_tf.py
```

reported:

```text
4 passed, 2 warnings in 71.49s
```

The warnings are TensorFlow Probability `distutils` deprecation warnings and do
not affect the compatibility conclusion.

## Interpretation

The Phase 1B primary criterion passes for the first MacroFinance static target:

- dense QR likelihood parity passes;
- masked QR likelihood parity passes under the static dummy-row convention;
- dense QR score and Hessian parity passes;
- masked QR score and Hessian parity passes;
- the donor fixture uses static matrices/covariances and parameter-major
  derivative tensors compatible with BayesFilter's TF derivative contracts.

No Phase 1C time-varying derivative gate is required before starting Phase 2.

## Remaining Caveats

This Phase 1B result does not claim full MacroFinance switch-over readiness.
It certifies the first static linear compatibility target.  Later
MacroFinance switch-over should still test:

- one-country provider fixtures;
- large-scale masked provider fixtures;
- cross-currency structural provider fixtures;
- any time-varying provider if one is selected for production.

## Next Phase Justification

Phase 2, SVD/eigen linear value backend, is justified.

Reason:

- the linear QR value/derivative spine has passed BayesFilter unit tests;
- the masked QR derivative gate has passed;
- the first MacroFinance static compatibility target has passed;
- no time-varying derivative blocker was found for the first target.

Phase 2 must remain value-only and must not claim SVD/eigen derivative
readiness.
