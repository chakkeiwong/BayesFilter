# Phase 2 Subplan: QR Value Dtype Cleanup

Date: 2026-07-09

## Phase Objective

Make QR Kalman value paths dtype-polymorphic before touching analytical score.

## Entry Conditions Inherited From Previous Phase

- Phase 1 helper tests pass.
- Mixed dtype policy is documented.
- Phase 2 write set is limited to QR value code and focused tests.
- Phase 1 added `bayesfilter/linear/dtypes_tf.py` with
  `common_floating_dtype(...)` and `as_float_tensor(...)`.

## Required Artifacts

- Source changes in `bayesfilter/linear/kalman_qr_tf.py` and, only if needed,
  `bayesfilter/linear/qr_factor_tf.py`.
- Tests covering FP32/FP64 output dtype for scalar compact, scalar while-loop,
  batched-static, and masked QR value paths.
- Phase 2 result and refreshed Phase 3 subplan.

Exact initial function targets:

- `tf_qr_sqrt_kalman_log_likelihood_compact`
- `tf_qr_sqrt_kalman_log_likelihood_while_loop`
- `tf_qr_sqrt_kalman_log_likelihood_batched_static`
- `tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop`
- `tf_qr_sqrt_masked_kalman_log_likelihood_compact`
- `tf_qr_sqrt_masked_kalman_log_likelihood_batched_static`
- local helpers they depend on in `kalman_qr_tf.py` and `qr_factor_tf.py`

## Required Checks, Tests, And Reviews

Run:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_dtype_contracts.py tests/test_linear_qr_compact_loglik_tf.py
git diff --check -- bayesfilter/linear tests docs/plans
```

Review if QR factor primitives or public value behavior changes materially.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Do QR value kernels preserve FP32 and FP64 input dtype while retaining existing FP64 parity? |
| Baseline/comparator | Existing FP64 QR value outputs and batched-static row parity tests. |
| Primary criterion | FP64 tests remain within existing tolerance; FP32 tests return FP32 outputs and match FP32 scalar/autodiff references within declared tolerance. |
| Veto diagnostics | Any value path still hard-coerces to FP64, parity failure, nonfinite output, or XLA CPU compile failure. |
| Explanatory diagnostics | Tolerance deltas and source inventory reduction. |
| Not concluded | Analytical score dtype cleanup, benchmark readiness, and batched score are not complete. |
| Artifact | Phase 2 result and refreshed Phase 3 subplan. |

## Forbidden Claims And Actions

- Do not claim analytical score FP32 support.
- Do not run long benchmarks.
- Do not change math beyond dtype handling.

## Exact Next-Phase Handoff Conditions

Advance to Phase 3 only if QR value paths pass FP32/FP64 dtype and parity
checks, and remaining hard-coded FP64 sites in analytical-score code are
listed.

## Stop Conditions

Stop if QR value dtype cleanup requires a broader numerical redesign or breaks
existing FP64 behavior.
