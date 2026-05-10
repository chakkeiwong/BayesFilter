# Audit: BayesFilter v1 Local Fixture Gaps

## Date

2026-05-10

## Purpose

This audit checks whether BayesFilter has local fixtures for the v1
external-compatibility lane.  The target is not client switch-over.  The target
is enough local evidence to stabilize BayesFilter v1 without requiring
MacroFinance or DSGE at CI time.

## Local Fixture Coverage

| Area | Current local fixture | Status |
|---|---|---|
| Dense Cholesky/covariance value | `tests/test_linear_kalman_tf.py` | sufficient |
| Masked covariance value | `tests/test_linear_kalman_tf.py` | sufficient |
| Dense QR value | `tests/test_linear_kalman_qr_tf.py` | sufficient |
| Masked QR value | `tests/test_linear_kalman_qr_tf.py` | sufficient |
| Dense QR score/Hessian | `tests/test_linear_kalman_qr_derivatives_tf.py` | sufficient |
| Masked QR score/Hessian | `tests/test_linear_kalman_qr_derivatives_tf.py` | sufficient |
| SVD/eigen dense value | `tests/test_linear_kalman_svd_tf.py` | sufficient |
| SVD/eigen masked value | `tests/test_linear_kalman_svd_tf.py` | sufficient |
| SVD/eigen implemented-law diagnostics | `tests/test_linear_kalman_svd_tf.py` | sufficient |
| Static-shape graph parity | `tests/test_compiled_filter_parity_tf.py` | sufficient for CPU graph parity |
| Structural affine value | `tests/test_structural_affine_lgssm_controls_tf.py` | sufficient |
| Structural SVD sigma-point value | `tests/test_structural_svd_sigma_point_tf.py` | sufficient |
| CUT4-G rule/value | `tests/test_cut_rule_tf.py`, `tests/test_svd_cut_filter_tf.py` | sufficient |
| SVD-CUT smooth-branch derivatives | `tests/test_svd_cut_derivatives_tf.py` | sufficient as branch-gated derivative fixture |
| Production no-NumPy policy | module text checks in linear tests | partial but sufficient for current linear v1 surface |

## Optional External Fixture Coverage

MacroFinance live compatibility:

```text
tests/test_macrofinance_linear_compat_tf.py
```

Status:

- useful optional live check;
- should not be required for BayesFilter CI;
- should not drive immediate MacroFinance switch-over.

## Gaps Found

### Gap 1: Stable Public API Documentation

The main missing local artifact was not a test; it was a v1 API freeze note.
This is now addressed by:

```text
docs/plans/bayesfilter-v1-api-freeze-criteria-2026-05-10.md
```

### Gap 2: External Compatibility Matrix

The old plan mixed compatibility and switch-over language.  This is now
addressed by:

```text
docs/plans/bayesfilter-v1-external-compatibility-matrix-2026-05-10.md
```

### Gap 3: Benchmark Artifacts

Correctness fixtures are sufficient for v1 planning, but benchmark artifacts
are still missing.  Close this through a benchmark harness plan before claiming
performance.

### Gap 4: Public API Import Stability Tests

Current tests cover behavior, but there is no dedicated test that imports the
declared v1 public API surface from `bayesfilter` and `bayesfilter.linear`.
This is useful but not a blocker for the current documentation pass.

## Verdict

Local fixture coverage is sufficient for v1 external-compatibility planning.
No immediate production code or test edits are required in this pass.

Recommended future implementation:

- add a small `tests/test_v1_public_api.py` that imports the v1-stable symbols
  named in the API freeze note;
- add benchmark scripts and artifacts before performance claims;
- keep optional live MacroFinance tests optional.
