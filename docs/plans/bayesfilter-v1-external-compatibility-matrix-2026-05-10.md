# BayesFilter v1 External Compatibility Matrix

## Date

2026-05-10

## Purpose

This matrix records what BayesFilter can certify against external projects
without coupling those projects to BayesFilter before v1.  MacroFinance and
DSGE remain external projects.  Live checks are optional and local; stable CI
should rely on BayesFilter-local fixtures.

## Compatibility Classes

```text
local_ci
optional_live_external
deferred_v1_integration
blocked
```

Definitions:

- `local_ci`: runs inside BayesFilter without external checkouts.
- `optional_live_external`: runs only when the external checkout is present.
- `deferred_v1_integration`: future adapter/switch-over work after v1.
- `blocked`: not a valid claim under current evidence.

## MacroFinance Matrix

| Capability | Status | BayesFilter-local evidence | Optional live evidence | Notes |
|---|---|---|---|---|
| Dense QR value | local_ci plus optional_live_external | `tests/test_linear_kalman_qr_tf.py` | `tests/test_macrofinance_linear_compat_tf.py` | Certify value parity, not switch-over. |
| Masked QR value | local_ci plus optional_live_external | `tests/test_linear_kalman_qr_tf.py` | `tests/test_macrofinance_linear_compat_tf.py` | Static dummy-row convention must remain diagnostic-visible. |
| Dense QR score/Hessian | local_ci plus optional_live_external | `tests/test_linear_kalman_qr_derivatives_tf.py` | `tests/test_macrofinance_linear_compat_tf.py` | Parameter-major derivative ordering. |
| Masked QR score/Hessian | local_ci plus optional_live_external | `tests/test_linear_kalman_qr_derivatives_tf.py` | `tests/test_macrofinance_linear_compat_tf.py` | Value and derivative claims stay separate from HMC readiness. |
| SVD/eigen value | local_ci | `tests/test_linear_kalman_svd_tf.py` | none required for v1 | Value-only; implemented-law diagnostics required. |
| SVD/eigen derivatives | blocked | none | none | Deferred until a real client need is proven. |
| MacroFinance adapter switch-over | deferred_v1_integration | none | future MacroFinance branch | Do not switch before BayesFilter v1. |

## DSGE Matrix

| Capability | Status | BayesFilter-local evidence | Optional live evidence | Notes |
|---|---|---|---|---|
| Generic structural affine value | local_ci | `tests/test_structural_affine_lgssm_controls_tf.py` | none required | Validates BayesFilter structural protocol, not DSGE production. |
| SVD cubature/UKF structural value | local_ci | `tests/test_structural_svd_sigma_point_tf.py` | future read-only inventory | DSGE economics stay external. |
| SVD-CUT value | local_ci | `tests/test_svd_cut_filter_tf.py` | future read-only inventory | Point-count and support diagnostics required. |
| SVD-CUT derivatives | local_ci branch gate | `tests/test_svd_cut_derivatives_tf.py` | future target-specific audit | HMC readiness remains blocked. |
| Rotemberg structural completion | optional_live_external candidate | generic structural protocol tests only | `docs/plans/bayesfilter-v1-dsge-readonly-target-inventory-result-2026-05-10.md` | DSGE-owned `dy` completion is tested; future bridge must stay test-only. |
| EZ all-stochastic metadata | optional_live_external diagnostic | generic structural protocol tests only | `docs/plans/bayesfilter-v1-dsge-readonly-target-inventory-result-2026-05-10.md` | Metadata/stability only; no BK, QZ, HMC, or posterior claim. |
| SGU production filtering | blocked | none | DSGE causal-locality gate required | SGU remains diagnostic-only until local causal target passes. |
| DSGE adapter switch-over | deferred_v1_integration | none | future DSGE branch | Do not switch before BayesFilter v1. |

## Certification Requirements

For any external compatibility label:

- state the fixture source;
- state the backend names;
- state tolerance and dtype;
- state mask convention if masks are involved;
- state derivative tensor ordering if derivatives are involved;
- state whether the check is local CI or optional live external;
- avoid claiming client default changes.

## Optional Live Test Policy

Optional live tests may import an external checkout only inside tests.  They
must skip cleanly when the checkout is absent.  Passing live tests certify
compatibility with the observed external checkout state, not a permanent
production dependency.

## Deferred v1 Integration Entry Criteria

Do not write a MacroFinance or DSGE adapter branch until:

1. v1 API freeze criteria pass;
2. local fixtures pass in BayesFilter CI;
3. optional live external checks pass on recorded external commits;
4. CPU benchmark artifacts exist;
5. GPU/HMC claims, if any, have target-specific evidence;
6. rollback and dependency direction are documented.
