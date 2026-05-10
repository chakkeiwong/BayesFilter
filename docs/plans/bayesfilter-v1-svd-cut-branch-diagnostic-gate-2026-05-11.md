# Gate: BayesFilter v1 SVD-CUT Branch Diagnostics

## Date

2026-05-11

## Purpose

This gate turns SVD-CUT derivative caution into measurable diagnostics.  It
does not promote SVD-CUT to HMC readiness.

## Current Evidence

Existing tests cover:

```text
tests/test_svd_cut_filter_tf.py
tests/test_svd_cut_derivatives_tf.py
```

They establish:

- SVD-CUT value matches the affine linear likelihood fixture;
- support and deterministic residual diagnostics exist;
- score/Hessian matches finite differences and TensorFlow autodiff on a smooth
  separated-spectrum branch;
- active floor raises `blocked_active_floor`;
- weak spectral gap raises `blocked_weak_spectral_gap`.

## Required Diagnostic Fields

Future SVD-CUT derivative artifacts should record:

1. `derivative_branch`;
2. `differentiability_status`;
3. active placement floor count;
4. active innovation floor count;
5. minimum spectral gap over all factorizations;
6. weak spectral-gap event count;
7. maximum integration rank;
8. point count;
9. support residual;
10. deterministic residual;
11. finite-difference or autodiff parity status.

## Promotion Rule

SVD-CUT derivative claims may be made only for:

```text
smooth_separated_spectrum
inactive_floor
finite_difference_or_autodiff_parity_passed
```

HMC remains blocked unless smooth-branch diagnostics dominate the target region
and a target-specific sampler smoke passes.

## Blocked Labels

```text
blocked_active_floor
blocked_weak_spectral_gap
blocked_missing_branch_telemetry
blocked_missing_sampler_evidence
```

## Next Test Work

Add a future diagnostic aggregation test that sweeps a small parameter box and
reports branch frequencies.  That future test should remain an extended CPU
diagnostic, not fast local CI, until runtime is known.
