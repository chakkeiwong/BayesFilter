# Advanced particle filter kernel PFF reproduction

## Date

2026-05-10

## Scope

This report covers Phase H2 of
`docs/plans/bayesfilter-student-dpf-baseline-hypothesis-closure-plan-2026-05-10.md`.
It isolates the earlier `advanced_particle_filter/tests/test_kernel_pff.py`
partial failure and non-completion.

Snapshot commit:
`d2a797c330e11befacbb736b5c86b8d03eb4a389`.

## Commands and outcomes

All commands used:

```text
PYTHONPATH=experiments/student_dpf_baselines/vendor
timeout 90s pytest <test-node> -q -s
```

| Test node | Status | Runtime / evidence | Interpretation |
| --- | --- | --- | --- |
| `test_kernel_pff_lgssm` | timeout | Exit code `124` after 90 seconds, no summary output | Reproduces non-completion.  This test is not suitable as a bounded baseline in the current environment. |
| `test_kernel_pff_convergence` | failed | 1 failed in 4.60s; average iterations `100.0`; assertion expected `< 100` | Deterministic convergence failure: every step hit `max_iterations=100`. |
| `test_scalar_vs_matrix_kernel` | passed, slow | 1 passed, 1 warning in 85.92s | Algorithm path can run, but the bounded runtime is high for routine panels. |

## Classification

H2 classification: `algorithm_test_sensitivity_and_long_runtime`.

The kernel PFF issue is not a missing dependency and not a production
BayesFilter issue.  It is localized to the `advanced_particle_filter` kernel
PFF test behavior under the current environment:

- one test times out;
- one test fails because convergence reaches the maximum iteration count;
- one test passes but takes nearly the full 90-second bound.

## Decision

Kernel PFF should remain excluded from cross-student comparison panels until a
separate bounded reproduction/debug plan is approved.  It is not acceptable as
routine comparison evidence in the current harness.
