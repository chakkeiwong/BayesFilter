# Phase 6 Subplan: Batch-Native Analytical Score Implementation

Date: 2026-07-09

## Phase Objective

Implement `tf_qr_sqrt_kalman_score_batched_static`, a true batch-native
analytical QR score kernel for independent model rows, preserving dtype and XLA
compatibility.

## Entry Conditions Inherited From Previous Phase

- Phase 5 contract is reviewed and accepted.
- Scalar analytical score is dtype-polymorphic.
- Batched-static QR value/autodiff reference path is available.
- Phase 5 result names exact source, test, and comparator paths.

## Required Artifacts

- New batch-native analytical score source path, likely in
  `bayesfilter/linear/kalman_qr_derivatives_tf.py` or a dedicated sibling.
- Tests comparing batch-native score to scalar analytical rows and small
  autodiff references.
- Source-contract test proving optimized kernel does not use `tf.vectorized_map`
  or `tf.map_fn`, a Python loop over batch rows, or scalar score calls as the
  final implementation path.
- Phase 6 result and refreshed Phase 7 subplan.

## Required Checks, Tests, And Reviews

Run:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_batched_analytical_score_tf.py tests/test_linear_qr_dtype_contracts.py
git diff --check -- bayesfilter/linear tests docs/plans
```

Read-only review is required for implementation boundary and source contract if
the source diff is material.  Because the Claude review gate is unavailable for
this run unless separately approved, use a fresh bounded Codex substitute
review and label it weaker than Claude review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the batch-native analytical score return `[B]` values and `[B, P]` scores matching scalar analytical and autodiff references? |
| Baseline/comparator | Scalar `tf_qr_sqrt_kalman_score` row loop and autodiff through batched-static QR value on small fixtures. |
| Primary criterion | FP32/FP64 batch-native outputs have requested dtype and match references within predeclared tolerance under CPU/XLA. |
| Veto diagnostics | Use of vectorized/scalar row wrapper as final kernel, shape mismatch, dtype mismatch, parity failure, nonfinite output, or XLA compile failure. |
| Explanatory diagnostics | Max value/score deltas, compile status, and source-contract findings. |
| Not concluded | Runtime superiority or full benchmark ladder. |
| Artifact | Phase 6 result and refreshed Phase 7 subplan. |

## Forbidden Claims And Actions

- Do not claim speedup without Phase 7 benchmark evidence.
- Do not export public API broadly unless reviewed in this phase.
- Do not change unrelated HMC or nonlinear SSM files.

## Exact Next-Phase Handoff Conditions

Advance to Phase 7 only if correctness, dtype, shape, and source-contract tests
pass and benchmark dimensions are refreshed.

## Stop Conditions

Stop if true batch-native implementation would require unreviewed mathematical
changes or if parity cannot be achieved on small fixtures.
