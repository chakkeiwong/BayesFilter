# Phase 3 Subplan: Analytical Score Dtype Cleanup

Date: 2026-07-09

## Phase Objective

Make the public QR analytical score path and required derivative helpers
preserve FP32/FP64 dtype.

## Entry Conditions Inherited From Previous Phase

- Phase 2 QR value dtype cleanup passes.
- Value/autodiff FP32 references are available for small fixtures.
- Analytical-score hard-coded dtype sites are inventoried.

## Required Artifacts

- Source changes in `bayesfilter/linear/kalman_qr_derivatives_tf.py`,
  derivative payload containers in `bayesfilter/linear/types_tf.py`, derivative
  result containers in `bayesfilter/results_tf.py`, and shared QR factor
  derivative helpers only where required.
- Tests comparing FP32 analytical score to FP32 autodiff on small fixtures.
- Tests showing derivative payload containers and `TFFilterDerivativeResult`
  preserve explicit FP32/FP64 dtype and reject unsupported or mixed floating
  dtypes where the helper contract requires it.
- CPU/XLA compile smoke for FP32 and FP64 analytical score.
- Phase 3 result and refreshed Phase 4 subplan.

## Required Checks, Tests, And Reviews

Run:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_dtype_contracts.py tests/test_linear_kalman_qr_derivatives_tf.py
git diff --check -- bayesfilter/linear bayesfilter/results_tf.py tests docs/plans
```

Read-only review is required if derivative helper, derivative payload, or
derivative result contracts change materially.  Because the Claude review gate
is unavailable for this run unless separately approved, use a fresh bounded
Codex substitute review and label it weaker than Claude review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does analytical QR score preserve requested dtype and remain correct against scalar/autodiff references? |
| Baseline/comparator | Current FP64 analytical score and FP32 autodiff through QR value on small fixtures. |
| Primary criterion | FP64 existing parity remains; FP32 analytical value/score outputs are FP32 and match FP32 autodiff within predeclared tolerance. |
| Veto diagnostics | Hidden FP64 coercion in score kernels, derivative payload containers, or derivative result envelopes; score dtype mismatch; parity failure; nonfinite score; or XLA compile failure. |
| Explanatory diagnostics | Max absolute/relative score error and remaining dtype inventory. |
| Not concluded | Batch-native analytical score or runtime speed ranking. |
| Artifact | Phase 3 result and refreshed Phase 4 subplan. |

## Forbidden Claims And Actions

- Do not claim the score is faster or superior.
- Do not use FP64 references to mask FP32 output dtype mismatch.
- Do not implement batch-native score in this phase except for test fixtures.

## Exact Next-Phase Handoff Conditions

Advance to Phase 4 only if analytical score dtype tests pass and benchmark
harness changes can fail closed on requested/observed dtype mismatch.  The
handoff must list any remaining intentional FP64-only containers outside the
QR analytical-score path.

## Stop Conditions

Stop if derivative helper/container dtype cleanup requires a new derivation, if
derivative payload/result containers cannot preserve explicit FP32 without
breaking existing FP64 contracts, or if FP32 score parity fails for unclear
numerical reasons.
