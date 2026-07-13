# Phase 1 Subplan: Dtype Infrastructure

Date: 2026-07-09

## Phase Objective

Add the minimal dtype helper infrastructure and tests needed for QR Kalman
TensorFlow kernels to preserve requested floating dtype instead of coercing to
`tf.float64`.

## Entry Conditions Inherited From Previous Phase

- Phase 0 result exists and records hard-coded dtype inventory.
- Governance review converged or weaker substitute review is documented.
- No source edits occurred before Phase 0 inventory.
- Phase 1 write set is limited to dtype helpers and focused tests.

## Required Artifacts

- Source changes, likely in one of:
  - `bayesfilter/linear/qr_factor_tf.py`
  - `bayesfilter/linear/kalman_qr_tf.py`
  - a new local helper module under `bayesfilter/linear/`
- Focused dtype contract tests, likely:
  `tests/test_linear_qr_dtype_contracts.py`
- Phase 1 result:
  `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase1-dtype-infrastructure-result-2026-07-09.md`
- Refreshed Phase 2 subplan.

## Required Checks, Tests, And Reviews

Run:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_dtype_contracts.py
git diff --check -- bayesfilter/linear tests docs/plans
```

Review material helper design with Claude or a documented substitute review if
the helper affects shared QR primitives.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Do the new helpers infer and preserve floating dtype without silently mixing dtypes? |
| Baseline/comparator | Current helpers that coerce tensors to `tf.float64`. |
| Primary criterion | Focused tests prove helper output dtype follows FP32 and FP64 inputs and rejects/handles mixed floating dtypes according to the documented contract. |
| Veto diagnostics | Helper silently downcasts/upcasts unexpectedly, mixed dtype policy is ambiguous, or tests depend on eager-only behavior. |
| Explanatory diagnostics | Helper source inventory and exact dtype assertion failures. |
| Not concluded | QR value/score kernels are not yet fully dtype-polymorphic; no benchmark claim is made. |
| Artifact | Phase 1 result and refreshed Phase 2 subplan. |

## Forbidden Claims And Actions

- Do not claim QR value or analytical score dtype cleanup is complete.
- Do not modify public defaults or backend selection.
- Do not run GPU benchmarks.
- Do not broaden refactors beyond dtype infrastructure.

## Exact Next-Phase Handoff Conditions

Advance to Phase 2 only if helpers and focused tests pass under CPU-hidden
execution, and Phase 2 subplan names the exact QR value functions to update.

## Stop Conditions

Stop if helper design cannot preserve both FP32 and FP64, if mixed dtype policy
is unresolved, or if shared helper changes break unrelated linear tests.
