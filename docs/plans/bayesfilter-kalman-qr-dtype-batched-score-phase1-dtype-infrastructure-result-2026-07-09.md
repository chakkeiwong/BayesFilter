# Phase 1 Result: Dtype Infrastructure

Date: 2026-07-09

## Status

`PASSED_WITH_CODEX_SUBSTITUTE_REVIEW`

## Phase Decision

Phase 1 added minimal TensorFlow dtype helper infrastructure and focused tests.
It did not change QR value or analytical-score kernel behavior.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Do the new helpers infer and preserve floating dtype without silently mixing dtypes? |
| Baseline/comparator | Current QR helpers that coerce tensors to `tf.float64`. |
| Primary criterion status | `PASSED`: focused helper tests cover FP32/FP64 preservation, historical literal default, mixed dtype rejection, unsupported dtype rejection, and one CPU/XLA trace behavior. |
| Veto diagnostic status | No helper silently promoted/demoted mixed floating dtype in tests; no QR kernel source behavior was changed. |
| Main uncertainty | Shared QR primitive call sites have not yet been migrated to the helpers. |
| Next justified action | Phase 2 QR value dtype cleanup can use `bayesfilter/linear/dtypes_tf.py` helpers. |
| What is not concluded | QR value and analytical-score kernels are not yet dtype-polymorphic; no benchmark or runtime speed claim is made. |

## Implementation

Added:

- `bayesfilter/linear/dtypes_tf.py`
- `tests/test_linear_qr_dtype_contracts.py`

Helper contract:

- `common_floating_dtype(...)` infers a shared explicit floating dtype from
  TensorFlow/NumPy typed inputs.
- Python literals alone preserve the historical default `tf.float64`.
- Supported floating dtypes are `tf.float32` and `tf.float64`.
- Mixed FP32/FP64 inputs are rejected instead of silently promoted.
- `as_float_tensor(...)` converts values to the already chosen dtype and rejects
  unsupported requested dtypes.

## Local Checks

Command:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_dtype_contracts.py
```

Result:

```text
8 passed
```

The command emitted TensorFlow/AutoGraph deprecation warnings from `gast`, but
no test failed.

Command:

```bash
git diff --check -- bayesfilter/linear/dtypes_tf.py tests/test_linear_qr_dtype_contracts.py docs/plans docs/reviews
```

Result: `PASSED`.

## Review Status

Review status: `AGREE_WEAKER_THAN_CLAUDE_REVIEW`.

Fresh Codex substitute review found no blocking issues.  Non-blocking
observations:

- focused tests are narrower than full proof of every graph/XLA failure mode;
- Phase 2 call sites must pass actual typed tensors/arrays, not containers that
  merely hold typed tensors.

## Next-Phase Handoff

Phase 2 may begin.  Handoff conditions satisfied:

- helper tests passed under CPU-hidden execution;
- CPU/XLA trace test passed;
- `git diff --check` passed;
- Phase 2 subplan names the helper module and exact QR value functions to
  update.

## Nonclaims

- Phase 1 did not remove hard-coded `tf.float64` from existing QR kernels.
- Phase 1 did not implement FP32 QR value or analytical-score support.
- Phase 1 did not implement batched analytical score.
- Phase 1 did not run GPU commands or benchmarks.
