# LEDH While-Loop Refactor Phase 1 Result

**Date**: 2026-09-03  
**Phase**: 1 — Python-to-while_loop Conversion  
**Status**: COMPLETE  
**Decision**: PASS — Proceed to Phase 2

---

## Execution Summary

Converted both Python for loops in `ledh_canonical_batch_fused_tf.py` to `tf.while_loop`:

1. **Inner flow loop** (substeps iteration, originally ~line 219)
2. **Outer horizon loop** (timestep iteration, originally ~line 146)

All three parity tests PASSED. The conversion preserves the analytical score contract by maintaining hand-derived tangent recursion inside loop bodies.

---

## Implementation Changes

### Inner Flow Loop Conversion

**Before** (Python for loop):
```python
for s in range(substeps):
    lam = (float(s) + 1.0) / float(substeps)
    # ... body with h_jac, innovation_chol, kalman gain
    actual = new_actual
    d_actual = new_d_actual
    # ... mutation of actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det
```

**After** (tf.while_loop):
```python
def _flow_body(s, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det):
    lam = (tf.cast(s, dtype) + 1.0) / tf.cast(substeps, dtype)
    # ... (same body logic)
    return (s + 1, new_actual, new_d_actual, new_aux, new_d_auxiliary,
            log_det + ldet_inc, d_log_det + dldet_inc)

_, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det = tf.while_loop(
    cond=lambda s, *_: s < substeps,
    body=_flow_body,
    loop_vars=(tf.constant(0, tf.int32), pre_flow, d_pre_flow, anchors, d_anchors,
               tf.zeros([m], dtype), tf.zeros([m], dtype)),
    shape_invariants=(tf.TensorShape([]), 
                      tf.TensorShape([m, n * substeps]),
                      tf.TensorShape([m, n * substeps]),
                      tf.TensorShape([m, n_a * substeps]),
                      tf.TensorShape([m, n_a * substeps]),
                      tf.TensorShape([m]),
                      tf.TensorShape([m])),
    maximum_iterations=substeps,
    parallel_iterations=1,
)
```

### Outer Horizon Loop Conversion

**Before** (Python for loop):
```python
for t in range(horizon):
    observation = observations[t, :]
    # ... UKF predict, flow, weight assembly, UKF update
    states = children
    total += increment
```

**After** (tf.while_loop):
```python
def _step_body(t, states, d_states, covariances, d_covariances, total, d_total):
    observation = tf.gather(observations, t)
    # ... (same logic for predict, flow, weight, update)
    new_total = total + increment
    new_d_total = d_total + tf.reduce_sum(softmax * d_logits, axis=1)
    return (t + 1, children, d_children, new_covariances, new_d_covariances,
            new_total, new_d_total)

_, states, d_states, covariances, d_covariances, total, d_total = tf.while_loop(
    cond=lambda t, *_: t < horizon,
    body=_step_body,
    loop_vars=(tf.constant(0, tf.int32), states, d_states, covariances, d_covariances,
               total, d_total),
    shape_invariants=(tf.TensorShape([]),
                      tf.TensorShape([m, n]),
                      tf.TensorShape([m, n]),
                      tf.TensorShape([m, n, n]),
                      tf.TensorShape([m, n, n]),
                      tf.TensorShape([m]),
                      tf.TensorShape([m])),
    maximum_iterations=horizon,
    parallel_iterations=1,
)
```

### Critical Fix

Changed all mutation patterns from `total += increment` to `new_total = total + increment` with explicit returns. TensorFlow while_loop cannot mutate captured variables; all state must flow through loop_vars and returns.

Applied to: `total`, `d_total`, `actual`, `d_actual`, `auxiliary`, `d_auxiliary`, `log_det`, `d_log_det`, `states`, `d_states`, `covariances`, `d_covariances`.

---

## Verification Results

### Test Execution

**Command**:
```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/pytest \
tests/highdim/test_ledh_canonical_batch_fused.py \
-xvs
```

**Results**:
- `test_fused_batch_size_one_parity`: **PASSED**
- `test_fused_rows_independent_and_distinct`: **PASSED**
- `test_fused_lane_is_tf_function_compilable`: **PASSED**

All three parity tests pass. The while_loop conversion preserves numerical behavior.

### Import Verification

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -c \
"from bayesfilter.highdim.ledh_canonical_batch_fused_tf import ledh_canonical_batch_fused_score_tf; \
print('Import successful')"
```

**Result**: Import successful, no errors.

---

## Phase 1 Success Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Both loops converted to tf.while_loop | ✓ PASS | Code inspection: inner flow loop (substeps) and outer horizon loop converted |
| File imports without error | ✓ PASS | Import verification successful |
| All fused parity tests pass | ✓ PASS | 3/3 tests PASSED |
| Tangent propagation preserved | ✓ PASS | Hand-derived d_actual, d_auxiliary, d_log_det, d_total recursions unchanged in loop bodies |

---

## Technical Notes

### Shape Invariants

Explicit shape invariants prevent dynamic shape changes during loop execution:

- Counter `t` or `s`: scalar `tf.TensorShape([])`
- States: `[m, n]` for actual states
- Tangents: same shape as primal (e.g., d_states `[m, n]`)
- Flow states: `[m, n * substeps]` for concatenated sigma points
- Covariances: `[m, n, n]`
- Scalars: `[m]` for per-particle totals

### Parallel Iterations

Set `parallel_iterations=1` to enforce sequential execution and preserve dependency order, particularly for tangent accumulation where each step depends on prior tangent state.

### Maximum Iterations

Explicit `maximum_iterations` parameter matches loop bounds (`substeps` for inner, `horizon` for outer) to assist XLA optimization and prevent unbounded iteration.

---

## Git State

**Branch**: `ledh-canonical-rebuild` (worktree)  
**Modified**: `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`  
**Commit**: Not committed yet (per campaign protocol, commit at campaign completion)

---

## Decision

**PASS**: All Phase 1 success criteria satisfied.

**Next Action**: Proceed to Phase 2 — Multi-direction tangent generalization.

---

## Wall Time

**Phase 1 Execution**: ~8 minutes (implementation + verification)

---

## Notes

The conversion maintains the analytical score contract by preserving all tangent recursion logic. No numerical approximation introduced. The while_loop structure prepares for:

- Phase 2: Multi-direction tangent support
- Phase 3: XLA optimization via closure hoisting
- Phase 4: Graph size reduction (target ~2.2K nodes vs current 663K)

Parity tests confirm behavioral equivalence before proceeding to capability expansion.
