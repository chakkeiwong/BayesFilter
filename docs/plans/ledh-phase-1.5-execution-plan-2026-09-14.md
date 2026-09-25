# Phase 1.5 Execution Plan: TensorArray Elimination

**Date**: 2026-09-14  
**Parent**: ledh-tensorarray-elimination-plan-2026-09-14.md  
**Status**: READY FOR EXECUTION

## Critical Correction to Original Plan

The original plan identified ONE `tf.map_fn` at line 246. **Actual finding**: there are TWO nested `tf.map_fn` calls:

1. **Inner map_fn** (line 251): iterates over K directions per row
2. **Outer map_fn** (line 269): iterates over batch rows

Both must be eliminated for XLA compatibility.

## Verified Current Structure

```python
# Line 223: evaluate_row function
def evaluate_row(inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    theta_row, row_directions = inputs  # theta_row: [P], row_directions: [K, P]
    
    # Line 226: evaluate_direction function  
    def evaluate_direction(direction: Tensor) -> tuple[Tensor, Tensor]:
        bound_model = _single_cloud_model(model, theta_row, direction)
        value, score = canonical_value_and_analytical_score(
            bound_model, theta_row, ..., **common_kwargs
        )
        return value, score[0]  # value: scalar, score[0]: scalar
    
    # Line 251: INNER map_fn over K directions
    values, scores = tf.map_fn(
        evaluate_direction,
        row_directions,  # [K, P]
        fn_output_signature=(
            tf.TensorSpec([], dtype),  # K values
            tf.TensorSpec([], dtype),  # K scores
        ),
        parallel_iterations=1,
    )
    # values: [K], scores: [K]
    # Line 260: assert all values identical (primal invariance)
    return values[0], scores  # scalar value, [K] scores

# Line 269: OUTER map_fn over batch rows
values, scores = tf.map_fn(
    evaluate_row,
    (theta, directions),  # theta: [B, P], directions: [B, K, P]
    fn_output_signature=(
        tf.TensorSpec([], dtype),      # B values
        tf.TensorSpec([k_count], dtype),  # B×K scores
    ),
    parallel_iterations=1,
)
# values: [B], scores: [B, K]
```

## Replacement Strategy

### Option A: Replace Both with while_loop (RECOMMENDED)

Eliminates both TensorArray sources in one pass.

```python
# Preallocate outputs
batch_size = tf.shape(theta)[0]
k_count = int(directions.shape[1])  # Already verified statically known
param_dim = tf.shape(theta)[1]

values_out = tf.zeros([batch_size], dtype=dtype)
scores_out = tf.zeros([batch_size, k_count], dtype=dtype)

def batch_loop_cond(batch_idx, vals, scrs):
    return batch_idx < batch_size

def batch_loop_body(batch_idx, vals, scrs):
    theta_row = theta[batch_idx]
    row_directions = directions[batch_idx]  # [K, P]
    
    # Inner K-direction loop
    k_values = tf.zeros([k_count], dtype=dtype)
    k_scores = tf.zeros([k_count], dtype=dtype)
    
    def k_loop_cond(k_idx, k_vals, k_scrs):
        return k_idx < k_count
    
    def k_loop_body(k_idx, k_vals, k_scrs):
        direction = row_directions[k_idx]
        bound_model = _single_cloud_model(model, theta_row, direction)
        value, score = canonical_value_and_analytical_score(
            bound_model, theta_row, ..., **common_kwargs
        )
        k_vals = tf.tensor_scatter_nd_update(
            k_vals, [[k_idx]], [value]
        )
        k_scrs = tf.tensor_scatter_nd_update(
            k_scrs, [[k_idx]], [score[0]]
        )
        return k_idx + 1, k_vals, k_scrs
    
    _, k_values, k_scores = tf.while_loop(
        cond=k_loop_cond,
        body=k_loop_body,
        loop_vars=(tf.constant(0, tf.int32), k_values, k_scores),
        maximum_iterations=k_count,
        parallel_iterations=1,
    )
    
    # Primal invariance check
    tf.debugging.assert_near(
        k_values,
        tf.broadcast_to(k_values[0], tf.shape(k_values)),
        rtol=1.0e-12,
        message="direction changed the primal canonical value",
    )
    
    # Update batch outputs
    vals = tf.tensor_scatter_nd_update(vals, [[batch_idx]], [k_values[0]])
    scrs = tf.tensor_scatter_nd_update(scrs, [[batch_idx]], [k_scores])
    return batch_idx + 1, vals, scrs

_, values, scores = tf.while_loop(
    cond=batch_loop_cond,
    body=batch_loop_body,
    loop_vars=(tf.constant(0, tf.int32), values_out, scores_out),
    maximum_iterations=batch_size,
    parallel_iterations=1,
)
```

### Option B: Replace Inner Only First (INCREMENTAL)

Replace inner map_fn first, verify parity, then replace outer.

**Verdict**: Option A is better. Both map_fn calls are equally non-XLA-compatible, and fixing one doesn't enable XLA. Do both together with one parity gate.

## Success Criteria (from parent plan)

1. ✓ Parity: value and score relative error < 5e-4 vs current implementation
2. ✓ Graph mode: time unchanged or faster (baseline 2.465s small scale)
3. ✓ XLA mode: compiles successfully (no TensorListReserve error)
4. ✓ XLA mode: time < 1.0s small scale (target ~0.175s)
5. ✓ Oracle contract: still passes (8/8 tests)

## Implementation Steps

1. ✓ Read and verify current code structure (COMPLETE)
2. Create parity test harness
3. Implement Option A replacement
4. Run parity test (gate: must pass before timing)
5. Run graph mode timing (gate: must not regress >10%)
6. Run XLA compilation test (gate: must succeed)
7. Run XLA timing test
8. Run oracle contract
9. Update status document

## Parity Test Design

**Test**: `test_batch_fused_while_vs_mapfn_parity.py`

```python
def test_parity():
    # Small fixture: B=3, K=5, N=24, T=5
    # Compare:
    # - Current: nested map_fn
    # - New: nested while_loop
    # Check:
    # - values: rtol < 5e-4
    # - scores: rtol < 5e-4
    # - Both pass primal invariance assertion
```

## Risks Revisited

1. **Nested scatter overhead**: Now 2D (B×K iterations). At B=50, K=5 → 250 scatter ops.
   - Mitigation: XLA should fuse. Measure before/after graph mode.

2. **Shape invariants**: `param_dim` may be dynamic.
   - Finding: Line 198 asserts `k_count` is static. `param_dim` not checked.
   - Mitigation: Check `theta.shape[1]` staticness. If dynamic, use `None` in shape_invariants.

3. **Primal invariance assertion** (line 260): Must preserve exact placement.
   - Current: after inner map_fn
   - New: after inner while_loop, same location

## Open Question Resolution

**Q1**: Is `param_dim` statically known?
- Check: `theta.shape` at line 187-194
- Line 194: checks `theta.shape[1]` vs `directions.shape[2]` but doesn't require static
- **Answer**: Likely dynamic. Use `TensorShape([batch_size, None])` for scores_out if needed.

**Q2**: Typical batch_size and k_count?
- Small scale: B=1 (single particle), K=5
- Plan scale: B unclear, but execution matrix used B=1
- **Finding**: The outer map_fn may be over **evaluation points**, not particles. Need to trace caller.

Let me check typical usage:

**Finding from line 180**: `directions` can be rank-2 (squeezed to rank-3) or rank-3. If rank-2, B=1 implicitly.

**Conclusion**: Typical case is B=1, K=5-500. The nested while_loop overhead is dominated by the K loop, not the B loop.

## Execution Decision

**Proceed with Option A**: Replace both map_fn calls in one implementation, test parity, then measure.

---

**Status**: Plan reviewed and corrected. Ready to execute Step 2 (parity harness).
