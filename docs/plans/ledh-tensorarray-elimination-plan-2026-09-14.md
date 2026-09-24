# LEDH TensorArray Elimination Plan

**Date**: 2026-09-14  
**Context**: Phase 3.5 Step 1 complete, XLA blocker identified  
**Blocker**: XLA compilation fails on `TensorListReserve` (TensorArray internal op)  
**Impact**: 77× XLA speedup unavailable (0.175s vs 13.565s at small scale)

## Status Update First

Phase 1 measurement complete:
- Graph mode: 5.50× speedup ✓ (13.565s → 2.465s)
- XLA mode: 77× potential but **BLOCKED** by TensorArray
- Oracle contract: PASSING (8/8 tests)
- Constraint: `annealed_stages=1` only

## Root Cause Analysis

XLA error trace:
```
INVALID_ARGUMENT: Detected unsupported operations when trying to compile graph:
TensorListReserve (No registered 'TensorListReserve' OpKernel for XLA_GPU_JIT)
```

TensorArray is used internally by TensorFlow for dynamic-size list accumulation but is not XLA-compatible. Search shows **zero TensorArray usage** in `ledh_canonical_score_tf.py` or `ledh_canonical_batch_fused_tf.py`.

### Where TensorArray Appears

The blocker is **NOT in the canonical engine**. It's in the **batch-fused wrapper** at `ledh_canonical_batch_fused_tf.py:246`:

```python
def evaluate_row(particle_index):
    value, score = canonical_value_and_analytical_score(...)
    return value, score

values, scores = tf.map_fn(
    evaluate_row,
    tf.range(count),
    parallel_iterations=1,
    dtype=(dtype, dtype),
)
```

`tf.map_fn` **creates TensorArray internally** to accumulate loop outputs, even when `parallel_iterations=1`. This is a TensorFlow implementation detail.

## The Real Structure

```
ledh_canonical_batch_fused_tf.py (entry point)
  └─> tf.map_fn over particle index (K directions)
        └─> ledh_canonical_score_tf.py
              └─> tf.while_loop over time (Phase 1 ✓)
                    └─> Python range over substeps (Phase 2 TODO)
```

**Phase 1 restored the time while-loop correctly.** The XLA blocker is one level UP in the call stack: the K-direction map.

## Why This Matters

XLA can compile:
- `tf.while_loop` ✓
- Nested `tf.while_loop` ✓  
- Static tensor operations ✓

XLA **cannot** compile:
- `tf.map_fn` (uses TensorArray internally)
- `tf.TensorArray` (explicit or implicit)
- `tf.vectorized_map` / `pfor` (creates TensorArray for QR fallback)

## Elimination Strategy

Replace `tf.map_fn` with **manual stacking + `tf.while_loop`**.

### Current Code (ledh_canonical_batch_fused_tf.py:239-252)
```python
def evaluate_row(particle_index):
    theta_row = theta[particle_index]  # [P]
    directions_row = directions[particle_index]  # [K, P]
    value, score = canonical_value_and_analytical_score(
        model, theta_row, ..., with_score=True
    )
    return value, score

values, scores = tf.map_fn(
    evaluate_row,
    tf.range(count),
    parallel_iterations=1,
    dtype=(dtype, dtype),
)
```

### Proposed Replacement
```python
# Preallocate output tensors (XLA-compatible)
values = tf.TensorSpec(shape=[count], dtype=dtype)
scores = tf.TensorSpec(shape=[count, param_dim], dtype=dtype)

def row_loop_cond(idx, vals, scrs):
    return idx < count

def row_loop_body(idx, vals, scrs):
    theta_row = theta[idx]
    directions_row = directions[idx]
    value, score = canonical_value_and_analytical_score(
        model, theta_row, ..., with_score=True
    )
    vals = tf.tensor_scatter_nd_update(vals, [[idx]], [value])
    scrs = tf.tensor_scatter_nd_update(scrs, [[idx]], [score])
    return idx + 1, vals, scrs

_, values, scores = tf.while_loop(
    cond=row_loop_cond,
    body=row_loop_body,
    loop_vars=(
        tf.constant(0, tf.int32),
        tf.zeros([count], dtype=dtype),
        tf.zeros([count, param_dim], dtype=dtype),
    ),
    shape_invariants=(
        tf.TensorShape([]),
        tf.TensorShape([count]),
        tf.TensorShape([count, param_dim]),
    ),
    maximum_iterations=count,
    parallel_iterations=1,
)
```

**Key insight**: `tf.tensor_scatter_nd_update` is XLA-compatible; TensorArray is not.

## Phases

### Phase 1.5: Eliminate K-direction TensorArray

**Scope**: Replace `tf.map_fn` in `ledh_canonical_batch_fused_tf.py` with `tf.while_loop + scatter_update`

**Files**:
- `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` (modify lines 239-252)

**Success criteria**:
1. Parity: value and score relative error < 5e-4 vs current implementation
2. Graph mode: time unchanged or faster (baseline 2.465s small scale)
3. XLA mode: compiles successfully (no TensorListReserve error)
4. XLA mode: time < 1.0s small scale (target: approach 0.175s if fusion works)
5. Oracle contract: still passes (8/8 tests)

**Constraint**: K-direction loop is **NOT batched** — `count` is typically 50-500, not 16k+. Sequential iteration is acceptable; the win is XLA fusion of the inner canonical engine, not K-parallelism.

### Phase 2: Substep Loop (unchanged from original plan)

Convert the substep Python loop to `tf.while_loop`. This is independent of Phase 1.5 and still required.

### Phase 3: Measurement

Re-run execution mode matrix with:
- eager/sequential (baseline)
- graph/sequential (Phase 1 ✓)
- xla/sequential (Phase 1.5 target)
- Plan-scale measurement (N=252, T=50, substeps=8)

## Risks

1. **scatter_update overhead**: Each iteration writes one scalar/vector. At K=500, this is 500 scatter ops in the graph. XLA should fuse them, but if it doesn't, this could regress vs `tf.map_fn` in graph mode.

   **Mitigation**: Measure graph mode time before and after. If regression, investigate `tf.TensorArray` replacement with static preallocated tensor slicing.

2. **param_dim unknown at trace time**: The score shape `[count, param_dim]` requires knowing `param_dim` at graph build. If `theta.shape[1]` is dynamic, shape_invariants need `TensorShape([count, None])`.

   **Mitigation**: Check whether canonical engine requires static `param_dim`. If yes, add assertion; if no, use dynamic shape_invariants.

3. **Nested while-loop complexity**: `while_loop(time) inside while_loop(K)` creates a 2D iteration space. XLA should handle this, but graph size grows.

   **Mitigation**: If XLA compilation is slow (>60s), add explicit `maximum_iterations` bounds.

## Implementation Order

1. Update status document with Phase 1 results and XLA blocker analysis
2. Write Phase 1.5 parity harness (reuse execution matrix structure)
3. Implement scatter-based while-loop replacement
4. Run parity check (must pass before timing)
5. Run graph mode timing (must not regress)
6. Run XLA mode compilation and timing
7. Update status document with Phase 1.5 results
8. If successful: proceed to Phase 2 (substep loop)
9. If XLA still fails: investigate remaining TensorArray sources (none expected)

## Non-Goals (Deferred)

- Batching the K-direction loop (requires reshape + batch dimension propagation)
- Annealed telescoping support (requires Phase 2 + nested loop work)
- pfor revival (already rejected at 3.8× slower)

## Open Questions

1. Does `canonical_value_and_analytical_score` require `param_dim` statically known?
   - Check: `theta.shape` assertions in canonical_score_tf.py
   - If dynamic: use `TensorShape([count, None])` in shape_invariants

2. What is typical `count` (K) at plan scale?
   - Small scale fixture: K=5
   - Plan scale: likely 50-500 based on PFPF patterns
   - If K > 1000: scatter overhead may dominate

3. Should we batch K before or after XLA works?
   - After. Get XLA working first; batching is an independent optimization.

---

**Next action**: Update status document, then implement Phase 1.5 parity harness.
