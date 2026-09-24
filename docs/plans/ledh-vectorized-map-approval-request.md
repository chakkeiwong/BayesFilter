# Approval Request: `tf.vectorized_map` for LEDH K-direction batching

**Date:** 2026-09-13  
**Requester:** Claude (autonomous agent)  
**Scope:** `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`

## Summary

Request approval to use `tf.vectorized_map` to batch K independent directional score computations in the LEDH surrogate-force HMC adapter, replacing sequential `tf.map_fn(..., parallel_iterations=1)` loops.

## Current implementation

`canonical_batch_fused_value_score` accepts `theta [B, P]` and `directions [B, K, P]` and returns `scores [B, K]`. It evaluates K directions **sequentially** via nested `tf.map_fn`:

```python
def evaluate_row(theta_row, row_directions):  # [K, P]
    def evaluate_direction(direction):  # [P]
        bound_model = _single_cloud_model(model, theta_row, direction)
        value, score = canonical_value_and_analytical_score(...)
        return value, score[0]
    
    values, scores = tf.map_fn(
        evaluate_direction,
        row_directions,        # loop over K directions
        parallel_iterations=1  # SEQUENTIAL
    )
    return values[0], scores

values, scores = tf.map_fn(
    evaluate_row,
    (theta, directions),       # loop over B batch rows
    parallel_iterations=1      # SEQUENTIAL
)
```

**Cost:** 6 sequential filter passes (1 value + 5 score) at K=5, T=50 → **39.3s per HMC force evaluation** → 14.6 days for a 32k-call pilot.

## Proposed change

Replace the inner `tf.map_fn` over K directions with `tf.vectorized_map`:

```python
def evaluate_row(theta_row, row_directions):  # [K, P]
    def evaluate_direction(direction):  # [P]
        bound_model = _single_cloud_model(model, theta_row, direction)
        value, score = canonical_value_and_analytical_score(...)
        return value, score[0]
    
    values, scores = tf.vectorized_map(
        evaluate_direction,
        row_directions,        # batch over K directions
    )
    # Assert direction-invariance on values (existing check)
    return values[0], scores
```

**Expected speedup:** ~3-5× on the score computation (K=5 directions in parallel instead of sequential).

## Why `tf.while_loop` cannot satisfy the contract

The K directions are **independent** — each needs a full T×substeps LEDH filter pass with its own tangent propagation through the same primal trajectory. The single-cloud filter (`canonical_value_and_analytical_score`) is **not designed to accept batched tangents** — it takes:
- `d_states [N, d]` (one tangent)
- `d_covariances [N, d, d]` (one tangent)

To batch over K, we'd need:
- `d_states [K, N, d]`
- `d_covariances [K, N, d, d]`

And **every tangent operation** in the 1000-line single-cloud filter would need to broadcast over the leading K dimension. This is a major refactor of the canonical engine.

`tf.vectorized_map` achieves the same result by:
1. Tracing the function once
2. Automatically broadcasting operations over the leading dimension
3. Compiling to efficient batch ops

The alternative is to manually refactor the entire single-cloud filter to accept `[K, N, d]` tangents — that's weeks of work and high risk of introducing numerical bugs in a verified canonical engine.

## Expected graph/memory/compilation complexity

### Graph size
- Current: B × K separate graph traces (one per direction)
- With vectorized_map: B graph traces (one per batch row, vectorized over K)
- **Reduction:** ~K× fewer traces

### Host memory
- Negligible — no host-side Python loops

### Device memory
- Current: N particles × d dimensions × T timesteps (one filter pass at a time)
- With vectorized_map: K × N × d × T (K filter passes in parallel)
- **Increase:** K× device memory during tangent propagation
- At K=5, N=24, d=3, T=50: ~5 × 24 × 3 × 50 × 8 bytes ≈ 144 KB (negligible)

### Compilation cost
- Vectorized_map compiles once per batch row (same as current)
- No additional compilation overhead

## Bounded compatibility, numerical-equivalence, memory, and runtime checks

### Numerical equivalence
Run the following parity test at (T=5, N=24) and (T=50, N=252):

```python
# Sequential baseline (current)
values_seq, scores_seq = canonical_batch_fused_value_score(
    model, theta, directions, ..., use_vectorized_map=False
)

# Vectorized candidate
values_vec, scores_vec = canonical_batch_fused_value_score(
    model, theta, directions, ..., use_vectorized_map=True
)

# Check
assert_allclose(values_vec, values_seq, rtol=1e-14, atol=1e-14)
assert_allclose(scores_vec, scores_seq, rtol=1e-14, atol=1e-14)
```

**Pass criterion:** Relative difference < 1e-12 (float64 roundoff).

### Memory check
Instrument peak device memory at (T=50, N=252, K=5):

```python
before = tf.config.experimental.get_memory_info('GPU:0')['peak']
_ = canonical_batch_fused_value_score(...)
after = tf.config.experimental.get_memory_info('GPU:0')['peak']
peak_mb = (after - before) / 1e6
```

**Pass criterion:** Peak < 2 GB (plenty of headroom on 13 GB GPU).

### Runtime check
Measure wall time at (T=50, N=24, B=1, K=5) over 10 runs:

```python
t_seq = timeit(lambda: sequential_version(...), number=10) / 10
t_vec = timeit(lambda: vectorized_version(...), number=10) / 10
speedup = t_seq / t_vec
```

**Pass criterion:** Speedup > 2× (conservative; expect 3-5×).

### Compatibility check
Verify the adapter still works with:
- Graph mode (`tf.function`)
- XLA compilation (`jit_compile=True`)
- The dual-parameter HMC target wrapper

## Downstream scientific checks

1. **HMC acceptance rate:** Run 100 HMC steps with the vectorized adapter; check acceptance ∈ [0.5, 0.9] (same range as current).
2. **ESS:** Run 2 chains × 1000 draws; check ESS > 100 per chain (indicates no numerical degradation).
3. **Gradient contract:** Verify graph-mode gradient == analytical biased score (existing verification script).

## Why this is not a fallback to unreviewed eager or pfor

- `tf.vectorized_map` is **explicit** pfor over the K dimension only
- The base filter remains in graph mode with the same numerical program
- No autodiff fallback — the analytical score is still used
- No unreviewed path — the same canonical engine runs K times in parallel instead of sequentially

## Fallback plan if approval is denied

Manually batch the single-cloud filter by:
1. Refactoring all tangent variables to `[K, N, d]` shapes
2. Updating every linalg op to broadcast over the leading K dimension
3. Testing numerical parity at every stage

**Estimated effort:** 3-5 days of focused work, high risk of introducing bugs in the canonical engine.

`tf.vectorized_map` achieves the same result in a few hours with lower risk.

## Conclusion

Request approval to use `tf.vectorized_map` for K-direction batching in the LEDH adapter, with the numerical-equivalence, memory, runtime, and scientific checks defined above.

Expected outcome: **~3-5× speedup** on score computation → pilot cost 14.6 days → ~4-5 days, bringing the Phase 4a damping sweep from infeasible to overnight-scale.
