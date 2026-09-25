# LEDH Surrogate HMC: Batch Refactor Plan

**Date:** 2026-09-13  
**Status:** PROPOSAL — awaiting implementation  
**Blocking:** Phase 4a execution cost (14.6 days → target hours)

## Problem

`canonical_batch_fused_value_score` accepts `[B, K, P]` directions but evaluates them **sequentially** via nested `tf.map_fn`:
- Outer loop: B batch rows (sequential, `parallel_iterations=1`)
- Inner loop: K directions per row (sequential, `parallel_iterations=1`)

For K=5 directions, this means **6 sequential filter passes** (1 value + 5 score). Each pass is a full T×substeps LEDH filter (currently T=50, substeps=8).

**Measured cost:** 39.3s per (value, score) pair at plan scale → 14.6 days for the pilot.

## Observation

The B×K evaluations are **mathematically independent** — they share:
- Initial particles `[N, d]`
- Initial covariances `[N, d, d]`
- Noises `[T, N, d]`
- Observations `[T, obs_dim]`

But each (theta[b], direction[b,k]) pair produces an independent value and score.

## Proposed refactor

**Flatten the [B, K] grid into a [B*K] pseudo-batch:**

```python
# Input: theta [B, P], directions [B, K, P]
B = tf.shape(theta)[0]
K = directions.shape[1]  # statically known

# Expand theta: [B, P] → [B, K, P] → [B*K, P]
theta_expanded = tf.repeat(theta[:, None, :], K, axis=1)  # [B, K, P]
theta_flat = tf.reshape(theta_expanded, [B*K, P])         # [B*K, P]

# Flatten directions: [B, K, P] → [B*K, P]
directions_flat = tf.reshape(directions, [B*K, P])

# Build [B*K] single-cloud models (each binds one theta+direction pair)
# Then call the single-cloud filter B*K times — BUT use tf.vectorized_map
# or a single vmap-like call if we can batch the filter itself.
```

## Two implementation paths

### Path A: Batch `tf.map_fn` over [B*K] (simplest)

Replace the nested `tf.map_fn` with a single outer loop over `[B*K]`:

```python
def evaluate_one(theta_row, direction_row):
    bound_model = _single_cloud_model(model, theta_row, direction_row)
    value, score = canonical_value_and_analytical_score(
        bound_model, theta_row, initial_states, ...
    )
    return value, score[0]

flat_inputs = (theta_flat, directions_flat)
values_flat, scores_flat = tf.map_fn(
    evaluate_one,
    flat_inputs,
    parallel_iterations=1,  # or higher if safe
)

# Reshape: [B*K] → [B, K]
values = tf.reshape(values_flat[::K], [B])  # take first of each K-group
scores = tf.reshape(scores_flat, [B, K])
```

**Speedup:** Zero (still sequential) — but removes one level of indirection.

### Path B: Vectorize the single-cloud filter (target)

The single-cloud filter operates on:
- `[N, d]` particles (shared)
- One theta, one direction (broadcast via callbacks)

To batch over [B*K], we need the filter to accept:
- `[B*K, N, d]` particles (replicated B*K times)
- `[B*K, P]` theta rows
- `[B*K, P]` directions

But the filter's internal operations (UKF predict/update, Sinkhorn, correction) are **already batch-native over N** — they use `tf.linalg.matmul`, `tf.linalg.solve`, etc. on `[N, d, d]` shapes.

**The batching axis is orthogonal to the particle axis.** We can introduce a leading `[M]` dimension:
- `[M, N, d]` particles
- `[M, N, d, d]` covariances
- Each of M rows runs an independent filter

This requires:
1. Adding a leading `M` dimension to all filter state variables
2. Ensuring all linalg ops broadcast correctly over `[M, N, d, d]`
3. Model callbacks receive `[M, P]` theta_rows and broadcast over `[N, d]` points

**Speedup:** TensorFlow's batch matmul/solve should parallelize the M dimension on GPU → **~K× faster** (K=5 → ~5× from removing the direction loop, plus GPU parallelism over B).

## Recommendation

**Start with Path A** (flatten to [B*K] with single `tf.map_fn`) as a **refactoring checkpoint** — it proves the math is correct without performance gains, then **implement Path B** (true batch filter) on top.

Path B is the real win, but it touches the entire single-cloud filter stack. Path A derisks that by first proving the flattened [B*K] grid is mathematically correct.

## Expected performance

- Path A: **No speedup** (still sequential), but cleaner code
- Path B: **~5-10× speedup** (batch over B*K, GPU parallelism)
- Path B + XLA: **~10-20× speedup** (compile the batch kernel)

14.6 days → 0.73-1.46 days (Path B) → 0.4-0.7 days (Path B + XLA).

Still doesn't hit 4 GPU-hours, but brings the pilot from **infeasible to overnight-scale**.

## Next steps

1. Implement Path A (flatten [B, K] → [B*K])
2. Verify numerical parity against current sequential implementation
3. Implement Path B (batch the single-cloud filter)
4. Measure speedup at T=50, N=24
5. Evaluate XLA compilation

## Open questions

- Can the single-cloud filter's `tf.while_loop` (if any) handle a leading M dimension?
- Do Sinkhorn, correction, and pairwise stages batch cleanly over M?
- Does XLA compile the batched filter, or does it hit graph size limits?
