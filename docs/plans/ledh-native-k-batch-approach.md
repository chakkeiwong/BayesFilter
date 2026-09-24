# LEDH K-direction batching: Native batch approach (no pfor/vectorized_map)

**Date:** 2026-09-13  
**Alternative to:** `tf.vectorized_map` approval request

## Core insight

The per-point model callbacks currently **broadcast one theta row** over N particles:

```python
def transition_mean_fn(theta_rows, points):  # theta_rows [M, P], points [N, d]
    return model.transition_mean_fn(theta_rows, points)
```

Where `M = 1` (one theta row broadcast to all N particles).

To batch over K directions, we can use `M = K`:
- `theta_rows [K, P]` — same theta repeated K times
- `direction_rows [K, P]` — K different directions
- `points [N, d]` — shared particles (unchanged)

The callbacks broadcast `[K, P]` × `[N, d]` → `[K, N, ...]` naturally via TensorFlow's broadcasting rules.

## Proposed implementation

### Step 1: Extend `PerPointScoreModel` to accept K-batched theta/direction rows

No API change needed — the callbacks already accept `[M, P]` theta_rows. We just need to pass `M = K` instead of `M = 1`.

### Step 2: Create a K-batched single-cloud model

```python
def _k_batched_single_cloud_model(
    model: PerPointScoreModel,
    theta: Tensor,          # [P]
    directions: Tensor,     # [K, P]
) -> NonlinearScoreModel:
    """Bind one theta and K directions to the single-cloud callback contract."""
    
    K = tf.shape(directions)[0]
    
    def theta_rows(points: Tensor) -> Tensor:  # [N, d]
        # Broadcast theta [P] → [K, P]
        return tf.broadcast_to(theta[None, :], [K, tf.shape(theta)[0]])
    
    def direction_rows(points: Tensor) -> Tensor:  # [N, d]
        # Return directions [K, P] as-is
        return directions
    
    def transition_mean_fn(_theta: Tensor, points: Tensor) -> Tensor:
        # model callback: [K, P] × [N, d] → [K, N, d]
        return model.transition_mean_fn(theta_rows(points), points)
    
    def transition_mean_tangent_fn(
        _theta: Tensor, points: Tensor, d_points: Tensor
    ) -> Tensor:
        # d_points: [K, N, d] (K tangents)
        return model.transition_mean_tangent_fn(
            theta_rows(points), points, d_points, direction_rows(points)
        )
    
    # ... same for observation callbacks
    
    return NonlinearScoreModel(...)
```

### Step 3: Call the single-cloud filter with K-batched "particles"

The trick: **treat K directions as K independent particle clouds**, each with N particles.

```python
# Reshape particles [N, d] → [K, N, d] → [K*N, d]
# The filter sees K*N "particles", but they're actually K copies of N particles
K = directions.shape[1]
N = tf.shape(initial_states)[0]

batched_states = tf.tile(initial_states[None, :, :], [K, 1, 1])     # [K, N, d]
batched_states_flat = tf.reshape(batched_states, [K*N, d])          # [K*N, d]

batched_covs = tf.tile(initial_covariances[None, :, :, :], [K, 1, 1, 1])  # [K, N, d, d]
batched_covs_flat = tf.reshape(batched_covs, [K*N, d, d])                  # [K*N, d, d]

# Similarly for noises [T, N, d] → [T, K*N, d]
batched_noises = tf.tile(noises[:, None, :, :], [1, K, 1, 1])       # [T, K, N, d]
batched_noises_flat = tf.reshape(batched_noises, [T, K*N, d])       # [T, K*N, d]

# Call the single-cloud filter
bound_model = _k_batched_single_cloud_model(model, theta_row, row_directions)
value, score = canonical_value_and_analytical_score(
    bound_model,
    theta_row,
    batched_states_flat,      # [K*N, d]
    batched_covs_flat,        # [K*N, d, d]
    batched_noises_flat,      # [T, K*N, d]
    observations,             # [T, obs_dim] (shared)
    ...
)

# The filter computes K independent scores (one per K-group of N particles)
# But wait — the filter returns ONE scalar value and ONE scalar score...
```

**Problem:** The single-cloud filter returns **one scalar** value and score, not K scalars. It doesn't know about the K-batch structure.

## Revised approach: Extend the single-cloud filter to understand K-batched particles

This requires modifying `_value_and_analytical_score_impl` to:
1. Accept a `direction_count` parameter (default 1)
2. When `direction_count = K > 1`, interpret the "particles" as K groups of N
3. Return `[K]` values and scores instead of scalars

**Changes needed:**
- Weight normalization: `[K*N]` weights → `[K, N]` groups, normalize within each group
- Log-likelihood accumulation: sum over N within each K-group → `[K]` totals
- Tangent propagation: already handles `[K*N, d]` naturally

**Estimated effort:** 1-2 days to refactor the canonical filter + testing.

## Comparison with vectorized_map

| Approach | Speedup | Approval needed | Risk | Effort |
|---|---|---|---|---|
| vectorized_map | ~3-5× | Yes | Low | Hours |
| Native batch | ~3-5× | No | Medium | 1-2 days |
| Manual K-tangent | ~3-5× | No | High | 3-5 days |

## Recommendation

Given that `CLAUDE.md` requires prior written approval for `tf.vectorized_map`, and the native batch approach is feasible in 1-2 days, **implement the native batch approach first**.

If it works, we avoid the approval gate entirely. If it fails or is too complex, fall back to requesting vectorized_map approval.

## Next steps

1. Prototype the K-batched single-cloud model wrapper
2. Test whether the canonical filter handles `[K*N, d]` "particles" correctly
3. Identify where K-group structure needs to be exposed (weight normalization, likelihood accumulation)
4. Refactor those sections to accept a `direction_count` parameter
5. Verify numerical parity against sequential implementation
