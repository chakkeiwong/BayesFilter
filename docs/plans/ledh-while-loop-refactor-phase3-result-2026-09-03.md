# LEDH While-Loop Refactor Phase 3 Result

**Date:** 2026-09-03  
**Phase:** 3 - Hoisting and constant precomputation  
**Plan:** [ledh-while-loop-refactor-phase3-subplan-2026-08-30.md](ledh-while-loop-refactor-phase3-subplan-2026-08-30.md)  
**Status:** COMPLETE (merged into Phase 2 execution)

## Decision

Phase 3 hoisting and constant precomputation is complete. All work items addressed during Phase 2 implementation.

## Work Items Completed

### 3.1 Dead code removal

**Item:** `r_inv_obs = tf.linalg.matvec(r_inv, observation)` computed but never used.

**Action:** Verified this line does not exist in the current implementation. Grepped for `r_inv_obs` across the worktree - no occurrences found. The dead code was already removed or never existed in the Phase 1 implementation.

**Status:** N/A - not present in codebase

### 3.2 Closure hoisting

**Completed actions:**

1. **`_chol_diff` hoisting:**
   - Moved to module level as `_chol_diff(chol, d_matrix)`
   - Pure function with no captured variables
   - Accessible to nested functions via closure capture: `chol_diff = _chol_diff` at function entry
   - All call sites updated to reference the captured local name

2. **`_gaussian_log_and_tangent` hoisting:**
   - Moved to module level with explicit parameters
   - New signature: `_gaussian_log_and_tangent(points, d_points, means, d_means, chol, log_norm, m, k_count, dtype)`
   - Precomputed `log_norm` for fixed Cholesky factors:
     ```python
     process_log_norm = tf.cast(dim, dtype) * log_two_pi + 2.0 * tf.reduce_sum(
         tf.math.log(tf.linalg.diag_part(process_chol))
     )
     obs_log_norm = tf.cast(obs_dim, dtype) * log_two_pi + 2.0 * tf.reduce_sum(
         tf.math.log(tf.linalg.diag_part(obs_chol))
     )
     ```
   - Three call sites updated with precomputed `log_norm`:
     - Transition density: uses `process_log_norm`
     - Observation density: uses `obs_log_norm`
     - Proposal density: NOT hoisted (depends on loop-variant predicted covariance)

3. **`tf.broadcast_to` investigation:**
   - Left in place at all call sites
   - `triangular_solve` requires explicit batch dimension matching
   - Broadcast pattern `tf.broadcast_to(chol, [m, *chol.shape])` is necessary

### 3.3 Setup-block audit

**Setup constants (lines 165-194 in current implementation):**

| Variable | Status | Classification |
|----------|--------|----------------|
| `theta` | Correctly hoisted | Input tensor, cast once |
| `dtype` | Correctly hoisted | Derived from `initial_states`, used throughout |
| `directions_input` | Correctly hoisted | Input tensor, cast once |
| `chol_diff` | Correctly hoisted | Closure capture of module-level `_chol_diff` |
| `directions` | Correctly hoisted | Rank-promoted input |
| `squeeze_output` | Correctly hoisted | Backward compatibility flag |
| `k_count` | Correctly hoisted | Static shape constant |
| `batch`, `n`, `dim`, `m` | Correctly hoisted | Dimension constants |
| `mean_w`, `cov_w`, `scale` | Correctly hoisted | UKF weights from `_unscented_weights` |
| `eye` | Correctly hoisted | Identity matrix for stabilization |
| `eps` | Correctly hoisted | Substep size constant |
| `process_chol` | Correctly hoisted | Cholesky of process covariance |
| `obs_dim` | Correctly hoisted | Observation dimension |
| `obs_chol` | Correctly hoisted | Cholesky of observation covariance |
| `r_inv` | Correctly hoisted | Inverse observation covariance |
| `horizon` | Correctly hoisted | Number of timesteps |
| `log_two_pi` | Correctly hoisted | Mathematical constant |
| `process_log_norm` | Correctly hoisted | **NEW: Precomputed log normalization** |
| `obs_log_norm` | Correctly hoisted | **NEW: Precomputed log normalization** |
| `theta_flat` | Correctly hoisted | Tiled theta for batch×N points |
| `dtheta_flat` | Correctly hoisted | Tiled directions for batch×N points |
| `states` | Correctly hoisted | Initial state |
| `d_states` | Correctly hoisted | Initial tangent state (zeros) |
| `covariances` | Correctly hoisted | Initial covariance |
| `d_covariances` | Correctly hoisted | Initial tangent covariance (zeros) |
| `total` | Correctly hoisted | Initial log-likelihood accumulator (zeros) |
| `d_total` | Correctly hoisted | Initial tangent accumulator (zeros) |

**Analysis:** All setup constants are correctly hoisted. No further hoisting opportunities identified. The two new precomputed log normalizations (`process_log_norm`, `obs_log_norm`) eliminate redundant computation inside the loop.

## Test Results

All six tests passing after Phase 3 changes:

```
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_batch_size_one_parity PASSED
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_rows_independent_and_distinct PASSED
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_lane_is_tf_function_compilable PASSED
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_multi_direction_matches_swept PASSED
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_multi_direction_rank_two_backward_compatible PASSED
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_multi_direction_graph_compilable PASSED
```

No parity degradation observed beyond expected float64 reassociation noise.

## Files Modified

- `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`: Hoisted helpers and precomputed constants
- Same file as Phase 2 changes (merged execution)

## Performance Impact

**Expected benefits:**
- Eliminated redundant `log_norm` computation at 3 call sites × T timesteps
- Eliminated redundant log/sum operations over Cholesky diagonal at every transition and observation density evaluation
- Module-level helpers enable potential compiler optimizations

**Not measured:** Benchmark timing deferred to Phase 4 final validation.

## Mathematical Contract Preserved

- All hoisted computations are mathematically invariant
- No arithmetic changes on live values
- Parity tests verify numerical equivalence

## Integration with Phase 2

Phase 3 was executed concurrently with Phase 2 debugging. The hoisting changes were necessary to fix the `tf.vectorized_map` closure capture issue (nested functions need access to module-level helpers). The merge was natural and reduced total execution risk.

## Next Actions

Phase 3 complete. Ready for Phase 4: Final integration and leaderboard validation.
