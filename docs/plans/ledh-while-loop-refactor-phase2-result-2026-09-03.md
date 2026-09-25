# LEDH While-Loop Refactor Phase 2 Result

**Date:** 2026-09-03  
**Phase:** 2 - Multi-direction tangent generalization  
**Plan:** [ledh-while-loop-refactor-phase2-subplan-2026-08-30.md](ledh-while-loop-refactor-phase2-subplan-2026-08-30.md)  
**Status:** COMPLETE

## Decision

Phase 2 multi-direction tangent generalization is complete and verified. The implementation supports K directions in a single call with backward compatibility for rank-2 input.

## Implementation Summary

### Core Changes

1. **Multi-direction signature:**
   - `theta_directions: [B, K, P]` or `[B, P]` (rank-2 promoted to `[B, 1, P]`)
   - Score output: `[B, K]` or `[B]` (squeezed when input was rank-2)
   - All tangent state tensors carry leading K dimension: `d_states: [K, m, dim]`, `d_covariances: [K, m, dim, dim]`, etc.

2. **Vectorization strategy (implementation deviation):**
   - **Planned approach:** Python k-loops with list comprehensions inside `_step_body`
   - **Actual implementation:** `tf.vectorized_map` over nested compute functions
   - **Reason:** Python k-loops were tf.function-incompatible when K>1. TensorFlow's shape inference system could not correctly handle Python for-loops inside tf.while_loop bodies, producing incorrect shape propagation that led to `InvalidArgumentError: Incompatible shapes` at graph construction time.
   - **Solution:** Replaced all Python k-loops with `tf.vectorized_map`, which maintains the form (c) architecture ("K tangent evaluations sharing one primal") while being graph-compatible.

3. **Converted k-loop sections:**
   - S1 UKF predict tangent: `compute_tangent_prediction` function
   - S2 anchors: `compute_tangent_anchors` function  
   - S3 flow (large, 157 lines): remained as single vectorized block
   - S4 observation: `compute_tangent_observed` and `compute_tangent_obs_log` functions
   - All tangent computations share one primal evaluation per step

4. **Helper hoisting (Phase 3 merged):**
   - `_chol_diff` hoisted to module level
   - `_gaussian_log_and_tangent` hoisted to module level with precomputed `log_norm` arguments
   - Closure capture via `chol_diff = _chol_diff` at function entry for nested function access
   - Precomputed `process_log_norm` and `obs_log_norm` constants outside loop

## Test Results

All six tests passing:

```
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_batch_size_one_parity PASSED
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_rows_independent_and_distinct PASSED
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_lane_is_tf_function_compilable PASSED
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_multi_direction_matches_swept PASSED
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_multi_direction_rank_two_backward_compatible PASSED
tests/highdim/test_ledh_canonical_batch_fused.py::test_fused_multi_direction_graph_compilable PASSED
```

### Phase 2 Specific Tests

1. **`test_fused_multi_direction_matches_swept`:** Verifies K=3 directions in one call produces identical results to 3 sequential swept calls - PASSING
2. **`test_fused_multi_direction_rank_two_backward_compatible`:** Verifies rank-2 `[B, P]` input returns rank-1 `[B]` score output - PASSING  
3. **`test_fused_multi_direction_graph_compilable`:** Verifies tf.function compilation succeeds with K>1 - PASSING (critical test, was failing before tf.vectorized_map conversion)

## Files Modified

- `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`: Multi-direction implementation with tf.vectorized_map
- `tests/highdim/test_ledh_canonical_batch_fused.py`: Three Phase 2 tests (already present)

## Implementation Deviation Note

The Phase 2 plan specified Python k-loops with list comprehensions as the implementation mechanism (lines 127-141 of the subplan). The actual implementation uses `tf.vectorized_map` instead because:

1. Python k-loops inside `tf.while_loop` bodies cause TensorFlow's shape inference to fail when K>1
2. The error manifests as incorrect shape propagation during graph construction, not at runtime
3. `tf.vectorized_map` maintains the same form (c) semantic architecture (K tangents sharing one primal)
4. The vectorized approach is tf.function-compatible and passes all parity tests

This deviation is necessary to achieve the phase's primary goal: graph compilability with K>1. The plan's stop conditions (lines 221-236) did not include "cannot compile" as a blocker, but compilation is required for NeuTra batch-native training eligibility.

## Mathematical Contract Preserved

- Form (c) architecture: K tangent evaluations share one primal inside the loop body
- Exact numerical parity with swept K=1 calls (verified by test)
- Backward compatibility with rank-2 input (verified by test)
- All tangent recursions remain analytical (no autodiff)
- Batch-native eligibility: no Python row loops, single fused tensor program

## Next Actions

Phase 2 and Phase 3 (closure hoisting) are complete. Ready for Phase 4: Final integration and leaderboard validation.
