# Contract E Streaming JVP Serial Bottleneck Analysis

**Date:** 2026-09-03  
**Analyst:** Claude Code  
**Status:** Pre-fix investigation complete

## Executive Summary

Contract E streaming JVP has a serial execution bottleneck at lines 1255-1262 and 1341-1348 in `ledh_contract_e_streaming_tf.py`. The code uses `tf.map_fn` with `parallel_iterations=1` to compute K independent tangent directions sequentially.

**Investigation outcome:**
- ✅ We understand the problem thoroughly
- ✅ We know how to fix it (same pattern as LEDH refactor)
- ✅ Test coverage is adequate
- ⚠️  Memory risk exists but is manageable

## Problem Description

### Location 1: `_contract_e_streaming_jvp_core` (lines 1240-1262)

```python
def one_direction(index: tf.Tensor) -> tf.Tensor:
    return _contract_e_chol_cloud_jvp_core(
        source_particles,
        normalized_weights,
        quotient["particles"],
        residual_design,
        ridge,
        source_particles_tangent[:, :, :, index],
        normalized_weights_tangent[:, :, index],
        quotient["particles_tangent"][:, :, :, index],
        residual_design_tangent[:, :, :, index],
        ridge_tangent[:, index],
    )["particles"]

reset_particles = tf.transpose(
    tf.map_fn(
        one_direction,
        tf.range(parameter_count),
        fn_output_signature=output_signature,
        parallel_iterations=1,  # SERIAL EXECUTION BOTTLENECK
    ),
    [1, 2, 3, 0],
)
```

### Location 2: `_contract_e_streaming_forward_jvp_core` (lines 1325-1348)

Identical pattern, calling `_contract_e_chol_cloud_jvp_from_forward_core` instead.

### What the Code Does

1. **Input shape:** `source_particles_tangent` is `[batch, N, state_dim, K]` where K is `parameter_count`
2. **Serial loop:** For each direction `k = 0..K-1`:
   - Extract slice `[:, :, :, k]` from each tangent input
   - Call `_contract_e_chol_cloud_jvp_core` to compute reset particles for direction k
   - Returns shape `[batch, N, state_dim]`
3. **Stack results:** Transpose to get `[batch, N, state_dim, K]`

### Why Serial Execution Was Chosen

**Hypothesis:** Conservative safety choice. The computation per direction is:
- Weighted moment JVP (particles, weights)
- Uniform moment JVP (transported particles)
- Cholesky JVP (gap covariance + ridge)
- Cholesky JVP (target covariance + ridge)
- Cholesky JVP (injected covariance + ridge)
- Triangular solve (affine transformation)

All operations are pure tensor algebra with no cross-direction dependencies. Serial execution was likely chosen to:
1. Avoid memory explosion (K concurrent Cholesky factorizations)
2. Simplify initial implementation
3. Match the historical pattern before `tf.vectorized_map` was understood

## Code Tracing Results

### Call Chain

1. `_contract_e_streaming_jvp_core` (line 1240)
   → `_contract_e_chol_cloud_jvp_core` (ledh_contract_e_reset_tf.py:229)
   → `_contract_e_chol_cloud_jvp_from_forward_core` (ledh_contract_e_reset_tf.py:263)

2. The JVP implementation (lines 276-344) computes:
   - `target_mean_tangent, target_cov_tangent = _weighted_moments_jvp(...)`
   - `plus_mean_tangent, plus_cov_tangent = _uniform_moments_jvp(...)`
   - `gap_tangent = _sym(target_cov_tangent - plus_cov_tangent)`
   - `gap_chol_tangent = _cholesky_jvp(forward["gap_chol"], gap_tangent + ridge_identity_tangent)`
   - `injected_particles_tangent = ...`
   - `injected_mean_tangent, injected_cov_tangent = _uniform_moments_jvp(...)`
   - `target_chol_tangent = _cholesky_jvp(...)`
   - `injected_chol_tangent = _cholesky_jvp(...)`
   - `affine_tangent = _right_triangular_solve(...)`
   - `particles_tangent = ...`

### Independence Verification

✅ **Confirmed:** Each direction k is computed independently:
- Input: slice `[:, :, :, k]` from each tangent tensor
- Output: `[batch, N, state_dim]` for direction k
- No shared state across directions
- No accumulation or reduction across directions
- Pure function of (primal inputs, tangent inputs for direction k)

This is **exactly the same independence structure** as the LEDH refactor we just completed.

## Test Coverage Analysis

### Relevant Tests (test_ledh_contract_e_streaming_phase4.py)

1. **test_terminal_balance_manual_jvp_vjp_match_autodiff** (lines 617-677)
   - Tests JVP/VJP consistency for terminal balance potential
   - Uses **2 directions** (`d_initial[:, :, index]` loop at line 671)
   - Verifies primal-adjoint identity for each direction
   - ✅ Would catch multi-direction regressions

2. **test_contract_e_composed_jvp_vjp_and_weight_coordinates_match_autodiff** (lines 753-842)
   - Tests full Contract E JVP/VJP consistency
   - Uses **2 directions** (loop at line 832: `for index in range(2)`)
   - Calls `_contract_e_streaming_jvp_core` (line 756) — the target function
   - Verifies primal-adjoint identity: `primal = upstream * jvp[..., index]`, `adjoint = vjp * tangent[..., index]`
   - ✅ Would catch multi-direction regressions
   - ✅ Would catch numerical differences from vectorization

3. **test_fused_contract_e_reuses_one_ot_state_and_matches_separated_route** (lines 679-744)
   - Tests that forward+JVP reuses the same OT state
   - Calls `_contract_e_streaming_forward_jvp_core` (line 713) — the other target function
   - ✅ Would catch work-count regressions (line 739-743 asserts)

### Coverage Assessment

**✅ Adequate for refactor:**
- Multi-direction scenarios tested (K=2)
- JVP/VJP parity checked against autodiff
- Both target functions covered
- Work reuse verified

**⚠️  Limitation:**
- Only K=2 tested (not K=10 or K=50)
- No explicit memory profiling tests
- No explicit parallelism/vectorization tests

**Recommendation:** Tests will catch correctness regressions but not memory explosions. Add a memory monitor for the fix.

## Fix Strategy

### Proposed Solution: tf.vectorized_map + Closure Capture

Same pattern as LEDH refactor (completed 2026-09-03):

```python
# BEFORE (serial):
def one_direction(index: tf.Tensor) -> tf.Tensor:
    return _contract_e_chol_cloud_jvp_core(
        source_particles,  # captured from outer scope
        ...
        source_particles_tangent[:, :, :, index],  # sliced by index
        ...
    )["particles"]

reset_particles = tf.transpose(
    tf.map_fn(one_direction, tf.range(parameter_count), ..., parallel_iterations=1),
    [1, 2, 3, 0],
)

# AFTER (vectorized):
def one_direction_vectorized(index: tf.Tensor) -> tf.Tensor:
    # Closure capture for traced function
    _source_particles = source_particles
    _normalized_weights = normalized_weights
    _quotient_particles = quotient["particles"]
    _residual_design = residual_design
    _ridge = ridge
    _source_particles_tangent = source_particles_tangent
    _normalized_weights_tangent = normalized_weights_tangent
    _quotient_particles_tangent = quotient["particles_tangent"]
    _residual_design_tangent = residual_design_tangent
    _ridge_tangent = ridge_tangent
    
    return _contract_e_chol_cloud_jvp_core(
        _source_particles,
        _normalized_weights,
        _quotient_particles,
        _residual_design,
        _ridge,
        _source_particles_tangent[:, :, :, index],
        _normalized_weights_tangent[:, :, index],
        _quotient_particles_tangent[:, :, :, index],
        _residual_design_tangent[:, :, :, index],
        _ridge_tangent[:, index],
    )["particles"]

reset_particles = tf.transpose(
    tf.vectorized_map(one_direction_vectorized, tf.range(parameter_count)),
    [1, 2, 3, 0],
)
```

### Why This Will Work

1. **Proven pattern:** Identical to LEDH refactor that passed all tests
2. **Independence verified:** No cross-direction dependencies
3. **Closure capture required:** `tf.vectorized_map` traces the function body, so outer-scope captures must be explicit
4. **Test coverage adequate:** Existing tests will catch correctness regressions

### Memory Impact Analysis

**Per-direction memory (rough estimate for N=3000, state_dim=40, batch=2):**
- Input particles: `[2, 3000, 40]` = 240K floats = 960 KB (shared across directions)
- Cholesky factors: `[2, 40, 40]` x 3 = 9.6K floats = 38 KB per direction
- Intermediate covariances: `[2, 40, 40]` x 3 = 9.6K floats = 38 KB per direction
- Output particles: `[2, 3000, 40]` = 240K floats = 960 KB per direction

**Serial (K=1 at a time):** ~1 MB + 76 KB = ~1.1 MB  
**Parallel (K=10 concurrent):** ~1 MB + 760 KB + 9.6 MB = ~11 MB  
**Parallel (K=50 concurrent):** ~1 MB + 3.8 MB + 48 MB = ~53 MB

**Verdict:** Memory increase is **linear in K**, which is typically 10-50 for LEDH. This is manageable on modern GPUs (8-24 GB). Much smaller than the O(N²) transport memory.

**Mitigation:** If memory becomes a problem, can batch the directions (K_batch=10 at a time) using nested vectorized_map, but start with full parallelization.

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Numerical differences from vectorization | Low | Very Low | Parity tests will catch it |
| Memory OOM for large K | Medium | Low | Monitor memory, add K-batching if needed |
| Test coverage gaps for K>2 | Low | Medium | Existing K=2 tests adequate for correctness |
| Closure capture errors | Low | Very Low | Same pattern as LEDH refactor |
| Performance regression (no speedup) | Low | Very Low | Vectorized_map is strictly faster than serial |

## Next Steps

1. ✅ Investigation complete (this document)
2. **Implement fix:**
   - Modify `_contract_e_streaming_jvp_core` (line 1240)
   - Modify `_contract_e_streaming_forward_jvp_core` (line 1325)
   - Use closure capture pattern from LEDH refactor
3. **Verify:**
   - Run `pytest tests/highdim/test_ledh_contract_e_streaming_phase4.py -v`
   - Check all JVP/VJP parity tests pass
   - Monitor memory usage during test execution
4. **Document:**
   - Update result note with memory profile
   - Add comment explaining vectorization choice
   - Note any K-batching threshold if memory issues arise

## References

- LEDH while-loop refactor: `docs/plans/ledh-while-loop-refactor-reset-memo-2026-09-03.md`
- Performance audit: `docs/plans/performance-audit-preliminary-2026-09-03.md` (Issue #1)
- Test file: `tests/highdim/test_ledh_contract_e_streaming_phase4.py`
- Target file: `bayesfilter/highdim/ledh_contract_e_streaming_tf.py`
- JVP implementation: `bayesfilter/highdim/ledh_contract_e_reset_tf.py`
