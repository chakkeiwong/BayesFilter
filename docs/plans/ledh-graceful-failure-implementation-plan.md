# LEDH Graceful Failure Mode Implementation Plan

## Objective

Implement graceful failure handling in LEDH canonical computation to allow HMC Metropolis-Hastings rejection of pathological parameter proposals, instead of crashing with Cholesky decomposition errors.

## Problem Statement

**Current behavior:**
1. HMC leapfrog explores parameter space
2. LEDH computation encounters ill-conditioned covariance matrix
3. Cholesky decomposition fails → InvalidArgumentError
4. HMC crashes before reaching MH accept/reject step

**Desired behavior:**
1. HMC leapfrog explores parameter space
2. LEDH computation detects ill-conditioned covariance matrix
3. Returns log π(θ) = -∞ with zero gradient
4. Leapfrog completes with H_new = +∞
5. MH rejects proposal with accept probability = 0
6. HMC continues from safe state

## Root Cause Analysis

### Cholesky Failure Locations

From code analysis, Cholesky decompositions occur at:

1. **`ledh_unified_reset_tf.py:195`**: Gap covariance factorization
   ```python
   gap = _sym(target_cov - plus_cov) + ridge_eye
   gap_chol = tf.linalg.cholesky(gap)  # Can fail
   ```

2. **`ledh_unified_reset_tf.py:224`**: Target covariance factorization
   ```python
   target_chol = tf.linalg.cholesky(target_cov + ridge_eye)  # Can fail
   ```

3. **`ledh_unified_reset_tf.py:226`**: Injected covariance factorization
   ```python
   injected_chol = tf.linalg.cholesky(injected_cov)  # Can fail
   ```

4. **`ledh_canonical_score_stages_tf.py:378`**: Sigma points generation
   ```python
   stabilized = 0.5 * (cov + cov^T) + jitter * I
   chol = tf.linalg.cholesky(scale_c * stabilized)  # Can fail
   ```

### Mathematical Requirement

For positive definiteness:
```
λ_min(Σ) > 0  for all eigenvalues λ
```

Under numerical errors:
```
λ_min(Σ_computed) = λ_min(Σ_exact) + O(ε * κ(Σ))
```

Where:
- ε ≈ 1e-16 (float64 machine epsilon)
- κ(Σ) = λ_max / λ_min (condition number)

Failure occurs when:
```
κ(Σ) > ridge / ε
→ For ridge = 1e-5: κ > 1e11 triggers failure
→ For ridge = 1e-3: κ > 1e13 triggers failure
```

## Implementation Strategy

### Design Principles

1. **Return sentinel values** instead of raising exceptions
2. **Preserve TensorFlow graph structure** (no Python control flow in tf.function)
3. **Propagate failure through tangent computations** correctly
4. **Minimize performance overhead** for normal (non-pathological) cases
5. **Maintain mathematical correctness**: ill-conditioned states have zero posterior probability

### Sentinel Value Convention

**For failed LEDH computation:**
- Primal value: `log π(θ) = -inf`
- Tangent/gradient: `∇_θ log π(θ) = 0` (arbitrary, will not be used after MH rejection)
- Score: `s(x|θ) = 0` (arbitrary)

### Implementation Approach

**Three-layer strategy:**

#### Layer 1: Safe Cholesky Wrapper (Innermost)

Create `safe_cholesky()` function that:
- Attempts Cholesky decomposition directly
- Detects NaN in result (TensorFlow returns NaN on failure)
- Returns `(success: bool, chol: Tensor)` tuple
- Minimal overhead: just NaN check, no eigenvalue decomposition

```python
def safe_cholesky(matrix: Tensor, name: str = "cholesky") -> tuple[Tensor, Tensor]:
    """
    Safe Cholesky decomposition with NaN detection.
    
    TensorFlow's tf.linalg.cholesky returns NaN when the matrix is not 
    positive definite. We detect this and return a validity flag.
    
    Returns:
        valid: bool tensor, True if decomposition succeeded (no NaN)
        chol: Cholesky factor if valid, zeros otherwise
    """
    # Attempt Cholesky - returns NaN on failure
    chol_attempt = tf.linalg.cholesky(matrix)
    
    # Check for NaN in result
    has_nan = tf.reduce_any(tf.math.is_nan(chol_attempt))
    is_valid = tf.logical_not(has_nan)
    
    # Return zeros when invalid (for clean propagation)
    chol = tf.where(
        is_valid,
        chol_attempt,
        tf.zeros_like(chol_attempt)
    )
    
    return is_valid, chol
```

**Rationale for NaN-detection over eigenvalue pre-check:**
- **Performance**: No expensive eigenvalue decomposition (O(n³) → O(1) check)
- **XLA-compatible**: Pure TensorFlow ops, no Python control flow or exceptions
- **Equally reliable**: Cholesky already produces NaN on ill-conditioned matrices
- **Observed in practice**: Phase 4a logs show Cholesky returning NaN, not crashing

#### Layer 2: Contract E Reset with Validity Flag (Middle)

Modify `batched_sinkhorn_contract_e_reset_triple_with_tangent()` to:
- Use `safe_cholesky()` for all decompositions
- Return `valid: bool` flag indicating success
- Return zeros for tangents when invalid

```python
def batched_sinkhorn_contract_e_reset_triple_with_tangent(
    ...
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Returns:
        particles, d_particles, carried_cov, d_carried_cov, transport, d_transport, valid
    """
    ...
    # Line 195: Gap Cholesky
    valid_gap, gap_chol = safe_cholesky(gap, "gap")
    
    # Line 224: Target Cholesky
    valid_target, target_chol = safe_cholesky(target_cov + ridge_eye, "target")
    
    # Line 226: Injected Cholesky
    valid_injected, injected_chol = safe_cholesky(injected_cov, "injected")
    
    # Combined validity
    valid = valid_gap & valid_target & valid_injected
    
    # When invalid, return zeros (will be masked at higher level)
    particles = tf.where(valid[:, None, None], particles_computed, tf.zeros_like(particles_computed))
    ...
    
    return particles, d_particles, carried_cov, d_carried_cov, transport, d_transport, valid
```

#### Layer 3: Canonical Score with -inf Propagation (Outermost)

Modify `canonical_score_parameter_jvp()` to:
- Check validity flag from reset policy
- Return -inf log probability when invalid
- Return zero gradient when invalid

```python
def canonical_score_parameter_jvp(...):
    ...
    # After reset policy (line 441-453):
    reset_states, d_reset_states, ..., reset_valid = sinkhorn_contract_e_reset_triple_with_tangent(...)
    
    # Check validity
    if not reset_valid:
        # Return sentinel values
        invalid_log_prob = tf.constant(-np.inf, dtype)
        invalid_score = tf.zeros_like(d_observations)
        return invalid_log_prob, invalid_score
    
    # Continue normal computation
    ...
```

### Tangent Computation Handling

**Challenge:** When primal fails, what should tangents be?

**Solution:** Use the **identity tangent rule**:
```
If f(θ) = -∞ for all θ in neighborhood, then df/dθ = 0
```

This is mathematically correct because:
- The function is constant (-∞) in the pathological region
- Zero gradient correctly signals "don't go this direction"
- MH will reject anyway, so gradient value doesn't matter

### XLA Compatibility

**Requirements:**
1. No Python exceptions in @tf.function
2. No Python control flow (if/else)
3. Use tf.cond() for conditionals
4. Use tf.where() for masking

**Implementation:**
```python
@tf.function
def canonical_batch_fused_value_score(...):
    # All operations are TensorFlow ops
    valid, log_prob, score = _compute_with_validity_check(...)
    
    # Use tf.where to mask invalid results
    log_prob = tf.where(valid, log_prob, tf.constant(-np.inf, dtype))
    score = tf.where(valid, score, tf.zeros_like(score))
    
    return log_prob, score, valid
```

## Implementation Steps

### Phase 1: Core Infrastructure (Estimated: 4 hours)

**Step 1.1:** Create `safe_cholesky()` wrapper
- File: `bayesfilter/highdim/ledh_numerical_safety_tf.py` (new)
- Function: `safe_cholesky(matrix, threshold=1e-12, name="cholesky")`
- Returns: `(valid: bool, chol: Tensor)`
- Test: Unit tests with well-conditioned, ill-conditioned, and indefinite matrices

**Step 1.2:** Add validity flag to reset policy
- File: `bayesfilter/highdim/ledh_unified_reset_tf.py`
- Modify: `batched_sinkhorn_contract_e_reset_triple_with_tangent()`
- Changes:
  - Replace 3 `tf.linalg.cholesky()` calls with `safe_cholesky()`
  - Add `valid` to return signature
  - Mask outputs with `tf.where()` when invalid
- Test: Unit tests with pathological covariances (κ > 1e12)

**Step 1.3:** Add validity flag to sigma points
- File: `bayesfilter/highdim/ledh_canonical_score_stages_tf.py`
- Modify: `_sigma_points_with_tangent()`
- Changes:
  - Replace `tf.linalg.cholesky()` with `safe_cholesky()`
  - Add `valid` to return signature
- Modify: `quadrature_predict_with_parameter_tangent()`
- Propagate validity through prediction step

### Phase 2: Canonical Score Integration (Estimated: 3 hours)

**Step 2.1:** Propagate validity through forward pass
- File: `bayesfilter/highdim/ledh_canonical_score_tf.py`
- Modify: `canonical_score_parameter_jvp()`
- Changes:
  - Collect validity flags from reset policy and sigma points
  - Combine with logical AND: `valid = reset_valid & predict_valid & update_valid`
  - Return -inf log probability when invalid
  - Return zero score when invalid

**Step 2.2:** Handle validity in batch fused computation
- File: `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`
- Modify: `canonical_batch_fused_value_score()`
- Changes:
  - Remove assertion at line 260 (direction check)
  - Add validity flag to return signature
  - Mask results with sentinel values when invalid

### Phase 3: Dual-Parameter Target Integration (Estimated: 2 hours)

**Step 3.1:** Handle -inf in custom gradient
- File: `bayesfilter/inference/ledh_dual_parameter_target.py`
- Modify: `target_with_surrogate_gradient()`
- Changes:
  - Check if exact_value is -inf
  - Return -inf with zero gradient when invalid
  - Ensure no NaN propagation

**Step 3.2:** Update target logging
- File: `bayesfilter/inference/ledh_dual_parameter_target.py`
- Add diagnostic logging:
  - Count of invalid evaluations
  - Parameter values that triggered invalidity
  - Fraction of HMC steps with invalid proposals

### Phase 4: Testing (Estimated: 4 hours)

**Step 4.1:** Unit tests for safe Cholesky
- Test well-conditioned matrices (κ < 1e6): should succeed
- Test ill-conditioned matrices (κ > 1e12): should fail gracefully
- Test indefinite matrices: should fail gracefully
- Test edge cases: singular, zero, negative definite

**Step 4.2:** Integration tests for LEDH
- Test pathological LGSSM parameters:
  - Tiny process noise → particle degeneracy
  - Extreme observation likelihood → weight collapse
  - Mismatched model parameters → divergent filter
- Verify -inf return value
- Verify zero gradient
- Verify no NaN propagation

**Step 4.3:** HMC integration test
- Create minimal HMC test with intentionally pathological target
- Verify leapfrog completes despite invalid intermediate states
- Verify MH rejects invalid proposals
- Verify chain continues from safe state

**Step 4.4:** Phase 4a re-run with graceful failure
- Revert regularization: ridge = 1e-5 (original value)
- Run Phase 4a diagnostic with graceful failure enabled
- Verify:
  - No crashes
  - Invalid proposals are rejected by MH
  - HMC explores and converges correctly
  - Diagnostic logs show rejection rate

### Phase 5: Documentation and Cleanup (Estimated: 2 hours)

**Step 5.1:** Code documentation
- Docstrings for all modified functions
- Comments explaining validity propagation
- Examples of pathological cases

**Step 5.2:** Update Phase 4a result document
- Document graceful failure implementation
- Compare with ridge-increase approach
- Show MH rejection statistics

**Step 5.3:** Update memory/notes
- Record numerical stability insights
- Document eigenvalue threshold choice
- Note performance implications

## Success Criteria

### Correctness
- [ ] No Cholesky decomposition crashes in HMC
- [ ] Pathological proposals return -inf log probability
- [ ] MH correctly rejects invalid proposals with probability 1
- [ ] HMC chain continues from safe state after rejection
- [ ] No NaN propagation in any computation path

### Performance
- [ ] Overhead < 5% for well-conditioned cases (normal operation)
- [ ] Eigenvalue check is vectorized and GPU-accelerated
- [ ] tf.function compilation succeeds with XLA

### Robustness
- [ ] Handles all condition numbers: κ ∈ [1, ∞]
- [ ] Handles indefinite matrices (negative eigenvalues)
- [ ] Handles singular matrices (zero eigenvalues)
- [ ] Handles edge cases (all-zero, identity, diagonal)

### Integration
- [ ] Phase 4a runs successfully with ridge = 1e-5
- [ ] HMC converges to correct posterior
- [ ] W₂ distance comparable to ridge = 1e-3 version
- [ ] Diagnostic logs show reasonable rejection rate (< 10%)

## Risk Assessment

### Risk 1: Performance Overhead (LOW - RESOLVED)
**Description:** Initially considered eigenvalue decomposition, but revised to NaN detection

**Resolution:**
- Use post-Cholesky NaN detection instead of pre-Cholesky eigenvalue check
- Overhead: O(n²) NaN scan vs O(n³) eigenvalue decomposition
- For n=10 state dimension: ~100 comparisons vs ~1000 FLOPS
- Expected overhead: < 1% for normal cases

### Risk 2: XLA Compilation (MEDIUM)
**Description:** tf.cond() and dynamic shapes may prevent XLA compilation

**Mitigation:**
- Use static shapes wherever possible
- Replace tf.cond() with tf.where() for arithmetic operations
- Test XLA compilation explicitly with `@tf.function(jit_compile=True)`
- Fallback: Disable XLA for this computation path if necessary

### Risk 3: Gradient Discontinuity (LOW)
**Description:** Zero gradient at boundary may affect HMC trajectory quality

**Mitigation:**
- This is mathematically correct behavior (posterior is zero in pathological region)
- MH rejection prevents acceptance of bad states
- Momentum reset after rejection helps escape
- Monitor ESS and acceptance rate to detect issues

### Risk 4: False Positives (LOW)
**Description:** Eigenvalue check may flag valid matrices as invalid

**Mitigation:**
- Use conservative threshold (1e-12 for float64)
- Add diagnostic logging of flagged matrices
- Manual inspection of flagged cases during testing
- Adjust threshold if false positives observed

## Testing Strategy

### Unit Test Matrix

| Component | Test Case | Expected Behavior |
|-----------|-----------|-------------------|
| safe_cholesky | Well-conditioned (κ < 1e6) | valid=True, correct factorization |
| safe_cholesky | Ill-conditioned (κ > 1e12) | valid=False, zero output |
| safe_cholesky | Indefinite (λ_min < 0) | valid=False, zero output |
| safe_cholesky | Singular (λ_min = 0) | valid=False, zero output |
| Reset policy | Normal particles | valid=True, correct moments |
| Reset policy | Degenerate weights | valid=False, zero particles |
| Reset policy | Ill-conditioned cov | valid=False, zero particles |
| Canonical score | Normal parameters | valid=True, finite log π |
| Canonical score | Pathological params | valid=False, log π = -inf |
| Dual target | exact valid, biased valid | Return exact value |
| Dual target | exact invalid | Return -inf, zero gradient |
| HMC | Invalid intermediate state | Reject via MH, continue |
| HMC | Valid trajectory | Accept via MH |

### Integration Test Scenarios

**Scenario 1: Tiny process noise**
```python
theta = [1.0, 1.0, 1.0, 0.001, 0.3]  # σ_q = 0.001
# Expect: Filter degeneracy → invalid → MH reject
```

**Scenario 2: Extreme observation noise**
```python
theta = [1.0, 1.0, 1.0, 0.5, 0.001]  # σ_r = 0.001
# Expect: Sharp likelihood → weight collapse → invalid → MH reject
```

**Scenario 3: Parameter sweep**
```python
for sigma_q in [1.0, 0.1, 0.01, 0.001, 0.0001]:
    log_prob, grad = target(theta)
    # Plot validity region in parameter space
```

**Scenario 4: Full HMC chain**
```python
# 2 chains × 1000 steps with minimal regularization (ridge=1e-5)
# Monitor: rejection rate, ESS, W₂ distance
# Compare: graceful failure vs strong regularization (ridge=1e-3)
```

## Performance Optimization (Post-Implementation)

### Optimization 1: Fast PD Check
Instead of full eigenvalue decomposition, use Cholesky attempt with try/catch:
```python
def safe_cholesky_fast(matrix):
    try:
        chol = tf.linalg.cholesky(matrix)
        return True, chol
    except:
        return False, tf.zeros_like(matrix)
```

Benchmark: eigenvalue check vs try/catch overhead

### Optimization 2: Early Exit
Add condition number check before expensive operations:
```python
def safe_cholesky_with_cond(matrix):
    # Fast upper bound: Frobenius norm / smallest diagonal
    cond_upper = tf.norm(matrix, 'fro') / tf.reduce_min(tf.linalg.diag_part(matrix))
    if cond_upper > 1e12:
        return False, tf.zeros_like(matrix)
    # Proceed with Cholesky
    ...
```

### Optimization 3: Validity Caching
Cache validity flag across HMC leapfrog steps for same θ:
```python
@tf.function
def cached_target(theta, cache_key):
    if cache_key in validity_cache:
        return cached_result
    result = compute_with_validity(theta)
    validity_cache[cache_key] = result
    return result
```

## Rollout Plan

### Stage 1: Development Branch
- Implement on `feature/ledh-graceful-failure` branch
- Complete all unit tests
- Verify integration tests pass

### Stage 2: Limited Testing
- Run Phase 4a with both approaches:
  - A: Graceful failure (ridge = 1e-5)
  - B: Strong regularization (ridge = 1e-3)
- Compare: W₂ distance, ESS, rejection rate, wall-clock time

### Stage 3: Validation
- If W₂(A) ≈ W₂(B) and no crashes: graceful failure is correct
- If rejection rate > 50%: investigate parameter prior or initialization
- If performance overhead > 10%: apply optimizations

### Stage 4: Merge
- Update Phase 4a result document
- Merge to `surrogate-hmc` branch
- Revert temporary ridge increase
- Proceed to Phase 4b with confidence

## Implementation Checklist

- [ ] Create `ledh_numerical_safety_tf.py` with `safe_cholesky()`
- [ ] Add unit tests for `safe_cholesky()`
- [ ] Modify `ledh_unified_reset_tf.py` reset policy
- [ ] Add unit tests for reset policy validity
- [ ] Modify `ledh_canonical_score_stages_tf.py` sigma points
- [ ] Modify `ledh_canonical_score_tf.py` canonical score
- [ ] Add integration tests for canonical score
- [ ] Modify `ledh_canonical_batch_fused_tf.py` batch computation
- [ ] Modify `ledh_dual_parameter_target.py` custom gradient
- [ ] Add HMC integration test
- [ ] Run Phase 4a with graceful failure
- [ ] Compare results: graceful vs regularization
- [ ] Document findings in Phase 4a result
- [ ] Merge and proceed to Phase 4b

## Timeline Estimate

- **Phase 1 (Core):** 4 hours
- **Phase 2 (Integration):** 3 hours
- **Phase 3 (Target):** 2 hours
- **Phase 4 (Testing):** 4 hours
- **Phase 5 (Docs):** 2 hours

**Total:** ~15 hours of development + testing time

**Wall-clock:** With interruptions and debugging, estimate 2-3 working days

## Approval and Next Steps

**This plan requires review and approval before implementation.**

Key questions for review:
1. Is the three-layer strategy (safe Cholesky → reset validity → canonical -inf) sound?
2. Is the eigenvalue threshold (1e-12) appropriate for float64?
3. Should we use eigenvalue check or try/catch for PD testing?
4. Are there additional failure modes not covered?
5. Is the testing strategy comprehensive enough?

**After approval, execute:**
```bash
git checkout -b feature/ledh-graceful-failure
# Implement Phase 1-5 per checklist
# Run tests and Phase 4a validation
# Review results and merge
```
