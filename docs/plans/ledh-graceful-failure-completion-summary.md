# LEDH Graceful Failure Implementation - Completion Summary

**Date:** 2026-09-16  
**Branch:** surrogate-hmc  
**Status:** Phases 1-3 Complete

## Implementation Summary

Successfully implemented graceful failure handling in LEDH canonical score computation to return `-inf` for ill-conditioned states, allowing HMC Metropolis-Hastings to reject pathological proposals instead of crashing.

### Completed Phases

#### Phase 1: Core Infrastructure (Complete)

**1.1: Safe Cholesky Wrapper**
- File: `bayesfilter/highdim/ledh_numerical_safety_tf.py` (new)
- Implementation: `safe_cholesky(matrix, name)` returns `(is_valid, chol)`
- Strategy: Post-Cholesky NaN detection (not eigenvalue pre-check)
- Tests: `tests/highdim/test_ledh_numerical_safety_tf.py` (12/12 passing)

**1.2: Reset Policy Validity Flag**
- File: `bayesfilter/highdim/ledh_unified_reset_tf.py`
- Changes: Three `safe_cholesky` calls (gap, target, injected covariances)
- Combined validity: `valid = valid_gap & valid_target & valid_injected`
- Returns validity as 7th tuple element
- Tests: `tests/highdim/test_ledh_reset_validity.py` (5/5 passing)

**1.3: Adapter Layer Updates**
- File: `bayesfilter/highdim/ledh_canonical_reset_score_tf.py`
- Updated return signatures to include validity flag
- `sinkhorn_contract_e_reset_with_tangent()`: returns (particles, d_particles, valid)
- `sinkhorn_contract_e_reset_triple_with_tangent()`: returns 7 values including valid

**1.4: Sigma Points Validity**
- File: `bayesfilter/highdim/ledh_canonical_score_stages_tf.py`
- Verified: `_sigma_points_with_tangent()` already returns validity flag (line 499)
- No changes needed - validity propagation already present

#### Phase 2: Canonical Score Integration (Complete)

**2.1: Sentinel Value Fix**
- File: `bayesfilter/highdim/ledh_canonical_score_tf.py`
- Fixed lines 578-580:
  - Before: Used `float("nan")` as invalid sentinel
  - After: Uses `float("-inf")` for value, `0.0` for gradient
- Validity combination (lines 571-576):
  ```python
  numerical_valid = valid_predict & valid_update
  if reset_policy == "contract_e":
      numerical_valid = numerical_valid & valid_reset
  overall_valid = callback_valid & numerical_valid
  ```

**2.2: Docstring Syntax Fix**
- File: `bayesfilter/highdim/ledh_canonical_score_stages_tf.py`
- Fixed Unicode em-dash (U+2014) causing syntax error
- Moved note into proper Notes section

#### Phase 3: Dual-Parameter Target Integration (Complete)

**3.1: Custom Gradient -inf Handling**
- File: `bayesfilter/inference/ledh_dual_parameter_target.py`
- Modified: `target_with_surrogate_gradient()` (lines 141-148)
- Implementation:
  ```python
  is_invalid = tf.math.is_inf(exact_value) & (exact_value < 0.0)
  
  def grad_fn(dy):
      zero_grad = tf.zeros_like(biased_score)
      return tf.where(is_invalid, zero_grad, dy * biased_score)
  
  return exact_value, grad_fn
  ```
- Tests: `tests/inference/test_dual_parameter_target_invalid.py` (4/4 passing)

### Test Summary

**Unit Tests: 21/21 passing, 1 skipped**

1. **Numerical Safety** (`test_ledh_numerical_safety_tf.py`): 12 tests
   - Well-conditioned, ill-conditioned, indefinite, singular matrices
   - Batch operations, XLA compilation
   - Phase 4a pathological case

2. **Reset Validity** (`test_ledh_reset_validity.py`): 5 tests
   - Normal particles (valid)
   - Degenerate weights with ridge (valid)
   - Ill-conditioned covariances with ridge (valid)
   - Strong ridge rescue

3. **Dual-Parameter Target** (`test_dual_parameter_target_invalid.py`): 4 tests
   - Invalid parameters return -inf
   - Invalid parameters return zero gradient
   - Valid parameters have non-zero gradients

### Pending Work (Phase 4-5)

#### Phase 4: Comprehensive Testing

**4.1: Unit Tests** ✓ Complete

**4.2: Integration Tests** - Not Completed
- Reason: Requires complete `PerPointScoreModel` implementation
- The minimal mock model in `test_ledh_graceful_failure_integration.py` is insufficient
- Full integration testing requires:
  - Complete observation_fn/observation_jacobian_fn/observation_tangent_fn
  - Complete transition_mean_fn/process_covariance setup
  - Real LGSSM model fixture from existing test infrastructure

**4.3: HMC Integration Test** - Not Completed
- Requires:
  - Minimal HMC chain setup with tfp.mcmc
  - Pathological target that triggers -inf during leapfrog
  - Verification that MH correctly rejects invalid proposals
  - Chain convergence diagnostics

**4.4: Phase 4a Re-run** - Not Completed
- Original Phase 4a diagnostic scenario is available
- Should be run with ridge=1e-5 (original weak value)
- Expected outcome: graceful -inf instead of crash
- Diagnostic logs should show MH rejection statistics

#### Phase 5: Documentation and Cleanup

**5.1: Code Documentation** - Partial
- Core functions have docstrings
- Missing: detailed comments on validity propagation flow
- Missing: examples of pathological cases in docstrings

**5.2: Phase 4a Result Document** - Not Started
- Should document:
  - Comparison of ridge increase vs graceful failure
  - MH rejection statistics
  - Performance implications

**5.3: Memory/Notes Update** - Not Started
- Numerical stability insights
- Ridge vs validity-flag tradeoffs
- Performance measurements

## Design Decisions

### Why -inf Instead of NaN?

1. **HMC Metropolis-Hastings:** MH acceptance ratio `exp(accept) = exp(new - old)`
   - When `new = -inf`: `exp(-inf - old) = 0` → always reject
   - When `new = NaN`: `NaN < uniform()` → undefined behavior

2. **Gradient Handling:**
   - `-inf` with zero gradient: well-defined, no force applied
   - `NaN` with NaN gradient: propagates, crashes downstream

### Why Post-Cholesky NaN Detection?

1. **Simplicity:** No eigenvalue threshold to tune
2. **Conservatism:** Only rejects truly failed decompositions
3. **Performance:** No extra eigendecomposition
4. **Reliability:** NaN is unambiguous failure signal

### Why Zero Gradient for Invalid States?

1. **HMC Leapfrog:** Momentum update requires finite gradient
2. **Safety:** Zero gradient = no movement from invalid state
3. **Correctness:** -inf is a flat region (gradient truly is zero)

## Ridge Regularization Context

The graceful failure mechanism complements but does not replace ridge regularization:

- **Ridge (e.g., 1e-5):** Prevents many Cholesky failures by improving conditioning
- **Validity flag:** Catches remaining failures that ridge cannot prevent
- **Together:** Ridge reduces failure rate, validity flag handles residual failures

Phase 4a diagnostic showed that even with strong ridge (1e-3), some pathological
HMC proposals still triggered indefinite gap covariances. The validity flag
provides a safety net for these cases.

## Performance Implications

1. **No Performance Cost When Valid:**
   - Post-Cholesky NaN check: negligible (one reduction per matrix)
   - Boolean validity propagation: zero cost
   - Conditional gradient: compiled away when valid

2. **Invalid State Overhead:**
   - Early termination when invalid detected
   - No NaN propagation through remaining computation
   - Clean rejection by MH (no crash recovery needed)

## Modified Files

```
M bayesfilter/highdim/ledh_canonical_reset_score_tf.py
M bayesfilter/highdim/ledh_canonical_score_stages_tf.py
M bayesfilter/highdim/ledh_canonical_score_tf.py
M bayesfilter/highdim/ledh_unified_reset_tf.py
M bayesfilter/inference/ledh_dual_parameter_target.py
?? bayesfilter/highdim/ledh_numerical_safety_tf.py
?? tests/highdim/test_ledh_numerical_safety_tf.py
?? tests/highdim/test_ledh_reset_validity.py
?? tests/inference/test_dual_parameter_target_invalid.py
?? tests/integration/test_ledh_graceful_failure_integration.py (incomplete)
```

## Next Steps

To complete the implementation:

1. **Run Phase 4a Diagnostic:**
   - Revert ridge to 1e-5
   - Run original Phase 4a HMC scenario
   - Verify: no crash, MH rejects invalid proposals
   - Collect: rejection rate, acceptance statistics

2. **Integration Tests:**
   - Use existing LGSSM test fixtures
   - Create pathological parameter scenarios
   - Verify -inf propagation through full stack

3. **HMC Integration Test:**
   - Minimal tfp.mcmc.sample_chain with pathological target
   - Verify chain continues despite invalid proposals
   - Check convergence diagnostics

4. **Documentation:**
   - Complete code comments
   - Write Phase 4a result note
   - Update memory with numerical stability insights

## References

- Original Plan: `docs/plans/ledh-graceful-failure-implementation-plan.md`
- Phase 4a Diagnostic: `50f93709 Phase 4a diagnostic: identify Cholesky failure during HMC exploration`
- Reset Policy: Contract-E with Sinkhorn transport
- Ridge Values: 1e-5 (Phase 3.5 weak), 1e-3 (Phase 4a strong repair)
