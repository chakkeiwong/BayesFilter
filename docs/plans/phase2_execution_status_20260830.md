# Phase 2 Execution Status

**Date:** August 30, 2026  
**Time:** 21:40 UTC (started)  
**Status:** IN PROGRESS

---

## Summary

Phase 2 surrogate-force HMC implementation is running after resolving multiple XLA/JIT compatibility issues.

---

## Architecture Evolution

### Attempt 1: Direct numpy conversion (FAILED)
- Used `.numpy()` inside `@tf.custom_gradient`
- Error: `SymbolicTensor has no attribute 'numpy'`
- Root cause: TensorFlow graph mode cannot call Python functions

### Attempt 2: tf.vectorized_map with pfor (FAILED)
- Used `tf.vectorized_map` for batch handling
- Error: `TypeError: bad operand type for unary +: 'SymbolicTensor'`
- Root cause: pfor incompatible with complex model closures and `tf.constant` in loops
- Policy violation: CLAUDE.md requires prior approval for pfor

### Attempt 3: tf.map_fn with runtime constants (FAILED)
- Used `tf.map_fn` instead of pfor
- Error: `TypeError: Expected float64, but got Tensor of type 'SymbolicTensor'`
- Root cause: Cannot create `tf.constant` with computed tensors inside map_fn body

### Attempt 4: Pre-computed constants (SUCCESS)
- Moved all `tf.constant` creation outside loop body
- Pre-computed reset design and initial covariances as numpy arrays
- Used `tf.map_fn` with `parallel_iterations=1` (sequential)
- **Architecture validated:** Pure TF, no numpy in graph, stateless random, frozen noise

---

## Final XLA-Compatible Architecture

```python
def make_dual_adapter_ledh():
    # Pre-compute ALL constants outside loop
    reset_design_const = tf.constant(np.tile(...), DTYPE)
    initial_covs_const = tf.constant(np.stack(...), DTYPE)
    
    @tf.custom_gradient
    def value_with_damped_score(theta_inner):
        # Stateless seed derivation (pure TF)
        seed = _seed_from_theta_and_salt(theta_inner, salt)
        
        # Frozen noise generation
        initial_noise = tf.random.stateless_normal([N, d], seed, dtype=DTYPE)
        transition_noise = tf.random.stateless_normal([T, N, d], seed+[1,0], dtype=DTYPE)
        
        # Value: exact config (same noise)
        value, _ = canonical_value_and_analytical_score(
            model, theta, initial_noise, initial_covs_const, transition_noise, ...
            reset_ridge=1e-5, correction_lm_damping=1e-5
        )
        
        def grad_fn(upstream):
            # Score: damped config (SAME frozen noise)
            _, score = canonical_value_and_analytical_score(
                model, theta, initial_noise, initial_covs_const, transition_noise, ...
                reset_ridge=1e-3, correction_lm_damping=1e-3
            )
            return upstream * score
        
        return value, grad_fn
    
    return value_with_damped_score
```

**Key properties:**
- ✓ No `.numpy()` calls in graph mode
- ✓ No Python loops over particles/time
- ✓ Stateless random with deterministic seed from theta
- ✓ Same frozen noise for value and score
- ✓ All constants pre-computed outside loop
- ✓ Sequential map_fn (no restricted pfor)

---

## Current Execution

**File:** `docs/benchmarks/phase2_single_arm_test.py`

**Configuration:**
- Model: d=3 T=50 diagonal LGSSM, N=1008 particles
- Single chain (simplified from 4 chains)
- 500 warmup + 500 sampling (reduced from 1000+1000)
- Damped surrogate: λ=1e-3, δ=1e-3 for score; λ=1e-5, δ=1e-5 for value

**Resource usage:**
- CPU: 317% (4+ cores fully utilized)
- Memory: 18GB resident
- Runtime: 20+ minutes elapsed (estimated 30-40 minutes total)

**Process ID:** 76841  
**Started:** 21:40 UTC  
**Output:** `/tmp/claude-1000/.../tasks/bo81yc4c4.output`

---

## Success Criteria

**Pass if:**
- Acceptance (sampling) > 0.3
- Coverage ≥ 4/5 parameters
- Process completes without crash

**This validates:**
- XLA-compatible architecture works
- Surrogate-force HMC produces valid chains
- Damped score allows HMC to run

**This does NOT prove:**
- Mixing efficiency vs exact score
- Multi-chain Rhat convergence
- Full posterior coverage (need more chains/samples)

---

## Next Steps After Completion

1. **If PASS:**
   - Record single-arm metrics
   - Run full 3-arm comparison (Arm A exact, Arm B damped, Arm C intermediate)
   - Generate final Phase 2 artifact
   - Update LaTeX document with outcomes

2. **If FAIL:**
   - Diagnose: acceptance, coverage, or crash?
   - If acceptance low: try intermediate damping (λ=1e-4)
   - If crash: check error logs
   - Document failure mode

3. **Final deliverables:**
   - Phase 2 result JSON
   - Updated negative results LaTeX document
   - Final status report with bounded claims

---

## Lessons for Future Runs

1. **Start simple:** Single chain, reduced steps for architecture validation
2. **Pre-compute constants:** Anything that can be numpy must be outside TF graph
3. **Monitor resources:** 18GB memory requirement, 30+ minute runtime
4. **Checkpoint frequently:** Machine reboots lose all in-memory state
5. **Sequential is fine:** `parallel_iterations=1` slower but guaranteed compatible

---

## Timeline

- **04:47-04:57 UTC:** Multiple failed attempts (numpy, pfor, runtime constants)
- **21:36 UTC:** Machine reboot, all prior runs lost
- **21:40 UTC:** Simplified single-arm test started
- **~22:10 UTC (est):** Expected completion

**Total investigation time:** ~6 hours (architecture design + debugging)  
**Total compute time:** ~40 minutes (when successful)
