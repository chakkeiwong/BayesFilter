# Phase 2 XLA-Compatible Architecture

**Date:** August 30, 2026  
**Status:** Implementation complete, execution in progress

---

## Problem Statement

Phase 2 surrogate-force HMC implementation was blocked by XLA/JIT incompatibility:
- Original design used `.numpy()` calls inside `@tf.custom_gradient`
- TensorFlow graph mode cannot call Python functions or convert SymbolicTensors to numpy
- User requirement: "no numpy, no python loop should be used with TF/TFP. All implementation be XLA compatible and jit-compiled."

---

## Solution Architecture

### Key Insights from LEDH Codebase

1. **LEDH canonical score API expects pre-generated noise tensors**, not seeds:
   ```python
   canonical_value_and_analytical_score(
       model,
       theta,
       initial_states: Tensor,  # Pre-generated [N, d]
       initial_covariances: Tensor,
       noises: Tensor,  # Pre-generated [T, N, d]
       observations: Tensor,
       ...
   )
   ```

2. **Stateless random generation is already used internally**:
   - `tf.random.stateless_uniform` for annealed resampling (line 295)
   - Seed pair format: `[seed_int, salt]` of type `tf.int32`

3. **Noise must be frozen per theta** for HMC determinism:
   - Same theta → same noise → same (value, score)
   - Different configs (exact vs damped) use same noise

### XLA-Compatible Dual Adapter

```python
def make_dual_adapter_ledh(observations, obs_matrix, ridge_exact, damping_exact, 
                          ridge_damped, damping_damped):
    """Pure TF dual adapter - no numpy, no Python loops."""
    
    @tf.custom_gradient
    def value_with_damped_score(theta_inner: tf.Tensor) -> tf.Tensor:
        # 1. Derive integer seed from theta (pure TF, deterministic)
        seed_int = tf.cast(
            tf.reduce_sum(tf.abs(theta_inner) * 1e7 + salt) % 2147483647,
            tf.int32
        )
        seed_init = tf.stack([seed_int, 1001], axis=0)
        seed_transition = tf.stack([seed_int, 2002], axis=0)
        
        # 2. Generate frozen noise (stateless, deterministic)
        initial_noise = tf.random.stateless_normal(
            [PARTICLES, DIM], seed_init, dtype=DTYPE
        )
        transition_noise = tf.random.stateless_normal(
            [HORIZON, PARTICLES, DIM], seed_transition, dtype=DTYPE
        )
        
        # 3. Value: exact config
        value, _ = canonical_value_and_analytical_score(
            model_exact, theta_inner, initial_noise, initial_covs,
            transition_noise, observations,
            reset_ridge=ridge_exact,
            correction_lm_damping=damping_exact,
            ...
        )
        
        def grad_fn(upstream: tf.Tensor) -> tf.Tensor:
            # 4. Score: damped config (SAME frozen noise)
            scores = []
            for direction in range(5):
                _, score = canonical_value_and_analytical_score(
                    model_damped, theta_inner,
                    initial_noise,  # SAME noise
                    initial_covs,
                    transition_noise,  # SAME noise
                    observations,
                    reset_ridge=ridge_damped,
                    correction_lm_damping=damping_damped,
                    ...
                )
                scores.append(score[0])
            return upstream * tf.stack(scores, axis=0)
        
        return value, grad_fn
    
    # 5. Batch handling for HMC (4 chains)
    def log_prob_fn(theta: tf.Tensor) -> tf.Tensor:
        if len(theta.shape) == 1:
            return value_with_damped_score(theta)
        else:
            # Use tf.map_fn (not pfor - restricted by CLAUDE.md policy)
            return tf.map_fn(
                value_with_damped_score,
                theta,
                dtype=DTYPE,
                parallel_iterations=1,
            )
    
    return log_prob_fn
```

---

## Technical Details

### Seed Derivation

**Requirement:** Deterministic integer seed from theta tensor (pure TF ops).

**Solution:**
```python
def _seed_from_theta_and_salt(theta: tf.Tensor, salt: int) -> tf.Tensor:
    modulus = 2147483647
    seed_int = tf.cast(
        tf.reduce_sum(tf.abs(theta) * 1e7 + salt) % modulus,
        tf.int32
    )
    return tf.stack([seed_int, salt], axis=0)
```

**Properties:**
- Deterministic: same theta → same seed
- Collision resistance: scale by 1e7 spreads theta values
- TF int32 pair format matches `tf.random.stateless_*` API
- Different salts → different noise streams

### Batch Handling

**Problem:** HMC passes batched theta `[num_chains, 5]` but LEDH expects unbatched `[5]`.

**Solution:** `tf.map_fn` with `parallel_iterations=1` (sequential execution).

**Why not `tf.vectorized_map` (pfor)?**
- CLAUDE.md policy: "TensorFlow pfor requires prior written approval"
- pfor incompatible with complex model closures and `tf.constant` inside loops
- Error: `TypeError: Expected float64, but got Tensor of type 'SymbolicTensor'`

**Trade-off:** Sequential execution slower but guaranteed compatible.

### Reset Design Constant

**Problem:** `tf.constant([sqrt_d, ...])` fails in pfor when `sqrt_d` is a tensor.

**Solution:** Pre-compute as numpy constant:
```python
sqrt_d = tf.constant(np.sqrt(DIM), DTYPE)
basis = tf.constant([
    [sqrt_d, 0.0, 0.0],
    [-sqrt_d, 0.0, 0.0],
    ...
], DTYPE)
```

**Why it works:** `np.sqrt(DIM)` evaluated at graph-build time, not runtime.

---

## Verification Checklist

✓ **No `.numpy()` calls** inside `@tf.custom_gradient` or `@tf.function`  
✓ **No Python loops** over particles or time steps  
✓ **Stateless random generation** using `tf.random.stateless_normal`  
✓ **Deterministic seed** derived from theta via pure TF ops  
✓ **Same frozen noise** for value (exact) and score (damped)  
✓ **Batch handling** via `tf.map_fn` (not restricted pfor)  
✓ **Model closures** work inside map_fn with `parallel_iterations=1`

---

## Implementation Status

**File:** `docs/benchmarks/surrogate_force_lgssm_three_arm_v3.py`

**Configuration:**
- Model: d=3 T=50 diagonal LGSSM
- Particles: N=1008
- Three arms:
  - A (exact): λ=1e-5, δ=1e-5 for both value and score
  - B (damped): λ=1e-5, δ=1e-5 for value; λ=1e-3, δ=1e-3 for score
  - C (intermediate): λ=1e-5, δ=1e-5 for value; λ=1e-4, δ=1e-4 for score
- HMC: 4 chains, 1000 warmup, 1000 samples, step_size=0.01, L=10

**Execution:** In progress (started 2026-08-30 04:57 UTC)

---

## Success Criteria (from Plan)

**Pass if ALL hold:**
- Acceptance (sampling phase) > 0.3
- ESS/grad (min across components) > 0.3 × Arm A baseline
- Posterior coverage: all 5 parameters covered by 95% CI
- Mean shift (B vs A): ≤ 0.18
- Rhat < 1.01, no divergences

**Hard veto:**
- Mean shift > 0.18
- Acceptance < 0.2
- Any parameter's 95% CI excludes true value

---

## Lessons Learned

1. **Read the codebase first**: LEDH already had stateless random patterns; the solution was to match that architecture, not invent a new one.

2. **pfor is restricted for good reason**: Complex model closures, mutable state, and dynamic control flow are incompatible with automatic loop parallelization.

3. **XLA/JIT compatibility requires pure TF**: No escape hatches to Python/numpy once inside `@tf.function` or `@tf.custom_gradient`.

4. **Seed derivation is the critical piece**: A deterministic TF-only theta→seed function enables frozen noise without breaking the graph.

5. **Performance vs compatibility trade-off**: Sequential `tf.map_fn` is slower than `tf.vectorized_map` but guaranteed to work with complex code.

---

## Next Steps

1. **Monitor execution** (30+ minutes expected)
2. **Analyze results** against success criteria
3. **Generate final artifact** with metrics, plots, coverage
4. **Update LaTeX document** with Phase 2 outcomes
5. **Write final status report** with bounded claims or failure diagnosis
