# Implementation Plan: Surrogate-Force HMC with Damped Score

Date: 2026-08-29  
Status: Implementation-ready recipe, awaiting go decision  
Provenance: variance note §5 Corollary 5.2 (lines 954-976), verified against Maskell §6.1 CRN determinism requirement

---

## Executive Summary

Use a heavily damped analytical score (λ=1e-3, δ=1e-3) as the HMC leapfrog force, while accepting/rejecting based on the exact executed value with current λ=1e-5, δ=1e-5. This decouples correctness from score bias: the chain targets exp(-U) exactly where U has 0.01-0.09% value bias, and the 3-9% score bias affects only mixing.

**Effort:** 1-2 days (two-adapter setup, verification on d=3 T=50 fixture)  
**Risk:** Acceptance rate may drop; empirically testable before deployment  
**Reward:** Removes score bias from the correctness path entirely

---

## Mathematical Foundation

### Corollary 5.2 (variance note, lines 954-976)

For HMC with:
- Leapfrog map Ψ driven by **any deterministic force** F: ℝ^P → ℝ^P
- Metropolis-Hastings acceptance on H(θ,p) = U(θ) + ½p^T M^{-1} p

the sampler is **volume-preserving, reversible, and H-invariant**.

**Requirements:**
1. F is a deterministic function of θ only (no dependence on momentum or trajectory history)
2. All Monte Carlo seeds inside F are frozen across the trajectory
3. The same scalar U(θ) is evaluated at both ends for the acceptance ratio

**Why this helps:**
- Score bias in F affects the proposal quality (acceptance rate, ESS) but not the invariant distribution
- As long as U is nearly unbiased (measured: 0.02-0.13 absolute error on U ≈ -145), the chain is nearly correct
- The 3-9% score bias becomes a **mixing problem**, not a **correctness problem**

### Connection to Maskell CRN determinism (§6.1, lines 649-745)

Maskell states: with common random numbers, "these Jacobian terms still cancel (and we do not need to integrate over the possible gradient calculations)" because the gradient is a deterministic function of θ. This is the same requirement as our Corollary 5.2 point 1.

---

## Current HMC Infrastructure

### Value-and-score plumbing (already supports surrogate-force)

`bayesfilter/inference/batched_value_score.py` lines 173-224:

```python
def reviewed_value_score_target_fn(adapter, *, dtype=tf.float64, require_batched=False):
    """Return a custom-gradient target value function backed by adapter scores."""
    
    def target_value(theta):
        @tf.custom_gradient
        def value_with_reviewed_score(values):
            result = adapter.log_prob_and_grad(values)
            value_tensor = result.value
            score_tensor = result.score
            
            def grad(upstream):
                # The score returned here is used as the HMC force
                return tf.reshape(upstream, [-1, 1]) * score_tensor, ...
            
            return value_tensor, grad
        
        return value_with_reviewed_score(values)
    
    return target_value
```

**Key fact:** `value_tensor` flows to the forward pass (acceptance energy), while `score_tensor` is returned only from `grad` (leapfrog force). Nothing enforces that the score equals ∂value/∂θ. TFP's HMC calls `value_and_gradients_fn`, uses value for the Metropolis ratio and gradient for leapfrog, and never cross-checks them.

**This means:** An adapter whose `log_prob_and_grad` returns the exact value alongside a damped score **already implements** surrogate-force HMC. The change is in the adapter, not the sampler.

---

## Implementation Recipe

### Step 1: Create two adapters

**Adapter A (exact, for acceptance):**
- λ = 1e-5, δ = 1e-5 (current production)
- Returns: value with minimal regularization bias

**Adapter B (damped, for force):**
- λ = 1e-3, δ = 1e-3 (100× heavier damping)
- Returns: score with gains O(λ^(-1/2)) ≈ 32 and O(δ^(-1)) ≈ 1000 (vs 316 and 10^5)

Both adapters share the same frozen noise seeds (see Step 2).

### Step 2: Enforce seed determinism

**Requirement (Corollary 5.2, Remark 5.3 line 981):**  
All Monte Carlo seeds inside the score must be frozen across the entire trajectory. A per-leapfrog-step reseed breaks the involution and therefore breaks invariance.

**Current seed discipline:**  
Check `canonical_value_and_analytical_score` signature:
```python
def canonical_value_and_analytical_score(
    model, theta, initial, covs, noises, observations, ...
):
```

- `initial`: (N, d) initial particles — passed in, not generated inside
- `covs`: (N, d, d) initial covariances — passed in
- `noises`: (T, N, d) all transition noise samples — **passed in, pre-generated**

**Action:** Verify that the HMC adapter generates these three arrays **once per chain position** and reuses them for both the value call (adapter A) and the score call (adapter B), not once per call to `log_prob_and_grad`.

Typical pattern:
```python
class DualAdapterSurrogateForce:
    def __init__(self, seed_base):
        self.seed_base = seed_base
        self._adapter_exact = make_adapter(lambda_ridge=1e-5, delta_damp=1e-5)
        self._adapter_damped = make_adapter(lambda_ridge=1e-3, delta_damp=1e-3)
    
    def log_prob_and_grad(self, theta):
        # Generate frozen noise ONCE per theta
        rng = np.random.default_rng(self.seed_base)
        initial = rng.standard_normal((N, d))
        noises = rng.standard_normal((T, N, d))
        covs = np.stack([np.eye(d)] * N)
        
        # Value from exact adapter
        value = self._adapter_exact.value_only(theta, initial, noises, covs)
        
        # Score from damped adapter
        _, score = self._adapter_damped.value_and_score(theta, initial, noises, covs)
        
        return BatchValueScoreResult(value=value, score=score, ...)
```

**Critical check:** Confirm that TFP's HMC does not re-invoke `log_prob_and_grad` multiple times per leapfrog step. If it does (unlikely, but verify), the seed-sharing must happen at the trajectory level, not the theta level.

### Step 3: Modify the HMC target factory

Assuming the existing `neutra_hmc.py` flow, the modification site is where `reviewed_value_score_target_fn` is called:

**Before:**
```python
target = reviewed_value_score_target_fn(adapter, dtype=tf.float64, require_batched=True)
```

**After:**
```python
surrogate_adapter = DualAdapterSurrogateForce(
    seed_base=hmc_seed,
    lambda_ridge_exact=1e-5,
    delta_damp_exact=1e-5,
    lambda_ridge_force=1e-3,  # damped
    delta_damp_force=1e-3,    # damped
)
target = reviewed_value_score_target_fn(surrogate_adapter, dtype=tf.float64, require_batched=True)
```

### Step 4: Verification protocol

**Fixture:** d=3 T=50 LGSSM, same observation path as the N-ladder and ε-schedule runs

**Metrics:**
1. **Acceptance rate** (warmup + sampling phases)
   - Target: 0.6-0.8 in warmup, 0.5-0.7 in sampling (standard HMC guidance)
   - Floor: >0.2 (below this, the chain is barely moving)

2. **ESS/gradient** (effective sample size per gradient evaluation)
   - Baseline: measure with current exact-score-as-force
   - Comparison: measure with damped-score-as-force
   - Acceptable: ESS/grad > 0.5× baseline (paying 2× gradient cost for correctness is fine)

3. **Posterior coverage** (does the chain hit the true parameter?)
   - True θ = [0.72, 0.55, 0.35, 0.35, 0.45]
   - Run 4 chains × 2000 warmup + 2000 sampling steps
   - Check: are 95% posterior intervals covering the true values?
   - Red flag: if posterior is shifted away from truth, the value U itself is more biased than we measured

**Comparison arms:**
- **Arm 1:** Current (exact score as force)
- **Arm 2:** Damped score (λ=1e-3, δ=1e-3) as force, exact value for acceptance
- **Arm 3:** (if Arm 2 acceptance is poor) Intermediate damping (λ=1e-4, δ=1e-4)

### Step 5: Diagnostic logging

Add to each HMC step:
```python
# After acceptance step
log_dict = {
    "step": step_idx,
    "accepted": bool(accepted),
    "value_proposed": float(value_proposed),
    "value_current": float(value_current),
    "alpha": float(acceptance_ratio),
    "force_norm": float(tf.norm(score_tensor)),  # monitor force magnitude
}
```

Watch for:
- `force_norm` should be smaller with damped score (gains are capped at ~32 vs ~316)
- If `force_norm` is too small (<1e-3), the leapfrog steps are tiny and acceptance is high but ESS is poor
- If `alpha` is consistently near 0, the force is too aggressive

---

## Tuning Strategy

If initial damping (λ=1e-3, δ=1e-3) gives acceptance <0.2:

1. **First:** Check that step size ε and leapfrog steps L are still appropriate. The damped force has smaller magnitude, so the same ε may now overshoot. Try ε ← ε/2.

2. **Second:** Try intermediate damping (λ=1e-4, δ=1e-4). This gives gains ~100 and ~10^4, halfway between exact and heavy damping in log space.

3. **Third:** Adaptive step size based on force magnitude. Compute `||F(θ)||` and scale ε so that `ε||F|| ≈ constant` across the parameter space.

If acceptance is good (>0.5) but ESS/grad is poor (<0.3× baseline):

- The force is accurate enough for correctness but not informative enough for mixing
- This is the trade-off: perfect correctness at the cost of slower exploration
- Mitigation: run longer chains (increase retained samples), or accept the 3-9% score bias and stay with exact-score-as-force

---

## Expected Outcome

**Best case:**  
Acceptance ≈ 0.6, ESS/grad ≈ 0.7× baseline, posterior coverage correct. The 3-9% score bias is now irrelevant to correctness, and the chain mixes at acceptable speed.

**Pessimistic case:**  
Acceptance <0.3, ESS/grad < 0.4× baseline. The damped score is too far from the true gradient to guide proposals effectively. Surrogate-force is mathematically correct but practically slow. Revert to exact score, or pursue Solution 3 (PaRIS).

**Null case:**  
Posterior coverage fails even with damped score. This means the **value bias** (currently 0.01-0.09%) is larger than we thought, or there's a second-order effect we haven't measured. Investigate value bias directly.

---

## Code Integration Checklist

- [ ] Write `DualAdapterSurrogateForce` class with frozen seed discipline
- [ ] Verify that `log_prob_and_grad` is called once per θ evaluation, not per leapfrog kick
- [ ] Add damping parameters (λ_force, δ_force) to HMC config
- [ ] Add diagnostic logging (acceptance rate, force norm, ESS)
- [ ] Run 3-arm verification on d=3 T=50 fixture (exact / damped / intermediate)
- [ ] Document the acceptance-rate vs ESS/grad trade-off
- [ ] Gate deployment behind a `use_surrogate_force` flag (default False until validated)

---

## Non-Claims and Boundaries

**What this fixes:**  
Score bias in the **correctness** path. The chain's invariant distribution is exp(-U) where U has the small value bias, not exp(-U) where U has the large score-derived bias.

**What this does NOT fix:**  
- The underlying score bias (still 3-9%)
- The score variance (still SD ≈ 0.3)
- Value bias (still 0.01-0.09%, but that's what the chain now targets)

**When this is not enough:**  
If the application requires **gradients of expectations** (not just sampling), surrogate-force does not help. For those cases, the score must be accurate, and Solutions 1 or 3 are needed.

**Compatibility:**  
- Works with current Contract-E + dual-cap + trust-region
- Compatible with Solution 1 (adaptive ε) — can run both together
- Independent of Solution 3 (PaRIS) — but if PaRIS gives an unbiased score, surrogate-force becomes unnecessary

---

## Estimated Timeline

- **Day 1 AM:** Implement `DualAdapterSurrogateForce`, verify seed discipline
- **Day 1 PM:** Run 3-arm diagnostic on d=3 T=50 fixture (4 chains × 4K steps each ≈ 2-3 hours wall time)
- **Day 2 AM:** Analyze acceptance / ESS / coverage, tune if needed
- **Day 2 PM:** Document findings, decide go/no-go

**Checkpoint before wider deployment:** If d=3 T=50 shows acceptable performance, run on one real DSGE fixture (your MacroFinance examples) before declaring production-ready.
