# Epsilon Schedule Diagnostic: Negative Result

Date: 2026-08-29  
Status: **Completed (negative result) — do not adopt adaptive ε**  
Fixture: d=3 T=50 LGSSM, N=1008, 16 seeds, exact score φ_1 = -6.2167  
Hypothesis: Corenflos Prop 3.3 requires ε_N = o(1/log N) for convergence; test whether shrinking ε removes the measured 3-9% score bias

---

## Results

| ε | Sinkhorn | Value error | Score error | z | Relative | Separated? |
|---|---|---|---|---|---|---|
| 1.0000 | 8 | +0.059 ± 0.074 | **-0.340 ± 0.101** | **-3.37** | **-5.46%** | yes |
| 0.5000 | 8 | +0.054 ± 0.072 | **-0.344 ± 0.093** | **-3.71** | **-5.53%** | yes |
| 0.2500 | 16 | +0.044 ± 0.072 | **-0.342 ± 0.088** | **-3.88** | **-5.50%** | yes |
| 0.1450 | 32 | — | — | — | — | **CRASH: dual-cap singular matrix** |
| 0.0725 | 64 | — | — | — | — | not reached |

---

## Finding 1: The bias is constant across ε

Score error mean: -0.340, -0.344, -0.342 (indistinguishable within SE ≈ 0.09)  
Relative error: -5.46%, -5.53%, -5.50% (no trend)  
z-score: -3.37, -3.71, -3.88 (not moving toward zero)

**This is Outcome B from the implementation plan: the entropic ε·log(N) term is not the dominant bias source.**

The measured bias arises from the **filtering-vs-smoothing conditioning gap** (Corenflos Prop 3.1):
- DPF score at any ε targets ∫ ∇log p(x_t, y_t | x_{t-1}) · p(x_{t-1:t} | y_{1:t}) dx
- Fisher identity needs ∫ ∇log p(x_t, y_t | x_{t-1}) · p(x_{t-1:t} | y_{1:T}) dx

The difference is the missing future observations y_{t+1:T} in the conditioning. No amount of ε tuning can close this gap, because both sides are exactly-correct filtering objects at their respective conditioning.

---

## Finding 2: Numerical instability at small ε

At ε = 0.145 = 1/log(1008), the dual-cap shape iteration crashed:

```
tensorflow.python.framework.errors_impl.InvalidArgumentError: Input matrix is not invertible.
```

Stack trace: `scaled_lm_coefficients_value` → `tf.linalg.solve(system, rhs)` in the **JVP path** (score lane).

**Mechanism:**  
As ε→0, the entropic-OT plan approaches the deterministic exact-OT plan, clustering particles. The empirical covariance develops near-zero eigenvalues. Even with ridge λ=1e-5, the linear system `(Σ + λI) x = b` in the dual-cap correction becomes singular in the tangent propagation.

This happened in the score lane, not the value lane, because the JVP amplifies conditioning issues: the tangent is `d/dθ` of an already-stiff solve.

**Why this is a hard boundary:**  
The c=1 schedule at N=1008 is ε ≈ 0.145. We crashed exactly there. To go further (ε=0.0725, c=0.5) would require either:
1. Raising λ (e.g., to 1e-4), which introduces a new bias and defeats the point of shrinking ε
2. Disabling the dual cap in the score lane, which removes a variance control
3. Switching to a different moment-correction algorithm

None of these are "just tuning ε" — they're architectural changes to stabilize a regime that doesn't help anyway.

---

## Interpretation: Why Corenflos Prop 3.3 didn't apply

The proposition's convergence |β̃_N - β| → 0 requires:
1. W_2(α_N, α) → 0 (particle approximation of the initial/propagated measure)
2. W_2(β_N, β) → 0 (particle approximation after reweighting)
3. ε_N = o(1/log N) (entropic regularization vanishes)
4. **Assumptions 1-4 hold:** compact state space, κ<1 Wasserstein contraction, weights in [Δ, Δ^{-1}], unique λ-Lipschitz OT map

Our LGSSM violates Assumption 1 (state space is unbounded, Gaussian tails extend to infinity). Assumption 4 (Lipschitz OT) is hard to verify and may also fail. So the theorem's preconditions are not met, and we should not have expected the O(ε·log N) bound to be tight.

More importantly: **even if the bound held**, it governs the filtering error |β̃_N - β|, not the score bias relative to the Fisher-identity target. Proposition 3.1 already shows that the gradient computed from the DPF (even with ε=0) differs from the Fisher score by a smoothing-vs-filtering gap. That gap is what we're measuring.

---

## Decision: Reject adaptive ε as a solution

**Costs:**
- 2-8× more Sinkhorn work (16 to 128 iterations vs 8)
- Numerical instability at the very ε values the Corenflos schedule prescribes
- Implementation complexity (ε-dependent iteration count, coupled ridge)

**Benefits:**
- Zero bias reduction (tested down to ε = 0.25, 2.5× the schedule value)
- No variance improvement visible in the SD column

**Conclusion:**  
Do not adopt adaptive ε in production. The bias is structural, not parametric. The right fixes are:
1. **Surrogate-force HMC** (move bias out of correctness)
2. **PaRIS forward-only score** (target the Fisher-identity directly)

---

## What this diagnostic achieved

It **ruled out the cheap fix**. Before running it, we had two competing hypotheses for the bias:
- H1: Entropic ε·log(N) term (fix: shrink ε)
- H2: Filtering-vs-smoothing gap (fix: PaRIS or surrogate-force)

The constant bias across ε ∈ [1.0, 0.5, 0.25] is strong evidence for H2. We now know that Solutions 2 and 3 are necessary, not contingent.

The crash at ε=0.145 is a bonus finding: even if the bias had fallen, we couldn't operate there stably without substantial changes to the dual-cap numerics.

---

## Artifact

Full log: `/tmp/eps_run.log`  
Runner: `docs/benchmarks/lgssm_d3_t50_epsilon_schedule.py`  
Results: First 3 rungs shown above, ε=0.145 crashed before completion

The runner is preserved for replicability, but **no further ε-schedule experiments are planned** unless the dual-cap is redesigned for small-ε stability and we obtain independent evidence that the bias should fall.

---

## Updated priority ranking

| Solution | Effort | Removes bias? | Numerical risk | Next step |
|---|---|---|---|---|
| **Surrogate-force HMC** | 1-2 days | Moves it out of correctness | Low (uses current filter) | Implement dual-adapter, test on d=3 T=50 |
| **PaRIS forward-only** | 4-5 days | Yes, targets Fisher-identity | Medium (accept-reject sampler) | Implement Algorithm 2, start with T=2 N=16 parity test |
| ~~Adaptive ε~~ | ~~1 day~~ | **No** | **High** (dual-cap singular) | **Rejected** |
| Nemeth λ-shrinkage | 2-3 days | Partially (introduces λ-bias) | Low | Deprioritized (adds a bias dial) |
| Fixed-lag | 3-4 days | No (mixing-dependent bias per Olsson-Westerborn) | Low | Deprioritized (Olsson-Westerborn §1.1 warns against) |

Surrogate-force is now the highest-ROI next step: lowest effort, mathematically proven to decouple correctness from score quality, and uses the existing filter infrastructure.
