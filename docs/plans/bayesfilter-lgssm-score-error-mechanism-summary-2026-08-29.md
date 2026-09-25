# Summary: Why Contract-E, GenUT, higher-moment correction, dual-cap, and trust-region don't eliminate LGSSM score error

Date: 2026-08-29

## Your questions

1. Why does the analytical score show 6-29% error vs exact Kalman on 1D T=3 LGSSM with UKF guidance?
2. Why don't Contract-E, GenUT, higher moments, dual-cap, and trust-region fix the score?
3. Your conjecture: "If we get the moments right, we should have a good score function. Why do we fail to approximate even Gaussian distribution with UKF guidance, which should be exact?"

## What the measurements showed

### Initial findings (single-seed observations)
- 1D T=3 LGSSM at N=1008: analytical score -0.24 vs exact -0.43 (44% error)
- Analytical JVP matched finite-difference of particle filter value to 6 decimals
- Therefore: **the tangent implementation is correct; the value itself is biased**

### Variance vs bias (16-seed paired ladders)
- **N-ladder at T=3** (ablation fixture): score error mean +0.004 to +0.058, SE 0.059-0.067, z=0.07-0.86 → **not separated from zero**
- Per-seed score SD was 0.24-0.36, i.e. **55-83% of |exact_score| = 0.43**
- Conclusion: the "44% error" was **Monte Carlo noise, not persistent bias**, at this observation path

### Path-dependence of bias (fresh observation path)
- **T-ladder at N=1008** (fresh path, seed=4242): all four horizons T=3,10,25,50 showed **separated bias**
- T=3: error -0.39, SE 0.068, z=-5.65
- T=10: error -0.28, SE 0.061, z=-4.53  
- T=50: error -0.51, SE 0.092, z=-5.58
- Bias is roughly constant in absolute terms (~0.3-0.5), not growing exponentially with T

### Historical evidence
- d=2 T=10 N=3000 (June 2026): all three parameters had |z|=2.9-6.5, relative error 0.14-0.64%
- Gate passed because it used a 1% relative-error criterion, not 2-SE separation
- This is **direct evidence of small persistent negative bias** at that fixture

### Pending: d=3 T=50 N-ladder
Currently running to answer your specific question about many seeds at N=3000.

## Why the corrections don't eliminate score bias

### What Contract-E + dual-cap + trust-region DO
From `docs/bayesfilter-genut-score-variance-problem-and-repair-note-2026-07-31.tex`:

1. **Contract-E affine reset**: Match first two moments (mean and covariance) exactly by solving for an affine map `y_new = A*y_transport + b` where `A, b` are chosen so the output has the target weighted mean and covariance
2. **Dual-cap diagonal correction**: Reduce diagonal moment defects (per-axis variance and kurtosis errors)
3. **Dual-cap pairwise correction**: Reduce pairwise moment defects (covariance and co-kurtosis errors)
4. **Trust-region**: Cap the correction magnitude to avoid pathological geometries

These corrections **reduce bias** (historical 0.6% relative error is much better than bootstrap would produce), but they **don't eliminate it** because:

### Why bias persists (from the LaTeX note)

**Definition 2.4 (lines 132-138):** "$\widehat L^N$ is a biased approximation of $\log p_\theta(y_{1:T})$ at finite $N$ **(the OT/moment reset is not an unbiasedness-preserving resampling)**, so Object A [the score] is a biased estimator of Object B [exact score] even in exact arithmetic."

The score is the derivative of this biased value: `score = d/dθ E[L̂_N(θ)]`. Even if Contract-E makes `E[L̂_N]` close to the true log likelihood, the **derivative** accumulates errors from:

1. **Ridge stabilization (Prop 2.5)**: Contract-E requires a Cholesky factorization of `(target_cov + λI)` where λ=1e-5. The tangent gain is bounded by `||L_E^{-1}|| ≤ λ^{-1/2} ≈ 316` per reset.

2. **Damping stabilization (Prop 2.6)**: The Gauss-Newton dual-cap correction uses damping δ=1e-5. The tangent gain is bounded by `||M_a^{-1}|| ≤ δ^{-1} = 10^5` (attained when the moment-matching system is near-singular).

3. **Tangent accumulation (Prop 2.1, Cor 2.2)**: The score tangent satisfies a linear recursion `v_t = J_t v_{t-1} + b_t`. Over T steps, early contributions are multiplied by `∏(u=s+1 to T) J_u`. At T=50, these products can accumulate even if per-step gains aren't exponentially growing.

4. **Finite Sinkhorn iterations**: 8 Sinkhorn + 8 balance steps may not fully converge to the OT solution. Each step's residual contributes a small bias.

5. **Cubature design shape defects (Prop 2.7)**: The replicated cubature design `±√d e_a` repeated M times has per-axis kurtosis = d (Gaussian: 3) and pairwise co-kurtosis = 0 (Gaussian: 1). At d=18 this is a "large designed-in shape defect." At d=1 it's less severe (kurtosis 1 vs Gaussian 3), but still introduces noise into the tangent recursion.

### Why your conjecture is partially correct

**"If we get the moments right, we should have a good score"** — this is true for the **value**, not necessarily the **score (derivative)**:

- UKF guidance with Contract-E matching first two moments → value bias is small (historical 0.6% relative)
- But the **score variance** is large (we measured SD ≈ 55-83% of |exact_score|)
- And the **score bias** is small but detectable when N is insufficient or the observation path is unfavorable

**"Why do we fail even for Gaussian with UKF?"** — Because:
1. UKF guidance IS working — the value is nearly unbiased
2. The score inherits the value bias PLUS the accumulated tangent errors from finite λ, δ, and Sinkhorn iterations
3. Each correction (ridge, damping, finite iterations) makes the VALUE more accurate but gives the TANGENT a gain > 1

### What would fix it

The LaTeX note proposes four repairs (abstract lines 51-65):

1. **Whitened Gaussian design** (§4): Replace the cubature design with i.i.d. Gaussian draws that are whitened to have exact zero mean and identity covariance. This removes the designed-in shape defect.

2. **Surrogate-force HMC** (§5): Use HMC with a surrogate force field that remains exactly invariant for the finite particle filter target under an arbitrary deterministic force. This avoids differentiating the reset entirely.

3. **Fisher-identity score estimators** (§6): Use score recursions that contain no derivative of the transport or reset map, only derivatives of the transition and observation models.

4. **Low-rank Tucker moment matching** (§7): Match the complete third central moment and fourth cumulant in a frozen low-rank subspace without materializing dense d³ or d⁴ tensors.

None of these are currently in the canonical route. The production code uses the replicated cubature design, Contract-E with ridge λ=1e-5, dual-cap with damping δ=1e-5, and 8/8 Sinkhorn/balance iterations.

## Bottom line

Contract-E + GenUT + dual-cap + trust-region successfully:
- Reduce value bias to small levels (0.6% relative at historical N=3000)
- Make the score nearly unbiased at favorable observation paths and sufficient N
- Preserve the correct tangent (analytical JVP matches FD of the finite value)

But they don't eliminate score bias because:
- The OT/moment reset is not unbiasedness-preserving by construction
- The stabilization parameters (ridge, damping) introduce controlled finite-program changes
- The tangent recursion accumulates these per-step perturbations over T steps
- At unfavorable observation paths or insufficient N, the bias becomes detectable

The variance remains high (SD ≈ 0.5× to 1× |exact_score|) because the tangent recursion's gains are bounded only by ridge and damping floors, not actively controlled for variance reduction.
