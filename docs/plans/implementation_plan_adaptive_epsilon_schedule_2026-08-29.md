# Implementation Plan: Adaptive Sinkhorn Epsilon Schedule

Date: 2026-08-29  
Status: Diagnostic experiment running, implementation recipe below  
Source: Corenflos et al. (2021) Proposition 3.3, lines 519-540  
Runner: `docs/benchmarks/lgssm_d3_t50_epsilon_schedule.py` (in flight)

---

## Why this test matters

It is the **cheapest discriminating diagnostic** that separates two mechanisms:

1. **The entropic-OT ε·log(N) term** (Corenflos Lemma 3.1, Proposition 3.3): At fixed ε=1.0, the
   entropic radius R_H ≤ 2log(N) introduces an O(ε·log N / √N) bias that does not vanish as N→∞.
   
2. **The filtering-vs-smoothing conditioning gap** (Corenflos Proposition 3.1): The DPF gradient at
   ε=0 targets ∫ ∇log p(x_t, y_t | x_{t-1}) p(x_{t-1:t} | y_{1:t}) dx, but the Fisher identity
   needs the smoothing distribution p(x_{t-1:t} | y_{1:T}). Even with exact OT, this gap remains.

If shrinking ε along the schedule ε_N = c/log(N) removes the measured 3-9% bias → mechanism 1 is
the dominant source.  
If the bias persists at ε ≈ 0.14 → mechanism 2 dominates, and we need PaRIS or surrogate-force HMC.

**One-line change, ~30 minutes wall time, decisive diagnostic value.**

---

## The convergence condition (Corenflos Proposition 3.3)

For the DPF filtering distribution β̃^(t)_N and log-likelihood log(p̂_N(y_{1:T})):

> Under Assumptions 1-4 [compact state space, κ<1 Wasserstein contraction, bounded weights, unique
> Lipschitz OT map], for any bounded 1-Lipschitz ψ, if W_2(α_N, α) → 0, W_2(β_N, β) → 0 and we
> choose **ε_N = o(1/log N)**, then in probability:
> 
> |β̃^(t)_N(ψ) - β^(t)(ψ)| → 0  
> |log(p̂_N(y_{1:T}) / p(y_{1:T}))| → 0

The statement "o(1/log N)" means: ε_N·log(N) → 0 as N→∞. The canonical choice is ε_N = c/log(N)
for a constant c > 0, giving exactly the boundary case.

At our particle counts:
- N = 1008: log(N) ≈ 6.92 → ε ≈ c/6.92
- N = 3000: log(N) ≈ 8.01 → ε ≈ c/8.01

For c=1: ε ∈ [0.125, 0.145]. Our production ε=1.0 is 7-8× larger than this schedule.

---

## The entropic radius mechanism (Lemma 3.1, lines 1141-1160)

The coupling simplex S(α_N, β_N) has entropic radius R_H ≤ 2log(N) by concavity of entropy:

```
H(P) = -sum_{i,j} p_{i,j} log(p_{i,j})
     ≤ N^2 H(1/N^2) = N^2 · (1/N^2) log(N^2) = 2log(N)
```

The DET approximation error (equation after Lemma 3.1) includes the term `√(2ε log N)`, which
dominates at small N if ε is held fixed. To make it vanish, ε must shrink faster than 1/log(N).

---

## Experiment design

**Runner:** `docs/benchmarks/lgssm_d3_t50_epsilon_schedule.py` (adapted from the validated N-ladder)

**Fixture:** d=3 T=50 diagonal LGSSM, θ = [0.72, 0.55, 0.35, 0.35, 0.45], obs seed 81100,
exact Kalman value -145.434, exact score φ_1 = -6.217

**Epsilon ladder:** [1.0, 0.5, 0.25, 0.145, 0.0725]  
**Sinkhorn iterations:** {1.0: 8, 0.5: 8, 0.25: 16, 0.145: 32, 0.0725: 64}

Rationale: As ε→0, the Sinkhorn kernel K_ε = exp(-C/ε) becomes more peaked and needs more
iterations to converge to the entropic-OT plan. Rather than holding iterations fixed and confounding
the ε effect with unconverged Sinkhorn, we raise the iteration count as ε falls. This is a resource
trade: we're willing to pay 4-8× more Sinkhorn work to test whether small ε buys bias reduction.

**N = 1008, 16 paired seeds** (same seeds as the N-ladder for cross-run comparability).

**Primary metric:** score error mean ± SE, z-score, and separation from zero at 2 SE.

**Hypothesis under test:**  
If the entropic ε·log(N) term is the dominant bias source, then shrinking ε from 1.0 to 0.145
(the c=1 schedule at N=1008) should reduce |z| from ~3-7 toward <2. If z remains large and negative
across the entire ladder, the conditioning gap (mechanism 2) is dominant.

---

## What each ε rung tells us

| ε | c·log(N) product | Sinkhorn iters | Interpretation |
|---|---|---|---|
| 1.0 | 1.0 · 6.92 ≈ **6.92** | 8 | Production baseline, outside the convergence schedule |
| 0.5 | 0.5 · 6.92 ≈ **3.46** | 8 | Halfway to the boundary in linear scale |
| 0.25 | 0.25 · 6.92 ≈ **1.73** | 16 | Within 2× of the boundary |
| 0.145 | 0.145 · 6.92 ≈ **1.00** | 32 | Exactly c=1 schedule |
| 0.0725 | 0.0725 · 6.92 ≈ **0.50** | 64 | Well inside the convergence regime, 2× safety |

If bias scales like ε·log(N), we expect score error to drop roughly proportionally to the third column.

---

## Three possible outcomes and their interpretations

### Outcome A: Bias falls cleanly with ε

Example: z ≈ -5 at ε=1.0 → z ≈ -3 at ε=0.5 → z ≈ -1.5 at ε=0.25 → z ≈ -1 at ε=0.145.

**Interpretation:** The entropic term dominates. The measured 3-9% bias is mostly the O(ε·log N)
contribution, not the conditioning gap.

**Action:** Set ε_N = c/log(N) with c≈1 as the new production default. Test at N=3000 and higher
dimensions. Monitor Sinkhorn convergence (residuals should stay bounded as ε falls and iterations
rise). Check Contract-E Cholesky conditioning (the covariance (Σ + λI) may become stiffer at small ε
if the transported particles cluster more).

**Reward:** Removes most of the bias with no architectural change, just a schedule and more Sinkhorn
work.

### Outcome B: Bias is constant across the ladder

Example: z ≈ -5 ± 0.3 at every ε rung, no trend.

**Interpretation:** The conditioning gap dominates. Even at ε=0.0725 (well inside the Corenflos
convergence regime), the bias persists because the DPF score fundamentally targets the filtering
distribution p(x_{t-1:t} | y_{1:t}), and the Fisher identity needs p(x_{t-1:t} | y_{1:T}).

**Action:** Adaptive ε does not help. Pursue PaRIS (which uses the smoothing distribution
forward-only) or surrogate-force HMC (which makes correctness independent of score bias). Do **not**
adopt small ε in production, because it costs Sinkhorn work for no gain.

**Interpretation boundary check:** If the Corenflos assumptions (compact state, Lipschitz OT,
bounded weights) are violated for our fixture, the convergence theorem may not apply even at small ε.
Verify: state space is unbounded (Gaussian tails), but the filter weights are bounded
(exp(-likelihood differences) stay in a reasonable range at these parameters). The Lipschitz-OT
assumption (Assumption 4) is the hardest to verify and may be where the theorem's preconditions fail.

### Outcome C: Bias falls but not enough

Example: z ≈ -5 at ε=1.0 → z ≈ -3.5 at ε=0.145 (improvement, but still separated).

**Interpretation:** Both mechanisms contribute. The entropic term accounts for part of the bias,
and the conditioning gap accounts for the rest. Shrinking ε helps but does not eliminate the problem.

**Action:** Adopt ε_N = c/log(N) as a partial fix (removes the entropic contribution), **and**
implement PaRIS or surrogate-force HMC to address the residual conditioning-gap bias. The two fixes
compose: small ε makes the filtering distribution more accurate, and PaRIS/surrogate-force handles
the gap between filtering and what we need.

---

## Numerical stability checks

Three failure modes to watch for as ε→0:

### 1. Sinkhorn divergence

**Symptom:** Sinkhorn residual stays large or grows even as iterations rise.

**Diagnosis:** The transport problem may be ill-posed at small ε (e.g., particles collapsing onto a
lower-dimensional manifold, making the coupling plan near-singular). Or the iteration count is still
insufficient.

**Mitigation:** Log the Sinkhorn residual `||K diag(u) K^T diag(v) - β||_1` at each ε. If it's
>1e-3 at ε=0.0725 with 64 iterations, either raise iterations further or declare ε=0.0725
numerically infeasible for this fixture.

### 2. Contract-E Cholesky failure

**Symptom:** Cholesky of (Σ + λI) fails or produces inf/nan, or the condition number explodes.

**Diagnosis:** At small ε the transported particles cluster more tightly (the entropic-OT plan
approaches the deterministic exact-OT plan), so the empirical covariance Σ may have one or more
near-zero eigenvalues. The ridge λ=1e-5 is meant to prevent this, but if particles collapse onto a
d-1 or lower dimensional surface, even λ=1e-5 may not be enough.

**Mitigation:** Log `cond(Σ + λI)` at each reset. If it exceeds 1e8 at any ε, raise λ adaptively
for that ε (e.g., λ = max(1e-5, ε/100)). This introduces a λ-ε coupling that makes the "pure ε
effect" harder to interpret, but it's better than a runtime crash.

### 3. Dual-cap Newton failure

**Symptom:** The Gauss-Newton correction in the dual cap fails to converge, or the damping floor
δ=1e-5 is hit repeatedly.

**Diagnosis:** Similar to Cholesky — at small ε the moment-matching problem may become
near-singular if particles cluster.

**Mitigation:** Log the dual-cap damping value and the Newton residual. If damping is at the floor
for >50% of resets, the correction is numerically stressed. Consider raising δ with ε
(e.g., δ = max(1e-5, ε/100)).

---

## Cost accounting

At ε=1.0: 8 Sinkhorn + 8 balance = 16 passes over the N×N kernel.  
At ε=0.145: 32 Sinkhorn + 32 balance = 64 passes → **4× the reset cost**.  
At ε=0.0725: 64 + 64 = 128 passes → **8× the reset cost**.

The reset is currently a small fraction of the total per-step cost (the dual cap and the guidance
UKF dominate at d=3). So 4-8× more Sinkhorn iterations is acceptable for a diagnostic. But if this
becomes the production default, the Sinkhorn cost shifts from negligible to noticeable, and we'd
need to profile whether the guidance or the reset is now the bottleneck.

Wall time estimate for the current run (N=1008, T=50, 16 seeds, 5 ε rungs):
- Production ε=1.0 takes ~25-30s per seed
- ε=0.0725 with 8× Sinkhorn work → ~40-50s per seed
- Total: 16 seeds × 5 rungs × 35s ≈ **47 minutes**

---

## Integration path if Outcome A (bias falls with ε)

### Phase 1: Validate the schedule at production N

Run the same ε ladder at N=3000 (the historical benchmark particle count). If the bias reduction
holds, proceed. If it doesn't (e.g., z=-5 at N=1008 becomes z=-4 at N=3000 but plateaus there),
the log(N) term is growing and we're not inside the o(1/log N) regime yet.

### Phase 2: Adaptive ε as a config option

Add to the reset config:
```python
reset_epsilon_schedule: str = "fixed"  # "fixed" | "adaptive"
reset_epsilon_fixed: float = 1.0
reset_epsilon_adaptive_c: float = 1.0
```

When `schedule="adaptive"`:
```python
epsilon = reset_epsilon_adaptive_c / tf.math.log(tf.cast(N, tf.float64))
```

Gate behind the flag so existing experiments are unaffected.

### Phase 3: Sinkhorn auto-tuning

As ε falls, iterations must rise to maintain convergence. The current fixed `reset_sinkhorn_steps=8`
becomes insufficient. Options:

1. **Lookup table** (what the diagnostic runner uses):
   ```python
   SINKHORN_ITERS = {1.0: 8, 0.5: 8, 0.25: 16, 0.145: 32, 0.0725: 64}
   ```
   Simple, but requires manual tuning per ε regime.

2. **Residual-based early stop**:
   Run Sinkhorn until `||residual|| < tol`, cap at `max_iters`. Costs one extra matmul per iteration
   to compute the residual, but adapts automatically.

3. **Fixed budget scaled by ε**:
   ```python
   sinkhorn_iters = max(8, int(16 / epsilon))
   ```
   Heuristic: as ε halves, double the iterations. Cheap and parameter-free.

Recommend option 3 for a first cut, with option 2 as a refinement if the heuristic proves unreliable.

### Phase 4: Monitor Contract-E / dual-cap conditioning

Add logging:
```python
sigma_cond = tf.linalg.norm(sigma_empirical, ord=2) / tf.linalg.norm(sigma_empirical, ord=-2)
if sigma_cond > 1e8:
    warnings.warn(f"High condition number {sigma_cond:.2e} at epsilon={epsilon:.4f}, raising ridge")
    lambda_ridge = max(lambda_ridge, epsilon / 100)
```

If this triggers frequently, couple λ to ε as a stability guard.

---

## Decision tree after the run completes

```
Outcome A (bias falls with ε)?
├─ Yes → Phase 1-4 above, new production default pending validation
├─ No, Outcome B (constant bias)?
│  └─ PaRIS or surrogate-force HMC, do not adopt small ε
└─ Outcome C (partial fall)?
   └─ Adopt adaptive ε + pursue PaRIS/surrogate-force for the residual
```

Checkpoint: wait for the diagnostic run to finish, then revisit this plan with the actual z-scores.

---

## Status: experiment running

Started: 2026-08-29 ~05:17 UTC  
Estimated completion: ~05:50 UTC (30-50 minutes wall time)  
Output: `/tmp/eps_run.log` → final artifact at
`docs/benchmarks/lgssm_d3_t50_epsilon_schedule.json`

The runner is a direct adaptation of the validated N-ladder (`lgssm_d3_t50_score_n_ladder.py`),
differing only in sweeping ε instead of N and raising Sinkhorn iterations with ε. Everything
else — seeds, observation path, exact reference, statistical protocol — is unchanged.
