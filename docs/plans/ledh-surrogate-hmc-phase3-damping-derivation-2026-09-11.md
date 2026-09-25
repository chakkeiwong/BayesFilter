# LEDH Surrogate-Force HMC: Phase 3 Task 3.1 — Damping Parameter Derivation

**Date:** 2026-09-11  
**Authority:** Phase 3 of ledh-surrogate-hmc-unified-program-2026-09-06.md  
**Status:** DRAFT

---

## Question

What does "damp the LEDH analytical score by 100×" mean in the parameters that actually exist in `canonical_batch_fused_value_score`?

---

## Background

The Phase 3 task description states:

> The program states the scientific question in terms of λ (process-covariance ridge) and δ (observation-covariance ridge). Those parameters do not exist. `canonical_value_and_analytical_score` exposes `reset_ridge` (Contract-E ridge) and `correction_lm_damping` / `correction_lm_scale_floor` (trust-region LM).

The goal of surrogate-force HMC (Corollary 5.2) is to use a **biased but cheaper score** while keeping the **value exact**. The chain targets exp(-U) where U is the executed filter value, so value must not be damped. Only the score (gradient) should be affected.

---

## Available Parameters in Unified Engine

From `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py:115-144`:

```python
def canonical_batch_fused_value_score(
    ...,
    reset_ridge: float = 1.0e-5,              # Contract-E numerical ridge
    reset_epsilon: float = 2.0,                # Sinkhorn entropic regularization
    reset_sinkhorn_steps: int = 8,             # Sinkhorn iteration count
    reset_balance_steps: int = 8,              # Balance iteration count
    correction_lm_damping: float = 1.0e-2,     # LM damping parameter
    correction_lm_scale_floor: float = 1.0e-4, # LM scale floor
    correction_trust_radius: float = 0.5,      # Trust-region radius cap
    pairwise_strength: float = 0.02,           # Pairwise moment correction strength
    ...
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
```

---

## Candidate Parameters for Score Damping

### Candidate 1: `reset_ridge` (Contract-E Ridge)

**What it does:** Adds ridge to moment matrices during GenUT reset stage.

**Where it enters:**
- GenUT sigma-point generation: `Σ + reset_ridge * I`
- Affects particle generation at the start of each horizon segment

**Effect on score:**
- Regularizes the particle cloud covariance
- Stabilizes Cholesky factorization
- Affects tangent propagation through reset stage

**Effect on value:**
- Changes particle locations → **changes value**
- NOT suitable for surrogate-force (violates "value unchanged" requirement)

**Verdict:** ❌ **Not a score-only damping parameter**

---

### Candidate 2: `reset_epsilon` (Sinkhorn Entropic Regularization)

**What it does:** Entropic regularization in optimal transport (Sinkhorn algorithm).

**Where it enters:**
- Regularizes the transport plan: `π* = argmin <C,π> + ε·KL(π||μ⊗ν)`
- Higher ε → more entropic, less sharp transport

**Effect on score:**
- Transport map is part of the filter dynamics
- JVP through transport affects score
- Larger ε → smoother transport → different tangent

**Effect on value:**
- Transport map affects particle evolution → **changes value**
- NOT suitable for surrogate-force (violates "value unchanged" requirement)

**Verdict:** ❌ **Not a score-only damping parameter**

---

### Candidate 3: `correction_lm_damping` (Levenberg-Marquardt Damping)

**What it does:** LM damping in trust-region dual-cap correction.

**Where it enters:**
- Trust-region update: `Δ = (J^T J + λ·diag(J^T J) + μ·I)^{-1} J^T r`
- `correction_lm_damping` is the LM parameter λ

**Effect on score:**
- Dual-cap correction modifies particle covariance
- JVP through dual-cap affects score
- Larger damping → less aggressive correction → different tangent

**Effect on value:**
- Dual-cap modifies particles → **changes value**
- NOT suitable for surrogate-force (violates "value unchanged" requirement)

**Verdict:** ❌ **Not a score-only damping parameter**

---

### Candidate 4: `correction_lm_scale_floor` (LM Scale Floor)

**What it does:** Scale floor in LM trust region.

**Where it enters:**
- Used in denominator of trust-region step computation
- Prevents division by very small scales

**Effect on score:**
- Similar to `correction_lm_damping` - affects dual-cap JVP
- Changes tangent through trust-region stage

**Effect on value:**
- Modifies dual-cap correction → **changes value**
- NOT suitable for surrogate-force (violates "value unchanged" requirement)

**Verdict:** ❌ **Not a score-only damping parameter**

---

## The Fundamental Problem

**All available parameters affect the value, not just the score.**

This is because LEDH is a particle filter where:
1. Particles evolve through: initialization → GenUT reset → transport → dual-cap
2. The **value** is a function of final particle weights: `log(mean(weights))`
3. The **score** is the JVP (tangent) of that function

Every parameter that changes particle evolution changes both value and score. There is no parameter in the current implementation that:
- Changes only the tangent computation (score)
- Leaves the primal computation (value) unchanged

---

## What the Program Actually Meant

Looking at the audit context and the v2 runner mentioned in Phase 3:

> The pre-existing runner v2 maps λ → `reset_ridge` and δ → `correction_lm_scale_floor`, where the second is a **tuned control** (selected value 1e-06 for LGSSM T50).

The program's "damping" likely referred to **running with less aggressive numerical protections** (larger ridges), which would:
- Make the filter less stable (closer to singularity)
- Reduce computational cost (fewer stabilization iterations)
- Change both value and score

But this violates Corollary 5.2's requirement that the chain target the same distribution.

---

## Resolution: Corollary 5.2 Is Not Yet Implemented

**Source authority:** `bayesfilter-genut-score-variance-problem-and-repair-note-2026-07-31.tex`, lines 953-994.

**Corollary 5.2 (lines 953-963):** Surrogate-force HMC with:
1. Exact executed potential: U = -L̂^N (computed with exact parameters)
2. Any deterministic momentum-independent force F (can be biased)
3. Metropolis-Hastings acceptance using exact H(θ,p)

→ Targets π(θ) ∝ exp(-U(θ)) regardless of F quality. Force affects only mixing.

**Remark 5.3 (lines 984-988):** Licenses computing score with "larger λ,δ than the value program."

**Lines 989-993 (implementation status):**
> "The current repository HMC wrapper passes a single custom-gradient target (one adapter call returns value and score), so this scheme is an **implementable proposal, not current behavior**."

**Current architecture:** `canonical_batch_fused_value_score` returns `(value, score)` from a single call with one parameter set. Both value and score use the same `reset_ridge`, `correction_lm_damping`, etc.

**What Corollary 5.2 requires:** Separate execution:
```python
# Exact value
value = canonical_batch_fused_value(theta, exact_params)

# Biased score (larger λ, δ)
score = canonical_batch_fused_score(theta, biased_params)

# HMC acceptance uses exact value
accept_prob = min(1, exp(H_old - H_new))  # H uses exact value
```

**This dual-execution architecture does not exist in the current codebase.**

---

## Implementation Path: Dual-Execution Adapter

To implement Corollary 5.2, we need:

### Architecture: Separate Value and Score Calls

```python
class DualParameterLEDHTarget:
    """HMC target with exact value, biased score."""
    
    def __init__(self, model, exact_params, biased_params):
        self.model = model
        self.exact_params = exact_params
        self.biased_params = biased_params
    
    def value_and_score(self, theta):
        # Value: exact parameters
        value, _ = canonical_batch_fused_value_score(
            model, theta, ..., **self.exact_params
        )
        
        # Score: biased parameters (larger damping)
        _, score = canonical_batch_fused_value_score(
            model, theta, ..., **self.biased_params
        )
        
        return value, score
```

### Parameter Mapping (λ, δ → Implementation)

From Remark 5.3 line 985-986: "computed with larger λ,δ than the value program"

**λ (process-covariance damping):** Maps to `reset_ridge`
- Exact: 1e-5 (typical)
- Biased: 1e-4, 1e-3, 1e-2 (10×, 100×, 1000× coarser)

**δ (observation-covariance damping):** No direct parameter exists. The closest is `correction_lm_scale_floor` (trust-region scale floor), but this is NOT an observation-covariance ridge.

**Alternative interpretation:** Both λ and δ could map to `correction_lm_damping` (LM damping parameter) if the dual-cap correction is the "damping" stage.

### Proposed Mapping for "100× damping"

**Option 1: Scale `reset_ridge` by 100×**
- Exact value: `reset_ridge=1e-5`
- Biased score: `reset_ridge=1e-3` (100× larger)

**Option 2: Scale `correction_lm_damping` by 100×**
- Exact value: `correction_lm_damping=1e-2`
- Biased score: `correction_lm_damping=1.0` (100× larger)

**Option 3: Scale both**
- Exact: `reset_ridge=1e-5`, `correction_lm_damping=1e-2`
- Biased: `reset_ridge=1e-3`, `correction_lm_damping=1.0`

---

## Recommendation: Implement Dual-Execution Adapter (Option A)

**Task 3.1 Answer:** "Damping by 100×" means using **`reset_ridge` 100× larger for score than for value**.

**Mapping:**
- **λ (process ridge)** → `reset_ridge`
- **δ (observation ridge)** → Not directly available; use `correction_lm_damping` as proxy

**Proposed implementation for Phase 3:**

### Step 3.1a: Create dual-execution adapter (0.5 day)

**File:** `bayesfilter/inference/ledh_dual_parameter_target.py`

```python
class DualParameterLEDHTarget:
    """Corollary 5.2 surrogate-force HMC target.
    
    Value computed with exact parameters.
    Score computed with biased (larger damping) parameters.
    """
    
    def __init__(
        self,
        model: PerPointScoreModel,
        exact_reset_ridge: float = 1e-5,
        biased_reset_ridge: float = 1e-3,  # 100× larger
        exact_lm_damping: float = 1e-2,
        biased_lm_damping: float = 1.0,    # 100× larger
        **shared_params
    ):
        self.model = model
        self.exact_params = {**shared_params, 
                            'reset_ridge': exact_reset_ridge,
                            'correction_lm_damping': exact_lm_damping}
        self.biased_params = {**shared_params,
                             'reset_ridge': biased_reset_ridge,
                             'correction_lm_damping': biased_lm_damping}
    
    def __call__(self, theta):
        """HMC target: exact value, biased score."""
        value, _ = canonical_batch_fused_value_score(
            self.model, theta, ..., **self.exact_params
        )
        _, score = canonical_batch_fused_value_score(
            self.model, theta, ..., **self.biased_params
        )
        return value, score
```

**Test:** `tests/inference/test_ledh_dual_parameter_target.py`
- Verify value uses exact parameters (compare against baseline)
- Verify score uses biased parameters (check force magnitude is different)
- Verify determinism (frozen seeds)

### Step 3.1b: Task 3.2 calibration becomes concrete

**Question:** What damping ratio (1×, 10×, 100×, 1000×) maintains acceptance ≥ 0.15?

**Sweep:** `biased_reset_ridge` ∈ {1e-5, 1e-4, 1e-3, 1e-2} (1×, 10×, 100×, 1000×)

**Measure:** Acceptance rate, ESS/grad, posterior coverage (LGSSM d=3 T=50)

### Step 3.1c: Task 3.3 seed policy is already specified

Corollary 5.2 lines 980-981: "all Monte Carlo seeds inside F must be frozen"

Implementation: Pass same `noises` tensor to both value and score calls. Already satisfied by dual-execution adapter design.

---

## Budget Update

**Original Task 3.1:** 0.5 day (derivation only)  
**Revised Task 3.1:** 1 day (derivation + adapter implementation + tests)

**Original Task 3.2:** 0.5-1 day (calibration curve)  
**Revised Task 3.2:** 0.5 day (simplified: sweep is now well-defined)

**Original Task 3.3:** 0.5 day (seed policy)  
**Revised Task 3.3:** 0.1 day (seed policy verified by adapter design)

**Total Phase 3:** 1.6 days (vs original 1-2 days)

---

## Status

**Task 3.1 COMPLETE** (derivation)

**Answer:** "Damping by 100×" means:
- **λ → `reset_ridge`**: Use 1e-3 for score (vs 1e-5 for value) = 100× larger
- **δ → `correction_lm_damping`**: Use 1.0 for score (vs 1e-2 for value) = 100× larger

**Implementation:** Requires dual-execution adapter (value and score from separate calls).

**Next:** Implement adapter (Task 3.1a, 1 day), then proceed to calibration (Task 3.2).
