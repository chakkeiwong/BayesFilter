# Phase 0 Execution Handoff

**Date:** 2026-09-07  
**Authority:** ledh-surrogate-hmc-executable-master-program-2026-09-07.md  
**Estimated Time:** 1 day  
**GPU Required:** No

---

## Goal

Implement the three Option A decisions: tolerance derivation, seed policy documentation, joint Mahalanobis coverage formula.

---

## Files to Modify

1. **Create:** `bayesfilter/inference/tolerance_derivation.py`
2. **Create:** `docs/memos/ledh-surrogate-hmc-seed-policy-2026-09-07.md`
3. **Modify:** `bayesfilter/inference/coverage.py` (if exists) OR create it

---

## Tasks

### Task 0.1: Tolerance Derivation (0.3 day)

**Goal:** Derive tolerance from condition number in float32 TF32 regime.

**Implementation:**
```python
# bayesfilter/inference/tolerance_derivation.py

def derive_parity_tolerance(
    condition_number: float,
    dtype: str = "float32",
    backend: str = "TF32"
) -> float:
    """
    Derive relative tolerance for cross-lane parity checks.
    
    Args:
        condition_number: Estimated condition number of the operation
        dtype: "float32" or "float64"
        backend: "TF32" or "FP32"
    
    Returns:
        Relative tolerance for np.allclose or tf.debugging.assert_near
        
    Formula:
        tolerance = condition_number × machine_epsilon × safety_factor
        
    For float32 TF32:
        - machine_epsilon ≈ 1.2e-7 (float32)
        - TF32 matmul reduces precision: effective eps ≈ 1e-5
        - safety_factor = 10 (account for accumulated rounding)
        - Typical condition_number = 1e2 to 1e4 for particle filters
        - Result: tolerance = 1e-3 to 1e-1
    """
    # Implementation here
    pass

def derive_golden_master_tolerance(
    condition_number: float,
    dtype: str = "float64"
) -> float:
    """
    Derive tolerance for golden master (bitwise-reproducibility check).
    
    For float64 CPU-only:
        - machine_epsilon ≈ 2.2e-16
        - Deterministic ops → tolerance = condition_number × eps × 10
        - Typical result: 1e-12 to 1e-10
    """
    pass
```

**Success Criterion:**
- Functions exist and have docstrings
- Smoke test: `derive_parity_tolerance(1e3, "float32", "TF32")` returns ~1e-2
- Smoke test: `derive_golden_master_tolerance(1e3, "float64")` returns ~1e-12

**Bounded I/O:**
- No file reads needed (pure math)
- Write one test file: `tests/inference/test_tolerance_derivation.py` (~30 lines)

---

### Task 0.2: Seed Policy Documentation (0.5 day)

**Goal:** Document the one-master-ω seed policy and verification tests.

**File:** `docs/memos/ledh-surrogate-hmc-seed-policy-2026-09-07.md`

**Content (1-2 pages):**

```markdown
# LEDH Surrogate-Force HMC Seed Policy

**Date:** 2026-09-07  
**Authority:** Corollary 5.2 (requires deterministic force)

## Policy

**One master ω, frozen across all phases.**

- Sample one particle-noise realization ω before any HMC run
- Freeze it in a file: `data/ledh_master_omega_YYYYMMDD.npz`
- All HMC runs (exact-force, damped-force, all models) use identical ω
- No reseeding, no θ-dependent seeding, no per-step seeding

## Rationale

Corollary 5.2 proves invariance of π_N^ω under any **deterministic** force.

θ-dependent reseeding makes the force **stochastic** (ω = f(θ)), violating the premise.

Proposition 6 (note line 1166): "a θ-dependent branch makes L^N(θ) discontinuous, 
and no same-scalar derivative claim holds across the crossing."

## Verification Tests (Phase 3)

### V1: Determinism
- Same ω → bitwise identical trajectories
- Run HMC twice with same ω, assert θ_final matches bitwise

### V2: Reversibility (involution)
- Forward-backward-forward = identity
- Run forward N steps, reverse N steps, forward N steps
- Assert final state = initial state

### V3: No call-count dependence
- force(θ, call=1) = force(θ, call=100)
- Compute force at same θ with different call counts
- Assert outputs match
```

**Success Criterion:**
- File exists, < 2 pages
- States policy, rationale, three tests
- Cites Corollary 5.2 and Proposition 6

**Bounded I/O:**
- Read Corollary 5.2 location: `docs/plans/contract-e-jvp-serial-bottleneck-analysis-2026-09-03.md` (lines 1100-1200 only)
- No other file reads needed

---

### Task 0.3: Joint Mahalanobis Coverage (0.1 day)

**Goal:** Implement joint 95% Mahalanobis region coverage check (diagnostic only).

**File:** `bayesfilter/inference/coverage.py` (create if doesn't exist)

**Implementation:**
```python
import tensorflow as tf
import tensorflow_probability as tfp

def joint_mahalanobis_coverage(
    samples: tf.Tensor,  # [num_samples, param_dim]
    true_theta: tf.Tensor,  # [param_dim]
    alpha: float = 0.05
) -> dict:
    """
    Check if true_theta is inside the joint 95% credible region.
    
    Uses Mahalanobis distance with empirical posterior covariance.
    
    Returns:
        {
            "covers": bool,
            "mahalanobis_distance": float,
            "threshold": float (chi-squared quantile),
            "p_value": float
        }
    
    Formula:
        D² = (θ - μ)ᵀ Σ⁻¹ (θ - μ)
        Covers if D² < χ²(p, 1-α) where p = param_dim
    """
    param_dim = tf.shape(samples)[1]
    
    # Empirical mean and covariance
    mu = tf.reduce_mean(samples, axis=0)
    cov = tfp.stats.covariance(samples)
    
    # Mahalanobis distance
    delta = true_theta - mu
    # Solve Σ x = δ for x, then compute δᵀ x
    chol = tf.linalg.cholesky(cov)
    x = tf.linalg.cholesky_solve(chol, delta[:, None])
    mahal_sq = tf.reduce_sum(delta * x[:, 0])
    
    # Chi-squared threshold
    chi2_dist = tfp.distributions.Chi2(df=tf.cast(param_dim, tf.float32))
    threshold = chi2_dist.quantile(1.0 - alpha)
    
    covers = mahal_sq < threshold
    p_value = 1.0 - chi2_dist.cdf(mahal_sq)
    
    return {
        "covers": bool(covers.numpy()),
        "mahalanobis_distance": float(mahal_sq.numpy()),
        "threshold": float(threshold.numpy()),
        "p_value": float(p_value.numpy())
    }
```

**Success Criterion:**
- Function exists with docstring
- Smoke test: samples from N(0, I_5), true_theta = 0 → covers with p ≈ 0.5

**Bounded I/O:**
- No file reads needed
- Write one test: `tests/inference/test_coverage.py` (~20 lines)

---

## Success Criteria (Phase 0 Complete)

1. ✅ `tolerance_derivation.py` exists, smoke test passes
2. ✅ Seed policy memo exists, < 2 pages, cites theorems
3. ✅ `coverage.py` exists, smoke test passes
4. ✅ All three tests pass:
   - `tests/inference/test_tolerance_derivation.py`
   - `tests/inference/test_coverage.py`
   - (No test for memo — it's documentation)

---

## Result Summary File

**Write:** `results/phase0-summary.json`

```json
{
  "phase": "0",
  "status": "PASS" | "FAIL",
  "date": "2026-09-07",
  "tasks_completed": [
    "tolerance_derivation",
    "seed_policy_memo",
    "joint_mahalanobis_coverage"
  ],
  "tests_passed": [
    "test_tolerance_derivation.py",
    "test_coverage.py"
  ],
  "artifacts": [
    "bayesfilter/inference/tolerance_derivation.py",
    "docs/memos/ledh-surrogate-hmc-seed-policy-2026-09-07.md",
    "bayesfilter/inference/coverage.py"
  ],
  "next_phase": "1"
}
```

---

## On Failure

**If any test fails:**
1. Record failure in `results/phase0-summary.json` with `"status": "FAIL"`
2. Write diagnosis to `results/phase0-failure-diagnosis.md` (< 1 page)
3. Stop and report to user

**Do not proceed to Phase 1 if Phase 0 fails.**

---

## Bounded I/O Instructions

**DO:**
- Create new files (tolerance_derivation.py, coverage.py, tests)
- Write result summary file
- Read theorem statement (lines 1100-1200 only from one file)

**DO NOT:**
- Read full files during implementation
- Read multiple memo files "for context"
- Print full test output to terminal (write to summary file instead)

**If you need to check existing code:**
- Use `grep -n "function_name" file.py` to find line numbers
- Read only those lines with offset/limit
- Do not read entire files

---

## Completion Note Template

After Phase 0 completes, write `docs/plans/phase0-complete.md`:

```markdown
# Phase 0 Complete — [DATE]

**Status:** PASS / FAIL

**What was done:**
- [3 bullet points]

**Tests passed:**
- [list]

**Artifacts:**
- [file paths]

**Next:** Phase 1 (diagnostics)

**Issues encountered:** [if any]
```

---

**Execute Phase 0 with these constraints. Write the completion note when done.**
