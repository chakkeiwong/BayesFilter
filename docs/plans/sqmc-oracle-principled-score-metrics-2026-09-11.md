# Principled Score Quality Metrics for Oracle Comparison

**Date:** 2026-09-11  
**Context:** SQMC Oracle Comparison Diagnostic  
**Purpose:** Document the standard metrics for assessing gradient quality without running HMC

---

## Motivation

When comparing SQMC analytical scores against an exact oracle (Kalman filter), we need principled ways to assess whether the score errors matter for HMC, beyond raw L2 error or relative error on low-magnitude components.

The user asked: "how can we have a better more principled oriented way of judging if the score matters without actually running HMC?"

---

## Standard Metrics (adopted 2026-09-11)

### 1. Gradient Direction: Cosine Similarity

**Formula:**
```
cos(θ) = (oracle_score · sqmc_score) / (||oracle_score|| × ||sqmc_score||)
```

**Interpretation:**
- cos(θ) ≈ 1.0 means the gradient direction is correct
- cos(θ) < 0.99 suggests the SQMC gradient may point in a significantly wrong direction
- For HMC, gradient direction is more important than magnitude for correctness

**SQMC Results (T=20, N=1008):**
- All routes: 0.9995-0.9996
- **Verdict:** Gradient direction correct to within 0.05%

---

### 2. Relative Gradient Norm Error

**Formula:**
```
rel_norm_error = |||sqmc_score|| - ||oracle_score||| / ||oracle_score||
```

**Interpretation:**
- Measures gradient magnitude accuracy
- Small errors (< 5%) indicate SQMC correctly estimates gradient scale
- For HMC, this affects step size calibration

**SQMC Results:**
- All routes: 0.9-1.7%
- **Verdict:** Excellent gradient magnitude accuracy

---

### 3. Error Scaled by Fisher Information

**Formula:**
```
For each parameter θ_i:
  Err_i / √|oracle_score_i|
```

**Rationale:**
- Fisher information I_ij ≈ E[∇_i log p(y|θ) × ∇_j log p(y|θ)]
- For LGSSM, Fisher scales roughly with observation noise and horizon
- |oracle_score| is a proxy for √(Fisher diagonal)
- Err / √I gives error in natural parameter scale

**Interpretation:**
- Values < 0.1: excellent
- Values 0.1-0.5: good to acceptable
- Values > 0.5: may accumulate in HMC

**SQMC Results:**
- Noise parameters (θ₀, θ₁, θ₂): 0.15-0.50 (excellent to good)
- Observation noise (θ₃, θ₄): 0.008-0.11 (excellent)
- **Verdict:** All parameters within acceptable range

---

### 4. Induced HMC Parameter Error

**Formula:**
```
Induced parameter error per leapfrog step ≈ ε × (gradient_error / |gradient|)
```

**Rationale:**
- HMC leapfrog: θ_{t+1} = θ_t + ε × ∇log p(θ|y)
- Gradient error directly translates to position error
- For typical ε = 0.01 and |gradient| ≈ 30-40:
  - Err ≈ 1.0 → parameter error ≈ 0.0003 per step
  - Err ≈ 0.3 → parameter error ≈ 0.0001 per step

**SQMC Results:**
- Err ≈ 0.2-1.2 across all parameters
- |Gradient| ≈ 30-40
- **Induced parameter error: 0.0003-0.0009 per leapfrog step**
- **Verdict:** Well within acceptable HMC error accumulation

---

## Implementation

**Analysis script:** `sqmc_principled_metrics.py` (worktree root)

**Usage:**
```bash
python3 sqmc_principled_metrics.py
```

**Input:** Requires `diagnostic_attempt02/result.json` with fields:
- `oracle_score`: exact Kalman gradient (5D vector)
- `score`: SQMC analytical score (5D vector)
- `oracle_value`: exact log-likelihood
- `value`: SQMC value estimate

---

## Decision Framework

### When are scores good enough for HMC?

1. **Direction test:** Cosine similarity > 0.999 (required)
2. **Magnitude test:** Relative norm error < 5% (strongly preferred)
3. **Fisher-scaled test:** Most Err/√|Oracle| < 0.5 (acceptable range)
4. **Induced error test:** Parameter error < 0.001 per step with ε=0.01 (rule of thumb)

### When should we worry?

- Cosine similarity < 0.99: gradient direction significantly wrong
- Relative norm error > 10%: gradient magnitude poorly calibrated
- Err/√|Oracle| > 1.0 for multiple parameters: error accumulation risk
- Induced parameter error > 0.01 per step: HMC may drift rapidly

---

## Comparison to Naive Metrics

### What we DON'T use (and why):

**❌ Relative error on low-magnitude scores:**
- "25% error when score is close to 0" → misleading
- Absolute error matters for HMC, not relative error on small numbers

**❌ Score L2 error alone:**
- Doesn't distinguish direction error from magnitude error
- Doesn't scale by Fisher information

**❌ Per-parameter relative errors:**
- Unfair to low-information parameters
- Doesn't account for parameter scale

---

## Authority and Scope

**Scope:** This document defines the standard metrics for oracle comparison diagnostics in the BayesFilter repository.

**Authority:** User-approved on 2026-09-11 after principled metrics discussion.

**Application:** These metrics should be reported for all future SQMC oracle comparisons, alongside raw L2 errors.

**Not a claim:** These metrics show whether gradients are usable for HMC. They do NOT establish statistical route ranking, production readiness, or HMC benefit without exact-scope tuning and multi-seed validation.

---

## References

- Master program: `sqmc-oracle-comparison-master-program-v2-2026-09-09.md`
- SQMC diagnostic artifact: `docs/benchmarks/artifacts/sqmc-oracle-characterization-canonical-20260909/diagnostic_attempt02/result.json`
- Fisher information perspective: standard statistical theory for LGSSM
- HMC error accumulation: leapfrog integration analysis

---

## Changelog

- 2026-09-11: Initial document created after user request for principled metrics
- 2026-09-11: Integrated into master program execution summary
