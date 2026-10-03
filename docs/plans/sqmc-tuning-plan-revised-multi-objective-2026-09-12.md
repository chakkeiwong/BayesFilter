# SQMC Tuning Plan: Multi-Objective Revision

**Date:** 2026-09-12  
**Status:** REVISED - Ready for execution  
**Branch:** `rqmc-sqmc-4route-comparison`

---

## Revision Summary

**Original audit concern:** Plan treated all metrics as if they needed improvement and used L2 error as sole tuning objective.

**User correction:** "We have a number of metrics, i think they should all be taking into account. Maintain those good ones and tune the bad ones."

**Revised strategy:** Multi-objective tuning that maintains excellent direction quality while reducing L2 errors.

---

## UNTUNED Performance Analysis

From LGSSM oracle diagnostic (T=20, N=1008, 2 seeds):

### Metrics That Are Already Excellent ✓

| Metric | UNTUNED Result | Target | Status |
|--------|---------------|--------|--------|
| **Gradient direction (cosine)** | 0.9995-0.9996 | > 0.999 | ✓ EXCELLENT |
| **Relative norm error** | 0.9-1.7% | < 5% | ✓ EXCELLENT |
| **Fisher-scaled errors (obs noise)** | 0.008-0.11 | < 0.5 | ✓ EXCELLENT |
| **Induced HMC error per step** | 0.0003-0.0009 | < 0.001 | ✓ EXCELLENT |

### Metrics That Need Improvement ⚠️

| Metric | UNTUNED Result | Issue |
|--------|---------------|-------|
| **Score L2 error** | 1.31-1.75 | Substantial absolute error |
| **Fisher-scaled errors (state noise)** | 0.15-0.50 | Good but could be better |
| **Value error** | 0.25-0.42 | Moderate error on log-likelihood |

### Key Insight

**Direction is correct, but magnitude/component-wise errors are substantial.**

- Oracle gradient magnitude: ~30-40
- L2 error: 1.31-1.75
- **Relative L2: 3-5%** of gradient magnitude

This matters because:
1. **HMC error accumulation:** Over 100-1000 leapfrog steps, even small per-step errors accumulate
2. **Component-wise balance:** Some parameters may have larger errors than others
3. **Tuning hypothesis:** Transport/correction controls can reduce L2 without degrading direction

---

## Revised Tuning Objective

### Primary Objective
**Reduce score L2 error** from 1.31-1.75 → target < 1.0

### Hard Constraints (Veto Criteria)
These must NOT degrade during tuning:

1. **Cosine similarity ≥ 0.9995** (maintain excellent direction)
2. **Relative norm error ≤ 5%** (maintain magnitude accuracy)
3. **No catastrophic component errors** (Fisher-scaled < 1.0 for all parameters)

### Secondary Objectives
If multiple configurations pass primary + constraints:

1. Minimize Fisher-scaled errors for state noise parameters
2. Minimize induced HMC parameter error
3. Minimize value error

---

## Multi-Objective Tuning Metric

**Tuning score function:**

```python
def tuning_score(result):
    # Hard vetoes
    if result['cosine_similarity'] < 0.9995:
        return float('inf')  # REJECT
    if result['relative_norm_error'] > 0.05:
        return float('inf')  # REJECT
    if any(fisher_scaled > 1.0 for fisher_scaled in result['fisher_scaled_errors']):
        return float('inf')  # REJECT
    
    # Primary objective: minimize L2 error
    primary = result['score_l2_error']
    
    # Secondary objectives (weighted)
    fisher_penalty = 0.1 * np.mean(result['fisher_scaled_errors'])
    hmc_penalty = 0.1 * result['induced_hmc_error']
    
    return primary + fisher_penalty + hmc_penalty
```

**Interpretation:**
- L2 error is primary (weight 1.0)
- Fisher-scaled and HMC errors are tie-breakers (weight 0.1 each)
- Direction and magnitude constraints are hard vetoes

---

## Tuning Grid (Revised)

### Rationale

UNTUNED controls were:
- epsilon = 8.0
- sinkhorn_steps = 8
- balance_steps = 8
- diagonal_strength = 0.2 (Austria SIR) or 0.02 (LGSSM/KSC)
- pairwise_strength = 0.02

**Hypothesis:** These are reasonable warm-starts, but may not be optimal for LGSSM T=20 N=1008.

### Grid Design

**Coarse grid (54 configurations):**

```python
epsilon_values = [4.0, 8.0, 16.0]
step_pairs = [(4, 4), (8, 8), (16, 16)]  # (sinkhorn, balance)
diag_strengths = [0.1, 0.15, 0.2]
pair_strengths = [0.02, 0.03]  # Current default is 0.02
```

**Rationale for ranges:**
- **epsilon:** 4.0-16.0 covers typical Sinkhorn convergence regime
- **steps:** 4-16 balances computation vs convergence
- **diagonal strength:** 0.1-0.2 is validated range from other models
- **pairwise strength:** 0.02-0.03 narrow range around current default

**Not varied (fixed at validated values):**
- `correction_steps = 4` (diagonal)
- `pairwise_steps = 4`
- `coordinate_cap = 0.98`
- `radial_cap = 2.0`
- `lm_damping = 0.01`
- `trust_radius = 0.5`

---

## Execution Plan

### Phase 1: Pilot Tuning (Validation)

**Purpose:** Verify that tuning actually improves L2 error without degrading direction quality

**Scope:**
- **Routes:** `iid_dual_cap` only (simplest, baseline)
- **Seeds:** 4 tuning seeds (50001-50004)
- **Grid:** Full 54 configurations
- **Total:** 54 × 4 = 216 cells

**Success criteria:**
- Best tuned config achieves L2 < 1.2 (improvement from 1.31-1.75)
- Cosine remains ≥ 0.9995
- At least 10 valid configurations survive vetoes

**Budget:** ~1-2 hours GPU

**Decision gate:**
- If pilot shows improvement → proceed to Phase 2 (full tuning)
- If pilot shows no improvement → investigate why, possibly revise grid
- If pilot degrades direction quality → abort, use warm-start controls

---

### Phase 2: Full Tuning (All Routes)

**Conditional on Phase 1 success**

**Scope:**
- **Routes:** All 4 (iid_dual_cap, previous_inverse_cdf, repaired_permutation, repaired_permutation_ablation)
- **Seeds:** 16 tuning seeds (50001-50016)
- **Grid:** Full 54 configurations per route
- **Total:** 4 × 54 × 16 = 3,456 cells

**Per-route tuning artifacts:**
- `docs/tuning/sqmc-lgssm-t20-n1008-<route>-20260912/tuning_artifact.json`

**Budget:** ~8-12 hours GPU

---

### Phase 3: TUNED vs UNTUNED Comparison

**Purpose:** Quantify tuning benefit across all metrics

**Design:**
- **Configuration:** TUNED (best from Phase 2) vs UNTUNED (warm-start)
- **Seeds:** 16 comparison seeds (97701-97716, disjoint from tuning)
- **Routes:** All 4
- **Total:** 4 routes × 2 configs × 16 seeds = 128 cells

**Metrics reported:**

| Metric | UNTUNED | TUNED | Improvement | Maintained? |
|--------|---------|-------|-------------|-------------|
| Score L2 error | 1.31-1.75 | ? | ? | N/A (target) |
| Cosine similarity | 0.9995-0.9996 | ? | ? | ✓ Must maintain |
| Relative norm error | 0.9-1.7% | ? | ? | ✓ Must maintain |
| Fisher-scaled (state) | 0.15-0.50 | ? | ? | Improve |
| Fisher-scaled (obs) | 0.008-0.11 | ? | ? | ✓ Maintain |
| Induced HMC error | 0.0003-0.0009 | ? | ? | ✓ Maintain |
| Value error | 0.25-0.42 | ? | ? | Improve |

**Statistical analysis:**
- Paired comparison (same seeds for TUNED vs UNTUNED)
- Bootstrap confidence intervals
- Sign tests for "tuning improved L2 without degrading direction"

**Budget:** ~2-3 hours GPU

---

### Phase 4: Route Comparison (TUNED)

**Purpose:** With exact-scope tuning, do routes differ?

**Design:**
- Use best tuned configuration per route
- 16 seeds (same as Phase 3, already run)
- Compare routes using all metrics

**Expected outcome (based on Austria SIR precedent):**
- Routes likely remain indistinguishable
- But with TUNED configs, all should have lower L2 errors

**Decision:**
- If routes indistinguishable → use simplest (iid_dual_cap) or most tested (repaired_permutation)
- If routes differ → use best-performing route for LGSSM-class models

---

## Success Criteria

### Phase 1 (Pilot) Success
- ✓ L2 error reduced by ≥10% (target < 1.2)
- ✓ Cosine similarity maintained ≥ 0.9995
- ✓ No degradation in relative norm error

### Phase 2-4 Success
- ✓ All routes show L2 improvement with tuning
- ✓ All excellent metrics maintained (cosine, rel norm, Fisher obs, HMC error)
- ✓ Good metrics improved (Fisher state, value error)
- ✓ Statistical validation complete (16 seeds, bootstrap)

### Overall Success
**Tuning achieves multi-objective improvement:**
- **Primary:** L2 error reduced
- **Constraints:** Direction and magnitude quality maintained
- **Secondary:** Component-wise balance improved

---

## Failure Modes and Responses

### Failure 1: Tuning doesn't reduce L2 error
**Cause:** Warm-start controls already near-optimal, L2 error is irreducible Monte Carlo noise

**Response:** 
- Document that warm-start controls are sufficient
- Use UNTUNED controls with confidence
- Close tuning investigation

### Failure 2: Tuning reduces L2 but degrades cosine
**Cause:** Grid explores region where transport accuracy trades off with direction quality

**Response:**
- Tighten grid around warm-start controls
- Investigate why tighter transport degrades direction
- May indicate warm-start is optimal balance point

### Failure 3: High grid rejection rate (>80% violate vetoes)
**Cause:** Veto thresholds too strict, or grid explores invalid region

**Response:**
- Review veto thresholds (cosine ≥ 0.9995 may be too strict)
- Narrow grid around warm-start
- May indicate warm-start is near optimum

### Failure 4: Routes show different optimal controls
**Cause:** Route-specific transport behavior

**Response:**
- Document per-route optima
- Assess whether differences are meaningful (>10% L2 improvement)
- May justify per-route tuning in production

---

## Revised Audit Response

### Original Audit Findings

The audit was correct that the plan had issues, but the user's correction clarifies:

**Audit claim:** "UNTUNED diagnostic shows success, all metrics excellent"

**User correction:** "One metric good does not mean all the metric is good. The L2 error was not good."

**Revised assessment:**
- ✓ Direction quality is excellent (cosine 0.9995-0.9996)
- ⚠️ L2 error is substantial (1.31-1.75)
- ✓ Magnitude accuracy is good (rel norm 0.9-1.7%)
- ⚠️ Component-wise errors are mixed (Fisher-scaled 0.15-0.50 for state noise)

**Conclusion:** Tuning is justified to improve L2 and component errors while maintaining direction quality.

### Revised Research Question

**"Can exact-scope tuning reduce score L2 errors and component-wise imbalance while maintaining excellent gradient direction and magnitude accuracy?"**

**Hypothesis:** Transport/correction controls can be tuned to reduce L2 error from ~1.5 → ~1.0 without degrading cosine from 0.9995.

**Baseline:** UNTUNED warm-start controls serve as the comparison baseline, not as the final answer.

---

## Timeline

| Phase | Description | Duration | Total |
|-------|-------------|----------|-------|
| 1 | Pilot tuning (iid_dual_cap, 4 seeds) | 1-2 hours | 2 hours |
| 2 | Full tuning (all routes, 16 seeds) | 8-12 hours | 14 hours |
| 3 | TUNED vs UNTUNED comparison | 2-3 hours | 17 hours |
| 4 | Route comparison analysis | 1 hour | 18 hours |
| **Total** | **End-to-end campaign** | | **~18 hours** |

**Calendar time:** 2-3 days with checkpoints for review

---

## Approval Required

**User approval needed to proceed with:**

1. **Phase 1 pilot** (2 hours, validates tuning improves L2 without degrading direction)
2. **Full campaign** (16 hours, only if pilot succeeds)

**Plain-language authorization sufficient:** "Execute Phase 1 pilot" or "Run the revised tuning plan"

---

## Changes from Original Plan

### What Changed

1. **Tuning objective:** Multi-objective (reduce L2, maintain direction) instead of L2-only
2. **Veto criteria:** Hard constraints on cosine, rel norm, Fisher-scaled instead of cosine-only
3. **Baseline comparison:** Added explicit TUNED vs UNTUNED comparison (Phase 3)
4. **Pilot phase:** Added Phase 1 to validate tuning benefit before full campaign
5. **Success definition:** Clear multi-metric criteria instead of "find best route"

### What Stayed the Same

1. Grid design (54 configs, same parameter ranges)
2. Seed allocation (16 tuning, 16 disjoint comparison)
3. Route scope (all 4 ancestry mechanisms)
4. Artifact structure (per-route tuning artifacts)
5. Statistical validation (bootstrap, pairwise tests)

---

## Summary

**The user is correct:** We need to see TUNED vs UNTUNED results across all metrics, not just one.

**Revised plan:**
1. Multi-objective tuning (improve L2, maintain direction)
2. Pilot phase validates tuning benefit (2 hours)
3. Full tuning if pilot succeeds (16 hours)
4. TUNED vs UNTUNED comparison shows improvement across all metrics
5. Route comparison with tuned configs

**Next step:** User approval to execute Phase 1 pilot

---

## Changelog

- 2026-09-12: Original tuning plan created
- 2026-09-12: Audit found issues, recommended closure
- 2026-09-12: User correction - "maintain good metrics, tune bad ones"
- 2026-09-12: Plan revised with multi-objective strategy and pilot phase
