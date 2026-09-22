# SQMC 4-Route Oracle Comparison Plan

**Date:** 2026-09-09  
**Branch:** `rqmc-sqmc-4route-comparison` (continue on this branch)  
**Status:** DRAFT - awaiting approval

---

## Problem Statement

The completed SQMC 4-route comparison on Austria SIR T=20 found all routes statistically indistinguishable, but **Austria SIR has no oracle**. We measured internal consistency (value and score variability) but not **accuracy**.

To properly evaluate SQMC routes, we need oracle-based comparison on models where ground truth is available:
- **LGSSM**: Kalman filter provides exact value and score
- **KSC SV**: Dense Kalman filter oracle available

---

## Objectives

1. **Measure accuracy** (not just internal consistency) of 4 SQMC routes against Kalman oracle
2. **Test scaling behavior** across multiple horizons (T=20, T=50, T=360)
3. **Compare both value and score errors** to identify if routes differ in gradient quality
4. **Establish oracle-based route ranking** (if one exists)

---

## Models and Horizons

### LGSSM (Linear Gaussian State-Space Model)

**Oracle:** Kalman filter (`bayesfilter/highdim/ledh_kalman_oracle_tf.py`)

**Configurations:**
- **T=20:** Moderate horizon, baseline comparison
- **T=50:** Longer horizon, tests error accumulation
- **T=360:** Full-year daily data, stress test

**State dimension:** 10 (standard configuration)  
**Observation dimension:** 10

### KSC SV (Kim-Shephard-Chib Stochastic Volatility)

**Oracle:** Dense Kalman filter (already implemented)

**Configuration:**
- **T=10:** Standard SV horizon from prior work

**State dimension:** 1 (scalar log-volatility)  
**Observation dimension:** 1

---

## Experimental Design

### Particle Counts
- **N=1008:** Same as Austria SIR comparison (for consistency)
- **N=2016:** (optional) Test if route differences emerge at higher N

### Seeds
- **16 seeds:** Same as Austria SIR (97701-97716)

### Routes
1. `repaired_permutation` (current default)
2. `iid_dual_cap` (best empirical mean on Austria SIR)
3. `previous_inverse_cdf`
4. `repaired_fixed_previous_controls`

### Reset
- **trust_region** (Contract-E + GenUT + dual-cap)

### Transport
- **streaming** for all (compliant with K≤3000 rule at N=1008+)

---

## Metrics

### Value Error
- **Absolute error:** `|SQMC_value - Kalman_value|`
- **Relative error:** `(SQMC_value - Kalman_value) / |Kalman_value|`

### Score Error (per dimension and L2 norm)
- **Absolute error:** `||SQMC_score - Kalman_score||_2`
- **Relative error:** `||SQMC_score - Kalman_score||_2 / ||Kalman_score||_2`
- **Per-dimension error:** Check if routes differ on specific parameters

### Statistical Analysis
- **Bootstrap 95% CI** for each route's mean error
- **Pairwise comparisons** of error between routes
- **Ranking:** Identify if any route is significantly more accurate

---

## Campaign Structure

### Phase 1: LGSSM T=20 (Baseline)
- 4 routes × 16 seeds × N=1008 = 64 cells
- Estimated time: ~30 minutes
- **Purpose:** Establish if oracle-based comparison differs from Austria SIR

### Phase 2: LGSSM T=50 (Scaling)
- 4 routes × 16 seeds × N=1008 = 64 cells
- Estimated time: ~60 minutes
- **Purpose:** Test error accumulation over longer horizon

### Phase 3: LGSSM T=360 (Stress Test)
- 4 routes × 16 seeds × N=1008 = 64 cells
- Estimated time: ~4 hours (estimate)
- **Purpose:** Full-year stress test, identify if routes diverge

### Phase 4: KSC SV T=10 (Alternative Model)
- 4 routes × 16 seeds × N=1008 = 64 cells
- Estimated time: ~20 minutes
- **Purpose:** Verify findings on non-Gaussian nonlinear model

**Total cells:** 256 (4 phases × 64 cells)  
**Total estimated time:** ~5-6 hours

---

## Success Criteria

### Scenario A: Routes Remain Indistinguishable
If all routes show **comparable accuracy** (error differences < seed variation):
- **Conclusion:** Austria SIR finding generalizes; no route preference
- **Recommendation:** Continue using `repaired_permutation` (most tested)

### Scenario B: Clear Winner Emerges
If one route shows **significantly lower error** (pairwise comparisons significant):
- **Conclusion:** Oracle reveals accuracy differences not visible in internal consistency
- **Recommendation:** Promote the most accurate route to default

### Scenario C: Horizon-Dependent Ranking
If route ranking **changes with horizon**:
- **Conclusion:** No universal best route; horizon-specific recommendation needed
- **Recommendation:** Provide conditional guidance (T<50 vs T≥50)

---

## Implementation Plan

### Step 1: Runner Adaptation
Adapt `run_sqmc_rerun_corrected_filter_20260906.py` to:
- Use LGSSM/KSC-SV models instead of Austria SIR
- Call Kalman oracle after each SQMC run
- Record both SQMC and oracle values/scores
- Compute errors inline

### Step 2: Phased Execution
Run phases sequentially:
1. LGSSM T=20 (fastest, establishes pattern)
2. If T=20 shows distinguishable differences → continue to T=50, T=360
3. If T=20 shows indistinguishable → run T=50 only as confirmation
4. KSC SV T=10 as final verification

### Step 3: Analysis
- Adapt `analyze_sqmc_4route_comparison.py` to handle oracle errors
- Bootstrap CI on errors (not raw values)
- Pairwise error comparisons

### Step 4: Decision
Based on results:
- Update SQMC route recommendation in master program
- Archive Austria SIR comparison as "internal consistency only"
- Document oracle-based route selection

---

## Risk Assessment

### Computational Cost
- **Total:** ~5-6 hours GPU time
- **Mitigation:** Run overnight or in background

### LGSSM T=360 Feasibility
- **Risk:** May exceed memory or time budget at N=1008
- **Mitigation:** Run T=360 pilot with 1 seed first; if infeasible, drop T=360

### Oracle Implementation Gaps
- **Risk:** Kalman oracle may not integrate cleanly with existing runner
- **Mitigation:** Test oracle on 1 seed smoke test before full campaign

---

## Open Questions for User

1. **Priority:** Should this run immediately, or after other work?
2. **Scope:** Run all 4 phases, or start with LGSSM T=20 and decide based on results?
3. **Particle count:** N=1008 only, or also N=2016?
4. **KSC SV:** Is the dense Kalman oracle implementation already available and tested?

---

## Deliverables

1. **Execution runner:** `run_sqmc_oracle_comparison.py`
2. **Analysis script:** `analyze_sqmc_oracle_errors.py`
3. **Results:** Per-phase error analysis with bootstrap CIs
4. **Final report:** Oracle-based route ranking and recommendation
5. **Master program update:** Authoritative SQMC route guidance

---

## Next Actions

Awaiting user approval to:
1. Adapt runner for LGSSM + Kalman oracle integration
2. Run Phase 1 (LGSSM T=20) as pilot
3. Proceed based on Phase 1 findings
