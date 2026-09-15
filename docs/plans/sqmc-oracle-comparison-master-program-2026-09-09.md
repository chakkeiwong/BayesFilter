# SQMC 4-Route Oracle Comparison Master Program

**Date:** 2026-09-09  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Status:** AUTHORITATIVE - Ready for execution

---

## Executive Summary

This program measures **accuracy** (not just internal consistency) of 4 SQMC transport routes against analytical Kalman filter oracles on LGSSM and KSC-SV models. We test 7 horizon configurations, 2 particle counts, 4 routes, and 16 seeds = **896 cells total**.

**Key difference from Austria SIR comparison:** We measure error against ground truth, revealing which route is most accurate.

---

## Models and Oracles

### Model 1: LGSSM (Linear Gaussian State-Space Model)

**State dimension:** 10  
**Observation dimension:** 10  
**Oracle:** Kalman filter (exact marginal log-likelihood and score)  
**Implementation:** `bayesfilter/highdim/ledh_kalman_oracle_tf.py`

**Horizons:**
- T=20 (baseline)
- T=50 (moderate scaling)
- T=360 (full-year stress test)

### Model 2: KSC-SV (Kim-Shephard-Chib Stochastic Volatility)

**State dimension:** 1 (scalar log-volatility)  
**Observation dimension:** 1  
**Oracle:** Dense Kalman filter (if exists) OR to be implemented  
**Status:** **REQUIRES VERIFICATION** - need to check if dense Kalman oracle exists

**Horizons:**
- T=10 (standard SV horizon)
- T=20 (baseline comparison with LGSSM)
- T=50 (moderate scaling)
- T=120 (quarterly data stress test)

---

## Experimental Configuration

### Particle Counts
- **N=1008:** Matches Austria SIR comparison
- **N=2016:** Tests if route differences emerge at higher resolution

### Seeds
- **16 seeds:** 97701-97716 (same as Austria SIR for consistency)

### SQMC Routes
1. `repaired_permutation` (current default)
2. `iid_dual_cap` (best empirical mean on Austria SIR)
3. `previous_inverse_cdf`
4. `repaired_fixed_previous_controls`

### Reset
- **trust_region** (Contract-E + GenUT + dual-cap)

### Transport
- **streaming** for all (compliant with K≤3000 rule)

---

## Campaign Structure

### Total Scope
```
Models:          2 (LGSSM, KSC-SV)
Horizons:        7 (LGSSM: 3, KSC-SV: 4)
Particle counts: 2 (N=1008, N=2016)
Routes:          4
Seeds:           16

Total cells: 7 × 2 × 4 × 16 = 896 cells
```

### Phase Breakdown

#### Phase 0: Pre-execution Verification (BLOCKING)
**Deliverables:**
1. Verify LGSSM Kalman oracle works (`ledh_kalman_oracle_tf.py`)
2. **Check if KSC-SV dense Kalman oracle exists**
3. **If not exists:** Implement KSC-SV dense Kalman oracle
4. Smoke test: 1 seed × all 7 horizons × 2 models × oracle verification

**Time estimate:** 1-2 hours (4-8 hours if KSC-SV oracle needs implementation)  
**Success criteria:** All oracles return finite values matching expected ranges

---

#### Phase 1: LGSSM Baseline (T=20, N=1008)
**Configuration:**
- Model: LGSSM T=20
- Particle count: N=1008
- Routes: 4
- Seeds: 16

**Cells:** 4 × 16 = 64  
**Time estimate:** ~30 minutes  
**Purpose:** Establish baseline oracle error pattern

**Exit criteria:**
- All runs complete with finite values
- Oracle comparison shows measurable errors
- Proceed to Phase 2 regardless of results

---

#### Phase 2: LGSSM Scaling (T=50, T=360, N=1008)
**Configuration:**
- Model: LGSSM T=50 and T=360
- Particle count: N=1008
- Routes: 4
- Seeds: 16

**Cells:** 2 × 4 × 16 = 128  
**Time estimate:** ~5 hours (T=50: ~1h, T=360: ~4h estimated)  
**Purpose:** Test error accumulation with horizon length

**Decision point after T=50:**
- If T=50 runs exceed 2 hours → mark T=360 as infeasible, skip it
- Otherwise proceed to T=360

---

#### Phase 3: LGSSM Higher Resolution (T=20, T=50, T=360, N=2016)
**Configuration:**
- Model: LGSSM T=20, T=50, T=360
- Particle count: N=2016
- Routes: 4
- Seeds: 16

**Cells:** 3 × 4 × 16 = 192  
**Time estimate:** ~12 hours (2× slower than N=1008)  
**Purpose:** Test if route differences emerge at higher particle count

**Decision point:**
- If Phase 1-2 show all routes indistinguishable → consider running N=2016 with fewer seeds (8 instead of 16) to save time
- User decision required if time budget is tight

---

#### Phase 4: KSC-SV Baseline (T=10, T=20, N=1008)
**Configuration:**
- Model: KSC-SV T=10 and T=20
- Particle count: N=1008
- Routes: 4
- Seeds: 16

**Cells:** 2 × 4 × 16 = 128  
**Time estimate:** ~1 hour  
**Purpose:** Verify findings on nonlinear non-Gaussian model

---

#### Phase 5: KSC-SV Scaling (T=50, T=120, N=1008)
**Configuration:**
- Model: KSC-SV T=50 and T=120
- Particle count: N=1008
- Routes: 4
- Seeds: 16

**Cells:** 2 × 4 × 16 = 128  
**Time estimate:** ~3 hours  
**Purpose:** Test SV-specific error scaling

---

#### Phase 6: KSC-SV Higher Resolution (All T, N=2016)
**Configuration:**
- Model: KSC-SV T=10, T=20, T=50, T=120
- Particle count: N=2016
- Routes: 4
- Seeds: 16

**Cells:** 4 × 4 × 16 = 256  
**Time estimate:** ~8 hours  
**Purpose:** Complete N=2016 testing on KSC-SV

---

### Campaign Timeline Summary

| Phase | Description | Cells | Est. Time | Cumulative |
|---|---|---|---|---|
| 0 | Pre-execution verification | 14 (smoke) | 1-8h | 1-8h |
| 1 | LGSSM T=20, N=1008 | 64 | 0.5h | 1.5-8.5h |
| 2 | LGSSM T=50/T=360, N=1008 | 128 | 5h | 6.5-13.5h |
| 3 | LGSSM all T, N=2016 | 192 | 12h | 18.5-25.5h |
| 4 | KSC-SV T=10/T=20, N=1008 | 128 | 1h | 19.5-26.5h |
| 5 | KSC-SV T=50/T=120, N=1008 | 128 | 3h | 22.5-29.5h |
| 6 | KSC-SV all T, N=2016 | 256 | 8h | 30.5-37.5h |

**Total:** 896 cells, **30-38 hours** (1.3-1.6 days continuous, or 3-5 days with overnight runs)

---

## Metrics and Analysis

### Primary Metrics

#### Value Error
```
absolute_error = |SQMC_value - Kalman_value|
relative_error = (SQMC_value - Kalman_value) / |Kalman_value|
```

#### Score Error (3 parameters for LGSSM, 2 for KSC-SV)
```
L2_norm_error = ||SQMC_score - Kalman_score||_2
relative_L2_error = L2_norm_error / ||Kalman_score||_2
per_dimension_error = SQMC_score[i] - Kalman_score[i]  (for each parameter)
```

### Statistical Analysis (per configuration)

1. **Descriptive statistics** (mean, std, min, max, range) across 16 seeds
2. **Bootstrap 95% CI** for each route's mean error
3. **Pairwise comparisons** between routes (6 comparisons)
4. **Verdict:** Superior / Inferior / Indistinguishable

### Cross-Configuration Analysis

1. **Horizon scaling:** Does error scale linearly, quadratically, or exponentially with T?
2. **Particle count effect:** Does N=2016 reduce errors? By how much?
3. **Model dependence:** Do route rankings differ between LGSSM and KSC-SV?
4. **Value vs Score:** Do routes that minimize value error also minimize score error?

---

## Implementation Plan

### Step 1: Pre-execution Verification (Phase 0)

**Task 1.1:** Verify LGSSM Kalman oracle exists and works
```bash
# Check implementation
ls -la bayesfilter/highdim/ledh_kalman_oracle_tf.py

# Check tests
ls -la tests/highdim/test_ledh_kalman_oracle_tf.py

# Run smoke test if tests exist
python -m pytest tests/highdim/test_ledh_kalman_oracle_tf.py -v
```

**Task 1.2:** Check if KSC-SV dense Kalman oracle exists
```bash
# Search for KSC-SV oracle implementations
grep -r "ksc.*oracle\|sv.*oracle\|dense.*kalman" bayesfilter/highdim/*.py
grep -r "ksc.*oracle\|sv.*oracle" tests/highdim/*.py
```

**Task 1.3:** If KSC-SV oracle missing, implement it
- Subplan: Create `bayesfilter/highdim/ksc_sv_dense_kalman_oracle_tf.py`
- Based on KSC-SV model specification
- Use dense Kalman filter (no approximations)
- Test against known values or synthetic data
- Time estimate: 2-6 hours

**Task 1.4:** Create smoke test runner
- Run 1 seed on all 7 configurations
- Verify oracle returns finite values
- Verify SQMC returns finite values
- Verify errors are computable and reasonable (<1e6)

**Exit criteria:** All smoke tests pass, proceed to Phase 1

---

### Step 2: Create Execution Runner

**File:** `docs/benchmarks/run_sqmc_oracle_comparison.py`

**Key features:**
1. Model selector: LGSSM vs KSC-SV
2. Horizon parameter: T
3. Particle count parameter: N
4. Route parameter: 4 routes
5. Oracle integration:
   - Call Kalman oracle after SQMC completes
   - Record both SQMC and oracle values/scores
   - Compute errors inline
6. Checkpoint/resume support (same as existing runners)
7. Result format:
```python
{
  'model': 'lgssm',
  'horizon': 20,
  'particle_count': 1008,
  'route': 'iid_dual_cap',
  'seed': 97701,
  'sqmc_value': -123.45,
  'oracle_value': -123.40,
  'value_error': 0.05,
  'sqmc_score': [-10.2, 5.3, 1.1],
  'oracle_score': [-10.1, 5.4, 1.0],
  'score_L2_error': 0.17,
  'elapsed_seconds': 45.2,
  # ... diagnostics ...
}
```

**Design decision:** Single unified runner vs separate LGSSM/KSC-SV runners?
- **Recommendation:** Single runner with model parameter (DRY principle)

---

### Step 3: Create Analysis Script

**File:** `docs/benchmarks/analyze_sqmc_oracle_errors.py`

**Inputs:**
- Result JSON files from all phases
- Configuration: which model/horizon/N to analyze

**Outputs:**
1. Per-configuration error analysis:
   - Descriptive stats (mean, std, range)
   - Bootstrap CIs
   - Pairwise comparisons
   - Route ranking
2. Cross-configuration analysis:
   - Horizon scaling plots
   - N=1008 vs N=2016 comparison
   - LGSSM vs KSC-SV comparison
   - Value error vs score error correlation
3. Final recommendation:
   - Best route overall (if exists)
   - Horizon-dependent recommendations (if ranking changes with T)
   - Confidence statement

---

### Step 4: Execution Strategy

**Option A: Sequential (Safer)**
- Run Phase 1, analyze, decide
- Run Phase 2, analyze, decide
- Continue sequentially

**Pros:** Can adapt based on results, save time if routes are indistinguishable  
**Cons:** Takes longer wall-clock time (no parallelism)

**Option B: Parallel within Phase**
- Run all 4 routes in parallel (4 GPUs if available, or batched on 1-2 GPUs)
- Complete full phase before moving to next

**Pros:** Faster wall-clock time  
**Cons:** Commit to full phase before seeing results

**Option C: All-at-once (Riskiest)**
- Launch all 896 cells as batch jobs
- Analyze after everything completes

**Pros:** Fastest wall-clock time  
**Cons:** No opportunity to adapt, may waste compute if routes are equivalent

**Recommendation:** **Option B** (Parallel within Phase, Sequential between Phases)
- Balance between speed and adaptability
- Each phase is a natural decision point

---

### Step 5: Result Archival

**Artifacts directory:**
```
docs/benchmarks/artifacts/sqmc-oracle-comparison-20260909/
├── lgssm_T20_N1008/
│   ├── repaired_permutation_result.json
│   ├── iid_dual_cap_result.json
│   ├── previous_inverse_cdf_result.json
│   └── repaired_fixed_previous_controls_result.json
├── lgssm_T50_N1008/
│   └── ...
├── lgssm_T360_N1008/
│   └── ...
├── lgssm_T20_N2016/
│   └── ...
├── ksc_sv_T10_N1008/
│   └── ...
└── ...
```

**Analysis outputs:**
```
docs/benchmarks/artifacts/sqmc-oracle-comparison-20260909/
├── analysis_lgssm_T20_N1008.json
├── analysis_lgssm_T50_N1008.json
├── ...
├── cross_configuration_analysis.json
└── final_recommendation.md
```

---

## Decision Points and Contingencies

### Decision Point 1: After Phase 0
**If KSC-SV oracle implementation takes >6 hours:**
- Defer KSC-SV to separate follow-up campaign
- Complete LGSSM-only comparison first (Phases 1-3)
- Report "LGSSM complete, KSC-SV pending oracle"

**Decision:** Proceed to Phase 1 if oracle ready, or implement oracle first

---

### Decision Point 2: After Phase 1 (LGSSM T=20, N=1008)
**Scenario A: All routes indistinguishable (like Austria SIR)**
- High confidence Austria SIR finding generalizes
- Recommendation: Run Phase 2 with T=50 only (skip T=360 to save time)
- Consider reducing seeds to 8 for N=2016 phases

**Scenario B: Clear winner emerges**
- Route differences are real and measurable
- Proceed to full Phase 2 and Phase 3 to confirm scaling behavior

**Scenario C: Marginal differences**
- Run full Phase 2 to see if differences grow with horizon

---

### Decision Point 3: After Phase 2 (LGSSM T=50, T=360, N=1008)
**If T=50 or T=360 take >2x estimated time:**
- Mark longer horizons as infeasible at current N
- Skip corresponding N=2016 configurations

**If routes remain indistinguishable:**
- Consider running Phase 3 with 8 seeds instead of 16 (50% time savings)
- User approval required before reducing seeds

**If clear ranking established:**
- Proceed to Phase 3 to verify ranking holds at N=2016

---

### Decision Point 4: After Phase 3 (LGSSM all T, N=2016)
**If LGSSM results are conclusive (all routes equivalent or clear winner):**
- Proceed to KSC-SV (Phases 4-6) as verification only
- Can reduce KSC-SV seeds to 8 if time is tight

**If LGSSM results are inconclusive:**
- KSC-SV becomes critical for tiebreaking
- Run full 16 seeds on KSC-SV

---

### Decision Point 5: After Phase 6 (Campaign Complete)
**Synthesize findings:**
- Do LGSSM and KSC-SV agree on ranking?
- Is ranking stable across horizons and particle counts?
- Is there a clear recommendation, or is choice still arbitrary?

**Outcomes:**
1. **Universal winner:** Promote to default, update master program
2. **Model-dependent:** Document conditional recommendations
3. **Horizon-dependent:** Document horizon-specific guidance
4. **Still indistinguishable:** Confirm Austria SIR finding, no route preference

---

## Risks and Mitigations

### Risk 1: T=360 Infeasible at N=1008
**Likelihood:** Medium (LEDH Phase 4A tested T=20 only)  
**Impact:** Phase 2 incomplete, cannot assess long-horizon scaling  
**Mitigation:**
- Run T=360 smoke test (1 seed) after T=50 completes
- If >10 minutes per seed, mark as infeasible and skip

### Risk 2: KSC-SV Oracle Implementation Delay
**Likelihood:** Medium (depends on whether oracle exists)  
**Impact:** KSC-SV phases delayed 4-8 hours  
**Mitigation:**
- Check oracle existence FIRST (Phase 0 Task 1.2)
- If missing, decide: implement now vs defer KSC-SV

### Risk 3: All Routes Remain Indistinguishable
**Likelihood:** Medium-High (Austria SIR showed this)  
**Impact:** No clear recommendation, 30+ hours confirms "no difference"  
**Mitigation:**
- Early decision points allow stopping after Phase 1-2
- Even "no difference" is a valuable finding (confirms robustness)

### Risk 4: Route Ranking Changes Across Configurations
**Likelihood:** Low-Medium  
**Impact:** No universal recommendation, complex conditional guidance  
**Mitigation:**
- Document all rankings clearly
- Identify if pattern exists (e.g., "best for T<50 is X, for T≥50 is Y")

### Risk 5: GPU Memory Exhaustion
**Likelihood:** Low at N=1008, Medium at N=2016  
**Impact:** Runs crash, need to reduce batch size or N  
**Mitigation:**
- Smoke test each configuration before full 16-seed run
- Monitor GPU memory during execution

---

## Success Criteria

### Phase 0 Success
- [ ] LGSSM Kalman oracle verified (smoke test passes)
- [ ] KSC-SV oracle exists or implemented (smoke test passes)
- [ ] 1 seed × 7 configurations × 2 models × 4 routes = 56 smoke runs complete
- [ ] All oracles return finite values
- [ ] Errors are computable and reasonable magnitude

### Phase 1 Success
- [ ] 64 cells complete (4 routes × 16 seeds)
- [ ] All runs pass validity checks
- [ ] Oracle errors computed for all cells
- [ ] Analysis complete with route ranking
- [ ] Decision made: proceed to Phase 2

### Phase 2 Success
- [ ] 128 cells complete (or fewer if T=360 infeasible)
- [ ] Horizon scaling pattern identified
- [ ] Decision made: proceed to Phase 3 or adjust plan

### Phase 3 Success
- [ ] 192 cells complete (or fewer if adjusted)
- [ ] N=1008 vs N=2016 comparison complete
- [ ] LGSSM-specific recommendation established

### Phase 4-6 Success
- [ ] 512 cells complete (or fewer if adjusted)
- [ ] KSC-SV-specific recommendation established
- [ ] Cross-model comparison complete

### Campaign Success
- [ ] All completed phases analyzed
- [ ] Final recommendation documented
- [ ] Master program updated
- [ ] Results archived and reproducible

---

## Deliverables

### Code
1. `bayesfilter/highdim/ksc_sv_dense_kalman_oracle_tf.py` (if needed)
2. `tests/highdim/test_ksc_sv_dense_kalman_oracle_tf.py` (if needed)
3. `docs/benchmarks/run_sqmc_oracle_comparison.py`
4. `docs/benchmarks/analyze_sqmc_oracle_errors.py`
5. Smoke test script: `docs/benchmarks/smoke_test_sqmc_oracle.py`

### Documentation
1. This master program (authoritative)
2. Per-phase analysis reports (7 reports)
3. Cross-configuration analysis
4. Final recommendation report
5. Updated SQMC guidance in main master program

### Data
1. Raw results: 896 cells (or fewer if adjusted)
2. Analysis outputs: error statistics, CIs, rankings
3. Plots: horizon scaling, N comparison, model comparison

---

## Execution Checklist

### Pre-execution
- [ ] Review this plan thoroughly
- [ ] Verify GPU availability (30-38 hours compute)
- [ ] Check disk space for artifacts (~10 GB estimated)
- [ ] Verify git branch is `rqmc-sqmc-4route-comparison`
- [ ] Commit this plan to repo

### Phase 0: Verification
- [ ] Task 1.1: Verify LGSSM Kalman oracle
- [ ] Task 1.2: Check KSC-SV oracle existence
- [ ] Task 1.3: Implement KSC-SV oracle if needed (BLOCKING)
- [ ] Task 1.4: Run smoke tests
- [ ] Decision: Proceed to Phase 1

### Phase 1: LGSSM T=20, N=1008
- [ ] Create runner script
- [ ] Run 64 cells
- [ ] Analyze results
- [ ] Decision: Proceed to Phase 2

### Phase 2: LGSSM T=50, T=360, N=1008
- [ ] Run T=50 (64 cells)
- [ ] Check T=50 timing, decide on T=360
- [ ] Run T=360 if feasible (64 cells)
- [ ] Analyze results
- [ ] Decision: Proceed to Phase 3

### Phase 3: LGSSM all T, N=2016
- [ ] Run T=20, T=50, T=360 (192 cells or fewer)
- [ ] Analyze results
- [ ] Synthesize LGSSM findings
- [ ] Decision: Proceed to KSC-SV

### Phase 4-6: KSC-SV
- [ ] Run Phase 4 (128 cells)
- [ ] Run Phase 5 (128 cells)
- [ ] Run Phase 6 (256 cells or fewer)
- [ ] Analyze results
- [ ] Synthesize KSC-SV findings

### Final
- [ ] Cross-configuration analysis
- [ ] Final recommendation report
- [ ] Update master program
- [ ] Commit all results
- [ ] Archive artifacts

---

## Approval and Authority

**Created by:** Agent (Opus 5)  
**Date:** 2026-09-09  
**Status:** AUTHORITATIVE after user approval

**User approval signature:**
- [ ] Plan reviewed and approved
- [ ] Ready to execute Phase 0

---

## Revision History

| Date | Version | Changes |
|---|---|---|
| 2026-09-09 | 1.0 | Initial plan created |
