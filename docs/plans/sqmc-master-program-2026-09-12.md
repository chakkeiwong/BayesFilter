# SQMC Route Comparison and Tuning: Master Program

**Date:** 2026-09-12  
**Status:** ACTIVE - Phase 1 Pilot Ready  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Latest Commit:** 46b12fa9

---

## Research Question

**Primary:** On LGSSM and Austria SIR models at N=1008, do different SQMC ancestry routes (IID, Hilbert inverse-CDF, Hilbert one-to-one permutation) produce statistically distinguishable analytical score quality for HMC?

**Secondary:** Can exact-scope tuning improve score quality (reduce L2 error while maintaining direction accuracy) over warm-start controls?

---

## Background and Motivation

### Context

LEDH particle filter uses SQMC (Sequential Quasi-Monte Carlo) for score estimation. The ancestry mechanism (how particles are resampled) can use different point sets:
- **IID Gaussian** (baseline, identity ancestry)
- **Randomized Halton + Hilbert inverse-CDF**
- **Randomized Halton + Hilbert one-to-one permutation**

**Question:** Does ancestry mechanism matter for gradient quality?

### Why This Matters

- **HMC correctness:** Gradient direction must be accurate (cosine near 1.0)
- **HMC efficiency:** Gradient magnitude affects step size calibration
- **HMC convergence:** Component-wise errors accumulate over many leapfrog steps
- **Production decision:** Which route should be the default?

### Prior Work

**Austria SIR 16-seed statistical comparison (Sep 9, 2026):**
- Result: All 4 routes **statistically indistinguishable**
- Evidence: Bootstrap CIs overlap, seed variation > route variation
- Configuration: Production tuning (trust-region, dual-cap, Contract-E)
- File: `docs/plans/sqmc-4route-comparison-final-report-2026-09-09.md`

**LGSSM UNTUNED oracle diagnostic (Sep 11, 2026):**
- Compared SQMC scores against exact Kalman oracle
- All routes achieved **cosine 0.9995-0.9996** (excellent direction)
- But **L2 error 1.31-1.75** (3-5% of gradient magnitude - substantial)
- Configuration: Warm-start controls (NOT exact-scope tuned)
- File: `docs/plans/sqmc-oracle-comparison-master-program-v2-2026-09-09.md`

**Principled metrics established (Sep 11, 2026):**
- Cosine similarity (direction correctness) - most important for HMC
- Fisher-scaled errors (parameter-specific accuracy)
- Induced HMC parameter error (practical impact)
- Score L2 alone is misleading
- File: `docs/plans/sqmc-oracle-principled-score-metrics-2026-09-11.md`

---

## Key Insight: Mixed Performance

**User correction (Sep 12, 2026):** "One metric good does not mean all the metric is good. The L2 error was not good."

**UNTUNED performance analysis:**

| Metric | UNTUNED Result | Assessment |
|--------|---------------|------------|
| Gradient direction (cosine) | 0.9995-0.9996 | ✓ Excellent |
| Relative norm error | 0.9-1.7% | ✓ Excellent |
| **Score L2 error** | **1.31-1.75** | ⚠️ **Substantial** |
| Fisher-scaled (obs noise) | 0.008-0.11 | ✓ Excellent |
| Fisher-scaled (state noise) | 0.15-0.50 | ⚠️ Good but improvable |
| Induced HMC error | 0.0003-0.0009 | ✓ Excellent |

**Interpretation:**
- Direction is correct (cosine near 1.0)
- But magnitude/component errors are substantial
- L2 error of ~1.5 on gradient magnitude ~35 = **4% relative error**
- Over 100-1000 HMC leapfrog steps, this accumulates

**Conclusion:** Tuning is justified to improve L2 and component balance while maintaining direction quality.

---

## Research Phases

### Phase 0: Harness Validation ✓ COMPLETE

**Purpose:** Verify oracles work, establish metrics

**Completed:**
- ✓ LGSSM Kalman oracle verified (T=20, exact gradient)
- ✓ KSC-SV dense Kalman oracle verified
- ✓ Canonical score call chain tested (10/10 CPU gates green)
- ✓ Principled gradient quality metrics defined
- ✓ GPU smoke tests passed (all routes finite)

**Artifact:** `docs/benchmarks/artifacts/sqmc-oracle-characterization-canonical-20260909/`

---

### Phase 1: UNTUNED Diagnostic ✓ COMPLETE

**Purpose:** Baseline gradient quality with warm-start controls

**Scope:**
- Model: 3D LGSSM canonical
- Horizon: T=20, Particles: N=1008
- Seeds: 2 (harness diagnostic)
- Routes: All 4 (IID, inverse-CDF, permutation, permutation-ablation)
- Controls: Warm-start (NOT exact-scope tuned)

**Results:**
- All routes finite and valid ✓
- Direction quality: cosine 0.9995-0.9996 ✓
- L2 errors: 1.31-1.75 (substantial) ⚠️
- Seed variation ≈ route variation

**Decision:** UNTUNED shows good direction but substantial L2 error → proceed to tuning

**Status:** ✓ COMPLETE (commit 7b23adbd, Sep 11)

---

### Phase 2: Exact-Scope Tuning ⏸️ IN PROGRESS

**Purpose:** Find Pareto-optimal transport/correction controls per route

#### Phase 2.1: Pilot Tuning ⏸️ READY FOR EXECUTION

**Scope:**
- Route: `iid_dual_cap` only (simplest baseline)
- Seeds: 4 tuning seeds (50001-50004)
- Grid: 54 configurations
  - epsilon: [4.0, 8.0, 16.0]
  - steps: [(4,4), (8,8), (16,16)]
  - diagonal_strength: [0.1, 0.15, 0.2]
  - pairwise_strength: [0.02, 0.03]
- Total: 54 × 4 = 216 cells

**Multi-objective formulation:**
- Objectives (all minimize): L2, 1-cosine, rel_norm, mean_fisher, hmc_error
- Hard constraints: cosine ≥ 0.9995, rel_norm ≤ 0.05, fisher ≤ 1.0
- Selection: Pareto-optimal via `nondominated()` from `~/python/src/common_utils/tf_multiobjective/population_switching.py`
- From Pareto frontier: lexicographic ordering (L2 primary)

**Success criteria:**
- ✓ Pareto frontier found (≥1 config survives constraints)
- ✓ Best L2 < 1.2 (improvement from 1.31-1.75 baseline)
- ✓ Direction maintained (cosine ≥ 0.9995)

**Budget:** 1-2 hours GPU

**Decision gates:**
- If L2 improves & direction maintained → proceed to Phase 2.2 (full tuning)
- If no improvement → investigate grid or accept warm-start
- If direction degrades → investigate trade-off or accept warm-start

**Artifact:** `docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/tuning_artifact.json`

**Status:** ⏸️ READY, awaiting user approval

---

#### Phase 2.2: Full Tuning ⏸️ PENDING PILOT

**Conditional on Phase 2.1 success**

**Scope:**
- Routes: All 4 (iid_dual_cap, previous_inverse_cdf, repaired_permutation, repaired_permutation_ablation)
- Seeds: 16 tuning seeds (50001-50016)
- Grid: Same 54 configs per route
- Total: 4 × 54 × 16 = 3,456 cells

**Per-route tuning artifacts:**
- `docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/`
- `docs/tuning/sqmc-lgssm-t20-n1008-previous-inverse-cdf-20260912/`
- `docs/tuning/sqmc-lgssm-t20-n1008-repaired-permutation-20260912/`
- `docs/tuning/sqmc-lgssm-t20-n1008-repaired-permutation-ablation-20260912/`

**Budget:** 8-12 hours GPU

**Decision gate:** User approval required after reviewing Phase 2.1 results

**Status:** ⏸️ PENDING

---

### Phase 3: TUNED vs UNTUNED Comparison ⏸️ PENDING TUNING

**Purpose:** Quantify tuning benefit across all metrics

**Design:**
- Configuration: TUNED (best from Phase 2) vs UNTUNED (warm-start)
- Seeds: 16 comparison seeds (97701-97716, disjoint from tuning)
- Routes: All 4
- Total: 4 routes × 2 configs × 16 seeds = 128 cells

**Metrics table:**

| Metric | UNTUNED | TUNED | Improvement | Maintained? |
|--------|---------|-------|-------------|-------------|
| Score L2 error | 1.31-1.75 | ? | ? | Target (reduce) |
| Cosine similarity | 0.9995-0.9996 | ? | ? | ✓ Must maintain |
| Relative norm error | 0.9-1.7% | ? | ? | ✓ Must maintain |
| Fisher-scaled (state) | 0.15-0.50 | ? | ? | Improve |
| Fisher-scaled (obs) | 0.008-0.11 | ? | ? | ✓ Maintain |
| Induced HMC error | 0.0003-0.0009 | ? | ? | ✓ Maintain |
| Value error | 0.25-0.42 | ? | ? | Improve |

**Analysis:**
- Paired comparison (same seeds TUNED vs UNTUNED)
- Bootstrap confidence intervals
- Sign tests for directional improvement

**Decision criteria:**
- If tuning improves L2 without degrading direction → tuning validated
- If tuning provides no benefit → warm-start sufficient
- If tuning degrades any excellent metric → investigate or reject tuning

**Budget:** 2-3 hours GPU

**Status:** ⏸️ PENDING

---

### Phase 4: Route Comparison with TUNED Configs ⏸️ PENDING PHASE 3

**Purpose:** With exact-scope tuning, do routes differ?

**Design:**
- Use best tuned config per route from Phase 2
- Seeds: Same 16 comparison seeds (already run in Phase 3)
- Compare routes using all principled metrics

**Expected outcome (based on Austria SIR precedent):**
- Routes likely remain indistinguishable
- But all should have lower L2 errors than UNTUNED

**Decision:**
- If routes indistinguishable → use simplest (iid_dual_cap) or most tested (repaired_permutation)
- If routes differ → document trade-offs, select by multi-objective criteria

**Status:** ⏸️ PENDING

---

## Methodology: Pareto-Optimal Multi-Objective Tuning

### Rationale

**User feedback (Sep 12):** "Under ~/python, we have extensive tools for multiple objective optimization, why don't we leverage those tools instead of coming up with ad hoc tools ourselves?"

**Previous approach (REJECTED):**
```python
score = L2 + 0.1*fisher + 0.1*hmc  # Arbitrary weights!
```

**Current approach (ADOPTED):**
```python
from common_utils.tf_multiobjective.population_switching import nondominated

# Find Pareto-optimal configurations
pareto_optimal = nondominated(entries, absolute_tolerance=1e-12)

# Select by lexicographic ordering
best = min(pareto_optimal, key=lambda e: e.objectives)
```

### Multi-Objective Formulation

**Objectives (all minimize):**
1. Score L2 error
2. 1 - cosine similarity (direction error)
3. Relative norm error
4. Mean Fisher-scaled error
5. Induced HMC parameter error

**Hard constraints (vetoes before Pareto analysis):**
- Cosine similarity < 0.9995 → REJECT
- Relative norm error > 0.05 → REJECT
- Any Fisher-scaled error > 1.0 → REJECT

**Pareto dominance:**
- Config X dominates Y if X is no worse on all objectives AND strictly better on ≥1 objective
- Pareto frontier = all non-dominated configs

**Selection from frontier:**
- Lexicographic ordering: L2 primary, then direction, magnitude, Fisher, HMC as tiebreaks

### Advantages

✅ Uses established, tested tools (`nondominated()` from ~/python)  
✅ Mathematically principled (Pareto dominance)  
✅ No arbitrary weights  
✅ Transparent trade-offs (reports full frontier)  
✅ Reproducible and auditable  

---

## Configuration Status

All cells labeled: `canonical_score_sqmc_float32_tuned_lgssm_t20_n1008` (when tuned)

**Production mechanisms active:**
- Contract-E reset ✓
- GenUT dual-cap correction ✓
- Trust-region damping ✓
- UKF covariance lifecycle ✓
- Validity guards ✓
- Reset source-marginal checks ✓

**Differences from production default:**
- Tuning scope: LGSSM T=20 N=1008 specific (per LEDH per-scope tuning rule)
- Backend: float32/TF32 GPU (production target)
- Execution: Per-direction stable graphs (XLA resource issue in attempt 1)

---

## Evidence Standards

### Hard Vetoes (Correctness)

These must NEVER be violated:
- Cosine similarity ≥ 0.9995 (direction quality)
- Relative norm error ≤ 5% (magnitude quality)
- All Fisher-scaled errors ≤ 1.0 (component quality)

### Statistical Evidence

**Descriptive only:** 2-4 seeds, no uncertainty intervals
- Can nominate configurations
- Cannot support ranking or superiority claims

**Statistical validation:** 16 seeds, bootstrap CIs, paired tests
- Can support route ranking
- Can claim tuning benefit
- Required for production recommendation

### Scope Boundaries

**LGSSM T=20 N=1008 tuning applies ONLY to:**
- 3D LGSSM canonical model
- T=20 horizon
- N=1008 particles
- float32/TF32 GPU
- Contract-E + dual-cap + trust-region

**Does NOT apply to:**
- Other models (Austria SIR, Predator-Prey, KSC-SV)
- Other horizons (T=10, T=50)
- Other particle counts (N=504, N=2016)
- Different reset/correction families

Per LEDH per-scope tuning rule: any changed bound field is a new tuning scope.

---

## Non-Claims

Even after full completion, this program does NOT establish:

❌ **Universal route preference** - Evidence limited to LGSSM and Austria SIR at N=1008  
❌ **Production readiness** - Requires multi-model, multi-horizon validation  
❌ **HMC convergence guarantees** - Gradient quality necessary but not sufficient  
❌ **Generalization to all models** - Only tested on linear and nonlinear epidemic models  
❌ **Optimal particle count** - N=1008 is one tested regime  

---

## Artifacts

### Completed

**UNTUNED diagnostic:**
- `docs/benchmarks/artifacts/sqmc-oracle-characterization-canonical-20260909/diagnostic_attempt02/result.json`
- 8 cells (4 routes × 2 seeds × 5 directions)
- All routes finite, cosine 0.9995-0.9996, L2 1.31-1.75

**Analysis tools:**
- `docs/benchmarks/run_sqmc_oracle_characterization.py` - LGSSM oracle runner
- `sqmc_principled_metrics.py` - Metrics calculator
- `docs/benchmarks/analyze_sqmc_4route_comparison.py` - Statistical analysis (Austria SIR)

**Documentation:**
- `docs/plans/sqmc-4route-comparison-final-report-2026-09-09.md` - Austria SIR statistical comparison
- `docs/plans/sqmc-oracle-comparison-master-program-v2-2026-09-09.md` - LGSSM UNTUNED diagnostic
- `docs/plans/sqmc-oracle-principled-score-metrics-2026-09-11.md` - Metrics standard
- `docs/plans/sqmc-tuning-plan-audit-2026-09-12.md` - Critical audit of initial plan
- `docs/plans/sqmc-tuning-plan-revised-multi-objective-2026-09-12.md` - Multi-objective revision
- `docs/plans/sqmc-tuning-pareto-approach-2026-09-12.md` - Pareto optimization approach

### Pending

**Phase 2.1 pilot tuning:**
- `docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/tuning_artifact.json`

**Phase 2.2 full tuning:**
- 4 route-specific tuning artifacts

**Phase 3 comparison:**
- `docs/benchmarks/artifacts/sqmc-tuned-vs-untuned-comparison-20260912/result.json`

**Phase 4 analysis:**
- `docs/plans/sqmc-tuned-route-comparison-decision-2026-09-12.md`

---

## Execution Commands

### Phase 2.1 Pilot (READY NOW)

```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909
source ~/anaconda3/bin/activate tftwogpu
export CUDA_VISIBLE_DEVICES=1  # 4080 SUPER

python docs/benchmarks/run_sqmc_tuning.py --mode pilot
```

### Phase 2.2 Full Tuning (PENDING PILOT)

```bash
python docs/benchmarks/run_sqmc_tuning.py --mode full
# Or specific routes:
python docs/benchmarks/run_sqmc_tuning.py --mode full --routes iid_dual_cap previous_inverse_cdf
```

### Phase 3 Comparison (PENDING TUNING)

```bash
python docs/benchmarks/run_sqmc_tuned_vs_untuned_comparison.py \
  --tuning-artifacts docs/tuning/sqmc-lgssm-t20-n1008-*/tuning_artifact.json \
  --seeds 97701-97716 \
  --output docs/benchmarks/artifacts/sqmc-tuned-vs-untuned-comparison-20260912
```

---

## Budget

| Phase | Description | Duration | Cumulative |
|-------|-------------|----------|------------|
| 0 | Harness validation | ✓ Complete | - |
| 1 | UNTUNED diagnostic | ✓ Complete | - |
| 2.1 | Pilot tuning (1 route, 4 seeds) | 1-2 hours | 2 hours |
| 2.2 | Full tuning (4 routes, 16 seeds) | 8-12 hours | 14 hours |
| 3 | TUNED vs UNTUNED comparison | 2-3 hours | 17 hours |
| 4 | Route comparison analysis | 1 hour | 18 hours |
| **Total** | **From current state to completion** | | **~18 hours** |

**Calendar time:** 2-3 days with checkpoints for review

---

## Decision Gates and Approval

### Required Approvals

**Phase 2.1 pilot execution:**
- Plain-language: "Execute Phase 1 pilot" or "Run pilot tuning"
- No magic tokens or hash-bound approvals (per Academic Research Governance policy)

**Phase 2.2 full tuning:**
- Conditional on Phase 2.1 success
- Requires explicit approval after reviewing pilot results
- Budget: 8-12 hours GPU

**Phase 3-4 continuation:**
- Conditional on Phase 2.2 completion
- Can proceed without separate approval (within campaign scope)

### Checkpoints for Review

1. After Phase 2.1: Review Pareto frontier, assess L2 improvement
2. After Phase 2.2: Review all 4 route tuning artifacts
3. After Phase 3: Review TUNED vs UNTUNED comparison
4. After Phase 4: Final decision on route recommendation

---

## Recovery Protocol

### If Session Interrupted

**Current state indicators:**
- Branch: `rqmc-sqmc-4route-comparison`
- Latest commit: 46b12fa9
- Worktree: `/home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909`

**Check execution status:**
```bash
# Check for tuning artifacts
ls docs/tuning/sqmc-lgssm-t20-n1008-*/tuning_artifact.json 2>/dev/null

# Check for comparison results
ls docs/benchmarks/artifacts/sqmc-tuned-vs-untuned-comparison-20260912/ 2>/dev/null
```

**Resume from:**
- No artifacts → Start Phase 2.1 pilot
- Pilot artifact only → Review pilot, decide on Phase 2.2
- All route artifacts → Start Phase 3 comparison
- Comparison results → Analyze and write decision

**Context recovery:**
1. Read this master program (current file)
2. Read latest principled metrics: `docs/plans/sqmc-oracle-principled-score-metrics-2026-09-11.md`
3. Read UNTUNED baseline: `docs/plans/sqmc-oracle-comparison-master-program-v2-2026-09-09.md`
4. Check git log for latest progress

---

## Current Status

**Phase:** 2.1 Pilot Tuning  
**Status:** ⏸️ READY FOR EXECUTION  
**Blocker:** User approval  
**Next action:** Execute pilot tuning command  
**Estimated time:** 1-2 hours  

**Command:**
```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909
source ~/anaconda3/bin/activate tftwogpu
export CUDA_VISIBLE_DEVICES=1
python docs/benchmarks/run_sqmc_tuning.py --mode pilot
```

**Approval request:** "Execute Phase 1 pilot"

---

## Changelog

- 2026-09-09: Austria SIR 16-seed comparison complete (all routes indistinguishable)
- 2026-09-11: LGSSM UNTUNED oracle diagnostic complete (good direction, substantial L2)
- 2026-09-11: Principled score quality metrics established
- 2026-09-12: Initial tuning plan created
- 2026-09-12: Audit found critical errors, recommended closure
- 2026-09-12: User correction: "one metric good doesn't mean all metrics good"
- 2026-09-12: Revised with multi-objective approach (L2 + vetoes)
- 2026-09-12: User feedback: leverage ~/python multi-objective tools
- 2026-09-12: Adopted Pareto-optimal selection using `nondominated()`
- 2026-09-12: Master program created for recovery and governance
