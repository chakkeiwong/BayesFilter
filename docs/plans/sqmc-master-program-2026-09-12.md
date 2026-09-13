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

## Execution Record: Defects Found And Repaired (2026-09-13)

The runner as planned could not execute a single cell. Five defects were found
and repaired before the pilot grid ran. They are recorded here because three of
them would have produced misleading results rather than a crash.

### D-1 Runner called a non-existent API (crash)

`run_sqmc_tuning.py` called `canonical_value_and_analytical_score` with
`model_factory=`, `ancestry_route=`, `controls=`, `particle_count=`. The real
signature is
`(model, theta, initial_states, initial_covariances, noises, observations, ...)`.
The runner also had to generate the per-route point sets itself (IID Gaussian vs
`randomized_halton_joint`) and pass `ancestry_policy`, `reset_design`,
`state_map_policy`, `hilbert_bits`. Repaired by mirroring the baseline runner
`run_sqmc_oracle_characterization.py`.

### D-2 Kalman oracle received the wrong object (crash)

`diagonal_lgssm_canonical_model` returns `(model, set_score_direction)`, but it
was passed directly to `kalman_oracle_value_and_score` as its
`theta_to_lgssm_params` callable, which expects a parameter dict. Repaired with
an explicit `_oracle_score` adapter that mirrors `_oracle` in the baseline
runner, so the tuning comparator is the same exact reference the UNTUNED
baseline used.

### D-3 Wrong target data — comparability defect (silent)

The runner generated its own observations from `np.random.RandomState(88001)`.
The UNTUNED baseline used the frozen canonical target
`_lgssm_frozen_observations()`. TUNED and UNTUNED cells would have been measured
on **different data**, making every TUNED-vs-UNTUNED comparison invalid while
still producing plausible-looking numbers. Repaired to consume the frozen
canonical observations.

### D-4 Hard constraint was a descriptive statistic — evidence defect (silent)

The plan set `cosine >= 0.9995` as a hard veto. That number is the UNTUNED
baseline's **two-seed observed mean**, not a correctness requirement. The
principled-metrics decision framework states the required threshold is
`cosine > 0.999`, with `< 0.99` as the "significantly wrong direction" boundary.

A veto no-fire check (`docs/benchmarks/calibrate_sqmc_tuning_vetoes.py`) run on
the known-good warm-start baseline showed the 0.9995 threshold **rejects 3 of 4
baseline seeds**. Had the grid run with it, the Pareto frontier would have come
back empty or near-empty and the correct reading — "the veto is miscalibrated" —
would have been easy to misread as "no configuration is good enough."

Two repairs:
1. Thresholds now come from the principled-metrics decision framework
   (`cosine >= 0.999`, `rel_norm <= 0.05`, `fisher <= 1.0`), declared as named
   constants and recorded in every artifact with their source.
2. Constraints are applied to the **seed-aggregated mean**, not to individual
   draws. Per-seed cosine is noisy (baseline spans 0.9991682-0.9997341 across
   four seeds), so a per-seed threshold rejects the baseline itself. Per-seed
   cells now veto only on non-finite or raised results.

This is the statistical-evidence-discipline failure the policy names directly: a
descriptive two-seed statistic was promoted into a hard screen.

### D-5 GPU memory policy did not fail closed (governance)

The runner caught `set_memory_growth` failure and printed a warning, then
continued — prohibited by the TensorFlow GPU Memory Rule, which requires serious
runs to fail closed and record the verified policy. Repaired to call
`configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)`. Because
importing `bayesfilter.highdim` initializes the GPU runtime, the call had to move
to module scope **before** those imports; called from `main()` it always raised
`Physical devices cannot be modified after being initialized`. The verified
policy is now recorded in every artifact as `gpu_memory_policy`.

Separately, `CUDA_DEVICE_ORDER=PCI_BUS_ID` was unset, so `CUDA_VISIBLE_DEVICES=1`
selected the RTX 5080 rather than the intended 4080 SUPER. Now set explicitly;
runs confirm `NVIDIA GeForce RTX 4080 SUPER, pci bus id 0000:09:00.0`.

### Dtype decision

The canonical LGSSM adapter is float64-internal (`ledh_canonical_models_tf.DTYPE`)
and the UNTUNED baseline ran float64. Tuning therefore runs **float64**, so
TUNED-vs-UNTUNED varies the controls only. This is a deliberate deviation from
the repository float32/TF32 production execution target: a production-dtype arm
is a separate tuning scope under the LEDH per-scope rule and carries no claim
from this campaign. Recorded in each artifact as `backend: float64_gpu` with a
`backend_note`.

### Measured warm-start baseline (the real comparison target)

Established by the calibration check — float64, frozen canonical target,
`iid_dual_cap`, T=20, N=1008, seeds 50001-50004, warm-start controls
(epsilon 8.0, sinkhorn 8, balance 8, diagonal 0.2, pairwise 0.02):

| Metric | min | mean | max |
|---|---|---|---|
| Score L2 error | 1.3255 | **1.5058** | 1.7043 |
| Cosine similarity | 0.9991682 | **0.9994670** | 0.9997341 |
| Relative norm error | 0.0043 | 0.0142 | 0.0219 |
| Fisher-scaled (max) | 0.4111 | 0.4657 | 0.5588 |

This four-seed baseline supersedes the two-seed range (L2 1.31-1.75) as the
reference for judging whether tuning improves L2. It is descriptive: four seeds
with no uncertainty interval support no ranking claim.

Pilot seeds were aligned to 50001-50004 so the pilot frontier is directly
comparable to this baseline on the same seeds.

---

## Current Status

**Phase:** Executing Phases 2.2 → 3 → 4 (full 4-route campaign)  
**Started:** 2026-09-13 23:50 CST  
**Estimated completion:** ~6.5 hours from start (~05:20 CST)  
**Budget:** 16 hours authorized  

**Progress:**
- Phase 2.1 pilot (iid_dual_cap): ✓ complete
- Phase 3 (iid_dual_cap): ✓ complete — tuned beats baseline by 1.8% on disjoint seeds
- Phase 2.2 (3 remaining routes): **in progress**
  - previous_inverse_cdf: running (config 5/18)
  - repaired_permutation: queued
  - repaired_permutation_ablation: queued
- Phase 3 (3 remaining routes): queued
- Phase 4 (route comparison): queued

**Artifacts in progress:**
- Tuning: `docs/tuning/sqmc-lgssm-t20-n1008-{route}-20260912/`
- Phase 3: `docs/benchmarks/artifacts/sqmc-tuned-vs-untuned-lgssm-20260913/{route}_final/`
- Phase 4: `docs/benchmarks/artifacts/sqmc-route-comparison-20260913/final/`

**Execution log:** `/tmp/sqmc_full_campaign.log`  
**Individual logs:** `/tmp/sqmc_campaign_logs/`

---

## What Phase 4 Will Establish

Which route gives the lowest L2 error **at its own tuned controls** on the same claim seeds. This isolates the route choice while holding tuning effort constant — each route is compared at its best-found settings, not at a common baseline.

**Not established by Phase 4:**
- No statistical test of route superiority (no uncertainty intervals, no paired comparison across routes)
- No cost comparison (routes may differ in per-seed wall-clock)
- No production readiness or HMC convergence benefit
- Does not satisfy the Heuristic Dominance Gate

---

---

## Design Property: The Baseline Is Inside The Grid (recorded 2026-09-13)

The warm-start baseline controls (`epsilon 8.0, sinkhorn 8, balance 8,
diagonal 0.2, pairwise 0.02`) are **grid index 28** of 54 — verified by
reconstructing `_tuning_grid()` ordering. Two consequences that constrain how any
tuning output may be read, both now enforced in
`docs/benchmarks/analyze_sqmc_pilot_frontier.py` rather than left to prose:

1. **Tuning cannot lose by construction.** Because the baseline is itself a
   candidate, the Pareto frontier necessarily contains a baseline-or-better point
   *on the tuning seeds*. "Frontier L2 below baseline L2" is therefore guaranteed
   up to seed noise and is **not** evidence of a tuning benefit. If the selected
   config is ever *worse* than baseline, that means the baseline was
   constraint-rejected or dominated on another objective, and must be inspected
   rather than reported as an improvement.

2. **Winner's curse.** Taking the argmin over 54 configurations scored on four
   seeds biases the selected L2 downward. The selected config's tuning-seed L2 is
   a biased-low estimate of its true L2 and must not be quoted as the tuning
   improvement. The unbiased read requires the disjoint claim seeds in Phase 3,
   which is the specific reason Phase 3 exists and why its seed-disjointness is
   enforced at runtime.

---

## Budget Reconciliation Against Measured Throughput (2026-09-13)

The plan's Phase 2 budget was an estimate made before any cell had run. Measured
throughput on the 4080 SUPER contradicts it and the plan is corrected here.

**Measured:** pilot start 03:40:30, six configs complete by 04:05:08 →
**~4.1 min per configuration** at 4 seeds, i.e. **~1.03 min per seed-cell**
(each seed-cell = 5 directional score evaluations at T=20, N=1008, float64).

| Stage | Cells | Measured projection | Plan estimate |
|---|---|---|---|
| 2.1 pilot (1 route, 54 configs, 4 seeds) | 216 | **~3.7 h** | 1-2 h |
| 2.2 full as specified (4 routes, 54 configs, 16 seeds) | 3,456 | **~59 h** | 8-12 h |

Phase 2.2 as written is **~5x over its stated budget**. Launching it unchanged
would consume roughly 60 GPU-hours, which exceeds what the plan authorised and is
not justified by the evidence the stage would produce.

**This is a budget finding, not a scientific failure.** It does not invalidate
the harness, the target, the oracle, or the pilot. Under the campaign
repair-and-retry rule the target, method, promotion criteria, vetoes and hardware
class are unchanged; only the stage sizing needs revision, and a materially
expanded compute request would need explicit approval rather than being absorbed
silently.

### Finding: one third of the grid was inert (resolved 2026-09-13)

The first pilot attempt was stopped at 13/54 because its own log exposed a defect
in the grid design. Rows differing **only** in `(reset_sinkhorn_steps,
reset_balance_steps)` printed identical L2 and cosine to every displayed decimal,
with a clean period of 6 — precisely the block size of the step dimension.

Direct measurement of the reset core
(`docs/benchmarks/check_sinkhorn_step_sensitivity.py`, CPU-only) settles the
mechanism. `_sinkhorn_contract_e_reset_core` runs `sinkhorn_steps +
balance_steps` iterations of a single fixed-point loop. At the smallest grid
setting the Sinkhorn marginals are **already converged** (row error ~1e-16,
column TV ~1e-7), so further iterations refine the transport only at ~1e-11 and
its tangent at ~1e-9:

| epsilon | steps vs (4,4) | transport delta | tangent delta |
|---|---|---|---|
| 4.0 | (8,8) | 1.07e-11 | 2.38e-09 |
| 4.0 | (16,16) | 3.21e-11 | 7.15e-09 |
| 8.0 | (8,8) | 3.51e-12 | 6.57e-10 |
| 16.0 | (16,16) | 4.90e-12 | 7.13e-10 |

Those deltas are ~9 orders of magnitude below the control effects on score L2
(~1e-2) and ~10 below per-seed L2 noise (~1.7e-1). The dimension is **numerically
nonzero but scientifically inert**: it cannot change which configuration is
selected.

**A methodological note on the check itself.** Its first verdict used an absolute
`1e-12` tolerance and reported SENSITIVE — which tested *bit-identity*, not
decision relevance, and would have justified retaining a useless dimension. The
threshold is now `MATERIAL_DELTA = 1e-6`, justified against the measured control
effect and noise scale. This is the same error class as D-4: a tolerance chosen
without reference to the quantity it is supposed to discriminate.

**Resolution.** The grid is now **18 configurations** (3 epsilon × 3 diagonal × 2
pairwise), with the step count fixed at `(8,8)` — the baseline's own value, which
keeps the baseline controls inside the grid and preserves the baseline-in-grid
self-check. The baseline moved from index 28 to **index 10**.

**Scope limit on the inertness claim.** Measured at the grid's epsilon values, on
a Gaussian cloud, at N=120, in float64. It does not license removing the step
controls from the API, nor assuming inertness at other epsilon, other cloud
geometry, or in float32/TF32 where the noise floor differs.

**Budget consequence.** Phase 2.2 drops from 3,456 cells (~59 h) to 1,152 cells
(~20 h); combined with 4 tuning seeds it is 288 cells (~5 h), back inside the
plan's original 8-12 h envelope. The abandoned log is preserved at
`/tmp/sqmc_pilot_run_ABANDONED_54grid.log`.

### Revised Phase 2.2 options (owner decision)

- **Option R-1 — fewer tuning seeds (4 instead of 16).** 864 cells, **~15 h**.
  Defensible on the design: tuning-seed selection is a *nomination* step whose
  output is validated on disjoint claim seeds in Phase 3, so extra tuning seeds
  buy precision in a quantity that is never itself claimed. Cheapest change,
  keeps the full grid.
- **Option R-2 — evidence-driven grid reduction.** Use the completed pilot to
  identify which control dimensions actually move L2 beyond seed noise, drop the
  inert dimensions, then tune the reduced grid. Requires the pilot to finish
  first; reduction is then justified by measurement rather than by convenience.
- **Option R-3 — R-1 and R-2 combined.** Lands inside the original 8-12 h
  envelope.
- **Option R-4 — one route only.** Complete Phases 2.1→3→4 for `iid_dual_cap`
  and defer the other three routes. Yields a complete, honest answer for one
  route instead of a partial answer for four.

**Interim course taken:** finish the pilot, then run Phase 3 on the pilot's
selected config (2 arms × 16 claim seeds ≈ 33 cells, **~35 min**). That produces
the campaign's actual scientific answer for `iid_dual_cap` — *does exact-scope
tuning beat warm-start on unseen seeds* — at negligible cost and without
pre-committing 60 GPU-hours. The Phase 2.2 sizing decision is then made with the
pilot's measured control sensitivity in hand rather than in advance.

**Not concluded from the interim course:** a single-route Phase 3 result carries
no route-comparison claim, and the three untuned routes remain unaddressed.

---

## Open Gap: Heuristic Dominance Gate Not Applied (recorded 2026-09-13)

The governing policy requires the Heuristic Dominance Gate before interpreting or
reporting any optimised or otherwise complex method. **This campaign has not
applied it.** The omission is recorded here as an open gap rather than quietly
left out, because it is exactly the failure mode the gate was written to catch:
every internal check passes while the question "is the complex machinery beating
the cheap alternative at all?" is never asked.

What the campaign currently compares:
- SQMC ancestry routes against **each other** (two complex methods), and
- SQMC analytical score against the **exact Kalman oracle** (distance-to-oracle).

The second is the certifying metric and it is genuinely strong — on this target an
exact oracle is computable, so distance-to-oracle is available and is reported.
What is missing is the adversary set: nothing establishes that the LEDH transport
machinery (flow, Contract-E reset, dual-cap correction, trust region) earns its
cost against a cheap transport-free alternative.

### Constructed adversary set (not yet run)

Derived from the structure of this problem, not cited from a template:

1. **Exact Kalman score** — the oracle. Already in place as the comparator.
   Certifying metric; on a linear-Gaussian target it is cheap and exact.
2. **Finite differences on the SQMC value** — no analytical tangent machinery at
   all. Tests whether the analytical score path buys accuracy over simply
   differencing the value it already computes.
3. **Bootstrap PF at equal N** — `bootstrap_lgssm_fixed_stream_value_and_directional_score`
   already exists in `ledh_younis_kdm_lgssm_reference_tf.py`. Transport-free at
   the same particle count.
4. **Bootstrap PF at equal wall-clock** — the sharp adversary. Transport is the
   dominant per-cell cost, so a transport-free filter can afford many more
   particles for the same time. If a bootstrap PF at equal wall-clock reaches
   comparable distance-to-oracle, the transport machinery is not earning its cost
   *on this target*.
5. **Transport at reduced N, cost-matched** — the converse direction of (4).

**Estimand caveat that must travel with (3) and (4):** the bootstrap reference's
own docstring states it returns "the total derivative of that finite fixed-stream
bootstrap program, not an unbiased score identity for discrete resampling." It is
therefore a legitimate adversary for *how close to the oracle score a
transport-free program gets*, but it must never be described as computing the same
estimand as the canonical analytical score. Comparing the two requires saying
plainly that the estimands differ.

### Salient situations for conditional evaluation

The gate requires evaluation *conditional on* salient situations, not pooled
averages. For this target:
- **low vs high observation noise** (`r_scale`) — where the filter is
  informative vs diffuse;
- **near-unit-root vs strongly mean-reverting** `phi` — where transport quality
  should matter most;
- **short vs long horizon** — where per-step error accumulates;
- **low vs high particle count** — where transport should pay off or not.

The current campaign evaluates one parameter point (`theta = [0.9, 0.8, 0.7, 0.6,
0.8]`) at one horizon and one N, so it cannot satisfy the conditional requirement
even if the adversaries were run.

### Status and consequence

**Not run. Not scheduled inside the current budget.** The gate is a promotion
veto, so its absence means:

- No claim that the LEDH transport route is a good way to estimate this score may
  be made from this campaign, regardless of how the TUNED-vs-UNTUNED comparison
  turns out.
- The tuning result remains meaningful in its own narrow terms — *does exact-scope
  tuning beat warm-start controls within this route* — because both arms use the
  same machinery, so the comparison is internally valid.
- Per the policy, the relative comparison between complex methods (routes) stays
  uninterpretable as progress until the adversary set is cleared.

A separate, cheap experiment plan should run adversaries (2), (3) and (4) at the
current parameter point before any transport-vs-alternative claim is made. Item
(4) is the one most likely to be informative and is inexpensive relative to the
59 h Phase 2.2 estimate.

**Honest framing of the LGSSM target:** on a linear-Gaussian model the exact
Kalman filter is both cheap and exact, so a particle method is not the method of
choice for this model on its own merits. LGSSM is used here as a *test target*
where ground truth is computable, for machinery intended for nonlinear models.
That is a legitimate use, but it means favourable LGSSM results transfer to the
nonlinear case only by argument, never automatically.

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
