# SQMC Phase 3 Investigation: Why Only iid_dual_cap Shows Tuning Benefit?

**Date:** 2026-09-16  
**Status:** INVESTIGATION PLAN  
**Question:** Why does iid_dual_cap show statistical tuning benefit while 3 other routes don't, despite near-identical tuned controls?

---

## Observed Pattern

**Phase 3 Results (16 claim seeds: 97701-97716):**
- **iid_dual_cap**: L2 reduction −0.0252, 95% CI [−0.043, −0.008] → **favours_tuned** ✓
- **previous_inverse_cdf**: L2 reduction −0.0073, 95% CI [−0.043, +0.029] → indistinguishable
- **repaired_permutation**: L2 reduction −0.0279, 95% CI [−0.059, +0.003] → indistinguishable (near-miss)
- **repaired_permutation_ablation**: L2 increase +0.0083, 95% CI [−0.012, +0.029] → indistinguishable

**Tuned Controls (nearly identical across routes):**
- All 4 routes: `pairwise_strength` 0.02→0.03, `reset_epsilon` 8→16
- 3 routes: `correction_strength` 0.2→0.15 (only iid_dual_cap kept 0.2)

**Pareto Frontier Evidence:**
- All routes show L2 sensitivity to controls on tuning seeds (spread 0.017-0.035)
- Not a flat tuning surface — controls do matter

---

## Competing Hypotheses

### H1: Seed-Specific Overfitting
**Claim:** Tuned controls improved L2 on tuning seeds (50001-50016) but failed to generalize to claim seeds (97701-97716) for 3 routes.

**Prediction:** 
- If we use **different claim seeds** (e.g., 88801-88816), the pattern changes
- If we use **tuning seeds as claim seeds** (50001-50016), all 4 routes show benefit

**Test 1A: Swap to new claim seeds (88801-88816)**
```bash
python docs/benchmarks/run_sqmc_tuned_vs_untuned_comparison.py \
  --route iid_dual_cap \
  --tuning-artifact docs/tuning/sqmc-lgssm-t20-n1008-iid_dual_cap-20260912/tuning_artifact.json \
  --seeds 88801 88802 88803 88804 88805 88806 88807 88808 88809 88810 88811 88812 88813 88814 88815 88816 \
  --output docs/benchmarks/artifacts/sqmc-seed-sensitivity-test-20260916
```
Run for all 4 routes. If H1 is true: different routes show benefit with different seed sets.

**Test 1B: Use tuning seeds as claim seeds (in-sample test)**
```bash
python docs/benchmarks/run_sqmc_tuned_vs_untuned_comparison.py \
  --route previous_inverse_cdf \
  --tuning-artifact docs/tuning/sqmc-lgssm-t20-n1008-previous_inverse_cdf-20260912/tuning_artifact.json \
  --seeds 50001 50002 50003 50004 50005 50006 50007 50008 50009 50010 50011 50012 50013 50014 50015 50016 \
  --output docs/benchmarks/artifacts/sqmc-insample-test-20260916
```
Run for all 4 routes. If H1 is true: ALL routes show strong benefit on their own tuning seeds.

---

### H2: Route-Specific Algorithmic Sensitivity
**Claim:** iid_dual_cap's algorithm (i.i.d. resampling) responds more robustly to Sinkhorn control changes than permutation/inverse-CDF routes.

**Prediction:**
- iid_dual_cap shows benefit across **multiple different seed sets**
- Other routes remain indistinguishable across multiple seed sets

**Test 2: Multi-seed-set replication**
Run Phase 3 comparison on 3 additional disjoint seed sets:
- Set A: 97701-97716 (original, DONE)
- Set B: 88801-88816 (proposed Test 1A)
- Set C: 77701-77716 (new)
- Set D: 66601-66616 (new)

If H2 is true: iid_dual_cap shows benefit in ≥3/4 sets, others show benefit in ≤1/4 sets.

---

### H3: Statistical Power Limitation
**Claim:** All routes actually improve, but 16 seeds is insufficient power to detect smaller effect sizes.

**Prediction:**
- Larger seed sets (32-64 seeds) reveal benefit for all routes
- Effect sizes similar across routes, just smaller for 3 routes

**Test 3: Extended replication (32 seeds)**
Run Phase 3 with 32 seeds (97701-97732) for all 4 routes.

If H3 is true: More routes cross significance threshold with larger N.

---

### H4: Control Convergence Artifact
**Claim:** Pareto optimization converged to suboptimal controls for 3 routes due to limited grid search or multi-objective trade-offs.

**Prediction:**
- Re-tuning with denser grid or single-objective L2 optimization finds different controls
- New controls show benefit on claim seeds

**Test 4: Targeted re-tuning**
For the 3 routes that showed no benefit, run single-objective L2 tuning (no Pareto) with denser grid around the current best controls.

If H4 is true: New controls differ materially and show benefit on claim seeds.

---

## Minimal Discriminating Test Ladder

### Immediate (cheapest, highest value):
1. **Test 1B (in-sample)**: Use tuning seeds as claim seeds for all 4 routes
   - Cost: ~2 hours (4 routes × 30 min)
   - Discriminates: H1 (overfitting) vs H2 (algorithmic)
   - If all 4 routes show benefit → H1 likely
   - If only iid_dual_cap shows benefit → H2 likely

### If Test 1B confirms H1 (all routes improve in-sample):
2. **Test 1A (new seeds)**: Run all 4 routes on fresh seed set 88801-88816
   - Cost: ~2 hours
   - Validates: Whether benefit pattern changes with different seeds
   - Expected: Different route shows benefit, or all show indistinguishable

### If Test 1B confirms H2 (only iid_dual_cap improves in-sample):
3. **Test 2 (multi-seed)**: Run iid_dual_cap and repaired_permutation on 2 more seed sets
   - Cost: ~1 hour (2 routes × 2 sets × 15 min)
   - Validates: Whether iid_dual_cap consistently outperforms across seeds

### If near-miss suggests power issue:
4. **Test 3 (extended N)**: Run repaired_permutation with 32 seeds (97701-97732)
   - Cost: ~45 min
   - Validates: Whether near-miss reaches significance with more power

---

## Decision Criteria

**Stop investigation if:**
- Test 1B shows all 4 routes improve in-sample → Confirms H1 (overfitting), no further tuning justified
- Test 1B shows only iid_dual_cap improves → Confirms H2 (route-specific), recommend iid_dual_cap

**Continue investigation if:**
- Results are ambiguous or suggest multiple mechanisms

---

## Resource Budget

- Total investigation budget: 6 hours GPU time
- Test 1B (in-sample): 2 hours (PRIORITY 1)
- Test 1A (new seeds): 2 hours (PRIORITY 2, conditional)
- Test 3 (extended N): 1 hour (PRIORITY 3, conditional)
- Reserve: 1 hour

---

## Execution Status

- [ ] Test 1B: In-sample validation (tuning seeds as claim seeds)
- [ ] Test 1A: New seed set (88801-88816)
- [ ] Test 2: Multi-seed replication
- [ ] Test 3: Extended replication (32 seeds)
- [ ] Test 4: Re-tuning

**Next action:** Execute Test 1B for all 4 routes to discriminate H1 vs H2.
