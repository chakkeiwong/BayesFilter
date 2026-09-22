# SQMC 4-Route Campaign Reset Memo, 2026-09-22

## Scope Boundary

**This worktree and branch:** `/home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909`, branch `rqmc-sqmc-4route-comparison`

**Not in scope:** The observation-aware TT work on `surrogate-hmc` branch in main checkout belongs to someone else. Do not work on it, read its plans, or reference its artifacts.

## Research Question

Evaluate four SQMC gradient estimation routes on LGSSM models:
1. `iid_dual_cap` (baseline)
2. `previous_inverse_cdf` 
3. `repaired_permutation`
4. `repaired_permutation_ablation`

**Primary question:** Do tuned controls from T=20 N=1008 campaign generalize to:
- Higher state dimensions (3D → 10D)
- Longer horizons (T=20 → T=120)

## Completed Work

### Phase 1: Tuning Campaign (T=20, N=1008, 3D state)
- Tuned all 4 routes on 3D LGSSM, T=20 horizon, 1008 particles
- Tuning artifacts saved in `docs/tuning/sqmc-lgssm-t20-n1008-<route>-20260912/`
- Each artifact contains: `best_controls`, `tuning_history`, `search_config`

### Phase 2: Test 1B In-Sample Validation (T=20)
- Used tuning seeds as claim seeds (in-sample validation)
- Initially overwrote results - fixed with route-specific tags
- Scripts: `docs/benchmarks/run_sqmc_test_1b_*.py` (4 separate scripts)
- Results in `docs/benchmarks/artifacts/sqmc-test-1b-20260917/`
- Status: All 4 routes launched, need to collect and analyze results

### Phase 3: T=120 Tests (3D state, WRONG DIMENSION)
- Fixed tuning artifact path resolution (relative → absolute using REPO_ROOT)
- Committed fix: `33e0a5d "Fix: Use absolute paths for tuning artifacts in 3D T=120 test"`
- Script: `docs/benchmarks/run_sqmc_3d_t120_tuned.py`
- **Problem:** User requested 10D state, but 3D tests were run
- All 4 routes completed successfully in 5.5-6.2 minutes per route
- Results saved to `docs/benchmarks/artifacts/sqmc-3d-t120-20260917/`

## Critical Problems Found

### Problem 1: Wrong State Dimension
**What happened:** User explicitly requested "Test LGSSM with state dimension 10 and T=120", but 3D tests were executed instead.

**Evidence:** 
- User request: "Test LGSSM with state dimension 10 and T=120 for the 4 different methods, tuned version"
- Script executed: `run_sqmc_3d_t120_tuned.py` (STATE_DIM = 3)
- Script available but not used: `run_sqmc_10d_t120_tuned.py` (has bug at line 87)

**Status:** 3D results exist but do not answer the research question. 10D tests not run.

### Problem 2: TensorFlow Graph Policy Violation
**What:** The canonical SQMC score executor violates repository TensorFlow Graph And Compilation Policy.

**Location:** `bayesfilter/highdim/ledh_canonical_score_tf.py:_value_and_analytical_score_impl`

**Violations:**
1. No `@tf.function` decorator on performance-critical T-step loop
2. Python `for time_index in range(horizon):` loop at line 292
3. Python list accumulation: `trace.append()`
4. Python array indexing: `observations[time_index]`
5. No XLA compilation (`jit_compile=True`)

**Policy requirement (CLAUDE.md):**
> "Repeated TensorFlow scientific kernels must execute through `tf.function` with an explicit, stable `input_signature`. Shape polymorphism and retracing must be bounded and justified. Eager execution is reserved for diagnostics, smoke checks, and documented exceptions."

**Impact:** 
- T=120 tests ran in eager mode: 5.5-6.2 minutes per route for 4 seeds
- Should compile to single fused XLA kernel
- Affects ALL SQMC benchmarks, tuning campaigns, and tests

**Blockers to fix:**
- Python `for` loop → needs `tf.while_loop` or `tf.scan`
- Python indexing → needs `tf.gather` or tensor slicing
- `trace.append()` → needs tensor accumulator or optional/external recording
- Conditional branches → needs `tf.cond` or static dispatch

### Problem 3: 10D Script Bug
**File:** `docs/benchmarks/run_sqmc_10d_t120_tuned.py`

**Bug:** Line 87 uses `R.DTYPE` where `R` is the observations array, not the module. Should be `DTYPE` or import from module properly.

**Status:** Not fixed, script not runnable.

## Next Actions (In Order)

### Immediate: Write This Reset Memo
✓ Done - this document

### Action 1: Fix 10D Script and Run 10D Tests
- Fix line 87 bug in `run_sqmc_10d_t120_tuned.py`
- Run all 4 routes with tuned controls at 10D T=120
- Use same seeds: [97801, 97802, 97803, 97804]
- Compare with 3D results to assess dimensionality generalization

### Action 2: Collect and Analyze All Results
- Test 1B (T=20, in-sample): 4 routes
- T=120 3D: 4 routes (wrong dimension, but data exists)
- T=120 10D: 4 routes (pending Action 1)

### Action 3: Address Graph Compilation Violation
**Decision required:** This is a systemic issue affecting all SQMC work, not just this campaign. Options:
1. File as technical debt and continue with eager execution
2. Fix now before further experiments
3. Defer to owner/maintainer

**If fixing now:**
1. Create experiment plan for graph conversion
2. Convert Python loop to `tf.while_loop` or `tf.scan`
3. Add `@tf.function` with stable `input_signature`
4. Add XLA compilation after compatibility check
5. Verify numerical equivalence with current eager path
6. Run regression tests

## Configuration

**Hardware:** GPU 1 (NVIDIA GeForce RTX 4080 SUPER, 13495 MB)
**Backend:** TensorFlow/TFP, float64, TF32 disabled
**Environment:** conda env `tf-gpu` (or `tftwogpu`)
**Particle count:** N=1008
**Seeds:** [97801, 97802, 97803, 97804] for T=120 tests

## Key Files

**Tuning artifacts (inputs):**
```
docs/tuning/sqmc-lgssm-t20-n1008-iid_dual_cap-20260912/tuning_artifact.json
docs/tuning/sqmc-lgssm-t20-n1008-previous_inverse_cdf-20260912/tuning_artifact.json
docs/tuning/sqmc-lgssm-t20-n1008-repaired_permutation-20260912/tuning_artifact.json
docs/tuning/sqmc-lgssm-t20-n1008-repaired_permutation_ablation-20260912/tuning_artifact.json
```

**Test scripts:**
```
docs/benchmarks/run_sqmc_test_1b_iid_dual_cap.py
docs/benchmarks/run_sqmc_test_1b_previous_inverse_cdf.py
docs/benchmarks/run_sqmc_test_1b_repaired_permutation.py
docs/benchmarks/run_sqmc_test_1b_repaired_permutation_ablation.py
docs/benchmarks/run_sqmc_3d_t120_tuned.py
docs/benchmarks/run_sqmc_10d_t120_tuned.py  (HAS BUG)
```

**Results (outputs):**
```
docs/benchmarks/artifacts/sqmc-test-1b-20260917/
docs/benchmarks/artifacts/sqmc-3d-t120-20260917/
```

**Core implementation:**
```
bayesfilter/highdim/ledh_canonical_score_tf.py  (POLICY VIOLATION)
bayesfilter/highdim/ledh_alg1_contract.py
bayesfilter/highdim/transport_sinkhorn.py
bayesfilter/inference/sqmc_tuning.py
```

## Evidence Contract

**Promotion criteria (for tuned controls):**
- Valid score computation (no NaN, no crash)
- Numerical stability across all seeds
- Reasonable wall time (< 2 hours per route at T=120)

**Explanatory diagnostics:**
- Score norm per seed
- Wall time per seed
- Comparison across dimensions (3D vs 10D)
- Comparison across horizons (T=20 vs T=120)

**What is NOT being concluded:**
- Statistical superiority of any route (insufficient seeds/replication)
- Production readiness (in-sample validation only)
- Oracle accuracy (no oracle comparison for T=120)
- HMC performance (score quality, not downstream use)

## Budget and Runtime

**Observed T=120 runtimes (3D, 4 seeds, eager mode):**
- iid_dual_cap: 329.4s (5.5 min)
- previous_inverse_cdf: 332.0s (5.5 min)
- repaired_permutation: 329.4s (5.5 min)
- repaired_permutation_ablation: did not complete in initial run

**Expected 10D runtimes:** Higher than 3D due to increased state dimension, but exact scaling unknown.

**No formal budget:** This is local research work, not a charged campaign.

## Git State

**Branch:** `rqmc-sqmc-4route-comparison`
**Latest commit:** `b69fba17 "Add unified correction/reset tests and oracle contract test"`
**Untracked files:** None (all committed)
**Dirty changes:** None

## Stop Conditions

**Stop if:**
- 10D tests reveal tuning artifact incompatibility requiring re-tuning
- Runtime exceeds 4 hours per route per seed
- Persistent NaN/crash despite tuned controls
- Graph compilation fix requires substantial redesign (escalate to owner)

**Do not stop for:**
- Descriptive differences in score norms across routes (expected)
- Single-seed anomalies (not enough replication for ranking)
- Eager-mode performance (known issue, not a blocker for results)
