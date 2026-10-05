# SQMC Phase 1 Pilot: Ready with Pareto Optimization

**Date:** 2026-09-12  
**Status:** READY FOR EXECUTION  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Commit:** 46b12fa9

---

## Final Approach: Pareto-Optimal Multi-Objective Selection

### What Changed (User Feedback)

**Initial approach:** Ad-hoc weighting `score = L2 + 0.1*fisher + 0.1*hmc`
- Problem: Arbitrary weights, hides trade-offs

**Revised approach:** Pareto dominance from `~/python/src/common_utils/tf_multiobjective`
- Uses established, tested `nondominated()` function
- Mathematically principled
- Reveals full Pareto frontier
- No arbitrary parameters

---

## Multi-Objective Formulation

### Objectives (All Minimize)

1. **L2 error** - score accuracy
2. **1 - cosine** - direction error
3. **Relative norm error** - magnitude error  
4. **Mean Fisher-scaled** - component balance
5. **Induced HMC error** - practical HMC impact

### Hard Constraints (Vetoes)

Before Pareto analysis, reject configs with:
- Cosine < 0.9995
- Rel norm > 0.05
- Any Fisher-scaled > 1.0

### Selection from Pareto Frontier

Lexicographic ordering:
1. Minimize L2 (primary)
2. Minimize direction error (tiebreak)
3. Minimize rel norm (tiebreak)
4. Minimize Fisher (tiebreak)
5. Minimize HMC error (tiebreak)

---

## What the Runner Does

```python
# 1. Evaluate all 54 configs × 4 seeds
for config in grid:
    for seed in [50001, 50002, 50003, 50004]:
        result = evaluate(config, seed)
        # Computes: L2, cosine, rel_norm, fisher, hmc

# 2. Find Pareto frontier using established tools
from common_utils.tf_multiobjective.population_switching import nondominated
pareto_optimal = find_pareto_optimal(grid_results, seeds)

# 3. Report full frontier
for entry in pareto_optimal:
    print(f"Config {entry.config_idx}:")
    print(f"  L2: {entry.objectives[0]:.4f}")
    print(f"  Direction error: {entry.objectives[1]:.7f}")
    print(f"  Rel norm: {entry.objectives[2]:.4f}")
    print(f"  Fisher: {entry.objectives[3]:.4f}")
    print(f"  HMC: {entry.objectives[4]:.6f}")

# 4. Select best by lexicographic ordering
best = min(pareto_optimal, key=lambda e: e.objectives)
```

---

## Expected Outputs

### Console Output

```
Grid size: 54 configurations
Seeds: 4
Total cells: 216

Grid [1/54]: Valid=4/4, L2=1.4523, Cos=0.999612, Score=1.5234
Grid [2/54]: Valid=3/4, L2=1.3821, Cos=0.999587, Score=1.4532
...
Grid [54/54]: Valid=4/4, L2=1.2145, Cos=0.999634, Score=1.2876

================================================================================
PARETO ANALYSIS
================================================================================

Pareto frontier: 8 configurations

Config 1/8 (grid index 23):
  L2 error: 1.2145
  Direction error (1-cosine): 0.0003660
  Rel norm error: 0.0132
  Mean Fisher-scaled: 0.2341
  Induced HMC error: 0.000345

Config 2/8 (grid index 31):
  L2 error: 1.2287
  Direction error (1-cosine): 0.0003550
  Rel norm error: 0.0128
  Mean Fisher-scaled: 0.2198
  Induced HMC error: 0.000351

...

================================================================================
SELECTED (lexicographic: L2 primary, then direction, magnitude, fisher, hmc)
================================================================================
Grid index: 23
L2 error: 1.2145
Cosine similarity: 0.999634
Relative norm error: 0.0132
Mean Fisher-scaled: 0.2341
Induced HMC error: 0.000345
Valid fraction: 100.00%

Controls:
  reset_epsilon: 8.0
  reset_sinkhorn_steps: 8
  reset_balance_steps: 8
  correction_strength: 0.15
  correction_steps: 4
  pairwise_strength: 0.02
  pairwise_steps: 4

✓ Artifact saved: docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/tuning_artifact.json
```

### Artifact Contents

```json
{
  "tuning_method": "pareto_optimal_grid_search",
  "tuning_objective": "Find Pareto-optimal configs minimizing (L2, 1-cosine, rel_norm, fisher, hmc)",
  "selection_criterion": "Lexicographic ordering from Pareto frontier (L2 primary)",
  "hard_constraints": {
    "cosine_similarity_min": 0.9995,
    "relative_norm_error_max": 0.05,
    "fisher_scaled_error_max": 1.0
  },
  "pareto_tools": "common_utils.tf_multiobjective.population_switching.nondominated",
  "pareto_frontier_size": 8,
  "pareto_frontier": [
    {
      "config_idx": 23,
      "controls": {...},
      "objectives": {
        "l2_error": 1.2145,
        "direction_error": 0.000366,
        "rel_norm_error": 0.0132,
        "fisher_scaled": 0.2341,
        "hmc_error": 0.000345
      }
    },
    ...
  ],
  "best_controls": {...},
  "best_metrics": {...}
}
```

---

## Execution Command

```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909

# Activate environment
source ~/anaconda3/bin/activate tftwogpu

# Set GPU (4080 SUPER)
export CUDA_VISIBLE_DEVICES=1

# Run Phase 1 pilot
python docs/benchmarks/run_sqmc_tuning.py --mode pilot
```

**Estimated time:** 1-2 hours  
**Output:** Tuning artifact + Pareto frontier report

---

## Success Criteria

✓ **Pareto frontier found:** At least 1 config survives hard constraints  
✓ **L2 improvement:** Best Pareto config has L2 < 1.2 (vs UNTUNED 1.31-1.75)  
✓ **Direction maintained:** All Pareto configs have cosine ≥ 0.9995  
✓ **Transparent trade-offs:** Full frontier reported, not just one score

---

## What Happens Next

### If Pilot Succeeds (Expected)
1. Review Pareto frontier (~10 min)
2. Assess L2 improvement magnitude
3. Check for interesting trade-offs in frontier
4. User approval for Phase 2 (full tuning, all routes, 16 seeds, 8-12 hours)

### If Multiple Pareto Configs with Interesting Trade-offs
**Example:**
- Config A: L2=1.15, cosine=0.99960 (better L2)
- Config B: L2=1.20, cosine=0.99965 (better direction)

**Action:** Report trade-off, selected Config A by lexicographic ordering, but user can override if direction is more important

### If Warm-Start on Pareto Frontier
**Interpretation:** Warm-start controls are already Pareto-optimal  
**Action:** Document that tuning doesn't improve over warm-start

---

## Key Advantages of This Approach

1. **Uses your existing tools** - `nondominated()` from ~/python
2. **Principled** - Pareto dominance is mathematically well-defined
3. **No magic numbers** - No arbitrary 0.1 weights
4. **Transparent** - Reports all trade-offs, not just one aggregate score
5. **Reproducible** - Same Pareto frontier every time
6. **Auditable** - Can verify each dominance claim

---

## Ready to Execute

**Status:** All code committed (46b12fa9)  
**Environment:** tftwogpu conda env, GPU 1 (4080 SUPER)  
**Budget:** 1-2 hours for pilot  
**Next:** User approval to execute Phase 1 pilot

**Say "Execute Phase 1 pilot" to begin.**
