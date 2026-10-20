# SQMC Tuning: Pareto-Optimal Multi-Objective Approach

**Date:** 2026-09-12  
**Status:** REVISED - Using established Pareto optimization  
**Branch:** `rqmc-sqmc-4route-comparison`

---

## Revision: Use Established Multi-Objective Tools

**User feedback:** "Under ~/python, we have extensive tools for multiple objective optimization, why don't we leverage those tools instead of coming up with ad hoc tools ourselves?"

**Correct approach:** Use Pareto dominance from `~/python/src/common_utils/tf_multiobjective/population_switching.py`

---

## Multi-Objective Problem Formulation

### Objectives (All Minimize)

1. **Score L2 error** - primary accuracy metric
2. **1 - Cosine similarity** - direction error (minimize means maximize cosine)
3. **Relative norm error** - magnitude error
4. **Mean Fisher-scaled error** - component-wise balance
5. **Induced HMC parameter error** - practical HMC impact

### Hard Constraints (Vetoes)

Before Pareto analysis, filter out configurations that violate:
- Cosine similarity < 0.9995
- Relative norm error > 0.05
- Any Fisher-scaled error > 1.0

These are correctness requirements, not optimization objectives.

### Pareto Optimality

A configuration X **dominates** configuration Y if:
- X is no worse than Y on all objectives
- X is strictly better than Y on at least one objective

A configuration is **Pareto-optimal** if no other configuration dominates it.

---

## Tuning Strategy

### Phase 1: Grid Evaluation
Evaluate all 54 configurations × 4 seeds:
1. Compute all 5 objectives per configuration
2. Apply hard constraint vetoes
3. Identify Pareto frontier using `nondominated()` from population_switching.py

### Phase 2: Pareto Analysis
Among Pareto-optimal configurations:
1. **If single solution:** Use it (unique best)
2. **If multiple solutions:** Select based on primary objective (L2 error)
3. **If empty frontier:** All configs violated vetoes - investigate

### Phase 3: TUNED vs UNTUNED
Compare best tuned config against warm-start baseline across all metrics

---

## Implementation Using Established Tools

```python
import sys
sys.path.insert(0, '/home/chakwong/python/src')

from common_utils.tf_multiobjective.population_switching import (
    dominates,
    nondominated,
    ReplicaEvaluation,
)

def evaluate_config_pareto(grid_results, tuning_seeds):
    """
    Find Pareto-optimal configurations using established tools.
    
    Args:
        grid_results: List of dicts with per-config seed_results
        tuning_seeds: List of seed values
    
    Returns:
        List of Pareto-optimal config indices
    """
    
    # Build evaluation entries
    entries = []
    for config_idx, config_result in enumerate(grid_results):
        # Average across seeds
        valid_seeds = [r for r in config_result['seed_results'] if r['valid']]
        
        if len(valid_seeds) == 0:
            continue
            
        # Check hard constraints (vetoes)
        mean_cosine = np.mean([r['cosine_similarity'] for r in valid_seeds])
        mean_rel_norm = np.mean([r['relative_norm_error'] for r in valid_seeds])
        fisher_errors = [r['fisher_scaled_errors'] for r in valid_seeds]
        max_fisher = max(max(fs) for fs in fisher_errors if fs)
        
        # Apply vetoes
        if mean_cosine < 0.9995:
            continue  # Direction quality veto
        if mean_rel_norm > 0.05:
            continue  # Magnitude quality veto
        if max_fisher > 1.0:
            continue  # Component quality veto
        
        # Compute objectives (all minimize)
        objectives = [
            np.mean([r['score_l2_error'] for r in valid_seeds]),  # L2 error
            1.0 - mean_cosine,  # Direction error (1 - cosine)
            mean_rel_norm,  # Magnitude error
            np.mean([np.mean(r['fisher_scaled_errors']) for r in valid_seeds]),  # Fisher
            np.mean([r['induced_hmc_error'] for r in valid_seeds]),  # HMC error
        ]
        
        # Create evaluation entry
        entry = ReplicaEvaluation(
            replica_id=config_idx,
            candidate_id=str(config_idx),
            objectives=tuple(objectives),
            method='grid_search',  # Not a MOO method, just grid search
            parameters=config_result['controls'],  # For reference
        )
        entries.append(entry)
    
    # Find Pareto frontier using established function
    pareto_optimal = nondominated(entries, absolute_tolerance=1e-12)
    
    return pareto_optimal

def select_best_from_pareto(pareto_optimal):
    """
    Select single best configuration from Pareto frontier.
    
    Strategy: Lexicographic ordering
    1. Primary: L2 error (objectives[0])
    2. Tiebreak: Direction error (objectives[1])
    3. Tiebreak: Magnitude error (objectives[2])
    """
    if not pareto_optimal:
        return None
    
    # Sort by objectives lexicographically
    sorted_pareto = sorted(pareto_optimal, key=lambda e: e.objectives)
    
    return sorted_pareto[0]
```

---

## Expected Outcomes

### Scenario A: Single Pareto-Optimal Config
**Interpretation:** Clear winner across all objectives

**Action:** Use that configuration

### Scenario B: Multiple Pareto-Optimal Configs
**Interpretation:** Trade-offs exist (e.g., lower L2 but slightly worse direction)

**Action:** 
1. Report entire Pareto frontier
2. Select based on lexicographic ordering (L2 primary)
3. Document trade-offs for user decision

**Example:**
- Config A: L2=1.15, cosine=0.99960 (Pareto-optimal)
- Config B: L2=1.20, cosine=0.99965 (Pareto-optimal)
- Trade-off: 4% more L2 error for 0.0005 better direction

### Scenario C: Warm-Start on Pareto Frontier
**Interpretation:** Warm-start controls are already Pareto-optimal

**Action:** Document that tuning doesn't improve over warm-start

### Scenario D: Empty Pareto Frontier
**Interpretation:** All configs violated hard constraints

**Action:** 
1. Relax vetoes (e.g., cosine >= 0.999 instead of 0.9995)
2. Or narrow grid around warm-start
3. Or accept warm-start as sufficient

---

## Comparison to Ad-Hoc Approach

### Old Approach (Ad-Hoc Weighting)
```python
score = L2_error + 0.1*fisher + 0.1*hmc_error
```
**Problems:**
- Arbitrary weights (why 0.1?)
- Assumes objectives commensurable
- Hides trade-offs
- No principled multi-objective reasoning

### New Approach (Pareto Dominance)
```python
pareto_optimal = nondominated(entries)
best = min(pareto_optimal, key=lambda e: e.objectives[0])
```
**Advantages:**
- No arbitrary weights
- Respects all objectives equally
- Reveals trade-offs explicitly
- Uses established, tested tool
- Mathematically principled

---

## Implementation Plan

### Step 1: Update Runner to Compute All Objectives
Already done - runner computes:
- score_l2_error ✓
- cosine_similarity ✓
- relative_norm_error ✓
- fisher_scaled_errors ✓
- induced_hmc_error ✓

### Step 2: Replace Ad-Hoc Scoring with Pareto Analysis
```python
# OLD (ad-hoc)
tuning_score = L2 + 0.1*fisher + 0.1*hmc

# NEW (Pareto)
pareto_optimal = evaluate_config_pareto(grid_results, tuning_seeds)
best = select_best_from_pareto(pareto_optimal)
```

### Step 3: Report Pareto Frontier
```python
print(f"\n{'='*80}")
print(f"PARETO FRONTIER ({len(pareto_optimal)} configurations)")
print(f"{'='*80}\n")

for i, entry in enumerate(pareto_optimal):
    print(f"Config {i+1}/{len(pareto_optimal)} (grid index {entry.replica_id}):")
    print(f"  L2 error: {entry.objectives[0]:.4f}")
    print(f"  Direction error (1-cosine): {entry.objectives[1]:.6f}")
    print(f"  Rel norm error: {entry.objectives[2]:.4f}")
    print(f"  Mean Fisher-scaled: {entry.objectives[3]:.4f}")
    print(f"  Induced HMC error: {entry.objectives[4]:.6f}")
    print()

if best:
    print(f"SELECTED (lowest L2 from Pareto frontier):")
    print(f"  Grid index: {best.replica_id}")
    print(f"  L2 error: {best.objectives[0]:.4f}")
```

---

## Execution Timeline

Same as before:
- Phase 1 pilot: 2 hours (iid_dual_cap, 4 seeds)
- Phase 2 full: 8-12 hours (all routes, 16 seeds)
- Phase 3 comparison: 2-3 hours (TUNED vs UNTUNED)

But now with principled Pareto analysis instead of ad-hoc scoring.

---

## Benefits of This Approach

1. **Uses established, tested code** from ~/python
2. **Mathematically principled** (Pareto dominance is well-defined)
3. **No arbitrary parameters** (no magic weights to tune)
4. **Transparent trade-offs** (reports entire frontier, not just one score)
5. **Reproducible** (same Pareto frontier for any run)
6. **Auditable** (can verify Pareto optimality by checking dominance)

---

## Next Step

Update `run_sqmc_tuning.py` to:
1. Import Pareto tools from ~/python
2. Replace ad-hoc tuning_score with Pareto analysis
3. Report full Pareto frontier
4. Select best by lexicographic ordering

Ready to implement?
