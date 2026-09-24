# SQMC Phase 1 Pilot: Execution Ready

**Date:** 2026-09-12  
**Status:** READY FOR EXECUTION  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Commit:** 879e71f3

---

## What Changed

### User Correction
"We have a number of metrics, i think they should all be taking into account. Maintain those good ones and tune the bad ones."

### Revised Understanding

**UNTUNED performance is MIXED, not uniformly excellent:**

| Metric | UNTUNED | Assessment |
|--------|---------|------------|
| Gradient direction (cosine) | 0.9995-0.9996 | ✓ Excellent |
| Relative norm error | 0.9-1.7% | ✓ Excellent |
| **Score L2 error** | **1.31-1.75** | ⚠️ **Substantial** (3-5% of gradient) |
| Fisher-scaled (obs noise) | 0.008-0.11 | ✓ Excellent |
| Fisher-scaled (state noise) | 0.15-0.50 | ⚠️ Good but improvable |
| Induced HMC error | 0.0003-0.0009 | ✓ Excellent |

**Key insight:** Direction is correct, but magnitude/component errors are substantial and could accumulate in HMC.

---

## Revised Tuning Strategy

### Multi-Objective Optimization

**Primary objective:** Reduce L2 error from 1.31-1.75 → target < 1.0

**Hard constraints (must maintain):**
- Cosine similarity ≥ 0.9995
- Relative norm error ≤ 5%
- Fisher-scaled errors ≤ 1.0 for all parameters

**Secondary objectives:**
- Minimize Fisher-scaled errors for state parameters
- Minimize induced HMC parameter error

### Tuning Score Function

```python
if cosine < 0.9995 or rel_norm > 0.05 or any(fisher > 1.0):
    return inf  # REJECT - violated constraints
else:
    return L2_error + 0.1*mean(fisher) + 0.1*hmc_error
```

---

## Phase 1 Pilot: Execution Plan

### Purpose
**Validate that tuning improves L2 error without degrading direction quality**

### Configuration

- **Route:** `iid_dual_cap` only (simplest baseline)
- **Seeds:** 4 tuning seeds (50001-50004)
- **Grid:** 54 configurations
  - epsilon: [4.0, 8.0, 16.0]
  - steps: [(4,4), (8,8), (16,16)] for (sinkhorn, balance)
  - diagonal_strength: [0.1, 0.15, 0.2]
  - pairwise_strength: [0.02, 0.03]
- **Total cells:** 54 × 4 = 216

### Success Criteria

✓ **Best config achieves L2 < 1.2** (improvement from 1.31-1.75 baseline)  
✓ **Cosine remains ≥ 0.9995** (maintain direction quality)  
✓ **At least 10 valid configs survive vetoes** (grid explores viable region)

### Budget
- **GPU time:** 1-2 hours
- **Output:** `docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/tuning_artifact.json`

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

# Expected output:
# - Grid progress: [1/54], [2/54], ... [54/54]
# - Per-config: Valid=X/4, L2=Y.YYYY, Cos=Z.ZZZZZZ, Score=W.WWWW
# - Best configuration summary
# - Artifact: docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/tuning_artifact.json
```

---

## Decision Gates After Phase 1

### Scenario A: Pilot Shows Improvement ✓
**Criteria:** Best L2 < 1.2, cosine ≥ 0.9995, ≥10 valid configs

**Action:** Proceed to Phase 2 (full tuning, all routes, 16 seeds)

**Interpretation:** Tuning successfully reduces L2 error while maintaining direction quality

---

### Scenario B: Pilot Shows Marginal Improvement
**Criteria:** Best L2 = 1.2-1.3, small improvement over baseline

**Action:** Investigate before proceeding
- Review which grid regions performed best
- Consider refining grid around promising region
- Assess cost/benefit: is 10% improvement worth 16 hours?

**Decision:** User approval required to proceed to Phase 2

---

### Scenario C: Pilot Shows No Improvement
**Criteria:** Best L2 ≥ 1.3, no better than UNTUNED warm-start

**Action:** STOP, investigate root cause
- Warm-start controls may already be near-optimal
- L2 error may be irreducible Monte Carlo variance
- Grid may not include optimal region

**Decision:** Either revise grid or accept UNTUNED controls as sufficient

---

### Scenario D: Pilot Degrades Direction Quality
**Criteria:** L2 improves but cosine drops below 0.9995

**Action:** STOP, investigate trade-off
- Tighter transport may trade L2 accuracy for direction error
- Veto threshold (0.9995) may need revision
- Warm-start may be optimal balance point

**Decision:** Either relax veto or accept UNTUNED controls

---

### Scenario E: High Rejection Rate
**Criteria:** < 10 valid configs (>80% rejected by vetoes)

**Action:** Review veto thresholds
- Cosine ≥ 0.9995 may be too strict (try 0.999)
- Relative norm ≤ 5% may be too strict (try 10%)
- Fisher ≤ 1.0 may be appropriate

**Decision:** Either relax vetoes or narrow grid around warm-start

---

## Monitoring During Execution

### Real-time Progress

Watch for:
1. **Valid fraction per config:** Should be >50% for most configs
2. **L2 progression:** Should see some configs with L2 < 1.3
3. **Cosine stability:** Should stay >0.999 throughout
4. **Tuning score:** Lower is better, inf means veto

### Red Flags

⚠️ **All configs rejected (tuning_score = inf):** Vetoes too strict or grid wrong  
⚠️ **L2 errors increasing:** Grid exploring worse region  
⚠️ **Cosine dropping:** Transport settings degrading direction  
⚠️ **High failure rate:** Implementation or environment issue

### Expected Behavior

- Some configs will violate vetoes (tuning_score = inf) - this is normal
- L2 should vary across grid (1.0 - 2.0 range expected)
- Best configs likely near warm-start values (epsilon=8, steps=8, diag=0.15-0.2)

---

## Outputs

### Tuning Artifact

**Location:** `docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/tuning_artifact.json`

**Contents:**
```json
{
  "schema": "bayesfilter.sqmc_tuning_artifact.v1",
  "route": "iid_dual_cap",
  "model": "diagonal_lgssm_canonical",
  "horizon": 20,
  "particle_count": 1008,
  "tuning_metric": "multi_objective",
  "tuning_objective": "minimize L2 error while maintaining cosine >= 0.9995...",
  "veto_criteria": {
    "cosine_similarity": 0.9995,
    "relative_norm_error": 0.05,
    "fisher_scaled_max": 1.0
  },
  "best_controls": { ... },
  "best_metrics": {
    "mean_tuning_score": ...,
    "mean_score_l2_error": ...,
    "mean_cosine_similarity": ...,
    "valid_fraction": ...
  },
  "grid_size": 54,
  "all_results": [ ... ]
}
```

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
BEST CONFIGURATION
================================================================================
Tuning score: 1.2876
Score L2 error: 1.2145
Cosine similarity: 0.999634
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

---

## Next Steps After Phase 1

### If Pilot Succeeds (Expected)

1. **Review pilot results** (~10 minutes)
   - Check best L2 improvement
   - Verify direction quality maintained
   - Assess grid coverage

2. **User approval for Phase 2** (full tuning)
   - 4 routes × 54 configs × 16 seeds = 3,456 cells
   - 8-12 hours GPU time
   - Requires explicit approval

3. **Execute Phase 2** (if approved)
   ```bash
   python docs/benchmarks/run_sqmc_tuning.py --mode full
   ```

4. **Phase 3: TUNED vs UNTUNED comparison**
   - Quantify tuning benefit across all metrics
   - 16 seeds, paired comparison
   - Statistical validation

---

## Environment Check

Before execution, verify:

```bash
# Working directory
pwd
# Should be: /home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909

# Conda environment
conda env list | grep tftwogpu
# Should show: tftwogpu  */home/chakwong/anaconda3/envs/tftwogpu

# GPU visibility
nvidia-smi
# Should show 4080 SUPER and 5080

# Git status
git status
# Should be on: rqmc-sqmc-4route-comparison, commit 879e71f3
```

---

## Approval Request

**Ready to execute Phase 1 pilot (2 hours, 216 cells, iid_dual_cap only)?**

**Authorization:** Plain language "Execute Phase 1" or "Run the pilot" is sufficient.

**What happens next:**
1. Pilot runs for ~1-2 hours
2. Results reviewed against success criteria
3. Decision gate: proceed to Phase 2 or adjust strategy
4. If successful, Phase 2 requires separate approval (8-12 hours)

---

## Checkpoint

**Current state:**
- ✓ Revised multi-objective tuning plan complete
- ✓ Runner implements all principled metrics
- ✓ Multi-objective scoring with hard vetoes
- ✓ Phase 1 pilot configuration ready
- ⏸️ Awaiting user approval to execute

**Question for user:** Execute Phase 1 pilot?
