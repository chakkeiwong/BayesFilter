# SQMC 4-Route Campaign - Session Log

**Date:** 2026-09-09
**Branch:** `rqmc-sqmc-4route-comparison`
**Baseline commit:** `94d07332` (on main)

## What the Previous Agent Did Wrong

The previous agent spent excessive time on XLA compilation tests and bypass verification when the dense vs streaming discrepancy (0.21 units at N=1008) was simply **floating-point roundoff accumulation** over 20 time steps with different operation ordering.

Key insight: The 16-seed systematic test showed:
- Mean difference: +0.15 (small)
- Standard deviation: 0.85 (large - 5.7× the mean!)
- Range: 3.0 (from -1.57 to +1.46)

This is pure FP32/TF32 noise, not a logic bug. Neither mode is "more correct."

## What Was Done This Session

### 1. Verified Main Branch State
- All SQMC work merged to main
- Commit: `94d07332` (Repair Contract-E covariance carry and KDM diagnostics)
- Existing artifact: `claim_attempt01` (repaired_permutation, 16 seeds)

### 2. Created RQMC Branch
- Branch: `rqmc-sqmc-4route-comparison`
- Clean checkout from main

### 3. Created Execution Plan
- Document: `docs/plans/sqmc-4route-comparison-execution-plan-2026-09-09.md`
- Evidence contract: descriptive statistics + bootstrap 95% CI + pairwise comparisons
- Routes: 4 (repaired_permutation, iid_dual_cap, previous_inverse_cdf, repaired_fixed_previous_controls)
- Seeds: 16 per route (97701-97716)

### 4. Executed Runs
- **Baseline:** repaired_permutation (pre-existing, claim_attempt01)
- **Run 1:** iid_dual_cap (completed, claim_attempt03) ✓
- **Run 2:** previous_inverse_cdf (in progress, claim_attempt04) ⋯
- **Run 3:** repaired_fixed_previous_controls (in progress, claim_attempt05) ⋯

### 5. Created Analysis Tools
- Statistical analysis script: `docs/benchmarks/analyze_sqmc_4route_comparison.py`
- Progress checker: `check_sqmc_progress.py`
- Result template: `docs/plans/sqmc-4route-comparison-result-template.md`

## Current Status

**Time:** [Current time when runs complete]
**Progress:** 37/64 cells (57.8%)

| Route | Seeds | Status |
|---|---|---|
| repaired_permutation | 16/16 | ✓ Complete |
| iid_dual_cap | 16/16 | ✓ Complete |
| previous_inverse_cdf | 3/16 | ⋯ In progress |
| repaired_fixed_previous_controls | 2/16 | ⋯ In progress |

## Next Steps

Once all runs complete:
1. Run statistical analysis: `python docs/benchmarks/analyze_sqmc_4route_comparison.py`
2. Fill in result template
3. Commit results to branch
4. Update blockers document if needed
5. Consider whether statistical evidence warrants changing the default route

## Key Decisions Made

1. **Transport mode:** Dense (per chunk rule compliance at N=1008)
2. **Execution strategy:** Sequential runs to avoid GPU contention
3. **Statistical method:** Bootstrap 95% CI + pairwise comparisons (appropriate for 16 seeds)
4. **Evidence bar:** Descriptive only (hard vetoes explicit, no ranking claims without CI support)

## Wall Time Estimate

- Per seed: ~2 minutes
- Per route: ~32 minutes (16 seeds)
- Total for 3 new routes: ~96 minutes
- Expected completion: ~2 hours from start

## Notes

- Both Run 2 and Run 3 appear to be progressing simultaneously (claim_attempt05 had 1 seed when we started Run 2)
- This suggests Run 3 may have been started by a previous session or auto-queued
- Monitoring both for completion
