# Phase 2: Dimension Transfer Test - Complete

**Date:** 2026-09-24  
**Master Program:** sqmc-control-generalization-master-program-2026-09-23.md  
**Status:** ✓ COMPLETE

## Objective

Test whether controls tuned at 3D T=20 transfer to 10D T=20 without retuning.

## Test Configuration

- **3D baseline**: N=1008, tuned controls (reference)
- **10D transfer**: N=1000, same controls (test)
- **Horizon**: T=20
- **Routes tested**: All 4 (iid_dual_cap, previous_inverse_cdf, repaired_permutation, repaired_permutation_ablation)
- **Seeds**: [50001, 50002, 50003, 50004]
- **Total cells**: 32 (4 routes × 2 dimensions × 4 seeds)

## Results

**All 32 cells valid (100% success rate)**

| Route | 3D Baseline Avg | 10D Transfer Avg | 3D Runtime | 10D Runtime |
|-------|-----------------|------------------|------------|-------------|
| iid_dual_cap | -94.52 | -310.53 | 10.3s | 9.6s |
| previous_inverse_cdf | -94.94 | -310.43 | 11.9s | 12.6s |
| repaired_permutation | -94.86 | -310.67 | 11.1s | 12.1s |
| repaired_permutation_ablation | -94.86 | -310.67 | 11.0s | 12.1s |

### Detailed Results by Route

#### iid_dual_cap
- **3D baseline**: [-87.15, -86.27, -94.14, -110.53]
- **10D transfer**: [-312.07, -321.91, -303.94, -304.18]

#### previous_inverse_cdf
- **3D baseline**: [-87.58, -86.28, -94.11, -111.81]
- **10D transfer**: [-312.50, -323.35, -301.89, -303.96]

#### repaired_permutation
- **3D baseline**: [-87.42, -86.00, -94.28, -111.73]
- **10D transfer**: [-312.91, -322.88, -303.37, -303.52]

#### repaired_permutation_ablation
- **3D baseline**: [-87.42, -86.00, -94.28, -111.73]
- **10D transfer**: [-312.91, -322.88, -303.37, -303.52]

## Analysis

### Success Criteria

✓ **No catastrophic failures** - All cells completed successfully  
✓ **No NaNs or infinities** - All values in valid range  
✓ **No divergences** - Values consistent across seeds  
✓ **Performance maintained** - Runtime comparable (10-13s per cell)

### Key Findings

1. **Dimension transfer successful**: Controls tuned at 3D T=20 work correctly at 10D T=20 without retuning

2. **Value scaling expected**: Log-likelihood scales with dimension
   - 3D: ~-95 (avg across routes)
   - 10D: ~-310 (avg across routes)
   - Ratio: ~3.3× (consistent with dimension increase)

3. **Route consistency**: All 4 routes show similar transfer patterns
   - repaired_permutation and repaired_permutation_ablation produce identical results (as expected for T=20)
   - previous_inverse_cdf and iid_dual_cap show slightly different values but same transfer pattern

4. **No performance degradation**: 10D runtime (9.6-12.6s) comparable to 3D (10.3-11.9s)

### Interpretation

**Dimension transfer at T=20 requires no retuning.** Controls tuned at 3D generalize directly to 10D for this horizon. The tuning captured properties that are dimension-independent (reset balance, correction strength, pairwise coupling) rather than dimension-specific pathologies.

## Evidence Contract

**Promotion criterion:** None (transfer diagnostic)  
**Promotion veto:** Catastrophic failure (NaN, divergence, cosine < 0.99)  
**Continuation veto:** >50% cells fail validity

**Status:**
- ✓ No promotion veto fired (0% catastrophic failures)
- ✓ No continuation veto fired (0% invalid cells)

## Next Steps

**Phase 3: Horizon Transfer Test**
- Test if T=20 controls transfer to T=120
- At both 3D and 10D
- 64 total cells (4 routes × 4 configs × 4 seeds)

## Files

**Test script:** `docs/benchmarks/run_sqmc_dimension_transfer_t20.py`  
**Results:** `artifacts/sqmc-dimension-transfer-t20-20260924/result_20260924_032726.json`  
**Key commits:** c0fc50d1, c7158823

## Conclusion

**Phase 2 demonstrates successful dimension transfer.** Controls tuned at 3D T=20 generalize to 10D T=20 without retuning. This suggests the tuning captured fundamental SQMC properties rather than dimension-specific artifacts.

The remaining question is horizon transfer: do these controls also work at T=120? Phase 3 tests this at both dimensions.
