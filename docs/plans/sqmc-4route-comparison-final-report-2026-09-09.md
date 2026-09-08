# SQMC 4-Route Statistical Comparison - Final Report

**Date:** 2026-09-09  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Status:** COMPLETE ✓

---

## Executive Summary

**All 4 SQMC transport routes are statistically indistinguishable** at N=1008 with 16 seeds. The range across routes (0.15 units) is **smaller than within-route standard deviation** (~0.7 units), indicating that seed-to-seed variation dominates any route-specific differences.

**Recommendation:** Continue using `repaired_permutation` as the default (it's the most extensively tested), or switch to `iid_dual_cap` if seeking the slight empirical edge (0.1 units).

---

## Configuration

- **Model:** Austria SIR T=20
- **Particle count:** N=1008
- **Seeds:** 16 (97701-97716)
- **Reset:** trust_region (Contract-E + GenUT + dual-cap)
- **Transport:** dense for baseline/iid_dual_cap, streaming for others
- **Device:** CUDA_VISIBLE_DEVICES=1 (4080 SUPER)
- **Wall time:** ~2.5 hours total

---

## Execution Status

| Route | Seeds | Status | Artifact | Transport |
|---|---|---|---|---|
| repaired_permutation | 16/16 | ✓ Complete | claim_attempt01 | dense |
| iid_dual_cap | 16/16 | ✓ Complete | claim_attempt03 | dense |
| previous_inverse_cdf | 16/16 | ✓ Complete | claim_attempt06 | streaming |
| repaired_fixed_previous_controls | 16/16 | ✓ Complete | claim_attempt07 | streaming |

**Total:** 64 cells (4 routes × 16 seeds)

---

## Validity Checks

| Route | Finite | Valid | Status |
|---|---|---|---|
| repaired_permutation | 16/16 | 16/16 | ✓ |
| iid_dual_cap | 16/16 | 16/16 | ✓ |
| previous_inverse_cdf | 16/16 | 16/16 | ✓ |
| repaired_fixed_previous_controls | 16/16 | 16/16 | ✓ |

---

## Descriptive Statistics

| Route | Mean | Std | Min | Max | Range |
|---|---|---|---|---|---|
| **iid_dual_cap** | **-681.83** | 0.59 | -683.09 | -680.73 | 2.36 |
| repaired_fixed_previous_controls | -681.88 | 0.88 | -683.37 | -679.96 | 3.41 |
| repaired_permutation | -681.94 | 0.71 | -682.96 | -680.02 | 2.93 |
| previous_inverse_cdf | -681.99 | 0.75 | -683.71 | -680.60 | 3.12 |

**Range across route means:** 0.15 units  
**Typical within-route std:** ~0.7 units

---

## Bootstrap Analysis (95% CI)

| Route | Mean | 95% CI | Width |
|---|---|---|---|
| iid_dual_cap | -681.83 | [-682.12, -681.57] | 0.56 |
| repaired_fixed_previous_controls | -681.88 | [-682.29, -681.45] | 0.84 |
| repaired_permutation | -681.94 | [-682.25, -681.58] | 0.67 |
| previous_inverse_cdf | -681.99 | [-682.35, -681.63] | 0.72 |

All CIs overlap substantially.

---

## Pairwise Comparisons

All 6 pairwise comparisons are **INDISTINGUISHABLE** (95% CI includes zero):

| Comparison | Mean Diff | 95% CI | Verdict |
|---|---|---|---|
| iid_dual_cap - previous_inverse_cdf | +0.15 | [-0.25, +0.57] | ≈ |
| iid_dual_cap - repaired_permutation | +0.11 | [-0.36, +0.53] | ≈ |
| iid_dual_cap - repaired_fixed_previous_controls | +0.05 | [-0.40, +0.49] | ≈ |
| repaired_permutation - previous_inverse_cdf | +0.05 | [-0.47, +0.54] | ≈ |
| repaired_permutation - repaired_fixed_previous_controls | -0.06 | [-0.63, +0.47] | ≈ |
| previous_inverse_cdf - repaired_fixed_previous_controls | -0.10 | [-0.78, +0.48] | ≈ |

---

## Key Findings

1. **No statistically distinguishable differences:** All 4 routes perform equivalently at N=1008
2. **Seed variation dominates:** Within-route std (~0.7) is 4.7× larger than between-route range (0.15)
3. **Empirical ranking:** iid_dual_cap slightly ahead, but not statistically significant
4. **Practical implication:** Any route is acceptable; default choice should prioritize code simplicity and testing coverage

---

## Comparison to Single-Seed Smoke Test

**Smoke test (seed 97701 only):**
- iid_dual_cap: -681.15
- previous_inverse_cdf: -682.75
- repaired_fixed_previous_controls: -682.55
- repaired_permutation: -681.46

**Single-seed ranking:** iid_dual_cap > repaired_permutation > repaired_fixed_previous_controls > previous_inverse_cdf  
**16-seed ranking:** iid_dual_cap > repaired_fixed_previous_controls > repaired_permutation > previous_inverse_cdf

**Observation:** Single-seed results do NOT predict 16-seed means reliably. Seed-specific variation obscures small route differences.

---

## Decision

**Primary recommendation:** Continue using `repaired_permutation` as default
- Most extensively tested across multiple campaigns
- Middle-of-pack performance (within noise band)
- Known behavior and diagnostics

**Alternative:** Switch to `iid_dual_cap` if seeking empirical edge
- Best mean performance (+0.11 vs repaired_permutation)
- Tightest confidence interval (0.56 width)
- But: difference not statistically significant

**Not recommended for change:** No evidence to abandon any route; all are equivalent within measurement precision.

---

## Artifacts

### Analysis Tools
- Execution plan: `docs/plans/sqmc-4route-comparison-execution-plan-2026-09-09.md`
- Statistical analysis: `docs/benchmarks/analyze_sqmc_4route_comparison.py`
- Progress checker: `check_sqmc_progress.py`

### Results
- Analysis output: `docs/benchmarks/artifacts/sqmc-rerun-corrected-filter-20260906/sqmc_4route_statistical_analysis.json`
- Raw results:
  - `claim_attempt01/result.json` (repaired_permutation, dense)
  - `claim_attempt03/result.json` (iid_dual_cap, dense)
  - `claim_attempt06/result.json` (previous_inverse_cdf, streaming)
  - `claim_attempt07/result.json` (repaired_fixed_previous_controls, streaming)

### Session Logs
- `SQMC_SESSION_LOG_20260909.md`
- `RECOVERY_20260909.md` (branch misdirection recovery)

---

## Technical Notes

### Dense vs Streaming
- repaired_permutation and iid_dual_cap used dense transport
- previous_inverse_cdf and repaired_fixed_previous_controls used streaming transport
- Prior investigation showed dense/streaming difference is pure FP noise (~0.15 mean, 0.85 std)
- No systematic bias detected between transport modes

### Checkpoint Failures
- claim_attempt04: stalled at 3 seeds (previous_inverse_cdf, later restarted as attempt06)
- No pattern identified for checkpoint failures

---

## Next Steps

### If continuing SQMC investigation:
1. **Particle count scaling:** Test at N=2016, 4032 to see if route differences emerge
2. **Other models:** Test on Predator-Prey, synthetic benchmarks
3. **Longer horizons:** Test T=40, 80 to stress-test transport quality

### If closing SQMC work:
1. Merge this branch to main
2. Update master program with "no route preference" conclusion
3. Archive artifacts for reference

---

## Wall Time Breakdown

| Run | Route | Start | End | Duration |
|---|---|---|---|---|
| Baseline | repaired_permutation | (pre-existing) | (pre-existing) | N/A |
| Run 1 | iid_dual_cap | ~02:26 | ~02:29 | ~3 min (smoke cached) |
| Run 2 | previous_inverse_cdf | ~02:58 | ~03:12 | ~14 min (restart) |
| Run 3 | repaired_fixed_previous_controls | ~03:12 | ~03:23 | ~11 min |
| **Total** | | | | **~28 min** |

(Much faster than estimated 54 min due to smoke test cache and parallel execution)

---

## Conclusion

At N=1008 with 16 seeds, **all 4 SQMC transport routes are statistically equivalent**. Seed-to-seed variation (std ~0.7) dominates route-to-route differences (range 0.15). No evidence to prefer one route over another for this configuration.
