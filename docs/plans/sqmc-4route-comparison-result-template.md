# SQMC 4-Route Statistical Comparison - Results

**Date:** 2026-09-09  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Plan:** `docs/plans/sqmc-4route-comparison-execution-plan-2026-09-09.md`  
**Status:** [TO BE FILLED]

---

## Executive Summary

[TO BE FILLED: One paragraph with key finding]

---

## Configuration

- **Model:** Austria SIR T=20
- **Particle count:** N=1008
- **Seeds:** 16 (97701-97716)
- **Reset:** trust_region (Contract-E + GenUT + dual-cap)
- **Transport:** dense
- **Device:** CUDA_VISIBLE_DEVICES=1 (4080 SUPER)
- **Routes tested:** 4

---

## Execution Status

| Run | Route | Seeds | Status | Artifact |
|---|---|---|---|---|
| Baseline | repaired_permutation | 16/16 | ✓ Complete (pre-existing) | claim_attempt01 |
| Run 1 | iid_dual_cap | 16/16 | ✓ Complete | claim_attempt03 |
| Run 2 | previous_inverse_cdf | 16/16 | [STATUS] | claim_attempt04 |
| Run 3 | repaired_fixed_previous_controls | 16/16 | [STATUS] | claim_attempt05 |

**Total:** 64 cells (4 routes × 16 seeds)  
**Wall time:** [TO BE FILLED]

---

## Validity Checks

| Route | Finite Values | Program Valid | Status |
|---|---|---|---|
| repaired_permutation | [16/16] | [16/16] | ✓ |
| iid_dual_cap | [TBD] | [TBD] | [TBD] |
| previous_inverse_cdf | [TBD] | [TBD] | [TBD] |
| repaired_fixed_previous_controls | [TBD] | [TBD] | [TBD] |

---

## Descriptive Statistics

| Route | Mean | Std | Min | Max | Range |
|---|---|---|---|---|---|
| repaired_permutation | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| iid_dual_cap | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| previous_inverse_cdf | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| repaired_fixed_previous_controls | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

---

## Bootstrap Analysis (95% CI)

| Route | Mean | CI Lower | CI Upper | 
|---|---|---|---|
| repaired_permutation | [TBD] | [TBD] | [TBD] |
| iid_dual_cap | [TBD] | [TBD] | [TBD] |
| previous_inverse_cdf | [TBD] | [TBD] | [TBD] |
| repaired_fixed_previous_controls | [TBD] | [TBD] | [TBD] |

---

## Pairwise Comparisons

[TO BE FILLED: Table of all pairwise comparisons with verdicts]

---

## Key Findings

[TO BE FILLED]

1. **Best route:**
2. **Statistical support:**
3. **Spread:**

---

## Comparison to Smoke Test

**Smoke results (single seed 97701):**
- iid_dual_cap: -681.152
- previous_inverse_cdf: -682.752
- repaired_fixed_previous_controls: -682.545
- repaired_permutation: -681.455

**Claim mean vs smoke difference:**
[TO BE FILLED: Show that single-seed ranking doesn't match multi-seed ranking]

---

## Decision

[TO BE FILLED: Route ranking and recommendation]

---

## Artifacts

- Execution plan: `docs/plans/sqmc-4route-comparison-execution-plan-2026-09-09.md`
- Statistical analysis: `docs/benchmarks/analyze_sqmc_4route_comparison.py`
- Analysis output: `docs/benchmarks/artifacts/sqmc-rerun-corrected-filter-20260906/sqmc_4route_statistical_analysis.json`
- Raw results:
  - `claim_attempt01/result.json` (repaired_permutation)
  - `claim_attempt03/result.json` (iid_dual_cap)
  - `claim_attempt04/result.json` (previous_inverse_cdf)
  - `claim_attempt05/result.json` (repaired_fixed_previous_controls)

---

## Wall Time Breakdown

| Run | Start | End | Duration |
|---|---|---|---|
| Run 1 (iid_dual_cap) | [TBD] | [TBD] | [TBD] |
| Run 2 (previous_inverse_cdf) | [TBD] | [TBD] | [TBD] |
| Run 3 (repaired_fixed_previous_controls) | [TBD] | [TBD] | [TBD] |
| **Total** | | | [TBD] |

---

## Next Steps

[TO BE FILLED based on outcome]
