# SQMC 4-Route Statistical Comparison - Execution Plan

**Date:** 2026-09-09  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Parent Work:** `docs/plans/sqmc-rerun-corrected-filter-2026-09-06.md`  
**Status:** READY TO EXECUTE

---

## Objective

Complete the SQMC 4-route statistical comparison at N=1008 with 16 seeds per route to enable paired bootstrap analysis and route ranking.

---

## Current State

**Already complete:**
- Smoke test: 4 routes × 3 particle counts × 1 seed = 12 cells ✓
- Claim stage: `repaired_permutation` × 1 particle count × 16 seeds = 16 cells ✓

**Missing:**
- `iid_dual_cap` × 16 seeds at N=1008
- `previous_inverse_cdf` × 16 seeds at N=1008
- `repaired_fixed_previous_controls` × 16 seeds at N=1008

**Total:** 3 routes × 16 seeds = 48 cells

---

## Execution Specification

### Configuration
- **Model:** Austria SIR T=20
- **Particle count:** N=1008 only (focus on statistical comparison first)
- **Seeds:** 97701-97716 (same 16 seeds as `repaired_permutation`)
- **Reset:** trust_region (Contract-E + GenUT + dual-cap)
- **Transport:** dense (use same mode as existing runs)
- **Device:** CUDA_VISIBLE_DEVICES=1 (4080 SUPER)

### Three Sequential Runs

**Run 1: iid_dual_cap**
```bash
CUDA_VISIBLE_DEVICES=1 conda run -n tftwogpu python \
  docs/benchmarks/run_sqmc_rerun_corrected_filter_20260906.py \
  --stage claim \
  --particle-counts 1008 \
  --routes iid_dual_cap \
  --resets trust_region \
  --transport-plan dense
```

**Run 2: previous_inverse_cdf**
```bash
CUDA_VISIBLE_DEVICES=1 conda run -n tftwogpu python \
  docs/benchmarks/run_sqmc_rerun_corrected_filter_20260906.py \
  --stage claim \
  --particle-counts 1008 \
  --routes previous_inverse_cdf \
  --resets trust_region \
  --transport-plan dense
```

**Run 3: repaired_fixed_previous_controls**
```bash
CUDA_VISIBLE_DEVICES=1 conda run -n tftwogpu python \
  docs/benchmarks/run_sqmc_rerun_corrected_filter_20260906.py \
  --stage claim \
  --particle-counts 1008 \
  --routes repaired_fixed_previous_controls \
  --resets trust_region \
  --transport-plan dense
```

**Expected outputs:**
- `claim_attempt03/result.json` (iid_dual_cap)
- `claim_attempt04/result.json` (previous_inverse_cdf)
- `claim_attempt05/result.json` (repaired_fixed_previous_controls)

---

## Budget

**Per run:**
- 16 seeds × ~60 seconds/seed = ~16 minutes
- Plus overhead ~2 minutes
- **Total per route:** ~18 minutes

**Total campaign:**
- 3 routes × 18 min = **54 minutes**

**Hardware:** Single 4080 SUPER, XLA/TF32 enabled

---

## Statistical Analysis Plan

After all 4 routes complete (64 cells total):

### Primary Analysis
1. **Load all 4 route results** (claim_attempt01, 03, 04, 05)
2. **Compute paired differences** (each route - MC baseline) per seed
3. **Bootstrap 95% confidence intervals** (5000 bootstrap samples)
4. **Statistical verdicts:**
   - Superior: CI entirely above zero
   - Inferior: CI entirely below zero
   - Indistinguishable: CI includes zero

### Descriptive Statistics
- Mean, std, min, max per route
- Per-seed values for variance assessment
- Spread across routes

### Comparison to August Baseline
- Mean stability check (current - August)
- Variance comparison

---

## Success Criteria

**Hard vetoes:**
- All 48 cells must return finite values
- All cells must pass `valid=true`

**Primary criterion:**
- At least one route shows statistically superior performance (bootstrap CI > 0)

**Explanatory diagnostics:**
- Per-route descriptive statistics
- Per-seed variance
- Comparison to smoke single-seed results

**What is NOT concluded:**
- Route ranking at N=2016, 4032 (not in scope)
- Variance scaling analysis (would need full ladder)
- Comparison to other models (Austria SIR only)

---

## Decision Table Template

| Route | Mean Diff | 95% CI | Verdict | Rank |
|---|---|---|---|---|
| iid_dual_cap | TBD | TBD | TBD | TBD |
| previous_inverse_cdf | TBD | TBD | TBD | TBD |
| repaired_fixed_previous_controls | TBD | TBD | TBD | TBD |
| repaired_permutation | −0.485 | TBD | TBD | TBD |

*(Values will be filled after analysis)*

---

## Execution Sequence

1. **Pre-execution checks:**
   - Verify on `rqmc-sqmc-4route-comparison` branch
   - Verify tftwogpu conda env available
   - Verify 4080 SUPER accessible (CUDA_VISIBLE_DEVICES=1)

2. **Execute Run 1** (iid_dual_cap, 16 seeds)
   - Monitor progress (16 JSON lines)
   - Verify claim_attempt03/ created
   - Check result.json has 16 rows

3. **Execute Run 2** (previous_inverse_cdf, 16 seeds)
   - Same monitoring as Run 1
   - Verify claim_attempt04/

4. **Execute Run 3** (repaired_fixed_previous_controls, 16 seeds)
   - Same monitoring
   - Verify claim_attempt05/

5. **Statistical analysis:**
   - Create analysis script
   - Compute paired differences
   - Bootstrap CIs
   - Generate decision table

6. **Result documentation:**
   - Write result memo
   - Update this plan with actual outcomes
   - Commit to branch

---

## Artifacts

**Input artifacts:**
- Existing smoke: `smoke_attempt01/result.json`
- Existing claim: `claim_attempt01/result.json` (repaired_permutation)

**Output artifacts:**
- `claim_attempt03/result.json` (iid_dual_cap, 16 rows)
- `claim_attempt04/result.json` (previous_inverse_cdf, 16 rows)
- `claim_attempt05/result.json` (repaired_fixed_previous_controls, 16 rows)
- `sqmc_4route_statistical_analysis.json` (bootstrap results)
- `sqmc-4route-comparison-result-2026-09-09.md` (final report)

---

## Approval Status

**Pre-approved by:** Owner (via Option A selection)  
**Execution authority:** Proceed immediately  
**Budget approved:** 54 minutes, 48 cells

---

## Ready to Execute

All preconditions met. Proceeding with Run 1.
