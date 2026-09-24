# Dense vs Streaming Transport Investigation

**Date:** 2026-09-08  
**Question:** Is the dense vs streaming discrepancy systematic or random?  
**Status:** Testing in progress (16 seeds, N=1008)  

## Background

The smoke test showed dense and streaming transport modes differ by **0.21** at N=1008, despite both using K=1008 (full chunk) where they should be bitwise identical.

## Previous Agent's Mistakes

1. **Created multiple broken test scripts** instead of using the existing working runner
2. **Got stuck in tool-use API errors** from complex monitoring setups  
3. **Didn't check existing data first** - smoke results already had 2 overlapping seeds
4. **Lost focus on the actual question** - spent time on XLA compilation tests that weren't directly relevant

## What We Actually Need

Just run the existing claim-stage runner (16 seeds) with both transport modes and compare the results statistically.

## Initial Evidence (2 seeds from smoke tests)

| Seed | N | Route | Dense | Streaming | Diff |
|------|---|-------|-------|-----------|------|
| 97701 | 1008 | repaired_permutation | -681.454712 | -681.665649 | **-0.211** |
| 97701 | 4032 | repaired_permutation | -681.995605 | -681.660156 | **+0.335** |

**Preliminary findings from 2 seeds:**
- Mean difference: +0.062 (≈ 0)
- Sign flips between N=1008 and N=4032
- Suggests **symmetric, random FP noise** rather than systematic bias

**But 2 seeds is insufficient!** Need 16 seeds for confidence.

## Hypotheses to Test

### H1: Bitwise Identical
- All 16 seeds show diff = 0
- Smoke discrepancy was measurement error
- **Prediction:** Very unlikely given smoke evidence

### H2: Symmetric Random Noise (most likely)
- Mean diff ≈ 0
- Std dev ~ 0.1-0.5
- Positive and negative diffs roughly balanced
- **Cause:** FP32/TF32 roundoff accumulation over 20 time steps
- **Implication:** Neither mode is "more correct"; choose based on chunk rule compliance

### H3: Systematic Bias
- Mean diff significantly non-zero (|mean| > 0.1)
- Low coefficient of variation (std/|mean| < 0.5)
- **Cause:** One mode has a consistent numerical error
- **Implication:** Need to identify which mode is correct

### H4: Large Divergence
- Mean diff > 0.5
- **Cause:** Logic bug or non-determinism
- **Implication:** Serious problem requiring investigation

## Current Test Design

**Runner:** `run_sqmc_rerun_corrected_filter_20260906.py`  
**Stage:** `claim` (seeds 97701-97716, 16 seeds)  
**Configuration:**
- N = 1008 (K=1008 for both modes)
- Route = repaired_permutation
- Reset = trust_region
- TF32 enabled, XLA enabled

**Runs:**
1. Dense: `CUDA_VISIBLE_DEVICES=1 ... --transport-plan dense`
2. Streaming: `CUDA_VISIBLE_DEVICES=1 ... --transport-plan streaming`

**Analysis metrics:**
- Mean difference (streaming - dense)
- Standard deviation
- Min/max range
- Sign distribution
- Coefficient of variation

**Expected runtime:** ~30 min per mode (~1-2 min per seed)

## Why K=N Should Be Identical

At N=1008, the streaming implementation uses:
- `row_chunk_size = 1008`
- `col_chunk_size = 1008`
- This gives `row_blocks = 1` and `col_blocks = 1`

The code has an explicit bypass at line 200 of `genut_guided_proposal_tf.py`:
```python
if row_blocks == 1 and col_blocks == 1:
    return _dense_sinkhorn_barycentric_value(...)
```

So streaming should literally call the dense implementation. Any difference must come from:
1. Different XLA graph topology across the full filter loop
2. Different operation ordering leading to different FP rounding
3. Non-associativity of floating-point arithmetic over 20 time steps

## Next Steps

Once both runs complete:
1. Run `analyze_systematic_claim.py` to extract statistics
2. Determine which hypothesis is supported
3. Document findings and recommendation
4. Update CLAUDE.md if a policy change is needed

## Decision Criteria

| Mean | Std | Decision |
|------|-----|----------|
| < 0.01 | < 0.1 | Pure FP noise; either mode OK |
| < 0.1 | any | Small bias; document and choose |
| > 0.1 | < 0.5×|mean| | Systematic; investigate which is correct |
| > 0.5 | any | Major problem; disable TF32 or investigate |
