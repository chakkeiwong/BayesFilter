# BayesFilter Current Blockers - 2026-09-08

## Status Overview

**Branch:** `ledh-refactor-with-policy-fix`  
**Recent work:** Phase 4A KDM campaign completed with negative result  
**Dense vs streaming investigation:** Completed - resolved as FP noise, no action needed

---

## Active Blockers

### 1. **batch_fused / NeuTra Lane Conformance** ⚠️ HIGH PRIORITY

**Status:** Implementation-conformance blocker  
**Location:** `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`  
**Issue:** The batch_fused lane bypasses Contract-E, GenUT, and dual caps

**Details:**
- The current `canonical_batch_fused_value_score` returns `children` directly after UKF update
- Does NOT execute:
  - Contract-E covariance reset
  - GenUT moment restoration  
  - Dual-cap trust-region correction
- This means it's not computing the same target as the canonical lane
- Tests pass but only verify batch mechanics, not full conformance

**Why it matters:**
- batch_fused is the NeuTra-eligible training lane
- NeuTra training requires batch-native execution (no Python row loops)
- Can't use this lane for HMC until it computes the actual canonical target

**Referenced in:**
- [phase4-integrated-plan:337-339](docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md#L337-L339)
- [score-research-reset:872-873](docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md#L872-L873)

**Fix required:**
1. Add Contract-E reset to batch_fused lane
2. Add GenUT correction
3. Add dual-cap trust-region
4. Maintain batch-native execution (no row loops)
5. Add parity test vs single-cloud canonical with all corrections

**Complexity:** HIGH - must preserve batch-native semantics while adding stateful corrections

---

### 2. **Phase 4B Mathematical Specification** 🔬 RESEARCH BLOCKED

**Status:** PHASE4B_MATH_BLOCKED  
**Issue:** Proposal law incomplete

**Details:**
- Phase 4B would test full-mixture IWSG resampling (complete Younis method)
- Requires complete sequential proposal law before implementation
- Phase 4A (observation weighting only) showed NO_PROMOTION_EVIDENCE
- No point in Phase 4B until Phase 4A would have passed

**Referenced in:**
- [phase4b-plan](docs/plans/bayesfilter-ledh-younis-kdm-phase4b-resampling-reference-plan-2026-09-08.md)
- Status line: `IMPLEMENTATION AUTHORIZED; CORRECTNESS GATES BEFORE CAMPAIGN`

**Current decision:** PAUSED - Phase 4A negative result suggests Phase 4B unlikely to help

---

### 3. **Dense vs Streaming Transport Discrepancy** ✅ RESOLVED

**Status:** Investigated and explained (FP noise, not a bug)  
**Artifact:** [DENSE_VS_STREAMING_INVESTIGATION.md](DENSE_VS_STREAMING_INVESTIGATION.md)

**Finding:**
- Mean difference: +0.15 (streaming - dense)
- Std dev: 0.85 (5.7× larger than mean)
- Range: -1.57 to +1.46 across 16 seeds
- **Root cause:** FP32/TF32 non-associative arithmetic over 20 time steps

**Resolution:**
- Use streaming with K=N≤3000 (complies with chunk rule)
- Document that modes differ by ~1 unit (pure FP noise)
- Don't mix dense and streaming in same comparison

**No action required** - this is expected FP behavior, not a blocker.

---

## Recently Completed (Not Blockers)

### ✅ Phase 4A KDM Campaign
- **Completed:** 2026-09-08
- **Result:** NO_PROMOTION_EVIDENCE
- **Verdict:** Kernelized observation weighting does not reduce score error vs Kalman oracle
- **Decision:** No further Phase 4A work justified; canonical LEDH score remains baseline
- **Artifact:** [results/ledh_younis_kdm_phase4a_campaign_result_20260908.md](results/ledh_younis_kdm_phase4a_campaign_result_20260908.md)

### ✅ Contract-E Covariance Carry Bug
- **Fixed:** 2026-09-07
- **Issue:** Covariance wasn't being transported through Contract-E reset
- **Fix:** Added covariance carry with same transport as state
- **Tests:** All canonical endpoints pass with covariance carry

### ✅ SQMC Rerun with Corrected Filter
- **Completed:** 2026-09-07
- **Artifact:** `docs/benchmarks/artifacts/sqmc-rerun-corrected-filter-20260906/`

---

## Priority Order for Unblocking

### 1. **Fix batch_fused conformance** (if NeuTra training is needed)
   - **Owner decision required:** Is NeuTra training a near-term priority?
   - **If yes:** Implement Contract-E + GenUT + dual-cap in batch_fused
   - **If no:** Document as deferred, continue with single-cloud canonical

### 2. **Close out Phase 4 work** (documentation)
   - Commit Phase 4A negative result
   - Archive Phase 4B as PAUSED pending new evidence
   - Update master plan status

### 3. **Other research directions?**
   - What's the next scientific question after Phase 4?
   - Are there other LEDH improvements to investigate?

---

## Questions for Owner

1. **Is NeuTra training a priority?** If so, batch_fused conformance is urgent.

2. **What's the next research direction?** Phase 4 is complete (negative result).

3. **Are there outstanding DSGE or HMC validation tasks?**

4. **Should Phase 4 code be merged to main, or stay on branch?**

---

## Files Modified (Not Committed)

```
M bayesfilter/highdim/ledh_canonical_score_tf.py
M bayesfilter/highdim/ledh_younis_kdm_tf.py
M docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md
M docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md

?? DENSE_VS_STREAMING_INVESTIGATION.md
?? analyze_systematic_claim.py
?? bayesfilter/highdim/ledh_kalman_oracle_tf.py
?? bayesfilter/highdim/ledh_younis_kdm_lgssm_reference_tf.py
?? docs/plans/bayesfilter-ledh-younis-kdm-phase4a-campaign-amendment-20260908.md
?? docs/plans/bayesfilter-ledh-younis-kdm-phase4b-resampling-reference-plan-2026-09-08.md
?? docs/plans/phase4a_campaign_approval_request_20260908.md
?? results/ledh_younis_kdm_phase4a_campaign_result_20260908.md
?? results/phase4a_implementation_complete_20260908.md
```

**Decision needed:** Commit Phase 4A work? Archive to separate branch?

---

## Summary

**One real blocker:** batch_fused conformance (if NeuTra is priority)  
**One research blocker:** Phase 4B (paused after 4A negative result)  
**One resolved non-issue:** Dense vs streaming (FP noise)

**Main question:** What's next after Phase 4?
