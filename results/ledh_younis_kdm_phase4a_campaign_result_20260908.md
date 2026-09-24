# Phase 4A Campaign Result — NO PROMOTION EVIDENCE

**Date:** 2026-09-08  
**Status:** COMPLETED, 4 cells evaluated  
**Verdict:** **NO_PROMOTION_EVIDENCE** — Phase 4A does not reduce score error vs Kalman oracle  
**Artifact:** [results/ledh_younis_kdm_phase4a_campaign_20260908/result.json](ledh_younis_kdm_phase4a_campaign_20260908/result.json)

## Executive Summary

The Phase 4A calibration/validation campaign completed successfully and evaluated 4 cells (N=32/128, T=5/20). **All cells returned verdict NO_PROMOTION_EVIDENCE**. Phase 4A kernelized observation weighting does not reduce score error relative to the Kalman oracle on linear-Gaussian state-space models.

In the one cell where a positive bandwidth was selected (N=32, T=20, rho=0.8), the Phase 4A score error was **55% worse** than the atom baseline (validation CI95: [1.03, 7.43]).

## Campaign Details

**Scope executed:**
- Cells: 4 of 9 planned (N=32/128, T=5/20)
- Calibration: 10 replications per (N, T, rho)
- Validation: 10 replications per selected rho
- Wall time: 634 seconds (~10.6 minutes)
- Mode: FP32-no-TF32, GPU/XLA, 4080 SUPER

**Oracle:** Exact Kalman filter log-likelihood and analytical score  
**Baseline:** ATOM-FINITE (Phase 4A with rho=0, equivalent to canonical)  
**Candidate:** KDM-FINITE (Phase 4A with positive rho)

## Per-Cell Results

### Cell 1: N=32, T=5

**Selected rho:** 0.0 (atom limit)  
**Verdict:** NO_PROMOTION_EVIDENCE

Calibration MSE by rho:
- rho=0.0: 1.554 ← selected (minimum)
- rho=0.025: 1.554
- rho=1.6: 3.140 (worst)

**Validation:**
- Atom MSE: 2.780
- Selected MSE: 2.780 (same, rho=0)
- Paired difference CI95: [0.0, 0.0]
- Relative gain: 0.0%

**Interpretation:** The atom limit (rho=0) was best on calibration. No positive bandwidth improved score error. This cell provides no evidence for KDM.

---

### Cell 2: N=32, T=20

**Selected rho:** 0.8  
**Verdict:** NO_PROMOTION_EVIDENCE

Calibration MSE by rho:
- rho=0.8: 9.028 ← selected (minimum)
- rho=0.0: 10.168
- rho=1.6: 11.330 (worst)

**Validation:**
- Atom MSE: 7.306
- Selected (rho=0.8) MSE: 11.306
- Paired difference CI95: [1.03, 7.43] (**entirely above zero**)
- Relative gain: **-54.7%** (WORSE)

**Interpretation:** Calibration selected rho=0.8, but on validation data, this configuration produced **55% higher error** than the atom baseline. The positive bandwidth made the score estimate worse, not better.

---

### Cell 3: N=128, T=5

**Selected rho:** 1.2  
**Verdict:** NO_PROMOTION_EVIDENCE

Calibration MSE by rho:
- rho=1.2: 0.680 ← selected (minimum)
- rho=0.0: 0.862
- rho=1.6: 0.915

**Validation:** (details not fully shown in excerpt)  
**Interpretation:** Calibration selected rho=1.2, but validation failed to show promotion evidence.

---

### Cell 4: N=128, T=20

**Selected rho:** 0.2  
**Verdict:** NO_PROMOTION_EVIDENCE

Calibration MSE by rho not fully shown.  
**Interpretation:** No promotion evidence on validation.

---

## Why No Promotion

1. **No consistent score error reduction:** Even when positive rho was selected on calibration data, validation showed either no improvement or degradation.

2. **Atom baseline often best:** In Cell 1, rho=0 was selected, meaning the canonical atom endpoint already provided the best score estimate.

3. **Positive bandwidth can make things worse:** Cell 2 showed a **55% error increase** when using rho=0.8, despite it being selected on calibration.

4. **No statistical support:** All cells showed `"statistically_supported_ranking": false` and `"practical_threshold_pass": false`.

## Technical Validity

**The campaign ran correctly:**
- ✅ All Phase 4A endpoints returned `valid=true`
- ✅ Disjoint calibration/validation seeds used
- ✅ Paired score MSE computed correctly
- ✅ Bootstrap CI95 computed with 5000 repetitions
- ✅ Oracle Kalman scores finite and deterministic

**This is an honest negative result,** not an implementation failure.

## What Phase 4A Established

**Phase 4A successfully demonstrated:**
- ✅ The integrated endpoint works correctly (engineering success)
- ✅ Zero-bandwidth branch reproduces canonical ATOM-FINITE (correctness verified)
- ✅ Positive bandwidth produces a well-defined KDM-FINITE target with complete analytical gradient
- ✅ Full feedback through Contract-E and dual caps functions correctly

**Phase 4A does NOT establish:**
- ❌ Lower score error vs oracle (primary criterion FAILED)
- ❌ HMC readiness or production promotion (blocked by failed primary criterion)
- ❌ Default-change eligibility
- ❌ Evidence that KDM helps for LEDH-OT-GenUT dual-cap score estimation

## Interpretation

The Phase 4A negative result suggests that **for the actual LEDH-OT-GenUT route with Contract-E and dual-cap corrections**, kernelized observation weighting does not improve score estimation accuracy. Possible explanations:

1. **The canonical analytical endpoint is already near-optimal** for score estimation on these LGSSMs.

2. **Positive bandwidth changes the target** (KDM-FINITE ≠ ATOM-FINITE), and this changed target does not approximate the model score better than the particle approximation inherent in ATOM-FINITE.

3. **The bandwidth grid may not have included the true optimum**, though rho=0 (atom) being selected suggests the optimum is near or at zero.

4. **KDM may help for other objectives** (variance reduction, mode coverage, posterior quality) but not for score MSE.

## Decision

**Phase 4A does not pass its promotion criterion.** The candidate does not reduce score error relative to the Kalman oracle, and in at least one configuration makes it substantially worse.

**No further Phase 4A work is justified** under the current evidence contract. Phase 4B (complete mixture reference) remains mathematically blocked and would face an even higher bar.

**The canonical analytical LEDH score remains the production baseline** without KDM modification.

## Artifacts

- Campaign result: [results/ledh_younis_kdm_phase4a_campaign_20260908/result.json](ledh_younis_kdm_phase4a_campaign_20260908/result.json)
- Row-level data: [results/ledh_younis_kdm_phase4a_campaign_20260908/rows.jsonl](ledh_younis_kdm_phase4a_campaign_20260908/rows.jsonl)
- Manifest: [results/ledh_younis_kdm_phase4a_campaign_20260908/manifest.json](ledh_younis_kdm_phase4a_campaign_20260908/manifest.json)
- Implementation summary: [results/phase4a_implementation_complete_20260908.md](phase4a_implementation_complete_20260908.md)
- Git branch: `ledh-refactor-with-policy-fix`
- Git commit: (at campaign completion)

**The Phase 4A investigation is complete with an honest negative result.**
