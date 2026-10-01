# SQMC Oracle Comparison: Tuned Continuation Plan

**Date:** 2026-09-12  
**Status:** PLANNED - Awaiting execution approval  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Prerequisite:** UNTUNED diagnostic complete (commit 7b23adbd)

---

## Executive Summary

This plan extends the SQMC oracle comparison to use **exact-scope tuned** configurations, enabling claim-bearing statistical route ranking. The UNTUNED diagnostic established the harness and principled metrics; this continuation produces promotion-eligible evidence.

**Key differences from UNTUNED diagnostic:**
1. **Exact-scope tuning artifacts** required for 3D LGSSM, T=20, N=1008
2. **Multi-seed validation** (16 seeds minimum for statistical ranking)
3. **Claim-bearing scope** (can support route preference, not just "all valid")
4. **Float32/TF32 GPU target** (production configuration, not FP64 diagnostic)

---

## Research Intent

**Question:** With exact-scope tuned controls, do the four SQMC ancestry routes produce statistically distinguishable gradient quality on the 3D LGSSM oracle comparison?

**Candidate mechanism:** Same four routes as UNTUNED diagnostic:
- `iid_dual_cap` (baseline IID Gaussian)
- `previous_inverse_cdf` (Hilbert inverse CDF)
- `repaired_permutation` (Hilbert one-to-one permutation, registry controls)
- `repaired_permutation_ablation` (Hilbert one-to-one permutation, conservative ablation)

**Expected behavior:** Routes may remain statistically indistinguishable (Austria SIR precedent), or exact-scope tuning may reveal gradient quality differences not visible in the UNTUNED diagnostic.

**Promotion criterion:** Statistically supported ranking (bootstrap confidence intervals, pairwise tests) with principled score quality metrics favoring one route.

**Promotion veto:** Any route with cosine similarity < 0.999, relative norm error > 5%, or Fisher-scaled errors > 1.0 for multiple parameters.

**Continuation veto:** Tuning artifact does not match exact scope, or validity guards fail on tuned configurations.

**Repair trigger:** Localized tuning failure, serialization error, or infrastructure failure within budget.

**Explanatory diagnostics:** All principled score quality metrics from UNTUNED diagnostic, plus runtime and memory usage.

**Do not conclude:** Production readiness, HMC convergence guarantees, or generalization to other models without separate validation.

---

## Phase 1: Exact-Scope Tuning (Required)

### Tuning Scope Contract

Create repository-issued tuning artifacts matching:
- **Model/target:** 3D LGSSM canonical (diagonal_lgssm_canonical_model)
- **Horizon:** T=20
- **Particle count:** N=1008
- **Dimensions:** state=3, observation=3
- **Route family:** Each of the 4 SQMC ancestry routes
- **Backend:** float32 tensors, TF32 execution, GPU
- **Chunk policy:** dpf_transport_exact_divisor_cap3000_v1
- **Reset:** Contract-E with GenUT, dual-cap enabled

### Tuning Protocol

For each route:
1. **Tuning grid:** Sinkhorn epsilon (4.0, 8.0, 16.0), Sinkhorn steps (4, 8, 16), balance steps (4, 8, 16)
2. **Diagonal correction:** strength (0.1, 0.15, 0.2), steps (3, 4, 5)
3. **Pairwise correction:** strength (0.01, 0.02, 0.03), steps (3, 4, 5)
4. **Tuning metric:** Oracle score L2 error (primary), cosine similarity (veto)
5. **Tuning seeds:** 50001-50016 (16 seeds, disjoint from claim seeds)
6. **Tuning horizon:** T=20 (exact match)

**Artifact location:** `docs/tuning/sqmc-lgssm-t20-n1008-<route>-20260912/`

**Budget:** 4 routes × ~50 grid cells × 16 seeds ≈ 3200 cells, estimated 8-12 hours

---

## Phase 2: Tuned Oracle Comparison

### Configuration

| Route | Point set | Ancestry | Controls |
|-------|-----------|----------|----------|
| iid_dual_cap | IID Gaussian | identity | tuned artifact A |
| previous_inverse_cdf | randomized Halton | Hilbert inverse CDF | tuned artifact B |
| repaired_permutation | randomized Halton | Hilbert one-to-one | tuned artifact C |
| repaired_permutation_ablation | randomized Halton | Hilbert one-to-one | tuned artifact D |

### Execution Ladder

**Stage 1: Smoke (2 seeds, 4 routes = 8 cells)**
- Seeds: 97701, 97702 (same as UNTUNED for comparison)
- Purpose: Verify tuned configurations remain valid
- Budget: 10 minutes
- Pass criterion: All cells finite, cosine similarity > 0.999

**Stage 2: Statistical comparison (16 seeds, 4 routes = 64 cells)**
- Seeds: 97701-97716 (matches Austria SIR precedent)
- Purpose: Statistical route ranking with uncertainty
- Budget: 2 hours
- Pass criterion: All cells finite, valid

**Stage 3: Bootstrap analysis**
- 10,000 bootstrap samples per route
- Pairwise confidence intervals (95%)
- Sign tests for gradient direction consistency
- Decision: Accept ranking if CIs separate, report indistinguishable otherwise

---

## Configuration Status

All cells labeled: `canonical_score_sqmc_float32_tuned_lgssm_t20_n1008`

**Differences from UNTUNED diagnostic:**
- ✅ Exact-scope repository-issued tuning artifacts (Phase 1 output)
- ✅ Float32/TF32 GPU execution (production target)
- ✅ 16-seed statistical validation (claim-capable)
- ✅ Fused five-direction analytical score (production lane)
- ⚠️  Still uses stable non-XLA graphs (if XLA resource issue persists)

**Production mechanisms active:** Contract-E reset, dual-cap correction, trust-region damping, UKF covariance lifecycle, validity guards, reset source-marginal checks.

---

## Principled Score Quality Metrics (Standard)

Report for each route:
1. **Gradient direction (cosine similarity):** Target > 0.999
2. **Relative gradient norm error:** Target < 5%
3. **Fisher-scaled errors (Err/√|Oracle|):** Target < 0.5 for most parameters
4. **Induced HMC parameter error:** Target < 0.001 per step (ε=0.01)
5. **Statistical comparison:** Bootstrap CIs, pairwise tests

**Comparison to UNTUNED:**
- Report side-by-side: UNTUNED (warm-start) vs TUNED (exact-scope)
- Quantify tuning benefit: improvement in score L2, cosine similarity, Fisher-scaled errors
- Decision: Is exact-scope tuning necessary for this comparison?

---

## Decision Framework

### Scenario A: Routes Still Indistinguishable (Expected)

If bootstrap analysis shows no statistically significant ranking:
- **Conclusion:** "All 4 routes produce equivalent gradient quality at N=1008 even with exact-scope tuning"
- **Recommendation:** Use simplest route (iid_dual_cap) or most tested (repaired_permutation)
- **Next step:** Close SQMC route comparison as resolved

### Scenario B: Clear Winner Emerges

If one route shows statistically superior gradient quality:
- **Conclusion:** "Route X produces superior gradients with [metric] advantage"
- **Promotion criterion:** Cosine similarity, Fisher-scaled errors, and induced HMC error all favor the winner
- **Recommendation:** Use winner as default for LGSSM-class models
- **Next step:** Validate on other models (Predator-Prey, KSC-SV) to assess generalization

### Scenario C: Tuning Makes Routes Worse

If exact-scope tuning degrades gradient quality vs UNTUNED:
- **Root cause:** Investigate tuning metric mismatch, overfitting, or grid insufficiency
- **Repair:** Revise tuning protocol, expand grid, or revert to validated warm-start defaults
- **Continuation veto:** If tuned configurations violate cosine similarity > 0.999 threshold

---

## Artifacts

**Tuning artifacts:**
- `docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/`
- `docs/tuning/sqmc-lgssm-t20-n1008-previous-inverse-cdf-20260912/`
- `docs/tuning/sqmc-lgssm-t20-n1008-repaired-permutation-20260912/`
- `docs/tuning/sqmc-lgssm-t20-n1008-repaired-permutation-ablation-20260912/`

**Comparison results:**
- `docs/benchmarks/artifacts/sqmc-oracle-tuned-lgssm-20260912/smoke_attempt01/result.json`
- `docs/benchmarks/artifacts/sqmc-oracle-tuned-lgssm-20260912/statistical_attempt01/result.json`
- `docs/benchmarks/artifacts/sqmc-oracle-tuned-lgssm-20260912/bootstrap_analysis.json`

**Analysis scripts:**
- `docs/benchmarks/analyze_tuned_vs_untuned_sqmc.py` (side-by-side comparison)
- `docs/benchmarks/sqmc_statistical_ranking.py` (bootstrap + pairwise tests)

**Decision document:**
- `docs/plans/sqmc-oracle-tuned-decision-2026-09-12.md`

---

## Budget

**Phase 1 (Tuning):** 8-12 hours GPU time  
**Phase 2 (Comparison):** 2-3 hours GPU time  
**Phase 3 (Analysis):** 1 hour CPU  
**Total:** 12-16 hours, 2-3 days calendar time

**Human review points:**
1. After Phase 1: Verify tuning artifacts look reasonable
2. After Stage 2: Review statistical results before decision
3. Final: Approve route recommendation if ranking emerges

---

## Relation to UNTUNED Diagnostic

This plan **continues** rather than replaces the UNTUNED diagnostic:
- UNTUNED established: harness works, all routes valid, principled metrics defined
- TUNED answers: does exact-scope tuning reveal route differences?
- UNTUNED remains: the baseline for "what if we don't tune per-model?"

**Documentation strategy:** Keep both reports, cross-reference, explicitly state tuning status in every table.

---

## Risk Assessment

**Risk 1: Tuning takes longer than budgeted**
- Mitigation: Start with single-route tuning to calibrate time estimates
- Fallback: Use coarser tuning grid (27 cells instead of 50)

**Risk 2: Tuned configurations violate validity guards**
- Mitigation: Keep warm-start as lower bound, only move controls within valid region
- Fallback: Report "tuning degraded validity" and recommend warm-start

**Risk 3: Routes remain indistinguishable even when tuned**
- Mitigation: This is a valid scientific result, not a failure
- Interpretation: Ancestry mechanism matters less than other factors at N=1008

**Risk 4: XLA resource exhaustion persists**
- Mitigation: Continue with stable non-XLA graphs from UNTUNED diagnostic
- Note: XLA compatibility already established in smoke tests

---

## Approval and Execution

**Owner approval required before:**
1. Phase 1 execution (tuning campaign starts)
2. Phase 2 execution (claim-bearing comparison starts)
3. Final decision (route recommendation if ranking exists)

**Plain-language authorization sufficient:** "Execute the tuned SQMC plan" or "Run Phase 1 tuning"

**No magic tokens or hash-bound approvals required** (per Academic Research Governance And Proportionality policy)

---

## Changelog

- 2026-09-12: Initial plan created after UNTUNED diagnostic complete
