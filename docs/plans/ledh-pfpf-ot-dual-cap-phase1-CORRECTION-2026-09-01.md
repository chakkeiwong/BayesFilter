# CORRECTION: Dual-Cap Tuning Artifacts DO Exist

**Date:** 2026-09-01 (evening)  
**Correction to:** Phase 1 execution summary (Gap 2 finding)  
**Status:** CRITICAL CORRECTION

## Original Finding (WRONG)

Phase 1 summary stated:

> **Gap 2: Per-Model Tuning Artifacts (CRITICAL, BLOCKING)**
> - **Finding:** Zero tuning artifacts exist for any model under dual-cap routes
> - **Blocking:** YES for per-model claims

## Corrected Finding (RIGHT)

**Dual-cap tuning artifacts DO EXIST** for at least 4 models in `docs/benchmarks/artifacts/genut_four_model_leaderboard_rerun_20260816/`:

### Confirmed Tuned Models with Dual-Cap

1. **Austria SIR T=20** - `single_austria_sir_T20_dual_cap/result.json`
2. **KSC SV T=10** - `single_ksc_sv_T10_dual_cap/`
3. **LGSSM T=50** - `single_lgssm_T50_dual_cap/`
4. **Predator-Prey T=20** - `single_predator_prey_T20_dual_cap/`

### Tuned Parameters (Austria SIR Example)

From `single_austria_sir_T20_dual_cap/result.json`:

```json
{
  "controls": {
    "balance_steps": 16,
    "coordinatewise_standardized_cap": 0.98,
    "coordinatewise_standardized_cap_power": 8,
    "epsilon": 8.0,
    "higher_moment_correction_steps": 4,
    "higher_moment_floor": 1e-05,
    "higher_moment_strength": 0.2,
    "pairwise_moment_correction_steps": 4,
    "pairwise_moment_floor": 1e-05,
    "pairwise_moment_strength": 0.02,
    "pairwise_particle_rms_cap": 2.0,
    "ridge": 1e-05,
    "sinkhorn_steps": 16
  },
  "scope_hash": "cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07",
  "calibration_valid": true,
  "claim_valid": true
}
```

### Observability Confirmed

The artifacts record **all 14 dual-cap diagnostics** per seed:
- `max_mean_residual`
- `minimum_covariance_gap_eigenvalue`
- `maximum_normalized_shape_displacement`
- `mean_normalized_shape_residual_objective`
- `mean_normalized_pairwise_shape_residual_objective`
- `maximum_pairwise_pre_cap_particle_rms`
- `maximum_pairwise_post_cap_particle_rms`
- `minimum_pairwise_particle_cap_scale`
- `maximum_pairwise_co_skew_residual`
- `maximum_pairwise_co_kurtosis_residual`
- `maximum_coordinatewise_pre_cap_absolute`
- `maximum_coordinatewise_post_cap_absolute`
- `mean_coordinatewise_cap_displacement`
- `fraction_coordinatewise_cap_active`
- `minimum_coordinatewise_cap_derivative`

Plus transport diagnostics:
- `max_row_residual`
- `max_col_residual`
- `minimum_row_mass`

### Additional Tuning Evidence

Related artifact directories found:
- `zhao_cui_genut_dual_cap_cross_model_20260807/` (4 attempts)
- `zhao_cui_genut_austria_t2_dual_cap_20260806/`
- `zhao_cui_genut_austria_t20_dual_cap_20260807/`
- `ledh_pfpf_genut_n1008_contract_e_tuning_20260808/`

## Revised Gap Status

### Gap 2: Per-Model Tuning Artifacts (RESOLVED)

**OLD STATUS:** CRITICAL, BLOCKING  
**NEW STATUS:** RESOLVED for 4 models (Austria SIR, KSC SV, LGSSM, Predator-Prey)

**Evidence:**
- Tuned controls recorded with `scope_hash`
- `calibration_valid: true` and `claim_valid: true` flags
- Multiple seeds per model (3+ replications visible)
- Full diagnostic observability

### Gap 4: Observability Integration (RESOLVED)

**OLD STATUS:** Non-blocking, integration status unclear  
**NEW STATUS:** RESOLVED - full diagnostic payload preserved in artifacts

**Evidence:** All 14 dual-cap diagnostics + transport diagnostics recorded per seed in result.json files

## Revised Production-Readiness Assessment

**Originally:** 3 critical gaps (wiring gate, tuning, safety evaluation)  
**Corrected:** 2 critical gaps remaining:

1. **Gap 1: Wiring Gates (CRITICAL)** - Still blocking, no change
2. **Gap 3: Safety Evaluation (CRITICAL)** - Still blocking, no change

**Gap 2 (tuning) and Gap 4 (observability) are RESOLVED.**

## Implication for Phase 2

The existence of tuned artifacts suggests:

1. **A leaderboard campaign already ran** with dual-cap arms across 4 models
2. **Parameters have been tuned offline** (epsilon=8.0, steps=4, strength=0.2/0.02)
3. **Scope hashes exist** for comparability verification
4. **Claim validity flags exist** suggesting a gate system is operational

**Next step:** Find the leaderboard runner that produced these artifacts to understand:
- How dual-cap is enabled (is it the default or an optional arm?)
- What the production program definition is
- Whether wiring gates exist in the runner
- What the claim-validity gate checks

## Apology

The Phase 1 audit searched for `LEDH_PRODUCTION_PROGRAM_V1` in code but did not search artifacts directories. This was a methodological error. The correct audit sequence should have been:

1. Search code for production program definition
2. **Search artifacts for evidence of tuning campaigns** ← missed this
3. Read benchmark runners to understand execution
4. Trace implementation

The corrected finding changes the assessment from "not production-ready" to "may already be production-ready, need to verify gates."

## Action Required

**Immediate:** Update Phase 1 execution summary with corrected Gap 2 and Gap 4 status.

**Phase 2 revised goal:** Locate the leaderboard runner, verify wiring gates exist, understand claim-validity conditions, then determine whether Safety Evaluation (Gap 3) has already been completed or still requires dedicated campaign.
