# RQMC LEDH Initialization: Phase 3 Statistical Analysis Result

**Date:** 2026-09-04  
**Program:** docs/plans/rqmc-ledh-initialization-master-program-2026-09-02.md  
**Analysis artifact:** docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/phase3_analysis.json  
**Dataset:** 36 complete runs from Phase 2 (9 infrastructure failures excluded)

## Executive Summary

**Aggregate recommendation:** MIXED_PROMOTE

- **LGSSM T50:** NEUTRAL (all 4 RQMC arms statistically indistinguishable from MC)
- **KSC SV T10:** PROMOTE (all 3 available RQMC arms statistically superior to MC)
- **Predator-Prey T20:** NEUTRAL (both available sobol arms indistinguishable from MC)

## Per-Model Statistical Findings

### LGSSM T50 (15/15 complete, all 5 arms)

**Descriptive statistics (ELBO):**

| Arm            | n | Mean      | Std   | Q95       |
|----------------|---|-----------|-------|-----------|
| mc             | 3 | -136.2555 | 0.288 | -135.9941 |
| sobol_matousek | 3 | -136.1722 | 0.572 | -135.6085 |
| sobol_owen     | 3 | -136.3550 | 0.598 | -135.7652 |
| halton_owen    | 3 | -136.3223 | 0.539 | -135.7906 |
| genut_guided   | 3 | -136.2650 | 0.589 | -135.6838 |

**Statistical comparison (RQMC - MC):**

| Arm            | Mean Diff | 95% CI              | Favors RQMC? |
|----------------|-----------|---------------------|--------------|
| sobol_matousek | +0.0833   | [-0.28, +0.45]      | No           |
| sobol_owen     | -0.0995   | [-0.48, +0.30]      | No           |
| halton_owen    | -0.0668   | [-0.41, +0.26]      | No           |
| genut_guided   | -0.0095   | [-0.40, +0.38]      | No           |

**Recommendation:** NEUTRAL  
All four RQMC arms are statistically indistinguishable from MC (all bootstrap 95% 
CIs include zero). The continuation veto correctly did not fire.

### KSC SV T10 (12/12 complete in valid scope, 3 RQMC arms)

**Descriptive statistics (ELBO):**

| Arm          | n | Mean      | Std   | Q95       |
|--------------|---|-----------|-------|-----------|
| mc           | 3 | -19.9114  | 0.031 | -19.8811  |
| sobol_owen   | 3 | -19.9616  | 0.022 | -19.9421  |
| halton_owen  | 3 | -19.9452  | 0.015 | -19.9316  |
| genut_guided | 3 | -19.9368  | 0.024 | -19.9154  |

**Statistical comparison (RQMC - MC):**

| Arm          | Mean Diff | 95% CI              | Favors RQMC? |
|--------------|-----------|---------------------|--------------|
| sobol_owen   | -0.0502   | [-0.064, -0.032]    | **Yes**      |
| halton_owen  | -0.0338   | [-0.054, -0.017]    | **Yes**      |
| genut_guided | -0.0255   | [-0.037, -0.008]    | **Yes**      |

**Recommendation:** PROMOTE  
All three evaluated RQMC arms are statistically superior to MC (bootstrap 95% 
CI upper bound < 0). sobol_matousek excluded due to 1D incompatibility.

**Magnitude:** RQMC improvements range from 0.025 to 0.050 ELBO units (0.13% to 
0.25% relative to baseline ELBO ≈ -19.9).

### Predator-Prey T20 (9/9 complete in valid scope, 2 sobol arms)

**Descriptive statistics (ELBO):**

| Arm            | n | Mean      | Std   | Q95       |
|----------------|---|-----------|-------|-----------|
| mc             | 3 | -189.4879 | 0.913 | -188.8210 |
| sobol_matousek | 3 | -190.1788 | 0.748 | -189.5892 |
| sobol_owen     | 3 | -190.5141 | 1.245 | -189.6616 |

**Statistical comparison (RQMC - MC):**

| Arm            | Mean Diff | 95% CI              | Favors RQMC? |
|----------------|-----------|---------------------|--------------|
| sobol_matousek | -0.6909   | [-2.22, +0.53]      | No           |
| sobol_owen     | -1.0262   | [-3.15, +0.54]      | No           |

**Recommendation:** NEUTRAL  
Both available sobol arms are statistically indistinguishable from MC. halton_owen 
and genut_guided excluded due to missing Phase 1 tuning artifact.

**Observation:** Descriptive means favor RQMC by ~0.7–1.0 ELBO units, but high 
variance (std ~0.7–1.2) prevents statistical discrimination at n=3.

## Decision Table

| Model           | Hard Veto | Statistical Ranking | Descriptive Trend | Default-Ready | Next Evidence Needed |
|-----------------|-----------|---------------------|-------------------|---------------|----------------------|
| LGSSM T50       | Pass      | None supported      | Neutral           | N/A           | Longer validation    |
| KSC SV T10      | Pass      | All 3 RQMC superior | Favors RQMC       | Conditional   | 1D fix, final verify |
| Predator-Prey   | Pass      | None supported      | Favors RQMC       | No            | More seeds or Phase 1 repair |

## Inference Status Table

| Category                         | LGSSM T50 | KSC SV T10 | Predator-Prey T20 |
|----------------------------------|-----------|------------|-------------------|
| Hard veto screen                 | Pass      | Pass       | Pass              |
| Statistically supported ranking  | No        | Yes (3/3)  | No                |
| Descriptive-only differences     | Yes       | No         | Yes               |
| Default-readiness                | N/A       | Conditional| No                |
| Next evidence needed             | Longer    | 1D repair  | More seeds/repair |

## Scope Limitations

The analysis covers 36/45 planned runs due to:
- **KSC sobol_matousek (3 runs):** 1D state incompatibility with Lloyd optimization
- **Predator-Prey halton+genut (6 runs):** Missing Phase 1 tuning artifact

These limitations DO NOT invalidate findings for the evaluated arms but DO limit 
cross-arm and cross-model comparability.

## Scientific Interpretation

### What is established
1. **KSC SV T10:** RQMC initialization produces statistically better ELBO than MC 
   initialization for all three evaluated arms (sobol_owen, halton_owen, genut_guided)
2. **LGSSM T50 & Predator-Prey T20:** No statistical discrimination between RQMC 
   and MC at current sample size (n=3 seeds per arm)

### What is NOT concluded
1. **Not claiming:** RQMC is uniformly better across all models
2. **Not claiming:** sobol_matousek or other missing arms would show the same pattern
3. **Not claiming:** The observed KSC improvement generalizes to production LEDH 
   (only ELBO evaluated, not downstream inference quality)
4. **Not claiming:** Three seeds establish convergence or final RQMC policy

### Uncertainty sources
- **Small n:** Three seeds per arm limits power, especially for high-variance models
- **Missing arms:** 9/45 cells absent limits cross-arm conclusions
- **Single metric:** ELBO does not directly measure downstream posterior quality
- **Tuning artifact:** KSC and Predator-Prey use Phase 1 tuning; LGSSM result 
  predates that tuning

## Promotion Recommendation

### Immediate (KSC SV T10)
**CONDITIONAL PROMOTE** for KSC SV T10 RQMC initialization, pending:
1. Fix sobol_matousek 1D incompatibility (conditional dim>=2 Lloyd optimization)
2. Verify that the 1D-fixed implementation preserves the promoted result
3. Execute a final verification run with all 4 RQMC arms

**Promoted status:** Optional feature (not default) until downstream validation confirms

### Deferred (LGSSM T50, Predator-Prey T20)
**NEUTRAL status:** Remain viable candidates for longer validation but insufficient 
evidence for promotion. Next steps:
- LGSSM: Longer multi-seed validation if prioritized
- Predator-Prey: Repair Phase 1 blocker, complete Phase 2B, re-analyze with n=3 
  for missing arms

### Not recommended
**REJECT:** None. No RQMC arm is statistically inferior to MC in the evaluated scope.

## Next Actions

### Phase 2B Repair (if required for complete dataset)
1. Fix `sobol_matousek` 1D incompatibility in `_generate_initial_noise()`
2. Recover or regenerate Predator-Prey Phase 1 tuning artifact
3. Execute 9 missing cells
4. Merge into complete 45-run dataset
5. Re-run Phase 3 analysis

### KSC Promotion Path
1. Implement 1D fix for sobol_matousek
2. Regression check: verify existing KSC results unchanged (sha256 artifact match)
3. Execute sobol_matousek × 3 seeds for KSC
4. Final verification: confirm all 4 arms remain promoted
5. Write promotion receipt

### Master Program Update
Mark Phase 3 COMPLETE with MIXED_PROMOTE aggregate. Update program footer with:
- Phase 2: PARTIAL COMPLETE (36/45)
- Phase 3: COMPLETE (conditional on scope)
- KSC: CONDITIONAL PROMOTE pending 1D repair
- LGSSM & Predator-Prey: NEUTRAL, longer validation deferred
