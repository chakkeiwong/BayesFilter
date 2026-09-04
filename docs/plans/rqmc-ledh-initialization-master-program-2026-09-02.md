# RQMC LEDH Initialization Master Program

**Date:** 2026-09-02 (execution started 2026-09-04)  
**Owner:** chakwong  
**Status:** Phase 3 COMPLETE (conditional), Phase 2B repair path defined  

## Research Question

Can quasi-random Monte Carlo (QRMC) sequences for LEDH particle initialization 
produce better ELBO than standard Monte Carlo, and does the improvement generalize 
across models?

## Scientific Contract

**Primary criterion:** Statistical comparison of ELBO (RQMC initialization vs MC 
initialization) via bootstrap 95% confidence intervals on paired seed differences

**Promotion criterion:** RQMC arm has bootstrap CI upper bound < 0 (statistically 
superior to MC baseline)

**Promotion veto:** None defined (no RQMC arm is worse than MC in practice)

**Continuation veto (Phase 2 → Phase 3):** All RQMC arms statistically inferior 
to MC on LGSSM T50 (the simplest model). If this fires, stop before KSC and 
Predator-Prey models.

**Explanatory diagnostics:**
- Per-arm descriptive statistics (mean, std, quantiles)
- Per-seed ELBO values for variance assessment
- Infrastructure failure classification

**What is NOT concluded:**
- RQMC initialization quality does not directly measure downstream posterior quality
- Three seeds per arm is an early-phase sample size, not convergence evidence
- Promotion based on ELBO alone requires downstream validation before default status

## Campaign Structure

**Phase 0:** Smoke test (4 arms on LGSSM, single seed) — verify runner infrastructure

**Phase 1:** Trust-region tuning for LEDH on each model (separate campaign, 
prerequisite artifact)

**Phase 2:** Full 3-model × 5-arm × 3-seed execution (45 runs)
- Models: LGSSM T50, KSC SV T10, Predator-Prey T20 (increasing complexity)
- Arms: mc (baseline), sobol_matousek, sobol_owen, halton_owen, genut_guided
- Seeds: 98301, 98302, 98303 (hashed via np.random.SeedSequence for independence)
- LGSSM executes first; continuation veto evaluated before KSC/Predator-Prey launch

**Phase 3:** Statistical analysis
- Bootstrap 95% CI for (RQMC - MC) differences per model
- Promotion recommendation per model
- Aggregate recommendation across models

**Phase 2B (conditional):** Infrastructure repair and completion
- Fix sobol_matousek 1D incompatibility
- Recover Predator-Prey Phase 1 tuning artifact
- Execute 9 missing cells
- Re-run Phase 3 analysis on complete 45-run dataset

## Execution Budget

**Wall time:** ~90 minutes (Phase 2) + 5 minutes (Phase 3) = 95 minutes total

**Compute:** GPU (tftwogpu conda env, RTX 4080 SUPER primary)

**Attempts:** 2 per run for transient infrastructure failures

**Total campaign budget:** 180 minutes wall time, 90 attempts maximum

## Arms

### mc (baseline)
Standard Monte Carlo initialization via `tf.random.normal()` with per-seed streams

### sobol_matousek
Scrambled Sobol sequence with Lloyd optimization (centroidal Voronoi tessellation). 
Uses `scipy.stats.qmc.Sobol` with `optimization='lloyd'`.

**Known limitation:** Lloyd optimization requires `state_dim >= 2`. For 1D models 
(KSC SV), must fall back to plain scrambled Sobol without Lloyd step.

### sobol_owen
Scrambled Sobol sequence with Owen scrambling. Uses `scipy.stats.qmc.Sobol` with 
`scramble=True` and Owen-style randomization.

### halton_owen
Halton sequence with Owen scrambling. Uses `scipy.stats.qmc.Halton` with Owen-style 
randomization.

### genut_guided
Experimental: guided low-discrepancy initialization via custom generator. Details 
in runner implementation.

## Models

### LGSSM T50
Linear Gaussian state-space model, T=50 timesteps, state_dim=10. Simplest model, 
used for continuation veto evaluation.

### KSC SV T10
Kim-Shephard-Chib stochastic volatility, T=10 timesteps, state_dim=1 (scalar 
log-volatility). Tests 1D state behavior.

### Predator-Prey T20
Nonlinear predator-prey dynamics, T=20 timesteps, state_dim=2. Tests nonlinear 
multivariate case.

## Phase-by-Phase Status

### Phase 0: Smoke Test
**Status:** COMPLETE (2026-09-04)  
**Outcome:** Infrastructure verified, artifact loader corrected

### Phase 1: Trust-Region Tuning
**Status:** PARTIAL COMPLETE  
**Outcome:**
- LGSSM T50: artifact present
- KSC SV T10: artifact present
- Predator-Prey T20: BLOCKER (Phase 1 campaign failed, artifact missing)

**Action required:** Recover or regenerate Predator-Prey tuning artifact for Phase 2B

### Phase 2: Full Execution
**Status:** PARTIAL COMPLETE (36/45 runs)  
**Execution window:** 2026-09-04 03:17–04:32 UTC (74.8 minutes)  
**Completion:**
- LGSSM T50: 15/15 (100%)
- KSC SV T10: 12/12 valid scope (sobol_matousek excluded due to 1D bug)
- Predator-Prey T20: 9/9 valid scope (halton+genut blocked by Phase 1 artifact)

**Infrastructure failures:** 9
- sobol_matousek × KSC (3 runs): 1D incompatibility with Lloyd optimization
- halton_owen × Predator-Prey (3 runs): Phase 1 tuning artifact missing
- genut_guided × Predator-Prey (3 runs): Phase 1 tuning artifact missing

**Continuation veto:** Did NOT fire. LGSSM result showed all RQMC arms statistically 
indistinguishable from MC (neutral), permitting KSC and Predator-Prey execution.

### Phase 3: Statistical Analysis
**Status:** COMPLETE (2026-09-04)  
**Dataset:** 36 complete runs (infrastructure failures excluded)  
**Artifact:** docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/phase3_analysis.json

**Per-model recommendations:**
- **LGSSM T50:** NEUTRAL (all 4 RQMC arms statistically indistinguishable from MC)
- **KSC SV T10:** PROMOTE (all 3 available RQMC arms statistically superior to MC)
- **Predator-Prey T20:** NEUTRAL (both available sobol arms indistinguishable from MC)

**Aggregate recommendation:** MIXED_PROMOTE

**Statistical findings:**
- KSC SV T10: RQMC improvements of 0.025–0.050 ELBO units (0.13–0.25% relative), 
  bootstrap 95% CI excludes zero for all three arms
- LGSSM T50: No statistical discrimination (high variance relative to effect size)
- Predator-Prey T20: Descriptive trend favors RQMC (~0.7–1.0 ELBO) but insufficient 
  power at n=3 seeds

### Phase 2B: Repair and Completion
**Status:** PLANNED (not yet executed)  

**Blockers to resolve:**
1. **sobol_matousek 1D incompatibility:**
   - Root cause: Lloyd optimization requires `state_dim >= 2`
   - Fix: Conditional logic in `_generate_initial_noise()`: if `state_dim == 1`, 
     use plain scrambled Sobol without Lloyd step
   - Regression: Verify KSC MC/sobol_owen/halton/genut results unchanged
   - Execute: sobol_matousek × KSC × 3 seeds

2. **Predator-Prey Phase 1 tuning artifact:**
   - Root cause: Phase 1 campaign (2026-09-02/03) wrote 116 per-config evaluations 
     but never wrote campaign summary `result.json`
   - Recovery path 1: Regenerate `result.json` from 116 existing files if salvageable
   - Recovery path 2: Re-run Phase 1 trust-region tuning (27 configs, ~10 minutes)
   - Execute: halton_owen + genut_guided × Predator-Prey × 3 seeds

**Phase 2B budget:** 15 minutes wall time, 18 attempts (9 missing runs × 2 attempts)

**Phase 2B outcome:** Complete 45-run dataset, re-run Phase 3 analysis, final recommendation

## Promotion Path

### KSC SV T10 (Conditional Promote)
**Status:** CONDITIONAL PROMOTE pending 1D repair verification

**Requirements:**
1. Implement sobol_matousek 1D fix
2. Regression check: existing KSC results unchanged (artifact sha256 match)
3. Execute sobol_matousek × KSC × 3 seeds
4. Verify all 4 RQMC arms remain statistically superior to MC
5. Write promotion receipt

**Promoted status:** Optional feature (not default) until downstream posterior-quality 
validation confirms. ELBO improvement does not automatically establish posterior 
correctness or HMC convergence improvement.

### LGSSM T50 & Predator-Prey T20 (Neutral, Deferred)
**Status:** Viable candidates, insufficient evidence for promotion

**Next steps:**
- LGSSM: Longer multi-seed validation if prioritized
- Predator-Prey: Complete Phase 2B, re-analyze with full arm set

**Rationale:** NEUTRAL does not mean "failed" — it means current sample size 
(n=3 seeds) lacks power to discriminate. Descriptive trends are favorable for 
Predator-Prey (~1 ELBO unit improvement) but require more seeds or lower-variance 
arms for statistical support.

## Evidence Contract Summary

| Category                     | Contract Requirement                          | Status       |
|------------------------------|-----------------------------------------------|--------------|
| Primary criterion            | Bootstrap 95% CI for RQMC vs MC              | Complete     |
| Promotion criterion          | At least one RQMC arm CI upper < 0           | Met (KSC)    |
| Continuation veto            | All RQMC inferior to MC on LGSSM             | Did not fire |
| Hard vetoes                  | Divergence, NaN, crash                       | None fired   |
| Statistical inference        | Paired seed bootstrap, n=3                   | Complete     |
| Scope limitations            | 36/45 runs (9 infrastructure failures)       | Documented   |
| Nonclaims                    | Downstream posterior quality not measured    | Stated       |

## Decision Table

| Model         | Hard Veto | Statistical Ranking | Descriptive | Default-Ready | Next Evidence       |
|---------------|-----------|---------------------|-------------|---------------|---------------------|
| LGSSM T50     | Pass      | None                | Neutral     | N/A           | Longer validation   |
| KSC SV T10    | Pass      | 3/3 RQMC superior   | Favors RQMC | Conditional   | 1D fix, verify      |
| Predator-Prey | Pass      | None                | Favors RQMC | No            | Phase 2B or seeds   |

## Artifacts

**Campaign root:** docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/

**Phase 2 outputs:**
- `campaign_summary.json`: 36 complete runs, per-run manifest
- Individual result files: `<model>_<arm>_seed<seed>/result.json`

**Phase 3 outputs:**
- `phase3_analysis.json`: Bootstrap CIs, recommendations, decision tables
- Result memo: docs/plans/rqmc-ledh-phase3-result-2026-09-04.md

**Phase 2 completion memo:** docs/memos/rqmc-ledh-phase2-completion-2026-09-04.md

**Runner:** docs/benchmarks/run_rqmc_ledh_initialization.py

**Analysis script:** docs/benchmarks/analyze_rqmc_ledh_phase3.py

## Program Footer

**Program version:** 1.0 (2026-09-02)  
**Last updated:** 2026-09-04 (Phase 3 complete)  
**Git commit (Phase 2 execution):** [to be recorded]  
**Git commit (Phase 3 analysis):** [to be recorded]  

**Current state:**
- Phase 0: COMPLETE
- Phase 1: PARTIAL (Predator-Prey blocker)
- Phase 2: PARTIAL (36/45, valid scope 100% complete)
- Phase 3: COMPLETE (conditional on 36-run dataset)
- Phase 2B: PLANNED (9 runs to complete)

**Human approval required for:**
- Phase 2B execution (infrastructure repair, 15 minutes)
- KSC SV T10 promotion after 1D verification
- Any change to statistical inference method or promotion criteria

**No approval required for:**
- Phase 2B infrastructure repairs (sobol_matousek 1D fix, artifact recovery)
- Phase 3 re-run after Phase 2B completion (same method, larger dataset)
- Optional longer validation for LGSSM or Predator-Prey
