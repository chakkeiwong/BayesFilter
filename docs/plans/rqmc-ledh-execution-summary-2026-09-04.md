# RQMC LEDH Initialization Campaign: Execution Summary

**Date:** 2026-09-04  
**Program:** docs/plans/rqmc-ledh-initialization-master-program-2026-09-02.md  
**Status:** Phase 3 COMPLETE, Phase 2B repair path defined

## Quick Status

**Phase 2 execution:** 36/45 runs complete (80%), 100% of valid scope after blockers discovered  
**Phase 3 analysis:** COMPLETE  
**Aggregate recommendation:** MIXED_PROMOTE  
**Wall time:** 75 minutes (Phase 2) + <1 minute (Phase 3)

## Key Findings

### KSC SV T10: CONDITIONAL PROMOTE ✓
All three available RQMC arms (sobol_owen, halton_owen, genut_guided) are 
**statistically superior** to MC initialization:
- Improvements: 0.025–0.050 ELBO units (0.13–0.25% relative)
- Bootstrap 95% CI upper bounds all < 0
- Pending: 1D fix for sobol_matousek, final verification

### LGSSM T50: NEUTRAL
All four RQMC arms statistically indistinguishable from MC (bootstrap 95% CIs 
include zero). No evidence of harm or benefit at n=3 seeds.

### Predator-Prey T20: NEUTRAL
Both available sobol arms (sobol_matousek, sobol_owen) statistically indistinguishable 
from MC. Descriptive trend favors RQMC by ~0.7–1.0 ELBO units, but high variance 
prevents statistical discrimination.

## Infrastructure Blockers (9 runs)

1. **sobol_matousek × KSC (3 runs):** Lloyd optimization requires `state_dim >= 2`, 
   KSC has `state_dim=1`
2. **halton/genut × Predator-Prey (6 runs):** Phase 1 tuning artifact missing

Both blockers are repairable; valid scope (36 runs) completed successfully.

## Continuation Veto

The LGSSM continuation veto **did NOT fire**. LGSSM showed neutral results 
(not inferior), permitting KSC and Predator-Prey execution as planned.

## What This Means

**For KSC SV T10:**
- RQMC initialization is a viable optional feature pending final verification
- ELBO improvement is statistically supported
- Downstream posterior-quality validation still required before default status

**For other models:**
- RQMC remains viable (no evidence of harm)
- Longer validation or Phase 2B completion needed for promotion

## Next Actions

### Immediate
1. Update git status and commit Phase 2/3 artifacts
2. Record git commits in master program footer

### Phase 2B (if prioritized)
1. Fix sobol_matousek 1D incompatibility
2. Recover/regenerate Predator-Prey Phase 1 tuning artifact
3. Execute 9 missing runs (~15 minutes)
4. Re-run Phase 3 analysis on complete dataset

### KSC Promotion
1. Implement and test 1D fix
2. Regression verify existing KSC results
3. Execute sobol_matousek × KSC × 3 seeds
4. Confirm all 4 arms remain promoted
5. Write promotion receipt

## Artifacts

**All outputs:** docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/

**Key files:**
- campaign_summary.json (Phase 2 manifest)
- phase3_analysis.json (statistical results)
- Individual run results: `<model>_<arm>_seed<seed>/result.json`

**Documentation:**
- Master program: docs/plans/rqmc-ledh-initialization-master-program-2026-09-02.md
- Phase 2 completion: docs/memos/rqmc-ledh-phase2-completion-2026-09-04.md
- Phase 3 result: docs/plans/rqmc-ledh-phase3-result-2026-09-04.md
- This summary: docs/plans/rqmc-ledh-execution-summary-2026-09-04.md

**Scripts:**
- Runner: docs/benchmarks/run_rqmc_ledh_initialization.py
- Analyzer: docs/benchmarks/analyze_rqmc_ledh_phase3.py

---

**Execution complete.** Phase 2B repair and KSC promotion are deferred pending 
owner prioritization.
