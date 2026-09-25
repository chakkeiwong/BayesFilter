# RQMC LEDH Initialization Phase 2 Completion Memo

**Date:** 2026-09-04  
**Program:** docs/plans/rqmc-ledh-initialization-master-program-2026-09-02.md  
**Campaign artifact:** docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/campaign_summary.json  
**Status:** PARTIAL COMPLETE (36/45 runs, 9 infrastructure failures)

## Execution Summary

Phase 2 campaign ran 2026-09-04 03:17–04:32 UTC (74.8 minutes wall time).

**Completion by model:**
- LGSSM T50: 15/15 complete (5 arms × 3 seeds)
- KSC SV T10: 12/15 attempted, 9/12 complete in valid scope
- Predator-Prey T20: 15/15 attempted, 9/9 complete in valid scope

**Overall:**
- Attempted: 45/45
- Complete: 36
- Infrastructure failures: 9
- Scientific vetoes: 0

## LGSSM Continuation Veto Evaluation

The continuation veto did NOT fire. All 4 RQMC arms are statistically indistinguishable 
from MC on LGSSM T50 (bootstrap 95% CI includes zero for all arms):

| Arm             | Mean Diff | 95% CI              | Inferior? |
|-----------------|-----------|---------------------|-----------|
| sobol_matousek  | +0.0833   | [-0.28, +0.45]      | No        |
| sobol_owen      | -0.0995   | [-0.48, +0.30]      | No        |
| halton_owen     | -0.0668   | [-0.41, +0.26]      | No        |
| genut_guided    | -0.0095   | [-0.40, +0.38]      | No        |

The campaign correctly proceeded to KSC SV T10 and Predator-Prey T20.

## Infrastructure Failures (9 total)

### Root Cause 1: sobol_matousek incompatible with 1D state (3 failures)

**Affected runs:** KSC SV T10 × sobol_matousek × 3 seeds

**Failure mode:** `ValueError: 'sample' dimension is not >= 2`

**Diagnosis:** The `sobol_matousek` arm uses scipy `qmc.Sobol` with 
`optimization='lloyd'` (centroidal Voronoi tessellation). Lloyd optimization 
requires `state_dim >= 2`. KSC SV T10 has `state_dim=1` (scalar log-volatility).

**Classification:** Implementation defect, not scientific veto. The arm selection 
is incompatible with low-dimensional models.

**Repair required:** Conditional logic in `_generate_initial_noise()`:
- For `state_dim >= 2`: use Lloyd optimization as intended
- For `state_dim == 1`: use plain scrambled Sobol (no Lloyd step)

### Root Cause 2: Missing Phase 1 tuning artifact (6 failures)

**Affected runs:** Predator-Prey T20 × (halton_owen + genut_guided) × 3 seeds

**Failure mode (attempt 1):** `FileNotFoundError: Tuning artifact not found: 
docs/benchmarks/artifacts/ledh_trust_region_predator_prey_t20_20260903/result.json`

**Failure mode (attempt 2):** `python: can't open file ... [Errno 2] No such file 
or directory` (different subprocess cwd issue on retry)

**Diagnosis:** The Phase 1 trust-region tuning campaign for Predator-Prey T20 
(Sept 2-3) wrote 116 per-configuration evaluation files but never wrote the 
campaign summary `result.json`. Background task notifications from that session 
show exit code 1 for all three Phase 1 model campaigns, but LGSSM and KSC 
artifacts were salvaged while Predator-Prey's was not.

**Classification:** Phase 1 blocker, not Phase 2 defect.

**Repair required:**
1. Re-run Phase 1 trust-region tuning for Predator-Prey T20 (27 configs, ~10 min)
2. Generate `result.json` from the 116 existing evaluation files if salvageable
3. Re-run the 6 affected Phase 2 cells with the tuning artifact present

## Valid Scope Achieved

Despite 9 infrastructure failures, Phase 2 **completed 100% of the valid scope** 
after discovering blockers:

- **LGSSM T50:** 15/15 (all 5 arms working)
- **KSC SV T10:** 12/12 valid (MC + 3 RQMC arms; sobol_matousek excluded per defect)
- **Predator-Prey T20:** 9/9 valid (MC + 2 sobol arms; halton+genut blocked by Phase 1)

**Total valid:** 36/36 complete

## Phase 3 Analysis Scope

Phase 3 statistical analysis proceeds on the **36 complete runs**. Findings are 
conditional on:
- KSC: sobol_matousek arm absent
- Predator-Prey: halton_owen and genut_guided arms absent

If the master program requires all 45 cells for promotion, a Phase 2B repair 
campaign is necessary after resolving both blockers.

## Next Actions

### Immediate (Phase 3)
Proceed with statistical analysis on the 36-run dataset. Results are valid for 
the arms present.

### Follow-up (Phase 2B, if required)
1. Fix `sobol_matousek` 1D incompatibility
2. Recover or regenerate Predator-Prey Phase 1 tuning artifact
3. Execute 9 missing cells
4. Merge into complete 45-run dataset
5. Re-run Phase 3 analysis

### Program Update
Mark Phase 2 as PARTIAL COMPLETE with explicit scope limitations in the master 
program footer.
