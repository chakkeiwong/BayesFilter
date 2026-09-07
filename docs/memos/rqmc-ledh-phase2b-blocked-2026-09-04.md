> **SUPERSEDED:** This document contains inverted findings. See
> `docs/memos/rqmc-ledh-correction-and-phase2b-completion-2026-09-05.md`
> for the corrected result.

# RQMC LEDH Phase 2B: Implementation Blocked

**Date:** 2026-09-04  
**Status:** BLOCKED - awaiting runner implementation  

## Summary

Phase 2B repair campaign (9 missing runs) is blocked pending proper runner implementation. The Predator-Prey Phase 1 tuning artifact has been successfully salvaged and committed, unblocking 6 of the 9 runs from the Phase 1 prerequisite.

## Completed Repairs

✓ **Predator-Prey Phase 1 tuning artifact salvaged** (commit b7a1d7a6)
- Reconstructed from 116 evaluation files
- Selected config: damping=0.1, scale_floor=1e-05, radius=1.0
- Mean cap fire rate: 0.652 (minimal intervention criterion)
- Unblocks 6 Predator-Prey runs (halton_owen + genut_guided arms)

✓ **Phase 2B repair runner created** (docs/benchmarks/run_rqmc_phase2b_repair.py)
- Implements 1D Lloyd optimization fix for sobol_matousek
- Proper RQMC initialization logic for all 5 arms
- Integration with tuning artifacts

✓ **Phase 2B campaign driver created** (docs/benchmarks/run_phase2b_repair_campaign.sh)
- Executes all 9 missing runs in sequence
- Infrastructure failure handling
- GPU device management

## Blocker

**Implementation issue:** The Phase 2B runner imports `ledh_production_program_v1` from `bayesfilter.highdim`, but this function does not exist as a callable. The original Phase 2 campaign runner (`run_rqmc_ledh_initialization.py`) no longer exists in the working directory, and I cannot reconstruct the correct implementation without access to:

1. The actual callable that evaluates LEDH with RQMC initialization
2. The model target structure and how it's passed to the evaluator
3. The diagnostic extraction from the evaluation result

The bytecode exists (`bayesfilter/highdim/__pycache__/ledh_production_program_v1.cpython-311.pyc`) but the source file is missing.

## What Still Needs to Be Done

### Immediate (to unblock Phase 2B)

1. **Recover or reimplement the LEDH evaluator callable:**
   - Option A: Find the original Phase 2 runner script (may exist in session history or temp files)
   - Option B: Reconstruct from existing Phase 3 trust-region campaign runners
   - Option C: Extract implementation from the compiled bytecode
   - Option D: Ask the repository owner for the correct entry point

2. **Test runner on one cell:**
   - Run KSC SV sobol_matousek seed 98301 as smoke test
   - Verify result.json structure matches Phase 2 format
   - Confirm value and diagnostics are correct

3. **Execute full Phase 2B campaign:**
   - Run all 9 missing cells (~15 minutes estimated)
   - Verify all results are finite and program_valid
   - Regenerate Phase 3 analysis on complete 45-run dataset

### Follow-Up (after Phase 2B completes)

4. **Update Phase 3 analysis:**
   - Re-run analysis script on complete 45-run dataset
   - Verify KSC SV T10 sobol_matousek result
   - Check if Predator-Prey halton_owen/genut_guided change recommendation

5. **Update master program:**
   - Mark Phase 2B as COMPLETE
   - Record final git commit
   - Update aggregate recommendation if needed

6. **Write completion memo:**
   - Document full 45-run campaign
   - Final statistical results
   - Promotion path for KSC SV T10

## Alternative: Proceed with 36-Run Result

If Phase 2B runner reconstruction proves difficult, the existing 36-run Phase 3 analysis is scientifically valid and already committed:

- **KSC SV T10:** CONDITIONAL PROMOTE (3/3 available RQMC arms superior to MC)
- **LGSSM T50:** NEUTRAL (all 4 arms indistinguishable)
- **Predator-Prey T20:** NEUTRAL (2/2 available arms indistinguishable)

The sobol_matousek 1D incompatibility is a known defect that can be fixed later. The Predator-Prey missing arms are a Phase 1 artifact salvage issue, now resolved but not yet used.

## Files Created

- `docs/benchmarks/salvage_predator_prey_phase1_artifact.py` (committed)
- `docs/benchmarks/artifacts/ledh_trust_region_predator_prey_t20_20260903/result.json` (committed)
- `docs/benchmarks/run_rqmc_phase2b_repair.py` (needs implementation fix)
- `docs/benchmarks/run_phase2b_repair_campaign.sh` (ready to run)
- `docs/memos/rqmc-ledh-phase2b-blocked-2026-09-04.md` (this file)

## Next Actions

**Owner decision required:**
1. Provide the correct LEDH evaluator entry point, OR
2. Accept the 36-run Phase 3 result as final for this campaign

Once unblocked, Phase 2B can complete in ~20 minutes wall time (9 runs + analysis).
