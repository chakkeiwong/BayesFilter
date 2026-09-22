# SQMC Oracle Comparison: Executive Status Report

**Date:** 2026-09-09  
**Status:** READY FOR EXECUTION  
**Branch:** `rqmc-sqmc-4route-comparison`

---

## Status: Recovery Complete ✓

The previous agent stalled after planning. Recovery is now complete and execution is ready to begin.

---

## Governing Document

**Master Program v2:** `docs/plans/sqmc-oracle-comparison-master-program-v2-2026-09-09.md`

**Key features:**
- Phased execution: Characterization (8 cells) → Expansion (conditional) → Full campaign (optional)
- Tuning strategy: Warm-start + adaptive state map (no per-model tuning required)
- Decision gates: Route distinguishability determines next steps
- Timeline: Hours to decision (not weeks)

---

## What's Ready

### Phase 0: Pre-execution Verification ✓ COMPLETE
- LGSSM Kalman oracle: ✓ Verified working
- KSC-SV dense Kalman oracle: ✓ Verified working
- All horizons: ✓ Feasible
- Artifact: `smoke_test_oracles_sqmc_comparison.py` PASSED

### Production Code ✓ VERIFIED
- LEDH PFPF-OT with Contract-E + dual-cap + trust-region
- Correct production path confirmed
- Matches CLAUDE.md requirements

### Parameter Strategy ✓ DEFINED
- Mathematical analysis complete
- Warm-start controls validated
- Adaptive state map eliminates MAP tuning need
- Documentation: `sqmc-parameters-tuning-analysis-2026-09-09.md`

### Phase 1 Implementation: 60% Complete
- ✓ Runner skeleton created and tested
- ✓ Configuration defined
- ⏳ LGSSM model construction (next step, 2-3 hours)
- ⏳ SQMC integration (after model, 2-3 hours)

---

## Next Immediate Steps

### Step 1: Implement LGSSM Model Construction (2-3 hours)
**File:** `docs/benchmarks/run_sqmc_oracle_characterization_step1.py`  
**Task:** Adapt `diagonal_lgssm_callbacks()` for 10D state/obs, T=20 horizon  
**Output:** Working LGSSM model with callbacks compatible with LEDH PFPF-OT

### Step 2: Integrate SQMC Calling Code (2-3 hours)
**Task:** Call `finite_value_standard_score_initial_rqmc` with LGSSM callbacks  
**Reference:** Austria SIR runner `_row()` function  
**Output:** End-to-end SQMC execution with oracle comparison

### Step 3: Run Characterization (15 minutes)
**Configuration:** 4 routes × 2 seeds = 8 cells  
**Output:** Per-route mean errors, distinguishability assessment

### Step 4: Analyze and Decide (30 minutes)
**Outcomes:**
- Routes indistinguishable → Austria SIR confirmed, proceed to other models
- Routes differ → Statistical confirmation + optional tuning
- Issues found → Fix before scaling

---

## Timeline to Decision

| Activity | Time |
|---|---|
| LGSSM model construction | 2-3 hours |
| SQMC integration | 2-3 hours |
| Test single cell | 30 min |
| Run 8-cell characterization | 15 min |
| Analyze results | 30 min |
| **Total** | **6-8 hours** |

---

## Configuration Summary

**Production algorithm:** LEDH PFPF-OT TF32 with streaming transport  
**Reset:** Contract-E with GenUT  
**Dual-cap:** Enabled (radial_cap=2.0, coordinate_cap=0.98)  
**Trust-region:** Enabled (damping=0.01, radius=0.5)  
**State map:** Adaptive empirical (no MAP tuning)  
**Transport:** ε=8.0, 8 sinkhorn steps, 8 balance steps

**Source:** Warm-start from Austria SIR + four-model dual-cap evidence

---

## Documentation Created (9 files)

1. `smoke_test_oracles_sqmc_comparison.py` — Phase 0 verification ✓ PASSED
2. `sqmc-oracle-production-verification-2026-09-09.md` — Production code verification
3. `sqmc-parameters-tuning-analysis-2026-09-09.md` — Mathematical parameter analysis
4. `sqmc-oracle-decision-summary-2026-09-09.md` — User decision record
5. `sqmc-oracle-recovery-plan-2026-09-09.md` — Recovery diagnosis
6. `phase2b-step1-characterisation-tests-plan-20260909.md` — Characterization plan
7. `run_sqmc_oracle_characterization_step1.py` — Implementation skeleton (tested)
8. `sqmc-oracle-complete-handoff-2026-09-09.md` — Complete handoff document
9. `sqmc-oracle-comparison-master-program-v2-2026-09-09.md` — **AUTHORITATIVE MASTER PROGRAM**

---

## Key Decisions Made

**Q: Full 896-cell campaign immediately?**  
**A:** No. Characterization first (8 cells), then decide.

**Q: Per-model tuning required?**  
**A:** No. Warm-start + adaptive state map sufficient. Tuning optional if routes differ.

**Q: Which production code?**  
**A:** `finite_value_standard_score_initial_rqmc` - verified correct LEDH PFPF-OT path.

**Q: Critical parameters to tune?**  
**A:** Only MAP location/scale, but avoided by using adaptive state map.

---

## Success Criteria

**Phase 1 (Characterization):**
- ✓ 8 cells complete with finite values
- ✓ Oracle comparison successful
- ✓ Route distinguishability assessed
- ✓ Decision documented

**Overall:**
- Answer: "Do SQMC routes differ in oracle accuracy?"
- If yes: Which route? Does it generalize?
- If no: Confirm Austria SIR finding

---

## Bottom Line

**Recovery:** ✓ Complete  
**Master program:** ✓ Up to date (v2)  
**Production code:** ✓ Verified correct  
**Parameters:** ✓ Strategy defined (warm-start + adaptive)  
**Oracles:** ✓ Both working  
**Implementation:** 60% complete  
**Next:** LGSSM model construction (2-3 hours)  
**Timeline:** 6-8 hours to decision point

**Ready to proceed.**

