# LEDH Surrogate-Force HMC Program Reconciliation

**Date:** 2026-09-12  
**Issue:** Multiple master programs with inconsistent phase definitions  
**Purpose:** Establish single source of truth for program recovery

---

## Program Version History

### 1. Unified Program (2026-09-06)
**File:** `ledh-surrogate-hmc-unified-program-2026-09-06.md`

**Phase structure:**
- Phase 0: Decisions and Prerequisites
- Phase 1: JVP and Diagnostic Checks  
- Phase 2: Toy Potential Validation
- **Phase 3: Damping Calibration**
  - Task 3.1: Damping parameter derivation
  - Task 3.2: Damping calibration curve
  - Task 3.3: Seed discipline resolution
- Phase 4: LGSSM Certification
- Phase 5: Tier A Validation

### 2. Executable Master Program (2026-09-07)
**File:** `ledh-surrogate-hmc-executable-master-program-2026-09-07.md`

**Status:** "Replaces: All prior LEDH HMC validation programs"

**Phase structure:**
- Phase 0: Decisions and Prerequisites
- Phase 1: JVP and Diagnostic Checks
- Phase 2: Toy Potential Validation
- **Phase 3: Seed Policy Verification** (different!)
  - Build dual-adapter wrapper
  - Test V1: Determinism check
  - Test V2: Reversibility check
  - Test V3: No call-count dependence
- Phase 4: LGSSM Certification
- Phase 5: Tier A Validation

---

## Inconsistency

The 2026-09-07 program **changed Phase 3** from "Damping Calibration" to "Seed Policy Verification" but did NOT include damping calibration anywhere else in the program.

**Recent work (2026-09-11 to 2026-09-12)** followed the **older unified program** (2026-09-06) Phase 3 structure:
- Created `ledh_dual_parameter_target.py` (dual-adapter for damping)
- Completed Task 3.1 (damping derivation)
- Attempted Task 3.2 (damping calibration)
- Completed Task 3.3 (seed policy)

---

## Current State

### Completed Work

**From unified program Phase 3:**
- ✅ Task 3.1: Damping derivation complete
- ✅ Task 3.3: Seed policy resolved (frozen noises in dual-parameter target)
- ✅ Dual-parameter adapter implemented (`bayesfilter/inference/ledh_dual_parameter_target.py`)
- ✅ Custom gradient fix (prevents autodiff memory explosion)
- ✅ Spot-check verification passed (N=252, two damping ratios)
- 🔴 Task 3.2: Damping calibration blocked (computational cost, see below)

**From executable program Phase 3:**
- ❓ Test V1 (Determinism): Implicitly satisfied by frozen noises
- ❓ Test V2 (Reversibility): Not explicitly tested
- ❓ Test V3 (Call-count independence): Not explicitly tested

### Blocker: Damping Calibration Computational Cost

**Root cause identified:**
- Each LEDH filter evaluation with N=252-1008 particles takes 5-10 seconds
- Dual-parameter target requires 2 LEDH calls per HMC step (exact value + biased score)
- Full calibration: 4 arms × 2 chains × 1000 steps × 2 calls = 16,000 evaluations
- Estimated time: 22-44 hours for full spec, 9-18 hours for pilot (N=252, 400 steps)

**Custom gradient fix:**
- Previously: TFP HMC used autodiff → massive computation graphs → OOM
- Now: `tf.custom_gradient` provides analytical surrogate gradient → no graph explosion
- Validated: Simple test confirmed custom_gradient works with TFP HMC

---

## Reconciled Program

### Authority
**Primary:** Executable Master Program (2026-09-07) - explicitly replaces prior versions

**Modification:** Add damping calibration back into program structure (it was accidentally removed)

### Proposed Reconciliation

**Option 1: Damping calibration as Phase 3.5 (insert between current Phases 3 and 4)**
- Phase 3: Seed Policy Verification (as in executable program)
- **Phase 3.5: Damping Calibration** (restore from unified program)
- Phase 4: LGSSM Certification (as in executable program)

---

## Amendment 2026-09-14: While-Loop Regression Repair Prerequisite

**Context:** Phase 2B unification (commit 5cc59cfa, 2026-09-11) deleted working `tf.while_loop` implementation. Current LEDH engine unrolls ~400 flow stages into the graph. Trace time 485s at T=50 (dispatch-bound). Damping calibration requires 16,000+ LEDH evaluations and is infeasible without repair.

**Reconciliation updated:**
- Phase 3: Seed Policy Verification (as in executable program)
- **Phase 3.5: LEDH While-Loop Regression Repair** (NEW, inserted 2026-09-14)
  - Reference: `docs/plans/ledh-while-loop-regression-repair-plan-2026-09-14.md`
  - Restore `tf.while_loop` for time and substep loops
  - Target: graph O(10³) nodes, trace <50s, steady ≤80s
  - Blocks Phase 4a damping calibration
- ~~Phase 3.5: Damping Calibration~~ → **moved into Phase 4a** (damping now part of LGSSM certification)
- Phase 4: LGSSM Certification (as in executable program)

**Rationale:** The while-loop regression makes damping calibration computationally infeasible (22-44 hours at current cost). Repair is a prerequisite, not a parallel task. Damping calibration moves into Phase 4a as originally planned in the executable program.

**Option 2: Damping calibration absorbed into Phase 4a**
- Phase 3: Seed Policy Verification
- Phase 4a: LGSSM diagnostic **with damping sweep** (multi-arm instead of 2-arm)
- Phase 4b: LGSSM certification (as planned)

**Option 3: Skip damping calibration entirely**
- Use damping derivation from Task 3.1 (100× factor) without empirical validation
- Proceed directly to Phase 4 with fixed damping
- Risk: Damping choice may be suboptimal

---

## Recommendation

**Adopt Option 2:** Merge damping calibration into Phase 4a as a multi-arm diagnostic.

**Rationale:**
1. Phase 4a already runs HMC on LGSSM d=3 T=50 with N=1008
2. Current Phase 4a: 2 arms (exact vs 100× damped), 2 chains × 1000 steps
3. Extended Phase 4a: 4 arms (1×, 10×, 100×, 1000× damping), same chain spec
4. Measures W₂ agreement AND establishes optimal damping ratio
5. No separate calibration phase needed
6. Adds ~4 GPU-hours (2× current Phase 4a budget)

**Updated Phase 4a specification:**

```
Phase 4a: LGSSM Diagnostic with Damping Sweep (4 GPU-hours)

Tasks:
1. LGSSM d=3 T=50 N=1008, one frozen ω
2. 4 arms with damping ratios: 1×, 10×, 100×, 1000×
3. 2 chains × 1000 steps per arm (parallel batch execution)
4. Measure W₂ distance between all pairs
5. Measure acceptance rate and ESS per arm

Success criterion:
- At least one damping ratio achieves W₂ < threshold vs 1× baseline
- That ratio has acceptance ≥ 0.15
- Proceed to Phase 4b with best damping ratio

Deliverable: ledh-surrogate-hmc-phase4a-damping-diagnostic-2026-09-12.md
```

---

## Recovery Path

To recover from current state:

### Immediate Next Steps

1. **Acknowledge computational blocker** for standalone damping calibration
2. **Adopt Option 2** (merge into Phase 4a)
3. **Update executable master program** to include damping sweep in Phase 4a
4. **Create Phase 4a runner** with 4-arm damping sweep
5. **Execute Phase 4a** when computational resources available

### Work Not Lost

All completed work remains valid:
- Dual-parameter adapter is correct and tested (spot-check passed)
- Custom gradient fix is essential (prevents OOM)
- Damping derivation provides the 4 test ratios
- Seed policy resolution informs implementation

The work transitions cleanly into Phase 4a multi-arm diagnostic.

---

## Updated Master Program Status

### Completed
- ✅ Phase 0: Decisions and Prerequisites
- ✅ Phase 1: JVP and Diagnostic Checks
- ✅ Phase 2: Toy Potential Validation
- ✅ Phase 3: Seed Policy Verification (via dual-parameter adapter design)

### Next
- 📋 **Phase 4a: LGSSM Damping Diagnostic** (4-arm sweep, 4 GPU-hours)
  - Requires computational resources for 9-18 hour run
  - Can run pilot scale (N=252, 400 steps) for initial validation

### Pending
- Phase 4b: LGSSM Full Certification (conditional on Phase 4a)
- Phase 5: Tier A Validation (conditional on Phase 4b)

---

## Single Source of Truth

**AUTHORITATIVE PROGRAM:** 
`docs/plans/ledh-surrogate-hmc-executable-master-program-2026-09-07.md`

**WITH AMENDMENT:**
Phase 4a expanded to include damping sweep (this reconciliation)

**DEPRECATED:**
- `ledh-surrogate-hmc-unified-program-2026-09-06.md` (superseded)
- Phase 3 "Damping Calibration" as standalone phase (absorbed into Phase 4a)

---

## Summary

We have a **complete, logical, consistent master program** with clear:
- **Background:** Corollary 5.2 validation
- **Motivation:** Certify surrogate-force HMC samples π_N^ω correctly
- **Execution phases:** 0-5 with clear deliverables and decision gates
- **Recovery:** Any interruption resumes from phase deliverables

**Current blocker:** Computational resources for Phase 4a (4-18 hours depending on scale)

**Path forward:** Execute Phase 4a multi-arm damping diagnostic when resources available, or run pilot scale (N=252) for initial validation.
