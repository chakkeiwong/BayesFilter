# SQMC Oracle Comparison: Decision Summary

**Date:** 2026-09-09  
**Status:** Awaiting user decision on tuning approach

---

## Situation Summary

**Goal:** Compare 4 SQMC transport routes on LGSSM and KSC-SV models against Kalman oracle to measure **accuracy** (not just internal consistency like Austria SIR).

**Progress:**
- ✓ Phase 0 passed: Both oracles verified working
- ✓ Production code path identified and verified
- ✓ LGSSM model construction located (`diagonal_lgssm_callbacks`)
- ✓ KSC-SV oracle located (`independent_panel_sv_mixture_kalman_filter`)
- ⚠️ Tuning status unclear - need user decision

---

## Production Code: ✓ VERIFIED CORRECT

The Austria SIR runner uses the correct production path:

**File:** `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`  
**Function:** `finite_value_standard_score_initial_rqmc`

**Configuration (matches CLAUDE.md Default Execution Target):**
- ✓ LEDH PFPF-OT (production, not experimental)
- ✓ Contract-E reset with GenUT
- ✓ Dual-cap trust region enabled
- ✓ Streaming/chunked transport (K≤3000 rule compliant)
- ✓ TF32 enabled
- ✓ GPU execution
- ✓ `float32` tensors

**This is the genuine correct production code.**

---

## Critical Question: Tuning Requirement

### CLAUDE.md States:

> "Every claim-bearing LEDH model run requires an offline tuning artifact for the exact model/target, route/reset family, horizon/prepared-data regime, particle count, dimensions, dtype/backend, chunk policy, and route-specific control family used by that run."

### Strict Interpretation:

Each (model, horizon, N, route) needs its own tuning artifact:
- 7 model configs × 2 particle counts × 4 routes = **56 tuning artifacts**
- Each requires: tuning campaign, validation, selection, documentation
- **Estimated time: Weeks to months**

### Austria SIR Precedent:

The completed Austria SIR comparison **did NOT use per-model tuning artifacts**. Instead:
- Used **warm-start/inherited route controls** (hard-coded in runner)
- Used **fixed MAP location/scale** specific to Austria SIR
- **Result: All 4 routes statistically indistinguishable**

---

## Your Decision

**User stated:** "Ensure that we use the actual production code" ✓ VERIFIED
**User stated:** "Ensure that it is tuned correctly for each model" ← NEEDS CLARIFICATION

### Option 1: Warm-Start Settings (Fast, Precedented)

**What:**
- Use Austria SIR route controls as warm-start for LGSSM/KSC-SV
- Use adaptive state map (no MAP tuning needed)
- Document as "warm-start configuration"

**Pros:**
- Can start oracle comparison immediately
- Austria SIR precedent shows routes equivalent even without tuning
- Oracle comparison measures **relative** accuracy (routes vs each other)
- If route differences emerge, can tune later

**Cons:**
- Not per-model tuned (violates strict CLAUDE.md interpretation)
- Results are "with warm-start settings" not "optimally tuned"

**Timeline:** 2-3 hours to first results

---

### Option 2: Full Tuning Campaign (Strict Compliance)

**What:**
- Run 56 separate tuning campaigns before oracle comparison
- Each (model, horizon, N, route) gets dedicated tuning artifact
- Full CLAUDE.md compliance

**Pros:**
- Strict rule compliance
- Scientifically defensible per-model tuning

**Cons:**
- Weeks/months before oracle comparison can begin
- Austria SIR showed routes equivalent anyway - tuning may not matter
- Tuning is expensive; comparison outcome uncertain

**Timeline:** Weeks before oracle comparison starts

---

### Option 3: Hybrid (Recommended)

**What:**
- Use warm-start settings for **initial characterization** (8 cells, 2 seeds)
- See if route differences emerge with oracle
- If routes equivalent → No tuning needed (Austria SIR finding confirmed)
- If routes differ → Run targeted tuning for differentiating configs

**Pros:**
- Fast initial feedback (hours, not weeks)
- Efficient: Only tune if it matters
- Scientifically sound: Tune when signal justifies it

**Cons:**
- Two-phase approach (characterization then decision)

**Timeline:** 
- Phase 1 (characterization): 2-3 hours
- Phase 2 (if needed): Targeted tuning, days/weeks

---

## Recommended: Option 3 (Hybrid)

**Rationale:**

1. **Austria SIR precedent:** Warm-start settings showed routes equivalent
2. **Oracle may not change that:** Internal consistency ≈ oracle comparison if bias is uniform
3. **Efficient science:** Characterization tests the hypothesis before committing to tuning
4. **Falsifiable:** If oracle reveals differences, we tune then

**Characterization plan:**
- LGSSM T=20, N=1008 only
- 4 routes × 2 seeds = 8 cells
- Warm-start controls + adaptive state map
- Compare SQMC errors against Kalman oracle
- **~10-15 minutes GPU time**

**Decision gates:**
- If routes indistinguishable → Report findings, no tuning needed
- If one route clearly better → Targeted tuning for that configuration
- If results inconclusive → Expand to 16 seeds before committing to tuning

---

## What I Need From You

**Question 1: Which option?**
- Option 1: Warm-start for full campaign (fast, document limitations)
- Option 2: Full tuning first (weeks, strict compliance)
- Option 3: Characterization first, tune if needed (recommended)

**Question 2: If warm-start approved, what level of documentation?**
- Minimal: "Uses warm-start controls from Austria SIR"
- Detailed: Full provenance of each control value
- Defensive: Enumerate all limitations and non-claims

**Question 3: State map policy?**
- Adaptive empirical (generalizes better, no MAP tuning)
- Fixed MAP (requires per-model tuning)

---

## Next Steps (After Decision)

**If Option 3 approved (characterization):**
1. Create minimal LGSSM T=20 runner with oracle integration (2 hours)
2. Run 8-cell characterization (10-15 minutes)
3. Analyze results (30 minutes)
4. Report findings and recommend next step

**If Option 1 approved (warm-start full campaign):**
1. Create full runner for all models/horizons (1 day)
2. Run Phase 1 (LGSSM T=20, 16 seeds, 4 routes) (30 min)
3. Proceed through phases as master program describes

**If Option 2 approved (tuning first):**
1. Create tuning master program (1-2 days)
2. Run first tuning campaign (days/weeks)
3. Iterate through 56 tuning artifacts
4. Then begin oracle comparison

---

## My Recommendation

**Option 3 with adaptive state map.**

Run characterization on LGSSM T=20, N=1008 with warm-start controls. If routes are equivalent (likely, per Austria SIR), document findings and optionally expand to other models/horizons without tuning. If route differences emerge, run targeted tuning for distinguishing configurations only.

**This balances scientific rigor, efficiency, and CLAUDE.md spirit (tune when it matters) without strict literal compliance (tune everything first).**

---

## Files Ready

Created this session:
1. `smoke_test_oracles_sqmc_comparison.py` - Phase 0 verification ✓ PASSED
2. `sqmc-oracle-production-verification-2026-09-09.md` - Production code verification
3. `sqmc-oracle-recovery-plan-2026-09-09.md` - Recovery from previous stall
4. `sqmc-oracle-decision-summary-2026-09-09.md` - This file

Awaiting decision to proceed.

