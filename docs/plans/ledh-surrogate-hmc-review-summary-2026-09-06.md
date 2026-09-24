# LEDH Surrogate-Force HMC Program — Review Summary

**Date:** 2026-09-06  
**Request:** Thorough review for procedurally correct master program  
**Status:** Review complete, corrected program ready for decision

---

## What You Asked For

1. **Procedurally correct master program** — experimental setup that can certify what it claims OR detect failure, with interpretable results either way
2. **Scientifically rigorous modifications** — not a wish list for desired outcomes
3. **Clear separation** — what we're testing vs what we hope to find

---

## What I Found

### The Core Problem

**Current program's Phase 4 primary criterion (line 586):**
> "Posterior 95% intervals cover true θ (all 5 parameters)"

**This criterion conflates three questions:**
1. Does the sampler work correctly? (π_N^ω sampling)
2. Is finite-N bias small? (π_N^ω ≈ π_∞)
3. Does this dataset's posterior cover true θ? (data property)

**What Corollary 5.2 actually proves (variance note line 964):**
> "The θ-marginal ∝ exp(-U(θ)) is preserved, for any quality of F"

The theorem certifies **question 1 only** — sampling π_N^ω (the finite-N pseudo-posterior at frozen noise ω).

**Why this is a procedural defect:**
- **If method works but coverage fails:** Could be finite-N bias or unlucky data → false negative
- **If method fails but coverage passes:** Over-dispersion widens intervals by accident → false positive
- **Either outcome is uninterpretable** relative to what Corollary 5.2 proves

This is **not** a prediction about scientific outcomes. It's a statement that the experiment cannot distinguish sampler failure from other causes, regardless of what happens.

---

## The Fix

### Replace Phase 4 Primary Criterion

**From (current):**
> Primary criterion: Arm 2 posterior covers true θ

**To (corrected):**
> Primary criterion: Arm 1 (exact-force HMC) and Arm 2 (damped-force HMC) posteriors agree within MCMC error, measured via 2-Wasserstein distance W₂(P₁, P₂) < 2× MCMC_SE

**Why this is procedurally correct:**
| Outcome | Interpretation | Falsifiable? |
|---|---|---|
| Arms agree | Damped-force samples π_N^ω correctly | ✅ Corollary 5.2 certified |
| Arms disagree | Surrogate-force fails on this target | ✅ Method rejected |

Both outcomes are interpretable. True-θ coverage becomes an **explanatory diagnostic** — we report it, but don't promote/reject based on it.

### Three Required User Decisions

Before any work can proceed, you must decide:

**Decision 1: Tolerance derivation** (blocks Phase 2B, Phase 3)
- Current program uses two incompatible tolerances (1e-12 vs 5e-4)
- **Proposed:** Derive from condition number + tuning-insensitivity + backend dtype
- **Recommendation:** Condition-number-derived, float32 TF32 production regime
- **Time:** 0.3 day

**Decision 2: Seed policy** (required for Corollary 5.2)
- Remark 5.3 requires "all Monte Carlo seeds inside F must be frozen"
- **Proposed:** One master ω, frozen for chain, verify with 3 tests (bitwise/involution/continuity)
- **Recommendation:** Approve (determined by theorem, not negotiable)
- **Time:** 0.5 day

**Decision 3: Coverage criterion math** (if coverage is diagnostic)
- Current "all 5 parameters" has 77% false-negative rate
- **Proposed:** Joint 95% Mahalanobis region
- **Recommendation:** Approve (but coverage is explanatory only after Fix 1)
- **Time:** 0.1 day

**Total to unblock:** 0.9 days (all CPU, no campaigns)

---

## What The Corrected Program Certifies

### If Phase 4 passes (W₂ < threshold):

**Certified by Corollary 5.2:**
- ✅ Surrogate-force HMC samples π_N^ω correctly
- ✅ Acceptance ≥ 0.15 (chain is mixing)
- ✅ Method viable on LGSSM d=3 T=50

**Nominated for Phase 5:**
- ⏳ Robust across Tier A models? (requires Phase 5)

**NOT certified (out of scope):**
- ❌ Samples from π_∞ (finite-N bias unmeasured)
- ❌ True-θ coverage (diagnostic only)
- ❌ "Better" than exact-force (no statistical ranking at 1 seed)

### If Phase 4 fails (W₂ ≥ threshold):

**Interpretable:**
- ✅ Surrogate-force does NOT sample π_N^ω correctly
- ✅ Mechanism broken (not just "LEDH bias too large")
- ✅ Corollary 5.2 premises violated or implementation bug

**Next action:**
- Investigate: reversibility, determinism, seed policy (Phase 3 tests)
- If all pass: mechanism failure → STOP
- If test fails: fix violation, re-run

---

## Procedural vs Scientific

### Procedural Correctness (this review's job)
- ✅ Experiment can certify success OR detect failure
- ✅ Either result is interpretable
- ✅ Criterion measures what the theorem proves
- ✅ All premises are verified (Phase 3 seed tests)

### Scientific Outcome (discovered by running)
- ❓ Does the method actually work? (unknown)
- ❓ Is acceptance acceptable? (unknown)
- ❓ Is ESS/gradient competitive? (unknown)

**The corrected program guarantees the first (procedural). The second (scientific) is what we learn from execution.**

---

## Documents Delivered

1. **Procedural review** (`ledh-surrogate-hmc-procedural-review-2026-09-06.md`)
   - Complete gap analysis (1 blocking, 3 required, 6 non-blocking)
   - Why the current criterion is wrong
   - What the corrected criterion measures
   - 8-part analysis with examples

2. **Corrected master program** (`ledh-surrogate-hmc-procedurally-correct-master-program-2026-09-06.md`)
   - Implements all procedural fixes
   - Clear procedural vs scientific distinction
   - Three user decisions up front (blocks Phase 1)
   - Reference-sampler agreement as primary criterion
   - True-θ coverage demoted to explanatory
   - Seed-policy verification (Phase 3)
   - Complete budget and gates

3. **This summary** (one-page version for decision)

---

## What I Need From You

**Three decisions to proceed:**

1. **Approve tolerance derivation?** (condition-number + tuning-band, float32 TF32 regime)
2. **Approve seed policy?** (one master ω, frozen, 3 verification tests)
3. **Approve coverage formula?** (joint Mahalanobis, diagnostic only)

**Default if no response:** Approve all three as recommended (0.9 days to implement).

**Alternative:** Review the full documents, modify any decision, then approve.

**After approval:** Program is procedurally correct and ready to execute. Scientific outcomes (does it work?) are discovered by running, not predicted by planning.

---

## What Changed From Previous Programs

### Superseded program (2026-09-06 unified)
- ❌ Phase 4 criterion: "posterior covers true θ"
- ❌ Measures compound question (sampler + bias + data)
- ❌ Pass and fail both uninterpretable

### Corrected program (this review)
- ✅ Phase 4 criterion: "Arm 1 and Arm 2 posteriors agree"
- ✅ Measures exactly what Corollary 5.2 proves
- ✅ Pass → certified correct, fail → method broken, both interpretable

### Also fixed
- ✅ Tolerance incoherence resolved (one derived tolerance)
- ✅ Seed policy explicit with verification (Corollary 5.2 premises)
- ✅ Coverage math corrected (but demoted to diagnostic)
- ✅ Procedural vs scientific separation clear throughout

---

## Bottom Line

**You were right:** The planning process was certifying the wrong thing. Coverage tests a compound question (sampler × bias × data luck) when the theorem proves one component (sampler correctness). A passing run could mean a broken sampler got lucky; a failing run could mean a correct sampler hit unlucky data.

**The fix:** Measure what the theorem proves. Two HMC runs on the same frozen target, compare posteriors directly. Agreement → sampler works. Disagreement → sampler broken. Both interpretable.

**Cost:** 0.9 days to implement three decisions, then program is procedurally correct.

**Your call:** Approve the three defaults, or review and modify. Once decided, the program guarantees interpretable results regardless of scientific outcome.
