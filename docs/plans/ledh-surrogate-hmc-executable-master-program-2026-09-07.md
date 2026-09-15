# LEDH Surrogate-Force HMC Validation — Executable Master Program

**Date:** 2026-09-07  
**Authority:** This program  
**Status:** Ready for execution with Option A defaults approved  
**Replaces:** All prior LEDH HMC validation programs

---

## Program Goal

**Certify:** Surrogate-force HMC with damped OT gradient samples from π_N^ω (frozen-noise finite-N posterior) as Corollary 5.2 proves.

**Does NOT certify:** Finite-N bias (π_N^ω ≈ π_∞), posterior coverage of true θ, or "better than baseline."

**Scientific question discovered by execution:** Does damped surrogate-force HMC work? This program is procedurally correct whether the answer is yes or no.

---

## Procedural vs Scientific Separation

### What This Program Certifies (Procedural Correctness)
- ✅ If posteriors agree → surrogate-force HMC samples π_N^ω correctly
- ✅ If posteriors disagree → surrogate-force HMC is broken
- ✅ Either outcome is interpretable and actionable

### What Execution Discovers (Scientific Outcome)
- ❓ Does the method actually work?
- ❓ What is acceptance rate?
- ❓ How large is finite-N bias?
- ❓ Does posterior cover true θ? (diagnostic only)

---

## User Decisions (Option A Approved)

All three decisions approved with defaults. No further approval needed for execution.

### Decision 1: Tolerance Derivation (0.3 day)
**Approved:** Condition-number-derived, float32 TF32 production regime  
**Rationale:** Certificate covers the regime we actually use  
**Implementation:** Phase 0.1

### Decision 2: Seed Policy (0.5 day)
**Approved:** One master ω, frozen across all phases, three verification tests  
**Rationale:** Corollary 5.2 requires deterministic force  
**Implementation:** Phase 0.2 + Phase 3

### Decision 3: Coverage Formula (0.1 day)
**Approved:** Joint 95% Mahalanobis region (diagnostic only, not primary criterion)  
**Rationale:** Avoids 77% false-negative rate of marginal coverage  
**Implementation:** Phase 0.3

**Total Option A implementation:** 0.9 days (included in Phase 0 below)

---

## Execution Phases

### Phase 0: Decisions and Prerequisites (1 day)

**Tasks:**
1. Implement tolerance derivation (0.3 day)
2. Document seed policy (0.5 day)
3. Implement joint Mahalanobis coverage (0.1 day)
4. One compilation smoke test (0.1 day)

**Deliverable:** `docs/memos/ledh-surrogate-hmc-phase0-decisions-2026-09-07.md`

**Success criterion:** All three decisions implemented, smoke test compiles

**No approval needed** — Option A already approved

---

### Phase 1: JVP and Diagnostic Checks (0.5 day)

**Tasks:**
1. Re-run existing JVP parity tests (Contract E streaming)
2. Re-run Sinkhorn convergence checks
3. Verify memory-growth policy on GPU

**Deliverable:** `docs/memos/ledh-surrogate-hmc-phase1-diagnostics-2026-09-07.md`

**Success criterion:** All existing tests pass, memory growth verified

**Approval:** None needed (existing tests, CPU-only)

---

### Phase 2: Toy Potential Validation (0.5 day, 0.5 GPU-hour)

**Tasks:**
1. Simple 3D quadratic potential
2. Exact-force baseline (analytical gradient)
3. Damped-force test (OT surrogate with ε damping)
4. Posterior agreement via W₂ distance

**Deliverable:** `docs/plans/ledh-surrogate-hmc-phase2-toy-result-2026-09-07.md`

**Success criterion:** Posteriors agree within tolerance

**Budget:** 0.5 GPU-hour (short chains, simple potential)

**Approval:** None needed (toy potential, minimal cost)

---

### Phase 3: Seed Policy Verification (1 day)

**Tasks:**
1. Build dual-adapter wrapper (exact-force + damped-force on same frozen ω)
2. **Test V1:** Determinism check (same ω → bitwise identical trajectories)
3. **Test V2:** Reversibility check (involution: forward-backward-forward = identity)
4. **Test V3:** No call-count dependence (force(θ, call=N) = force(θ, call=N+1))

**Deliverable:** 
- `bayesfilter/inference/ledh_dual_force_adapter.py`
- `tests/inference/test_ledh_seed_policy.py`

**Success criterion:** All three tests pass

**Approval:** None needed (tests only, CPU-only)

---

### Phase 3.5: LEDH While-Loop Regression Repair ✓ COMPLETE (2026-09-15)

**Status:** ✓ SUCCESS (commit 230f3295)

**Result Summary:**
- Trace time: 6.70s < 50s target ✓ (72× speedup vs 485s baseline)
- Steady state: 4.56s ≤ 80s target ✓ (8.6× speedup vs 39.3s baseline)
- Graph size: O(10³) nodes (8× reduction from 400 unrolled stages)
- Oracle contract: 8/8 tests PASSING

**Implementation:**
- Time-loop `tf.while_loop` restored (Phase 3.5.1)
- Substep-loop `tf.while_loop` restored (Phase 3.5.2)
- Measurement validated at plan scale (N=252, T=50, substeps=8)
- Phase 1 constraint enforced: `annealed_stages=1` only

**Deliverables:**
- Modified `bayesfilter/highdim/ledh_canonical_score_tf.py`
- Measurement script: `docs/benchmarks/ledh_phase_3_5_2_measurement.py`
- Result summary: `docs/benchmarks/ledh_phase_3_5_result.md`

**Note:** Phase 3.5.3 (XLA evaluation) and 3.5.4 (K-batch) deferred. Current performance meets all targets for Phase 4 execution.

---

### Phase 4: LGSSM Certification (2 days, 8-11 GPU-hours)

Split into two stages with decision gate.

#### Phase 4a: Ultra-Short Diagnostic (2 GPU-hours)

**Tasks:**
1. LGSSM d=3 T=50 N=1008, one frozen ω
2. Arm 1: Exact-force HMC (Contract E, zero damping)
3. Arm 2: Damped-force HMC (Contract E, ε=0.01)
4. **2 chains × 1000 steps per arm** (parallel batched execution)
5. Measure W₂ distance between posteriors

**Budget:** 2 GPU-hours (1 hour per arm, chains in parallel)

**Success criterion (diagnostic only):** 
- If W₂ < threshold → proceed to Phase 4b
- If W₂ > threshold → method broken, STOP and diagnose

**Deliverable:** `docs/plans/ledh-surrogate-hmc-phase4a-diagnostic-result-2026-09-07.md`

**Approval:** None needed (diagnostic, 2 GPU-hours)

---

#### Phase 4b: Full Certification (6-9 GPU-hours, conditional)

**Only runs if Phase 4a passes.**

**Tasks:**
1. Same LGSSM setup, same frozen ω
2. Arm 1: Exact-force HMC
3. Arm 2: Damped-force HMC
4. **4 chains × 3000 steps per arm** (parallel batched execution)
5. Measure W₂ distance with full MCMC error quantification

**Budget:** 6-9 GPU-hours (3-4.5 hours per arm, chains in parallel)

**Primary criterion:** W₂(posterior₁, posterior₂) < tolerance

**Veto diagnostics:**
- Acceptance < 0.15 (mixing threshold)
- R-hat > 1.05 (convergence failure)
- Any divergence (integration error)

**Explanatory diagnostics (NOT primary):**
- True-θ coverage (joint Mahalanobis)
- ESS per gradient
- Runtime

**Deliverable:** `docs/plans/ledh-surrogate-hmc-phase4b-certification-result-2026-09-07.md`

**Approval:** None needed (Phase 4a already passed, budget pre-approved)

---

### Phase 5: Tier A Validation (2 days, 24-33 GPU-hours, conditional)

**Only runs if Phase 4b certifies correctness.**

**Models:** LGSSM (extended), Lotka-Volterra, Keen-Standish-Coxe (3 models)

**Per-model setup:**
- Same dual-arm design (exact vs damped)
- 4 chains × 3000 steps per arm
- **8-11 GPU-hours per model**

**Budget:** 24-33 GPU-hours total (3 models)

**Success criterion:** All three models pass W₂ agreement

**Deliverable:** `docs/plans/ledh-surrogate-hmc-phase5-tier-a-result-2026-09-07.md`

**Approval:** None needed (Phase 4b certified, budget pre-approved)

---

## Budget Summary

### CPU Time
- Phase 0: 1 day (decisions + smoke test)
- Phase 1: 0.5 day (diagnostics)
- Phase 2: 0.5 day (toy potential)
- Phase 3: 1 day (seed tests)
- Phase 4: 2 days (implementation + analysis)
- Phase 5: 2 days (conditional)
- **Total: 5 days (Phases 0-4), 7 days (with Phase 5)**

### GPU Time (Corrected for Parallel Chain Execution)
- Phase 2: 0.5 GPU-hour (toy)
- Phase 4a: 2 GPU-hours (diagnostic)
- Phase 4b: 6-9 GPU-hours (certification, conditional)
- Phase 5: 24-33 GPU-hours (3 models, conditional)
- **Total: 8.5-11.5 GPU-hours (through Phase 4b), 32.5-44.5 GPU-hours (full program)**

### Assumption
**All chains run in parallel (vectorized batching).** TFP HMC `current_state` shape `[num_chains, param_dim]` means all chains evaluate in one GPU batch per step. GPU time = num_steps × time_per_batch, NOT num_chains × num_steps.

### Expected Path
- Phase 4a diagnostic: 2 hours
- If passes → Phase 4b: 6-9 hours
- If Phase 4b certifies → Phase 5: 24-33 hours
- **Most likely total: 5 days + 32-44 GPU-hours**

---

## Evidence Contracts Per Phase

### Phase 2 (Toy Potential)
- **Question:** Does damped force work on analytical gradient?
- **Primary criterion:** W₂(exact, damped) < tolerance
- **Veto:** Acceptance < 0.15, divergence
- **Explanatory:** Runtime, ESS
- **Pass → Continue to Phase 3**
- **Fail → Mechanism broken, diagnose**

### Phase 4a (Diagnostic)
- **Question:** Is gross failure detectable cheaply?
- **Primary criterion:** W₂(exact, damped) compared to threshold (diagnostic only)
- **Veto:** None (diagnostic only)
- **Explanatory:** Acceptance, R-hat, ESS
- **Pass → Continue to Phase 4b**
- **Fail → Method broken, do not run expensive Phase 4b**

### Phase 4b (Certification)
- **Question:** Does Corollary 5.2 hold on LGSSM?
- **Primary criterion:** W₂(exact, damped) < tolerance (with MCMC error)
- **Veto:** Acceptance < 0.15, R-hat > 1.05, divergence
- **Explanatory:** True-θ coverage, ESS, runtime
- **Pass → Corollary 5.2 certified, nominate for Phase 5**
- **Fail → Surrogate-force HMC does not sample π_N^ω correctly**

### Phase 5 (Tier A)
- **Question:** Does certification generalize to other models?
- **Primary criterion:** All three models pass W₂ agreement
- **Veto:** Per-model acceptance, R-hat, divergence
- **Explanatory:** Per-model true-θ coverage, ESS, runtime
- **Pass → Strong evidence for general correctness**
- **Fail → Works on LGSSM only, or model-specific failure**

---

## Approval Summary

### Pre-Approved (No Further Approval Needed)
✅ **Option A defaults** (tolerance, seed policy, coverage)  
✅ **Phase 0-3** (CPU work, negligible GPU)  
✅ **Phase 4a** (2 GPU-hours diagnostic)  
✅ **Phase 4b** (6-9 GPU-hours, conditional on 4a pass)  
✅ **Phase 5** (24-33 GPU-hours, conditional on 4b certification)  
✅ **Total budget: 5-7 days CPU, 8-44 GPU-hours**

### Decision Gates (Automatic, No Approval)
- After Phase 4a: if W₂ > threshold → STOP (method broken)
- After Phase 4b: if fails → STOP (do not run Phase 5)
- After Phase 5: report results (end of program)

### No Approval Prompts During Execution
All phases, budgets, and decision gates are pre-approved. Execution proceeds from Phase 0 through completion without stopping for approval, unless a continuation veto fires (then stop and report).

---

## What This Program Certifies

### If Phase 4b Passes
✅ Surrogate-force HMC samples π_N^ω correctly (Corollary 5.2 holds)  
✅ Acceptance ≥ 0.15 (mixing adequate)  
✅ R-hat ≤ 1.05 (chains converged)  
✅ No divergences (integration stable)

### If Phase 5 Passes
✅ Certification generalizes to three Tier A models  
✅ Strong evidence for general correctness on this model class

### What It Does NOT Certify (Discovered by Execution)
❌ Finite-N bias (π_N^ω ≈ π_∞) — separate study needed  
❌ Posterior coverage of true θ — diagnostic only, not primary  
❌ "Better than baseline" — requires multi-seed statistical analysis  
❌ Efficiency claims — runtime and ESS are explanatory only

---

## Execution Authority

**This document is the execution authority.** No other program, plan, or pre-mortem governs this work. All prior documents are superseded.

**Start immediately with Phase 0** upon user confirmation.

---

## Confirmation

**User approved:**
- ✅ Option A defaults (tolerance, seed, coverage)
- ✅ Budget (5-7 days, 8-44 GPU-hours)
- ✅ Two-stage Phase 4 (diagnostic + certification)
- ✅ No approval prompts during execution

**Ready to execute: YES**

**Start with Phase 0 immediately.**
