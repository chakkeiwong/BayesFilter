> **SUPERSEDED 2026-09-06** by
> `docs/plans/ledh-surrogate-hmc-unified-program-2026-09-06.md`.
> Retained as historical record. Its Phase 3 lane assumption and its
> "audit resolved" reading of the 2026-08-29 score audit were both found
> wrong by the 2026-09-06 pre-execution audit. Do not execute from this file.

# LEDH Surrogate-Force HMC Program — Single Authority Document

**Date Created:** 2026-09-04  
**Last Updated:** 2026-09-04 (Phase 0 complete)  
**Status:** PHASE0_COMPLETE_AWAITING_MERGE  
**Current Branch:** `ledh-refactor-with-policy-fix`  
**Purpose:** Single authoritative document describing the complete state of the LEDH surrogate-force HMC program

---

## EXECUTIVE SUMMARY

**THE GOAL:**
Run surrogate-force HMC with LEDH filtering posteriors, using damped analytical score (λ=1e-3, δ=1e-3) for leapfrog force and exact value (λ=1e-5, δ=1e-5) for acceptance. This decouples correctness from score bias: 3-9% score bias affects mixing only, not the invariant distribution.

**THE PROBLEM:**
LEDH trust-region dual-cap score accuracy cannot be improved sufficiently (concluded after extensive Phase 1-4 work, Aug-Sept 2026). Score bias makes direct HMC usage questionable. Additionally, surrogate-force HMC has 6× graph explosion (663K nodes, 101.8s trace time) due to Python-unrolled loops.

**THE SOLUTION:**
1. ✅ **COMPLETE:** LEDH while-loop refactor (Sept 3) - fixes graph explosion
2. ✅ **COMPLETE:** Fixed 6 `tf.vectorized_map` policy violations (Sept 4, commit 38a5631a)
3. ⏳ **NEXT:** Merge to main → execute surrogate HMC Phase 1-4

**CURRENT STATE:**
- On branch `ledh-refactor-with-policy-fix` (commit 38a5631a)
- Refactored code complete and policy-compliant
- All 6 parity tests pass
- Phase 0 complete, awaiting merge to main
- Two master programs ready for surrogate HMC execution

---

## THE CHRONOLOGY: WHAT LED US HERE

### August-September 2026: Trust-Region Work

**Dates:** Aug-Sept 2026 (Phase 1-4)  
**Goal:** Improve LEDH-OT dual-cap trust-region score accuracy  
**Outcome:** ❌ **FAILED** - Cannot improve accuracy sufficiently  
**Evidence:**
- Phase 3 complete memo: `docs/memos/ledh-trust-region-phase3-complete-2026-09-02.md`
- Phase 4 complete memo: `docs/memos/ledh-trust-region-phase4-complete-2026-09-03.md`
- All 4 models (Austria SIR, LGSSM, KSC SV, Predator-Prey) selected identical minimal trust-region hyperparameters
- Score bias remains 3-9%, unacceptable for direct HMC

**Decision:** Abandon trust-region accuracy improvement, pivot to surrogate-force HMC

### August 29, 2026: Surrogate-Force HMC Plan

**Document:** `docs/plans/implementation_plan_surrogate_force_hmc_2026-08-29.md`  
**Status:** Implementation-ready recipe  
**Key Insight:** Score bias affects **mixing**, not **correctness** if:
1. Force F is deterministic function of θ only
2. All MC seeds frozen across trajectory  
3. Same exact U(θ) for acceptance at both ends

**Mathematical Foundation:** Variance note §5 Corollary 5.2 (lines 954-976) + Maskell §6.1 CRN determinism

### August 30, 2026: Three-Phase Validation Protocol

**Document:** `docs/plans/surrogate_force_hmc_three_phase_implementation_plan.md`  
**Phases:**
- **Phase 0:** Route identity and wiring repair (prerequisite)
- **Phase 1:** Deterministic mechanics check (toy potential)
- **Phase 2:** LEDH filter application (full complexity)

### August 30, 2026: Graph Explosion Diagnosed

**Discovery:** Surrogate-force HMC sweeps P gradient directions → 6× graph explosion  
- 1 forward + 5 swept directions = 6 LEDH calls per HMC gradient
- Each call produces 110,628 nodes (horizon=50)
- Total: 663,766 nodes, 61.24 MB GraphDef, 101.8s trace time

**Root Cause:** Python `for time_index in range(horizon)` unrolling → independent subgraph per timestep

**Solution:** LEDH while-loop refactor master program created

### September 3, 2026: While-Loop Refactor COMPLETE

**Governing Document:** `docs/plans/ledh-while-loop-refactor-master-program-2026-08-30.md`  
**Result Document:** `docs/plans/ledh-while-loop-refactor-program-result-2026-09-03.md`  
**Status:** ✅ **COMPLETE** - All 4 phases executed, all 6 parity tests pass

**What Was Delivered:**
- Bounded `tf.while_loop` for horizon and substep loops
- Multi-direction tangent support (K directions in one call)
- Form (c) architecture: K tangents share one primal
- Target: ~2,200 node bounded body vs 110,628 nodes unrolled

**Critical Issue:** Uses `tf.vectorized_map` in 6 places (lines 316, 334, 442, 490, 502, 615)

**Policy Violation:** CLAUDE.md requires prior written approval for `tf.vectorized_map` (implicit pfor). No approval exists.

### September 4, 2026: Context Lost, Governance Failure

**Problem:** No single authority document. Work spread across:
- Multiple master programs (HNN July - wrong program, LEDH refactor Aug-Sept, surrogate plans Aug)
- Refactor complete but on unmerged branch
- 87 commits of other work mixed in the same lineage
- Main branch still has the original problem

**This Document Created:** To serve as single source of truth

---

## CURRENT BRANCH STATE

### Branch: `ledh-refactor-with-policy-fix`

**Created From:** Commit 5752bb10 "docs: Add preliminary performance and XLA audit"  
**Parent Chain:** 5752bb10 ← 3ed4c86f ← f8a19e42 (the refactor) ← 2478d3da (tagged as `archive/ledh-canonical-rebuild-2026-09-01`)

**Current Working Directory:** `/home/chakwong/BayesFilter`

**Git Status:**
```
On branch ledh-refactor-with-policy-fix
M	bayesfilter/highdim/__init__.py
```

### File: `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`

**Status:** Refactored, policy violations present

**Uses of `tf.vectorized_map` (6 occurrences):**
- Line 316: `d_predicted_means, d_predicted_covs = tf.vectorized_map(...)`
- Line 334: `d_anchors, d_pre_flow = tf.vectorized_map(...)`  
- Line 442: `new_d_actual, new_d_auxiliary, d_ldet_inc = tf.vectorized_map(...)`
- Line 490: `d_observation_log = tf.vectorized_map(...)`
- Line 502: `d_observed = tf.vectorized_map(...)`
- Line 615: `new_d_covariances = tf.vectorized_map(...)`

**Uses of `tf.while_loop` (2 occurrences):**
- Line 450: Substep loop
- Line 623: Horizon loop

**Policy Requirement:**
```
TensorFlow pfor requires prior written approval. This includes direct pfor
calls and implicit pfor selected by APIs such as tf.vectorized_map,
GradientTape.jacobian, or GradientTape.batch_jacobian. The approval must
explain why tf.while_loop or another native TensorFlow loop with a single
traced body cannot satisfy the mathematical and engineering contract.
```

**No such approval exists.**

### File: `bayesfilter/highdim/ledh_contract_e_streaming_tf.py`

**Status:** Uses `tf.map_fn` (policy-compliant on this branch)

**Lines 1255-1262, 1341-1348:**
```python
tf.map_fn(
    one_direction,
    tf.range(parameter_count),
    fn_output_signature=output_signature,
    parallel_iterations=1,  # Serial execution
)
```

**Note:** Main branch (NOT this branch) has `tf.vectorized_map` in Contract E, but this branch uses `tf.map_fn` which is fine.

---

## ARTIFACTS AND EVIDENCE

### Completed Work (On This Branch)

1. **LEDH While-Loop Refactor**
   - Master program: `docs/plans/ledh-while-loop-refactor-master-program-2026-08-30.md`
   - Phase results: `docs/plans/ledh-while-loop-refactor-phase{1,2,3,4}-result-2026-09-03.md`
   - Program result: `docs/plans/ledh-while-loop-refactor-program-result-2026-09-03.md`
   - Reset memo: `docs/plans/ledh-while-loop-refactor-reset-memo-2026-09-03.md`
   - Tests: `tests/highdim/test_ledh_canonical_batch_fused.py` (6 tests, all pass)

2. **Performance Audit**
   - XLA audit: `docs/plans/ledh-while-loop-refactor-phase3-result-2026-09-03.md`
   - Graph size measurements in master program

3. **Codex Review**
   - Review memo: `docs/plans/ledh-while-loop-refactor-codex-review-2026-09-01.md`
   - Verdict: `APPROVE_WITH_CORRECTIONS`
   - Found 20 defects (D1-D20) in the original plan
   - Work was completed on branch despite defects

### Trust-Region Work (Main Branch)

1. **Phase 3 Complete:** Austria SIR T20
   - Memo: `docs/memos/ledh-trust-region-phase3-complete-2026-09-02.md`
   - Artifact: `docs/benchmarks/artifacts/ledh_trust_region_austria_sir_t20_20260902/`

2. **Phase 4 Complete:** LGSSM T50, KSC SV T10, Predator-Prey T20
   - Memo: `docs/memos/ledh-trust-region-phase4-complete-2026-09-03.md`
   - Artifacts: `docs/benchmarks/artifacts/ledh_trust_region_{lgssm_t50,ksc_sv_t10,predator_prey_t20}_20260903/`

3. **Conclusion:** All 4 models selected identical minimal-intervention configuration (damping=0.001, scale_floor=1e-06, radius=0.1). Score accuracy not improved enough.

### Surrogate-Force HMC Plans (Ready to Execute)

1. **Implementation Recipe:** `docs/plans/implementation_plan_surrogate_force_hmc_2026-08-29.md`
   - Status: Implementation-ready, awaiting go decision
   - Effort: 1-2 days
   - Dual-adapter structure defined
   - Verification protocol specified

2. **Three-Phase Validation:** `docs/plans/surrogate_force_hmc_three_phase_implementation_plan.md`
   - Status: Implementation-ready, reviewed against Codex feedback
   - Effort: 3-5 days total
   - Phase 0: Route identity and wiring (1-3 days)
   - Phase 1: Toy potential mechanics (1 day)
   - Phase 2: LEDH filter application (1 day)

---

## THE BLOCKING ISSUE: POLICY VIOLATIONS

### What CLAUDE.md Requires

**From `CLAUDE.md` TensorFlow Graph and Compilation Policy section:**

> TensorFlow pfor requires prior written approval. This includes direct pfor
> calls and implicit pfor selected by APIs such as `tf.vectorized_map`,
> `GradientTape.jacobian`, or `GradientTape.batch_jacobian`. The approval must
> explain why `tf.while_loop` or another native TensorFlow loop with a single
> traced body cannot satisfy the mathematical and engineering contract; state
> expected graph, host-memory, device-memory, and compilation complexity; and
> define bounded compatibility, numerical-equivalence, memory, and runtime
> checks.

### Why This Matters

`tf.vectorized_map` uses pfor (parallel-for) internally. It auto-vectorizes a scalar function by:
1. Creating complex graph transformations
2. Parallelizing independent loop iterations
3. Potentially exploding graph size for complex operations

The policy requires either:
- **Option A:** Prior written approval explaining why `tf.while_loop` won't work
- **Option B:** Replace `tf.vectorized_map` with `tf.while_loop`

### Why the Refactor Used `tf.vectorized_map`

**From `ledh-while-loop-refactor-program-result-2026-09-03.md` §"Implementation Deviations":**

> **Plan specified:** Python `for k in range(K)` loops for tangent evaluation
> 
> **Implemented:** `tf.vectorized_map` over K dimension
> 
> **Rationale:**
> - Python loops over tensors are not graph-compilable
> - `tf.vectorized_map` produces a single traced body
> - Graph compilation is a Phase 1 requirement
> - Implementation remains batch-native and NeuTra-eligible
> 
> **Trade-off:** `tf.vectorized_map` may allocate intermediate K copies internally, but this is TensorFlow's standard pattern for vectorization and maintains graph semantics.

**The problem:** This rationale does NOT explain why `tf.while_loop` cannot work. It only explains why Python loops don't work.

### The Correct Fix: Use `tf.while_loop`

`tf.while_loop` is the policy-preferred native TensorFlow loop. It:
- Traces a single body
- Is graph-compilable
- Avoids pfor complexity
- Is explicitly approved in CLAUDE.md

**Pattern:**
```python
# CURRENT (policy violation):
reset_particles = tf.transpose(
    tf.vectorized_map(one_direction, tf.range(parameter_count)),
    [1, 2, 3, 0],
)

# CORRECT (policy-compliant):
def loop_body(k, accumulated):
    particles = one_direction(k)
    return k + 1, accumulated.write(k, particles)

_, result_array = tf.while_loop(
    lambda k, _: k < parameter_count,
    loop_body,
    (0, tf.TensorArray(dtype=source_particles.dtype, size=parameter_count))
)
reset_particles = tf.transpose(result_array.stack(), [1, 2, 3, 0])
```

---

## NEXT STEPS: THE EXECUTION PLAN

### Step 1: Fix Policy Violations (THIS TASK)

**What:** Replace 6 occurrences of `tf.vectorized_map` with `tf.while_loop` in `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`

**How:**
- Each `tf.vectorized_map` call loops over K directions
- Replace with `tf.while_loop` using `tf.TensorArray` to accumulate results
- Closure capture remains the same
- Mathematical semantics unchanged

**Verification:**
- Run all 6 parity tests: `pytest tests/highdim/test_ledh_canonical_batch_fused.py -v`
- All tests must pass with same rtol=5e-4 tolerance
- No numerical differences expected (purely engineering change)

**Estimated Time:** 2-3 hours

### Step 2: Test Thoroughly

**Tests to Run:**
1. **Parity tests:** 6 tests in `test_ledh_canonical_batch_fused.py`
2. **Score suite integration:** Tests that depend on the refactored kernel
3. **Graph size verification:** Measure nodes, should remain ~2,200 vs 110,628
4. **Compilation check:** Verify `@tf.function` with `input_signature` still works

**Success Criteria:**
- All parity tests pass
- Graph size < 10,000 nodes (vs 110,628 unrolled)
- No new test failures introduced
- Batch-native property preserved

**Estimated Time:** 1-2 hours

### Step 3: Merge to Main

**Prerequisites:**
- Step 1 complete (policy violations fixed)
- Step 2 complete (all tests pass)
- Clean git status (no uncommitted changes)

**Merge Strategy:**
We need to decide:
- **Option A:** Cherry-pick only the refactor commits (clean but loses some context)
- **Option B:** Merge the entire branch (brings 87 commits of other work)
- **Option C:** Create fresh PR with only refactor changes (safest)

**Recommendation:** Option C - create clean PR to main with:
- Only `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` changes
- Only `tests/highdim/test_ledh_canonical_batch_fused.py` changes
- Updated documentation
- This authority document

**Estimated Time:** 1 hour (if Option C)

### Step 4: Execute Surrogate-Force HMC Program

**Once merged to main:**

**Phase 0: Route Identity and Wiring (1-3 days)**
- Verify canonical LEDH configuration
- Analytical-JVP parity check
- Self-consistency diagnostics
- Document baseline metrics
- See: `docs/plans/surrogate_force_hmc_three_phase_implementation_plan.md` §Phase 0

**Phase 1: Toy Potential Mechanics (1 day)**
- Simple quadratic potential U(θ) = 0.5 θᵀ Σ⁻¹ θ
- Test deterministic mechanics
- Acceptance across damping ladder
- See: `docs/plans/surrogate_force_hmc_three_phase_implementation_plan.md` §Phase 1

**Phase 2: LEDH Filter Application (1 day)**
- Implement `DualAdapterSurrogateForce`
- Exact value (λ=1e-5, δ=1e-5) + damped score (λ=1e-3, δ=1e-3)
- Verify on d=3 T=50 LGSSM
- 3-arm comparison: exact / damped / intermediate
- See: `docs/plans/implementation_plan_surrogate_force_hmc_2026-08-29.md` §Implementation Recipe

**Total Estimated Time:** 5-7 days from Step 4 start

---

## WHAT THIS DOCUMENT IS NOT

**This is NOT:**
- A proposal (the work is already done, just needs policy fix)
- A request for approval (the while-loop refactor was already approved and executed)
- A design document (design was done in Aug 2026)

**This IS:**
- The single source of truth for current state
- The record of how we got here
- The specification of what happens next
- The document to read FIRST when context is lost

---

## HISTORICAL PROGRAMS (NOT GOVERNING)

### July 17, 2026: HNN Surrogate-Force HMC (COMPLETE, DIFFERENT PROGRAM)

**Document:** `docs/plans/bayesfilter-hnn-surrogate-hmc-master-program-2026-07-17.md`  
**Status:** `COMPLETE`  
**Scope:** Corrected neural-force HMC (learned residual potential)  
**NOT THE CURRENT PROGRAM:** This used learned forces, not analytical LEDH scores

**Why it's not relevant:**
- Used neural network to learn surrogate force
- Required NeuTra charts and training
- Different mathematical foundation (Chapter 48 construction)
- 5/5 Tier A cells passed validity, but only 1/5 has complete performance ledger
- This is a different research direction

### RQMC Work (September 2026, PARALLEL TRACK)

**Documents:**
- `docs/plans/rqmc-ledh-initialization-master-program-2026-09-02.md`
- `docs/memos/rqmc-ledh-phase2-completion-2026-09-04.md`

**Scope:** Quasi-Monte Carlo initialization methods for LEDH

**Why it's not relevant:**
- Different research question (initialization, not HMC)
- Parallel track to surrogate-force HMC
- Does not block or depend on the current program

---

## GOVERNANCE LESSONS

### What Went Wrong

1. **No single authority:** Work spread across multiple docs with no unified view
2. **Branch never merged:** Completed work sat isolated for months
3. **Context scattered:** Had to read 15+ documents to understand current state
4. **Naming confusion:** Multiple "surrogate HMC" programs (HNN vs LEDH)
5. **No handoff memo:** Agent context compaction causes complete re-discovery

### What This Document Fixes

**Rule: One Authority Document per Active Program**

This document is now THE authority for LEDH surrogate-force HMC. Any agent (or human) should:
1. Read this document FIRST
2. All other documents are supporting evidence
3. This document gets updated as state changes
4. Never start work without confirming this document is current

### Maintenance Protocol

**When to Update This Document:**
- [ ] After Step 1 complete (policy violations fixed)
- [ ] After Step 2 complete (tests pass)
- [ ] After Step 3 complete (merged to main)
- [ ] After each Phase of surrogate HMC execution
- [ ] If blocking issues arise
- [ ] If program is suspended or cancelled

**Who Updates:** The agent or person doing the work

**Location:** Always `docs/plans/LEDH_SURROGATE_HMC_PROGRAM_AUTHORITY_2026-09-04.md`

---

## CONTACT POINTS

**Owner:** chakwong (user)

**Current Executor:** Claude Code agent (this session)

**Blocking Questions:** Stop and ask user, do not infer or proceed blindly

**Emergency Context Recovery:** Read this document, then ask user for clarification

---

**END OF SINGLE AUTHORITY DOCUMENT**

Last Updated: 2026-09-04 (document creation)
