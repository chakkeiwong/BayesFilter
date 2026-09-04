# LEDH Surrogate-Force HMC Master Program

**Program ID:** `ledh-surrogate-force-hmc-2026-09-04`  
**Date:** 2026-09-04  
**Status:** ACTIVE  
**Authority Document:** `LEDH_SURROGATE_HMC_PROGRAM_AUTHORITY_2026-09-04.md`

---

## SCIENTIFIC QUESTION

Can surrogate-force HMC with damped LEDH analytical score (λ=1e-3, δ=1e-3 for force, λ=1e-5, δ=1e-5 for acceptance) achieve:
1. **Correctness:** Valid sampling from the executed finite-particle pseudo-posterior
2. **Efficiency:** Acceptable ESS/gradient (≥0.5× exact-score baseline)
3. **Acceptance:** Reasonable acceptance rate (≥0.2)

on BayesFilter Tier A models?

---

## EXECUTIVE SUMMARY

**Background:** LEDH-OT dual-cap trust-region score has 3-9% bias (Phase 1-4 Aug-Sept 2026 concluded cannot improve). Direct HMC use is questionable.

**Insight:** Surrogate-force HMC (Corollary 5.2, variance note lines 954-976) decouples correctness from score bias. The chain targets exp(-U) exactly where U is the executed filter value. Score bias affects mixing only, not the invariant distribution.

**Requirements:**
1. Force F = deterministic function of θ only
2. All MC seeds frozen across trajectory
3. Same exact U(θ) evaluated at both trajectory ends for acceptance

**Infrastructure:** LEDH while-loop refactor complete (Sept 3), fixes 6× graph explosion. Requires policy compliance fix before execution.

---

## PROGRAM STRUCTURE

### Phase 0: Infrastructure Preparation (BLOCKING)

**Status:** IN_PROGRESS  
**Goal:** Make refactored LEDH kernel policy-compliant and merge to main

**Tasks:**
1. ✅ Create single authority document
2. ✅ Create master program
3. ⏳ Fix 6 `tf.vectorized_map` policy violations → `tf.while_loop`
4. ⏳ Test: 6 parity tests, graph size verification
5. ⏳ Merge to main (clean PR recommended)

**Success Criteria:**
- All 6 parity tests pass with rtol=5e-4
- Graph size < 10,000 nodes (vs 110,628 unrolled baseline)
- No `tf.vectorized_map` in `ledh_canonical_batch_fused_tf.py`
- Code on main branch

**Estimated Time:** 1 day

**Promotion Criterion:** All tasks complete, tests pass

**Veto Conditions:**
- Any parity test fails
- Graph size > 50,000 nodes (regression)
- Numerical differences > rtol=1e-3 vs original refactor

**Artifacts:**
- Updated `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`
- Test results: `docs/plans/ledh-surrogate-hmc-phase0-result-2026-09-04.md`
- Merge commit on main

---

### Phase 1: Route Identity and Wiring

**Status:** AWAITING_PHASE0  
**Goal:** Verify LEDH configuration and establish baseline metrics before surrogate-force

**Governing Plan:** `docs/plans/surrogate_force_hmc_three_phase_implementation_plan.md` §Phase 0

**Tasks:**
1. Verify canonical LEDH configuration (dtype=float64, seed policy, Sinkhorn params)
2. Analytical-JVP parity check (FD vs JVP residual < 1e-6)
3. Self-consistency diagnostics:
   - Sinkhorn marginal TV < 1e-3
   - Contract-E moment residual
   - Cholesky condition number baseline
   - Dual-cap convergence: iterations, floor hits
4. Document all baseline metrics

**Evidence Contract:**
- **Baseline:** None (diagnostic phase)
- **Primary Criterion:** All self-consistency checks pass
- **Veto:** Any check fails
- **Explanatory Only:** Timing, memory usage
- **Non-claim:** This phase establishes baselines only, not surrogate-force validation

**Success Criteria:**
- JVP parity: residual < 1e-6
- Sinkhorn marginal TV < 1e-3
- All diagnostics within historical ranges
- Configuration documented

**Estimated Time:** 1-3 days

**Promotion Criterion:** All diagnostics pass, baselines recorded

**Veto Conditions:**
- JVP parity fails (>1e-6 residual) - indicates implementation problem
- Sinkhorn marginal TV > 1e-2 - indicates transport failure
- Any diagnostic shows regression vs historical

**Continuation Veto:** None (diagnostics cannot invalidate surrogate-force idea)

**Artifacts:**
- Diagnostic results: `docs/plans/ledh-surrogate-hmc-phase1-result-2026-09-04.md`
- Baseline metrics table (JVP residuals, Sinkhorn TV, condition numbers, timing)

---

### Phase 2: Toy Potential Mechanics Check

**Status:** AWAITING_PHASE1  
**Goal:** Isolate surrogate-force mechanics from filter complexity using simple quadratic potential

**Governing Plan:** `docs/plans/surrogate_force_hmc_three_phase_implementation_plan.md` §Phase 1

**Fixture:**
- Simple quadratic: U(θ) = 0.5 θᵀ Σ⁻¹ θ
- Σ = diag([1, 4, 9]) (different scales for anisotropy test)
- True posterior: N(0, Σ)
- No particle filter involved

**Implementation:**
- `DualAdapterToy` class: exact value + damped force
- Damping scale: 0.1 (analogous to λ=1e-3 / λ=1e-5 = 100× ratio)

**Evidence Contract:**
- **Baseline:** Damping=1.0 (exact score as force)
- **Primary Criterion:** Damping=0.1 acceptance ≥ 0.2 (chain is mixing)
- **Promotion Veto:** Damping=0.1 acceptance < 0.2 (force too weak)
- **Explanatory Only:** ESS, force norm, trajectory energy conservation
- **Non-claim:** Toy potential success does not guarantee LEDH filter success

**Tests:**
1. **T1: Deterministic repeated calls** - same θ → same (value, force)
2. **T2: Endpoint energy equality** - H(start) ≈ H(end) up to leapfrog error
3. **T3: Acceptance across damping ladder** - [1.0, 0.5, 0.1]
4. **T4: Force-norm diagnostic** - ||F_damped|| < ||F_exact||

**Success Criteria:**
- All 4 tests pass
- Damping=1.0: acceptance 0.7-0.8
- Damping=0.5: acceptance 0.5-0.6
- Damping=0.1: acceptance ≥ 0.2

**Estimated Time:** 1 day

**Promotion Criterion:** All tests pass, acceptance ≥ 0.2 for damping=0.1

**Veto Conditions:**
- Damping=0.1 acceptance < 0.2 - force too weak, surrogate approach questionable
- T1 fails - determinism broken, violates Corollary 5.2
- T2 fails badly (|ΔH| > 1.0) - leapfrog integration problem

**Continuation Veto:** Promotion veto is also continuation veto (toy failure suggests LEDH will fail worse)

**Artifacts:**
- Implementation: `bayesfilter/inference/toy_surrogate_force_adapter.py`
- Test suite: `tests/inference/test_toy_surrogate_force.py`
- Results: `docs/plans/ledh-surrogate-hmc-phase2-result-2026-09-04.md`
- Acceptance table across damping ladder

---

### Phase 3: LEDH Filter Application

**Status:** AWAITING_PHASE2  
**Goal:** Test surrogate-force HMC with full LEDH filtering on d=3 T=50 LGSSM

**Governing Plan:** `docs/plans/implementation_plan_surrogate_force_hmc_2026-08-29.md` §Implementation Recipe

**Fixture:**
- Model: d=3 T=50 LGSSM
- True θ = [0.72, 0.55, 0.35, 0.35, 0.45]
- Same observation path as historical N-ladder and ε-schedule runs
- 4 chains × 2000 warmup + 2000 sampling steps

**Implementation:**
- `DualAdapterSurrogateForce` class
- Adapter A (exact): λ=1e-5, δ=1e-5
- Adapter B (damped): λ=1e-3, δ=1e-3
- Frozen noise: generate once per θ, share across both adapters
- Seed discipline: verify TFP HMC doesn't re-invoke `log_prob_and_grad` per leapfrog step

**Evidence Contract:**
- **Baseline:** Exact score as force (current HMC infrastructure, Arm 1)
- **Primary Criterion:** 3-arm descriptive comparison
- **Promotion Veto:** 
  - Damped-score (Arm 2) acceptance < 0.2
  - Posterior 95% intervals fail to cover true θ (>1 parameter)
- **Explanatory Only:** ESS/gradient, force norm, trajectory diagnostics
- **Statistical Evidence:** None (descriptive comparison only, small sample)
- **Non-claim:** 
  - Cannot claim "better" without multi-seed uncertainty analysis
  - Cannot claim "production-ready" without Tier A suite
  - Passing this phase nominates surrogate-force for wider validation only

**Comparison Arms:**
- **Arm 1:** Exact score as force (λ=1e-5, δ=1e-5 for both value and force)
- **Arm 2:** Damped score as force (λ=1e-3, δ=1e-3 for force, λ=1e-5, δ=1e-5 for value)
- **Arm 3:** Intermediate damping (λ=1e-4, δ=1e-4 for force) - run only if Arm 2 acceptance < 0.2

**Metrics:**
1. **Acceptance rate:** Warmup and sampling phases
   - Target: 0.6-0.8 warmup, 0.5-0.7 sampling
   - Floor: ≥0.2 (below = promotion veto)

2. **ESS/gradient:** Descriptive only
   - Baseline: Arm 1
   - Comparison: Arm 2 vs Arm 1
   - Acceptable: > 0.5× baseline (2× cost is fine)

3. **Posterior coverage:** Does chain hit true θ?
   - Compute 95% credible intervals per parameter
   - Check: do intervals cover true values?
   - Veto: >1 parameter fails coverage (value bias larger than expected)

4. **Force norm:** Monitoring only
   - Arm 2 should have smaller ||F|| than Arm 1
   - Watch for force collapse (<1e-3) or explosion (>1e3)

**Success Criteria:**
- Arm 2 acceptance ≥ 0.2
- Arm 2 posterior 95% intervals cover true θ (all 5 parameters)
- Arm 2 ESS/gradient > 0.5× Arm 1 (descriptive, not statistical)
- No divergences, no crashes, no NaN values

**Estimated Time:** 1-2 days (4 chains × 4K steps × 3 arms ≈ 3-6 hours wall time + analysis)

**Promotion Criterion:** Surrogate-force is viable candidate for Tier A validation (not "production-ready")

**Veto Conditions:**
- **Promotion Veto:** Arm 2 fails acceptance or coverage criteria (blocks Tier A expansion)
- **Continuation Veto:** Same as promotion veto (no point trying Tier A if d=3 T=50 fails)

**Repair Triggers:**
- If Arm 2 acceptance < 0.2 but ≥ 0.1: try Arm 3 (intermediate damping)
- If force norm < 1e-3: step size ε too large, try ε/2
- If force norm > 1e3: damping insufficient, try λ=1e-2, δ=1e-2

**Artifacts:**
- Implementation: `bayesfilter/inference/ledh_surrogate_force_adapter.py`
- Runner: `docs/benchmarks/run_ledh_surrogate_force_lgssm_d3t50_20260904.py`
- Results: `docs/plans/ledh-surrogate-hmc-phase3-result-2026-09-04.md`
- Output: `docs/benchmarks/artifacts/ledh_surrogate_force_lgssm_d3t50_20260904/`
- Decision table: decision, criteria status, next action, non-claims

---

### Phase 4: Tier A Suite Validation (CONDITIONAL)

**Status:** AWAITING_PHASE3_PROMOTION  
**Goal:** IF Phase 3 promotes surrogate-force, validate on full BayesFilter Tier A suite

**Contingent On:** Phase 3 promotion criterion met

**Models:**
1. LGSSM-KF (T=50, d=3)
2. Predator-Prey UKF (T=20, d=4)
3. Predator-Prey SGQF (T=20, d=4)
4. Austria SIR SGQF (T=20, d=5)
5. STR UKF (T=?, d=?)

**Evidence Contract:**
- **Baseline:** Exact score as force (current for each model)
- **Primary Criterion:** All 5 models pass acceptance and coverage
- **Promotion Criterion:** Surrogate-force becomes default-eligible (not automatic default)
- **Promotion Veto:** ≥2 models fail acceptance or coverage
- **Statistical Evidence:** Still descriptive (1 seed per model)
- **Non-claim:** Cannot claim "better than exact" without multi-seed uncertainty analysis

**Per-Model Criteria:**
- Acceptance ≥ 0.2
- Posterior 95% intervals cover true θ (all parameters)
- ESS/gradient > 0.3× baseline (relaxed from Phase 3)

**Success Criteria:**
- 5/5 models pass acceptance and coverage
- 4/5 models ESS/gradient > 0.5× baseline (descriptive)
- No evidence of value bias exceeding 0.1%

**Estimated Time:** 3-5 days (5 models × 4 chains × 4K steps ≈ 1-2 days wall time + analysis)

**Promotion Criterion:** Surrogate-force recommended for default consideration (requires separate tuning campaign)

**Veto Conditions:**
- **Promotion Veto:** ≥2 models fail (surrogate-force not robust enough)
- **Continuation Veto:** ≥3 models fail (approach fundamentally problematic)

**Artifacts:**
- Runner: `docs/benchmarks/run_ledh_surrogate_force_tier_a_suite_20260904.py`
- Results per model: `docs/plans/ledh-surrogate-hmc-phase4-{model}-result-2026-09-04.md`
- Output: `docs/benchmarks/artifacts/ledh_surrogate_force_tier_a_suite_20260904/`
- Summary table: model, acceptance, coverage, ESS/grad ratio, decision

---

## BUDGET

### Compute Budget
- **Phase 0:** Negligible (tests only)
- **Phase 1:** Negligible (diagnostics)
- **Phase 2:** 1 GPU-hour (toy potential, short chains)
- **Phase 3:** 10 GPU-hours (3 arms × 4 chains × 4K steps on d=3 T=50)
- **Phase 4:** 50 GPU-hours (5 models × 4 chains × 4K steps)
- **Total:** ~61 GPU-hours

### Attempt Budget
- **Phase 0:** 3 attempts (fix, test, merge)
- **Phase 1:** 2 attempts (diagnostics should be routine)
- **Phase 2:** 3 attempts (toy potential + up to 2 repairs)
- **Phase 3:** 5 attempts (3 arms + up to 2 tuning iterations)
- **Phase 4:** 10 attempts (5 models + up to 5 repairs)
- **Total:** 23 attempts

**Campaign stops if:**
- Attempt budget exhausted
- Continuation veto fires
- User suspends program

### Time Budget
- **Phase 0:** 1 day
- **Phase 1:** 1-3 days
- **Phase 2:** 1 day
- **Phase 3:** 1-2 days
- **Phase 4:** 3-5 days (conditional)
- **Total:** 7-12 days (assumes Phase 4 executes)

---

## DECISION GATES

### Gate 1: Phase 0 → Phase 1
- **Criterion:** All parity tests pass, code merged to main
- **Veto:** Any test failure, policy violations remain
- **Authority:** Automatic (no human decision needed)

### Gate 2: Phase 1 → Phase 2
- **Criterion:** All diagnostics pass, baselines recorded
- **Veto:** Any diagnostic shows regression
- **Authority:** Automatic (diagnostic only)

### Gate 3: Phase 2 → Phase 3
- **Criterion:** Toy potential acceptance ≥ 0.2 with damping=0.1
- **Veto:** Acceptance < 0.2 (continuation veto)
- **Authority:** Automatic if criterion met, user decision if veto fires

### Gate 4: Phase 3 → Phase 4
- **Criterion:** d=3 T=50 LGSSM passes acceptance and coverage
- **Veto:** Acceptance < 0.2 or coverage fails (continuation veto)
- **Authority:** User decision (Phase 4 is expensive, 50 GPU-hours)

**If Gate 4 veto fires:**
- Document why surrogate-force failed on d=3 T=50
- Classify failure: acceptance problem, coverage problem, or ESS problem
- Decide: pursue PaRIS (unbiased score) or accept current LEDH limitations

---

## SUCCESS DEFINITION

### Minimal Success (Phase 3 promotes)
- Surrogate-force viable on d=3 T=50 LGSSM
- Acceptance ≥ 0.2, coverage correct, ESS/grad > 0.5× baseline
- Nominated for wider validation

### Full Success (Phase 4 promotes)
- 5/5 Tier A models pass acceptance and coverage
- Surrogate-force recommended for default consideration
- Separate tuning campaign authorized

### Stretch Success (beyond this program)
- Multi-seed uncertainty analysis supports "better than exact"
- Production-ready with automatic damping selection
- Tier S (DSGE) validation passes

---

## FAILURE MODES AND RESPONSES

### F1: Phase 2 fails (toy potential acceptance < 0.2)
- **Diagnosis:** Damped force too weak to guide proposals
- **Response:** Stop program, document mechanics failure
- **Next:** Consider PaRIS or accept LEDH limitations
- **Authority:** User decision

### F2: Phase 3 fails acceptance (d=3 T=50 acceptance < 0.2)
- **Diagnosis:** LEDH filter complexity breaks surrogate-force
- **Repair:** Try Arm 3 (intermediate damping λ=1e-4, δ=1e-4)
- **If still fails:** Stop program, document filter incompatibility
- **Authority:** Automatic repair once, then user decision

### F3: Phase 3 fails coverage (posterior misses true θ)
- **Diagnosis:** Value bias larger than expected (>0.1%)
- **Repair:** None (this is a fundamental LEDH issue)
- **Response:** Stop program, re-evaluate LEDH value bias measurements
- **Authority:** User decision (may trigger new investigation)

### F4: Phase 4 fails (≥2 Tier A models fail)
- **Diagnosis:** Surrogate-force not robust across models
- **Response:** Document which models failed and why
- **Next:** Classify by failure mode, consider model-specific tuning
- **Authority:** User decision

### F5: Infrastructure failure (graph explosion, OOM, crash)
- **Diagnosis:** While-loop refactor didn't fix the problem
- **Response:** Debug, repair, re-run from current phase
- **Does not consume attempt budget:** Infrastructure failures are out-of-scope
- **Authority:** Automatic repair

---

## NON-CLAIMS AND BOUNDARIES

### What This Program Tests
- Viability of surrogate-force HMC with damped LEDH score
- Acceptance rate with 100× damping ratio
- Posterior coverage with exact value, damped force
- Tier A robustness (if Phase 4 executes)

### What This Program Does NOT Test
- Statistical superiority over exact-score HMC (requires multi-seed analysis)
- Production performance at scale (requires tuning campaign)
- Tier S (DSGE) applicability (out of scope)
- Optimal damping ratio (100× is fixed, not tuned)

### What This Program Does NOT Fix
- Underlying 3-9% LEDH score bias (still present)
- LEDH score variance (still SD ≈ 0.3)
- LEDH value bias 0.01-0.09% (chain targets this)

### Compatibility
- Works with current LEDH-OT dual-cap trust-region
- Compatible with Contract-E streaming transport
- Independent of NeuTra (no chart learning needed)
- Does NOT require PaRIS (but PaRIS would make this unnecessary)

---

## ARTIFACTS AND DELIVERABLES

### Phase 0
- Updated `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`
- Test results: `docs/plans/ledh-surrogate-hmc-phase0-result-2026-09-04.md`
- Merge commit on main

### Phase 1
- Diagnostic results: `docs/plans/ledh-surrogate-hmc-phase1-result-2026-09-04.md`
- Baseline metrics table

### Phase 2
- Implementation: `bayesfilter/inference/toy_surrogate_force_adapter.py`
- Tests: `tests/inference/test_toy_surrogate_force.py`
- Results: `docs/plans/ledh-surrogate-hmc-phase2-result-2026-09-04.md`
- Acceptance table

### Phase 3
- Implementation: `bayesfilter/inference/ledh_surrogate_force_adapter.py`
- Runner: `docs/benchmarks/run_ledh_surrogate_force_lgssm_d3t50_20260904.py`
- Results: `docs/plans/ledh-surrogate-hmc-phase3-result-2026-09-04.md`
- Output: `docs/benchmarks/artifacts/ledh_surrogate_force_lgssm_d3t50_20260904/`
- Decision table

### Phase 4 (conditional)
- Runner: `docs/benchmarks/run_ledh_surrogate_force_tier_a_suite_20260904.py`
- Per-model results: `docs/plans/ledh-surrogate-hmc-phase4-{model}-result-2026-09-04.md`
- Output: `docs/benchmarks/artifacts/ledh_surrogate_force_tier_a_suite_20260904/`
- Summary table

### Program-Level
- This master program
- Authority document: `LEDH_SURROGATE_HMC_PROGRAM_AUTHORITY_2026-09-04.md`
- Final program result: `docs/plans/ledh-surrogate-hmc-program-result-2026-09-04.md` (at completion)

---

## GOVERNANCE

### Authority Chain
1. **This master program:** Scientific and execution authority
2. **Authority document:** Context and historical record
3. **CLAUDE.md + AGENTS.md:** Engineering and policy authority
4. **User (chakwong):** Final decision authority on gates, vetoes, repairs

### Update Protocol
- Update this document after each phase completion
- Mark phase status: IN_PROGRESS → COMPLETE / FAILED / SKIPPED
- Update budget consumed (attempts, compute, time)
- Update decision gates with actual outcomes

### Emergency Suspension
- User can suspend program at any time
- Document reason for suspension
- Mark status: SUSPENDED
- Preserve all artifacts and state for potential resume

---

**END OF MASTER PROGRAM**

Program Status: ACTIVE  
Current Phase: Phase 0 - ✅ **COMPLETE** (2026-09-04)  
Next Milestone: Merge to main, then Phase 1 (Route Identity and Wiring)  
Blocking: User decision on merge strategy  
Last Updated: 2026-09-04 (Phase 0 completion)
