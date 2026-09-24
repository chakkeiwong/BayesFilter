# LEDH Surrogate-Force HMC — Procedurally Correct Master Program

**Program ID:** `ledh-surrogate-force-hmc-procedurally-correct-2026-09-06`  
**Date:** 2026-09-06  
**Status:** DRAFT — requires user approval of 3 decisions before execution  
**Supersedes:** `ledh-surrogate-hmc-unified-program-2026-09-06.md`  
**Authority:** Procedural correctness review 2026-09-06

---

## Procedural vs Scientific Distinction

**Procedural correctness (this program's responsibility):**
- The experiment can certify success OR detect failure
- Either result is interpretable
- Pass/fail criteria measure what the governing theorem proves

**Scientific outcome (discovered by execution, not controlled by planning):**
- Whether the method works
- Whether acceptance is acceptable
- Whether ESS/gradient is competitive

This program guarantees procedural correctness. Scientific outcomes are unknown until run.

---

## Scientific Question

Can surrogate-force HMC with damped LEDH analytical score correctly sample the finite-N pseudo-posterior π_N^ω (Corollary 5.2), with acceptable mixing (acceptance ≥ 0.15)?

**What this question IS:**
- Does the sampler leave π_N^ω(θ) ∝ exp(-U_N^ω(θ)) invariant?
- If yes, is the acceptance rate high enough for practical use?

**What this question IS NOT:**
- Does π_N^ω ≈ π_∞? (finite-N bias — separate question, see Non-Claims)
- Does the posterior cover true θ? (data property + finite-N bias, not sampler correctness)
- Is the method "better" than exact-force? (requires multi-seed statistical comparison, out of scope)

---

## Governing Theorem

**Corollary 5.2** (variance note lines 954-976):

> Consider the algorithm: (i) refresh p ~ N(0,M); (ii) propose S(θ,p) with S = F ∘ Ψ^L built from **any deterministic momentum-independent force F**; (iii) accept with probability min{1, exp(H(θ,p) - H(S(θ,p)))} where H uses the exact executed potential U = -L̂^N. Every step leaves π(θ,p) ∝ exp(-H(θ,p)) invariant, hence the θ-marginal ∝ exp(-U(θ)) is preserved, **for any quality of F**. The choice of F affects only the acceptance probability and mixing.

**Remark 5.3** (line 980):

> F must be a deterministic function of θ alone: all Monte Carlo seeds inside F must be frozen, and F may not depend on momentum or on the trajectory's history.

**What the theorem certifies:** Surrogate-force HMC with **any** deterministic F samples π_N^ω correctly. Score quality affects mixing only, not invariance.

**What the theorem requires:** F(θ) deterministic, all seeds frozen, F history-independent.

**Target distribution:** π_N^ω(θ) ∝ exp(-U_N^ω(θ)), where:
- U_N^ω = executed LEDH filter log-likelihood
- N = 1008 particles (finite, not infinite)
- ω = one frozen noise draw (not ω-averaged)

**NOT governed by this theorem:** Whether π_N^ω ≈ π_∞, whether π_∞ covers true θ, whether mixing is fast.

---

## USER DECISIONS REQUIRED BEFORE EXECUTION

Three decisions block all work. Program cannot proceed without them.

### Decision 1: Tolerance Derivation (required for Phase 2B, Phase 3)

**Question:** Current program uses two incompatible tolerance regimes (golden master 1e-12, parity 5e-4). Derive one tolerance for all checks before generating fixtures or running experiments.

**Proposed derivation:**
```
κ = condition_number(reset_Jacobian)  # from existing LGSSM T=50 tuning artifact
ε_backend = 1e-4  # float32 TF32 machine epsilon
ε_amplified = κ × ε_backend
ε_tuning = tuning_insensitivity_band()  # from trust-region tuning
tol_derived = max(ε_amplified, ε_tuning)
```

**Also required:** Production regime choice:
- **Option A (RECOMMENDED):** Phase 4 runs in float32 TF32 on GPU (actual production target), accept that float64-CPU tests don't cover production dtype
- **Option B:** Phase 4 runs in float64 on GPU (tested regime), accept ~30% slowdown

**Default if no response:** Option A (production regime), condition-number-derived tolerance.

**Estimated time:** 0.3 day (derivation note).

---

### Decision 2: Seed Policy (required for Corollary 5.2 applicability)

**Question:** Corollary 5.2 Remark requires "all Monte Carlo seeds inside F must be frozen." Approve the one-seed policy and verification tests?

**Proposed policy:**
- One trajectory-level master seed ω
- ω frozen for entire HMC chain
- ω may depend on initial θ₀ (part of context), NOT on current θ during leapfrog
- No reseeding per step, per trajectory, or per adapter call

**Proposed verification (Phase 3 Task 3.3):**
- T3.3a: Bitwise-identical repeat at same (θ, ω)
- T3.3b: Involution test (leapfrog reversibility)
- T3.3c: Continuity test (no O(1) jumps at nearby θ under same ω)

**Default if no response:** Approve as written (determined by Remark 5.3, not negotiable).

**Estimated time:** 0.5 day (instrumentation + 3 tests).

---

### Decision 3: Coverage Criterion Math (if coverage remains a diagnostic)

**Question:** Current "95% intervals cover true θ (all 5 parameters)" has P(pass | correct sampler) = 0.95^5 ≈ 0.774. Which coverage definition for the explanatory diagnostic?

**Options:**

**A. Joint 95% Mahalanobis region (RECOMMENDED)**
```
T² = (θ_post - θ_true)ᵀ Σ_post⁻¹ (θ_post - θ_true)
Accept if T² < χ²₀.₉₅(5)
```
- Exactly 95% coverage for correct multivariate sampler

**B. Marginal with expected-failures**
```
Accept if ≥4 of 5 marginals cover
```
- Easier to explain

**C. Bonferroni-corrected marginals**
```
Use 99% per parameter
```
- Most conservative

**Default if no response:** Option A (joint Mahalanobis).

**Estimated time:** 0.1 day (one function).

**NOTE:** Coverage is an **explanatory diagnostic only** after the primary criterion fix. This decision affects reporting, not promotion/rejection.

---

## Program Structure

### Phase 0: Planning and Tolerance Derivation

**Status:** AWAITING USER DECISIONS  
**Goal:** Resolve three blocking decisions, derive tolerance, document seed policy

**Tasks:**

**Task 0.1: User decision collection**
- Present Decision 1 (tolerance), Decision 2 (seed policy), Decision 3 (coverage)
- Document choices in program
- Proceed only when all three decided

**Task 0.2: Tolerance derivation** (requires Decision 1)
- Obtain κ from existing LGSSM T=50 trust-region tuning artifact
- Obtain tuning-insensitivity band from same artifact
- Compute tol = max(κ × 1e-4, tuning_band)
- Document derivation and production regime choice

**Task 0.3: Seed policy documentation** (requires Decision 2)
- Document one-seed policy: master ω, frozen for chain, θ₀-dependent only
- List all seed sources in LEDH value/force paths
- Specify verification tests (T3.3a, T3.3b, T3.3c)

**Task 0.4: Coverage criterion implementation** (requires Decision 3)
- Implement chosen coverage formula (diagnostic only, not primary criterion)
- Test on synthetic LGSSM draw

**Success Criteria:**
- All three decisions documented
- Tolerance derived and recorded
- Seed policy specified with test definitions
- Coverage diagnostic implemented

**Estimated Time:** 1 day (0.3 + 0.5 + 0.1 + 0.1)

**Deliverable:**
- `docs/plans/ledh-surrogate-hmc-phase0-decisions-2026-09-06.md`
- Tolerance constant in program
- Seed policy in Phase 3 Task 3.3
- Coverage function ready for Phase 4

---

### Phase 1: Route Identity and Baseline Metrics

**Status:** AWAITING_PHASE0  
**Goal:** Verify LEDH configuration, establish baseline metrics

**Tasks:**

**Task 1.1: Configuration verification**
- Canonical LEDH setup: Contract-E, dual-cap trust-region, N=1008
- Dtype: float64 (Phase 1) or float32 TF32 (if Decision 1 chose production regime)
- Sinkhorn parameters: documented per tuning artifact
- Trust-region: documented per tuning artifact

**Task 1.2: Analytical-JVP parity check**
- FD vs JVP residual < derived tolerance (from Phase 0 Task 0.2)
- 10 θ points in posterior support
- Same test as historical Phase 1, now with principled tolerance

**Task 1.3: Self-consistency diagnostics**
- Sinkhorn marginal TV < 1e-3
- Contract-E moment residual
- Cholesky condition number
- Dual-cap convergence: iterations, floor hits

**Success Criteria:**
- JVP parity: residual < derived tolerance
- Sinkhorn marginal TV < 1e-3
- All diagnostics within historical ranges

**Veto Conditions:**
- JVP parity fails (> derived tolerance) — implementation problem
- Sinkhorn marginal TV > 1e-2 — transport failure

**Estimated Time:** 1 day

**Deliverable:**
- `docs/plans/ledh-surrogate-hmc-phase1-result-2026-09-06.md`
- Baseline metrics table

---

### Phase 2: Toy Potential Mechanics Check

**Status:** AWAITING_PHASE1  
**Goal:** Validate surrogate-force mechanism on simple quadratic, independent of filter complexity

**Fixture:**
- U(θ) = 0.5 θᵀ Σ⁻¹ θ
- Σ = diag([1, 4, 9])
- True posterior: N(0, Σ)
- No particle filter

**Implementation:**
- `DualAdapterToy`: exact value + damped force
- Damping scale: 0.1 (analogous to 100× λ ratio)

**Tests:**

**T2.1: Deterministic repeat (Corollary 5.2 premise)**
```python
(u1, f1) = adapter(θ, seed=ω)
(u2, f2) = adapter(θ, seed=ω)
assert u1 == u2 and f1 == f2  # bitwise
```

**T2.2: Endpoint energy equality**
```python
H_start = U(θ) + 0.5 pᵀ M⁻¹ p
(θ_end, p_end) = leapfrog(θ, p, L, ε)
H_end = U(θ_end) + 0.5 p_endᵀ M⁻¹ p_end
assert |H_end - H_start| < leapfrog_error_bound
```

**T2.3: Acceptance across damping ladder**
- Run HMC at damping = [1.0, 0.5, 0.1]
- 4 chains × 500 warmup × 500 sampling
- Measure acceptance per damping level

**T2.4: Force-norm diagnostic**
- ||F_damped|| < ||F_exact||
- Watch for collapse (< 1e-3) or explosion (> 1e3)

**Evidence Contract:**
- **Baseline:** Damping=1.0 (exact score as force)
- **Primary Criterion:** Damping=0.1 acceptance ≥ 0.2
- **Promotion Veto:** Acceptance < 0.2 (force too weak)
- **Continuation Veto:** Same as promotion veto (toy failure suggests LEDH will fail worse)
- **Explanatory:** ESS, force norm, trajectory energy

**Success Criteria:**
- All 4 tests pass
- Damping=1.0: acceptance 0.7-0.8
- Damping=0.5: acceptance 0.5-0.6
- Damping=0.1: acceptance ≥ 0.2

**Veto Conditions:**
- Damping=0.1 acceptance < 0.2 → surrogate approach questionable, STOP
- T2.1 fails → determinism broken, Corollary 5.2 inapplicable
- T2.2 fails badly (|ΔH| > 1.0) → leapfrog integration problem

**Estimated Time:** 1 day

**Deliverable:**
- Implementation: `bayesfilter/inference/toy_surrogate_force_adapter.py`
- Tests: `tests/inference/test_toy_surrogate_force.py`
- Results: `docs/plans/ledh-surrogate-hmc-phase2-result-2026-09-06.md`
- Acceptance table

---

### Phase 3: LEDH Adapter Implementation and Verification

**Status:** AWAITING_PHASE2  
**Goal:** Build dual-adapter wrapper, verify Corollary 5.2 premises

**Tasks:**

**Task 3.1: Dual-adapter implementation**
- `DualAdapterSurrogateForce` class
- Adapter A (value): λ=1e-5, δ=1e-5 (or tuning-artifact values)
- Adapter B (force): λ=1e-3, δ=1e-3 (100× damping from tuning values)
- Frozen noise: generate once per θ, share across both adapters
- Return: (value_from_A, force_from_B)

**Task 3.2: Damping parameter selection**
- Read trust-region tuning artifact for LGSSM T=50 N=1008
- Extract λ_value, δ_value (the promoted values)
- Compute λ_force = 100 × λ_value, δ_force = 100 × δ_value
- Document rationale: 100× chosen to balance bias vs robustness (100² ≈ 10⁴ variance reduction per Remark 5.3 note line 950)

**Task 3.3: Seed policy verification** (implements Decision 2)

**T3.3a: Bitwise-identical repeat**
```python
for θ in [θ_interior, θ_boundary, θ_tail]:
    (u1, f1) = dual_adapter(θ, seed=12345)
    (u2, f2) = dual_adapter(θ, seed=12345)
    assert u1 == u2 and f1 == f2  # bitwise
```

**T3.3b: Involution test**
```python
(θ_fwd, p_fwd) = leapfrog(θ, p, L=10, ε=0.01)
(θ_back, p_back) = leapfrog(θ_fwd, -p_fwd, L=10, ε=0.01)
assert |θ_back - θ| < derived_tolerance
assert |p_back - p| < derived_tolerance
```

**T3.3c: Continuity test**
```python
u_base = dual_adapter(θ, seed=12345)[0]
for direction in [e_1, e_2, ..., e_5]:
    u_nearby = dual_adapter(θ + 1e-6 * direction, seed=12345)[0]
    assert |u_nearby - u_base| < C * 1e-6  # O(ε), not O(1)
```

**Task 3.4: TFP HMC seed discipline**
- Verify TFP HMC doesn't re-invoke `log_prob_and_grad` redundantly per leapfrog step
- If it does: wrap adapter with `@tf.function` + tracing guard

**Success Criteria:**
- Dual-adapter implemented
- Damping parameters documented with source
- All three seed-policy tests pass
- TFP seed behavior documented

**Veto Conditions:**
- T3.3a fails → determinism broken, Corollary 5.2 inapplicable, STOP
- T3.3b fails → reversibility broken, Corollary 5.2 inapplicable, STOP
- T3.3c fails → θ-dependent branching (Proposition 6), Corollary 5.2 inapplicable, STOP

**Estimated Time:** 1 day

**Deliverable:**
- Implementation: `bayesfilter/inference/ledh_surrogate_force_adapter.py`
- Seed-policy verification note with test results
- Damping-parameter selection note

---

### Phase 4: LGSSM Validation (Procedurally Corrected)

**Status:** AWAITING_PHASE3  
**Goal:** Certify that surrogate-force HMC samples π_N^ω correctly (Corollary 5.2)

**Model:**
- LGSSM d=3 T=50
- True θ = [0.72, 0.55, 0.35, 0.35, 0.45]
- N = 1008 particles
- One frozen ω (master seed from Phase 3 Task 3.3)
- Same observation path as historical runs

**Arms:**

| Arm | Value (λ/δ) | Force (λ/δ) | Role |
|---|---|---|---|
| 1 | exact (tuning artifact) | exact (same) | Reference baseline |
| 2 | exact (tuning artifact) | damped (100× from Task 3.2) | Surrogate-force test |

**PRIMARY CRITERION (Corollary 5.2 certification):**

Arm 1 and Arm 2 posteriors agree within MCMC error.

**Measurement:**
- 2-Wasserstein distance: W₂(P₁, P₂) between empirical distributions
- Acceptance threshold: W₂ < 2 × MCMC_SE(Arm 1)
- Alternative: KL divergence via histogram, or energy distance

**Interpretation:**
- **W₂ < threshold → PASS:** Surrogate-force HMC samples π_N^ω correctly (Corollary 5.2 certified)
- **W₂ ≥ threshold → FAIL:** Method does not sample π_N^ω correctly (Corollary 5.2 violated or premises broken)

**PROMOTION VETOES:**
- Arm 1 acceptance < 0.15 (baseline broken)
- Arm 2 acceptance < 0.15 (chain not mixing)
- W₂(P₁, P₂) ≥ 2 × MCMC_SE (posteriors disagree)

**EXPLANATORY DIAGNOSTICS (reported, not used for promotion):**
- Acceptance rate per arm
- ESS/gradient per arm (descriptive, no ranking claim)
- True-θ coverage (both arms) — see Decision 3 for formula
- Mean score bias (Arm 2 vs exact Kalman)
- R-hat (< 1.05 per parameter)

**HEURISTIC ADVERSARY SET** (CLAUDE.md Heuristic Dominance Gate):

| Adversary | What it is | Salient situation |
|---|---|---|
| H1 | Random-walk Metropolis, matched cost | No gradient, cheapest |
| H2 | Coarser unbiased LEDH (N=504), matched wall time | Unbiased alternative |
| H3 | Exact Kalman HMC | Correctness oracle |
| H4 | Fixed-preconditioner HMC (identity mass) | Ignores θ-dependence |

Evaluated conditionally on: (i) θ near boundary (φ close to 1); (ii) θ interior.

**Heuristic adversary verdict:** If Arm 2 loses to any adversary in any salient situation, that is the headline and a promotion veto.

**NON-CLAIMS:**
- Passing nominates surrogate-force for wider validation (Phase 5) only
- Single-seed run → no statistical ranking ("better/worse") supported
- LGSSM coverage does not prove correctness on nonlinear models
- Passing does NOT prove π_N^ω ≈ π_∞ (finite-N bias unmeasured)

**Execution:**
- 4 chains × 5000 warmup × 5000 sampling per arm
- One seed (from Phase 3 Task 3.3)
- Device: GPU (tftwogpu, CUDA_VISIBLE_DEVICES=1 → 4080 SUPER)
- Regime: float32 TF32 or float64 per Decision 1

**Success Criteria:**
- W₂(P₁, P₂) < 2 × MCMC_SE(Arm 1)
- Both arms acceptance ≥ 0.15
- R-hat < 1.05 (all parameters, both arms)
- Heuristic adversaries: Arm 2 not dominated in salient situations

**Promotion Criterion:**
- All success criteria met → surrogate-force nominated for Phase 5 (Tier A suite)

**Veto Conditions:**
- **Promotion veto:** Any success criterion fails → method not viable on LGSSM
- **Continuation veto:** Same as promotion veto (no point trying Tier A if LGSSM fails)

**Budget:**
- 1 day implementation (Arm 1 baseline, W₂ metric, adversary comparison)
- 12 GPU-hours (6 per arm × 2)

**Deliverable:**
- Runner: `docs/benchmarks/run_ledh_surrogate_lgssm_t50_20260906.py`
- Results: `docs/plans/ledh-surrogate-hmc-phase4-result-2026-09-06.md`
- Output: `docs/benchmarks/artifacts/ledh_surrogate_lgssm_t50_20260906/`
- Decision table: W₂ verdict, heuristic adversary table, Phase 5 proceed/stop decision

---

### Phase 5: Tier A Suite (Conditional on Phase 4 Promotion)

**Status:** AWAITING_PHASE4_PROMOTION  
**Goal:** Validate surrogate-force robustness across non-LGSSM models

**Entry Condition:** Phase 4 promoted (W₂ < threshold, heuristic adversaries passed)

**Models:**
1. Austria SIR T=20 d=5
2. KSC SV T=10 d=2
3. Predator-Prey T=20 d=4

(The three non-LGSSM models with 2026-09-02/03 trust-region tuning artifacts)

**Per-Model Criterion:**
- Arm 1 (exact-force) vs Arm 2 (damped-force) posterior agreement
- W₂(P₁, P₂) < 2 × MCMC_SE(Arm 1)
- Both arms acceptance ≥ 0.15
- Heuristic adversaries (per-model salient situations)

**Evidence Contract:**
- **Primary Criterion:** All 3 models pass W₂ agreement
- **Promotion Veto:** ≥2 models fail (method not robust)
- **Continuation Veto:** Same as promotion veto
- **Explanatory:** ESS/gradient, coverage, score bias per model

**Success Criteria:**
- 3/3 models pass W₂ agreement
- 3/3 models acceptance ≥ 0.15 (both arms)
- Heuristic adversaries: no model-specific domination

**Promotion Criterion:**
- All 3 models pass → surrogate-force recommended for default consideration (requires separate tuning campaign)

**Budget:**
- 3 days implementation + analysis
- 36 GPU-hours (12 per model × 3)

**Deliverable:**
- Per-model results: `docs/plans/ledh-surrogate-hmc-phase5-{model}-result-2026-09-06.md`
- Summary table: model, W₂ verdict, heuristic adversaries, decision
- Program-level result: `docs/plans/ledh-surrogate-hmc-program-result-2026-09-06.md`

---

## Budget Summary

### Compute Budget
- **Phase 0:** Negligible (decisions, derivation, 1 test)
- **Phase 1:** Negligible (diagnostics)
- **Phase 2:** 1 GPU-hour (toy potential)
- **Phase 3:** Negligible (CPU tests, seed verification)
- **Phase 4:** 12 GPU-hours (2 arms × 6 hours each)
- **Phase 5:** 36 GPU-hours (3 models × 12 hours each) — conditional
- **Total:** 49 GPU-hours (13 if Phase 5 doesn't run)

### Attempt Budget
- **Phase 0:** 1 (user decisions, no retries)
- **Phase 1:** 2 (diagnostics routine)
- **Phase 2:** 3 (toy + up to 2 repairs)
- **Phase 3:** 2 (adapter + seed tests)
- **Phase 4:** 3 (baseline + test + 1 repair)
- **Phase 5:** 6 (3 models × 2 attempts each) — conditional
- **Total:** 17 attempts

### Time Budget
- **Phase 0:** 1 day (decisions + derivation + seed policy + coverage)
- **Phase 1:** 1 day (diagnostics)
- **Phase 2:** 1 day (toy potential)
- **Phase 3:** 1 day (adapter + seed tests)
- **Phase 4:** 1 day (baseline implementation + W₂ metric + analysis)
- **Phase 5:** 3 days (3 models + analysis) — conditional
- **Total:** 8 days (5 days if Phase 5 doesn't run)

---

## Decision Gates

### Gate 0: User Decisions → Phase 1
- **Criterion:** All 3 decisions documented (tolerance, seed policy, coverage)
- **Authority:** User

### Gate 1: Phase 1 → Phase 2
- **Criterion:** All diagnostics pass
- **Authority:** Automatic

### Gate 2: Phase 2 → Phase 3
- **Criterion:** Toy potential acceptance ≥ 0.2 at damping=0.1
- **Veto:** Acceptance < 0.2 (continuation veto, STOP)
- **Authority:** Automatic if pass, user if veto fires

### Gate 3: Phase 3 → Phase 4
- **Criterion:** All 3 seed-policy tests pass
- **Veto:** Any test fails (Corollary 5.2 inapplicable, STOP)
- **Authority:** Automatic if pass, user if veto fires

### Gate 4: Phase 4 → Phase 5
- **Criterion:** W₂(P₁, P₂) < threshold, heuristic adversaries passed
- **Veto:** W₂ ≥ threshold or adversary domination (continuation veto, STOP)
- **Authority:** User (Phase 5 is 36 GPU-hours)

---

## Success Definition

### Minimal Success (Phase 4 promotes)
- Surrogate-force HMC samples π_N^ω correctly on LGSSM (Corollary 5.2 certified)
- Acceptance ≥ 0.15, heuristic adversaries passed
- Nominated for Tier A validation

### Full Success (Phase 5 promotes)
- 3/3 Tier A models pass W₂ agreement
- Surrogate-force robust across model types
- Recommended for default consideration (requires tuning campaign)

### What Full Success Does NOT Certify
- Samples from π_∞ (finite-N bias unmeasured)
- "Better" than exact-force (single-seed, no statistical ranking)
- Production-ready (requires tuning campaign + ω-ensemble analysis)

---

## Failure Modes and Interpretation

### F1: Phase 2 fails (toy acceptance < 0.2)
- **Diagnosis:** Damped force too weak
- **Response:** STOP, document mechanics failure
- **Next:** Consider PaRIS or accept LEDH limitations

### F2: Phase 3 seed tests fail
- **Diagnosis:** Corollary 5.2 premises violated
- **Which test?**
  - T3.3a fails → determinism broken (TFP reseeding, state leak)
  - T3.3b fails → reversibility broken (asymmetric force, wrong Jacobian)
  - T3.3c fails → θ-dependent branching (Proposition 6)
- **Response:** Fix violation, document, re-run Phase 3

### F3: Phase 4 fails W₂ agreement
- **Diagnosis:** Surrogate-force HMC does NOT sample π_N^ω correctly
- **Response:** Investigate which premise failed (re-check Phase 3 tests on LGSSM fixture)
- **Next:** If premises hold, mechanism failure → STOP

### F4: Phase 4 fails heuristic adversaries
- **Diagnosis:** Damped-force loses to simpler method in salient situation
- **Response:** Document which adversary won where, classify as efficiency issue (not correctness)
- **Decision:** User decides whether to proceed to Phase 5 (correctness holds but mixing poor)

### F5: Phase 5 fails (≥2 models fail W₂)
- **Diagnosis:** Surrogate-force not robust across model types
- **Response:** Document which models failed, classify by failure mode
- **Next:** Model-specific tuning or accept limited scope

---

## Non-Claims and Scope

### What This Program Tests
- ✓ Does surrogate-force HMC sample π_N^ω correctly? (Corollary 5.2)
- ✓ Is acceptance ≥ 0.15? (mixing threshold)
- ✓ Does it pass heuristic adversaries? (basic sanity)
- ✓ Is it robust across Tier A models? (Phase 5)

### What This Program Does NOT Test
- ✗ Does π_N^ω ≈ π_∞? (finite-N bias — requires N-convergence study)
- ✗ Is it "better" than exact-force? (requires multi-seed statistical comparison)
- ✗ Is it production-ready? (requires tuning campaign + ω-ensemble analysis)
- ✗ Does it work on Tier S (DSGE)? (out of scope)

### What Passing Phase 4 Certifies
- ✓ Corollary 5.2 applies (premises verified in Phase 3)
- ✓ Surrogate-force HMC samples π_N^ω correctly (W₂ agreement)
- ✓ Acceptance ≥ 0.15 (chain is mixing)
- ✓ Heuristic adversaries passed (not dominated by simpler methods)

### What Passing Phase 4 Does NOT Certify
- ✗ Posterior covers true θ (finite-N bias + data property)
- ✗ π_N^ω ≈ π_∞ (value quality separate from sampler correctness)
- ✗ Method is "good" or "ready" (nomination only)

---

## Governance

### Authority Chain
1. **This master program:** Scientific and execution authority
2. **Procedural review 2026-09-06:** Correctness basis
3. **Corollary 5.2 (variance note lines 954-976):** Mathematical authority
4. **CLAUDE.md + AGENTS.md:** Engineering and policy authority
5. **User:** Final decision authority on gates, vetoes, repairs

### Update Protocol
- Update this document after each phase completion
- Mark phase status: AWAITING → IN_PROGRESS → COMPLETE / FAILED / STOPPED
- Update budget consumed (attempts, compute, time)
- Update decision gates with actual outcomes

### Emergency Suspension
- User can suspend program at any time
- Document reason, mark status SUSPENDED
- Preserve all artifacts for potential resume

---

**END OF PROCEDURALLY CORRECT MASTER PROGRAM**

**Program Status:** DRAFT — awaiting user approval of 3 decisions  
**Next Milestone:** User decides D1 (tolerance), D2 (seed policy), D3 (coverage)  
**Blocking:** Three user decisions required before Phase 1 can start  
**Last Updated:** 2026-09-06 (procedural correctness review complete)
