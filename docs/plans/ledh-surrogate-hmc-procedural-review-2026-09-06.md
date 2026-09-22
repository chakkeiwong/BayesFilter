# LEDH Surrogate-Force HMC — Procedural Correctness Review

**Date:** 2026-09-06  
**Authority:** User request for procedurally correct master program  
**Scope:** Experimental setup correctness, not scientific outcome prediction

---

## Executive Summary

**Procedural vs Scientific Distinction:**
- **Procedural correctness:** The experiment can certify success OR detect failure, and either result is interpretable
- **Scientific outcome:** Whether the method works — unknown until run, not controllable by planning

**Review Verdict:** The current unified program (2026-09-06) has **one blocking procedural defect** and **three required clarifications** before execution. The defect makes the primary criterion uninterpretable regardless of outcome.

**The Blocking Defect (estimand mismatch):**

Phase 4 primary criterion is "posterior 95% intervals cover true θ (all 5 parameters)." This criterion confounds three independent questions:
1. Does the sampler correctly sample π_N^ω (the finite-N pseudo-posterior at frozen noise ω)?
2. Is π_N^ω close to π_∞ (the exact posterior)? — finite-N bias
3. Does π_∞ cover true θ for this dataset? — data property

**What Corollary 5.2 Actually Certifies:**

Line 964: "Every step leaves π(θ,p) ∝ exp(-H(θ,p)) invariant, hence the θ-marginal ∝ exp(-U(θ)) is preserved."

The corollary targets **π_N^ω only** — the distribution induced by the executed finite-N filter with frozen noise. It says nothing about π_∞ or true θ coverage.

**Why This Is a Procedural Defect, Not a Scientific Prediction:**

- **If the method works but coverage fails:** Could be finite-N bias (question 2) or unlucky data (question 3). Cannot conclude sampler is broken. False negative.
- **If the method fails but coverage passes:** Could be over-dispersion from broken sampler happens to widen intervals enough. False positive.
- **Either outcome is uninterpretable** — the criterion doesn't measure what Corollary 5.2 proves.

**The Procedural Fix:**

Replace Phase 4 primary criterion with **reference-sampler agreement**:
- Arm 1: long-run exact-force HMC on frozen (ω, N) target → samples from π_N^ω via exact gradient
- Arm 2: damped-force HMC on **same** frozen (ω, N) target → samples from π_N^ω via damped gradient
- Primary criterion: Arm 1 and Arm 2 posteriors agree (KL, Wasserstein, or overlap < MCMC error threshold)

This directly measures question 1 (sampler correctness) and is both necessary and sufficient for Corollary 5.2. True-θ coverage becomes an **explanatory diagnostic** — we report it, but don't promote/reject based on it.

**Three Required Clarifications:**

1. **Coverage criterion math (D1):** Current "all 5 parameters" has P(pass | correct sampler) = 0.95^5 ≈ 0.774. Must specify: joint Mahalanobis region, or marginal-with-expected-failures, or Bonferroni-corrected marginals.

2. **Tolerance policy (C3):** Program uses two incompatible regimes (golden master 1e-12, cross-lane parity 5e-4). Must derive one tolerance from condition number + tuning-insensitivity band + backend dtype.

3. **Seed policy (B4):** Corollary 5.2 Remark line 980 requires "all Monte Carlo seeds inside F must be frozen." Program must verify: one trajectory-level master seed ω, frozen for entire HMC chain, not reseeded per step or per trajectory.

---

## Part 1: What The Program Claims To Test

**From unified program Phase 4 (line 586):**

> Primary criterion (promotion): Arm 2 posterior 95% intervals cover true θ (all 5 parameters).

**What this measures:** Whether samples from the damped-force chain produce credible intervals that contain the data-generating θ.

**What Corollary 5.2 actually proves (line 964):**

> Every step leaves π(θ,p) ∝ exp(-H(θ,p)) invariant, hence the θ-marginal ∝ exp(-U(θ)) is preserved, for any quality of F.

Where U = -L̂^N is the executed finite-N log-likelihood with frozen noise ω.

**The target of the theorem:** π_N^ω(θ) ∝ exp(-U_N^ω(θ)), the pseudo-posterior induced by:
- N particles (finite sample, not infinite limit)
- One frozen noise draw ω (not averaged over ω-ensemble)
- The executed filter (OT transport, dual-cap trust-region, Contract-E reset)

**The estimand mismatch:**

| Object | Definition | What it represents | Does Corollary 5.2 govern it? |
|---|---|---|---|
| π_∞(θ\|y) | Exact posterior of state-space model | The "true" Bayesian posterior | ❌ No |
| π_N(θ\|y) | E_ω[π_N^ω(θ\|y)] | Finite-N pseudo-posterior, ω-averaged | ❌ No |
| π_N^ω(θ\|y) | ∝ exp(-U_N^ω(θ)) | Finite-N at one frozen ω | ✅ Yes — this only |

**True-θ coverage depends on all three:**

P(θ_true ∈ credible interval) = P₁ × P₂ × P₃, where:
- P₁ = P(sampler draws from π_N^ω correctly) — **this is Corollary 5.2**
- P₂ = P(π_N^ω ≈ π_N for this ω) — ω-variance, acknowledged as open question
- P₃ = P(π_N ≈ π_∞) — finite-N bias, **unmeasured** in current program
- P₄ = P(π_∞ covers θ_true for this y) — data luck, not method property

Coverage failing could be any of (P₂, P₃, P₄) failing while P₁ = 1 (correct sampler).  
Coverage passing could be P₁ = 0 (broken sampler) with over-dispersion compensating.

**This is a procedural defect because:**
- The experiment cannot distinguish sampler failure from finite-N bias or data luck
- Pass and fail are both uninterpretable relative to Corollary 5.2
- It measures a compound question when the theorem addresses one component

---

## Part 2: The Procedurally Correct Criterion

**Principle:** Measure what the theorem proves.

Corollary 5.2 proves: surrogate-force HMC leaves π_N^ω invariant. So measure that directly.

**Reference-Sampler Agreement:**

**Setup:**
1. Fix one ω (frozen noise) and one N (particle count)
2. Run two HMC chains on **identical** frozen target U_N^ω:
   - **Arm 1 (reference):** Exact-force HMC, long run (ESS ≥ 1000 per parameter)
   - **Arm 2 (test):** Damped-force HMC, same length
3. Both arms see the same (ω, N, y, U) — only the force differs

**Primary criterion:**

Arm 1 and Arm 2 posteriors agree within MCMC error. Measure via:
- **2-Wasserstein distance** between empirical distributions: W₂(P₁, P₂) < ε_MCMC
- **KL divergence** via histogram: KL(P₁ || P₂) < threshold
- **Overlap coefficient** or **energy distance**

Acceptance threshold: agreement within 2× the Arm 1 MCMC standard error.

**Why this is procedurally correct:**

| Outcome | Interpretation | Falsifiable? |
|---|---|---|
| Arm 1 ≈ Arm 2 | Damped force samples π_N^ω correctly | ✅ Yes — Corollary 5.2 certified |
| Arm 1 ≠ Arm 2 | Surrogate-force HMC fails on this target | ✅ Yes — method rejected |

Both outcomes are interpretable. The criterion is both necessary and sufficient for Corollary 5.2.

**True-θ coverage becomes explanatory:**

Still compute and report:
- Does Arm 1 cover θ_true? (tests: is the baseline reasonable?)
- Does Arm 2 cover θ_true? (tests: does damped-force preserve coverage?)
- If both cover → compatible with correctness (but not proof)
- If Arm 1 covers but Arm 2 doesn't → suggests damped force under-disperses
- If neither covers → finite-N bias or data issue, not sampler defect

But **do not promote/reject based on coverage** — it's a diagnostic, not the criterion.

**Cost:**

Adds Arm 1 (reference baseline), ~6 GPU-hours. Already budgeted in unified program Phase 4 (12 GPU-hours for 2 arms).

---

## Part 3: The Three Required Clarifications

### C1. Coverage Criterion Math (if coverage becomes a diagnostic)

**Problem:** "95% intervals cover true θ (all 5 parameters)" has P(pass | correct sampler) = 0.95^5 ≈ 0.774.

A correct sampler fails this check ~23% of the time (false negative).

**Required decision:** Which coverage definition?

**Option A: Joint 95% Mahalanobis region (RECOMMENDED)**
```
T² = (θ_post - θ_true)ᵀ Σ_post⁻¹ (θ_post - θ_true)
Accept if T² < χ²₀.₉₅(5)
```
- One joint test, exactly 95% coverage for correct multivariate sampler
- Requires posterior covariance estimate

**Option B: Marginal with expected-failures**
```
Accept if ≥4 of 5 marginals cover
P(≥4 of 5) ≈ 0.95 under independence
```
- Easier to explain, robust to moderate correlation

**Option C: Bonferroni-corrected marginals**
```
Use 99% per parameter (1 - 0.05/5)
Joint coverage ≈ 95%
```
- Most conservative

**Procedural requirement:** Choose one before Phase 4, document it, use it consistently.

**Recommendation:** Option A (joint Mahalanobis) if coverage is reported; but demote coverage to explanatory regardless.

---

### C2. Tolerance Policy (Phase 2B prerequisite)

**Problem:** Unified program references two incompatible tolerance regimes:
- Golden-master fixtures: 1e-12 relative error (bitwise-identical in float64)
- Cross-lane parity: 5e-4 relative error

Eight orders of magnitude apart. A refactor with 1e-5 drift fails golden and passes parity — which verdict governs?

**Procedural requirement:** Derive one tolerance before generating any fixtures.

**Derivation:**

```
κ = condition_number(reset_Jacobian)  # from existing LGSSM T=50 artifact
ε_backend = machine_epsilon(production_dtype)  # 1e-4 for float32 TF32, 1e-16 for float64
ε_amplified = κ * ε_backend
ε_tuning = tuning_insensitivity_band()  # from trust-region tuning: largest drift that doesn't trigger retuning
tol_derived = max(ε_amplified, ε_tuning)
```

**Also required:** Choose production regime:
- **Option A:** Phase 4 runs in float32 TF32 on GPU (actual production target), accept that tests in float64-CPU don't cover production dtype
- **Option B:** Phase 4 runs in float64 on GPU (tested regime), accept ~30% slowdown and certificate doesn't cover production

**Recommendation:** Option A (production regime), derive tolerance for float32 TF32.

**Estimated time:** 0.3 day (one derivation note).

---

### C3. Seed Policy Verification (Corollary 5.2 premise)

**Problem:** Remark 5.3 line 980 requires:

> F must be a deterministic function of θ alone: all Monte Carlo seeds inside F must be frozen, and F may not depend on momentum or on the trajectory's history.

**What must be verified:**

1. **All seeds frozen to one master ω:**
   - Initial-noise matrix: deterministic slice of ω
   - Innovation rows e_{t,i}: deterministic slice of ω
   - Sinkhorn entropic noise: deterministic slice of ω
   - Residual design Ξ: deterministic (Contract-E)
   - No other TF random ops in value or force path

2. **Master seed ω is frozen for entire HMC chain:**
   - ω may depend on initial θ₀ (part of context)
   - ω must NOT depend on current θ during leapfrog
   - ω must NOT be reseeded per step, per trajectory, or per adapter call

3. **Force is history-independent:**
   - No cached particle cloud from prior evaluation
   - No call-count conditional
   - No step-index conditional
   - No accumulated float state (running mean, EMA)

**Procedural tests required:**

**T1. Bitwise-identical repeat:**
```python
for θ in test_grid:
    u1, f1 = evaluate(θ, seed=12345)
    u2, f2 = evaluate(θ, seed=12345)
    assert u1 == u2 and f1 == f2  # bitwise
```

**T2. Involution test (standard HMC check):**
```python
(θ_fwd, p_fwd) = leapfrog(θ, p, L, ε)
(θ_back, p_back) = leapfrog(θ_fwd, -p_fwd, L, ε)
assert |θ_back - θ| < tol and |p_back - p| < tol
```
If force depends on anything but θ, this fails.

**T3. Continuity test (no θ-dependent branching):**
```python
u_base = evaluate(θ, seed=12345)
u_nearby = evaluate(θ + ε*direction, seed=12345)
assert |u_nearby - u_base| < C * ε  # O(ε), not O(1) jump
```

**Estimated time:** 0.5 day (instrumentation + 3 tests on toy and LGSSM).

**Why this is procedural, not scientific:** If these fail, Corollary 5.2 does not apply regardless of coverage or acceptance results. The experiment cannot certify anything without verifying the premises.

---

## Part 4: Additional Procedural Gaps (Non-Blocking)

These do not block Phase 4 execution but should be addressed before claiming production-readiness.

### G1. No exact-Kalman surrogate-force rung

**Problem:** The validation ladder jumps from 3-D quadratic toy (Phase 2) to full LEDH T=50 (Phase 4). Without an intermediate rung where the **value is exact Kalman log-likelihood** and force is that gradient deliberately corrupted, a Phase 4 failure cannot be attributed.

**Why gap:** If Phase 4 fails:
- Is the surrogate-force mechanism broken? (would fail on Kalman rung too)
- Is LEDH score bias too large for the damping ratio? (would pass Kalman rung)

Cannot distinguish without the rung.

**Repair:** New Phase 2.5 — LGSSM d=3 T=20, exact Kalman log-likelihood as value, that gradient with frozen seed ≠ value seed as force. Acceptance: posterior agreement with exact Kalman (reference).

**Estimated time:** 0.5 day + 2 GPU-hours.

**Status:** Non-blocking for Phase 4 (can interpret pass, just not failure).

---

### G2. No volume-preservation test

**Problem:** Corollary 5.2 Proposition 1 requires leapfrog preserves Lebesgue measure. A sign error, wrong Jacobian, or trust-region projection could break this. Toy tests might still pass if the error is small.

**Repair:** Volume-preservation test (estimate phase-space volume before/after leapfrog on 1000-point grid, assert |(V_after - V_before)/V_before| < 1e-6).

**Estimated time:** 0.2 day.

**Status:** Non-blocking (trust-region is smooth so likely safe, but unchecked).

---

### G3. No gradient correctness test

**Problem:** Force F must equal -∇U. A wrong chain-rule, missing term, or sign error breaks HMC but might pass acceptance tests if error is in low-sensitivity direction.

**Repair:** Finite-difference check:
```python
for i in range(dim):
    fd_i = (U(θ + ε*e_i) - U(θ - ε*e_i)) / (2*ε)
    auto_i = F(θ)[i]
    assert |fd_i - auto_i| < tol * (1 + |fd_i|)
```

**Estimated time:** 0.1 day (standard practice).

**Status:** Non-blocking (analytical JVP tested in Phase 1, but not against FD).

---

### G4. No N-convergence check

**Problem:** Corollary 5.2 applies to π_N^ω, which has bias vs π_∞. Only planned check that **U_N converges to the right limit** as N → ∞.

**Why procedural:** Without this, Phase 4 could pass (sampler works) while U_N converges to the wrong function (filter is systematically biased).

**Repair:** Phase 2.5 (exact-Kalman rung): run N ∈ {50, 100, 200, 500, 1000}, plot ||μ_N - μ_Kalman|| and ||Σ_N - Σ_Kalman||_F vs 1/√N. Acceptance: monotone decreasing, passes through zero-intercept confidence band.

**Estimated time:** Included in Phase 2.5 (no separate budget).

**Status:** Non-blocking for Phase 4 (sampler correctness is independent of value quality).

---

### G5. Device-level determinism not verified

**Problem:** GPU results are deterministic only if TF32, cuDNN autotune, and non-deterministic reduce are controlled. Program assumes this but doesn't check.

**Repair:** Run Phase 2A toy twice on GPU with same seed, assert bitwise-identical (U, F) at 10 θ points.

**Estimated time:** 0.1 day.

**Status:** Non-blocking (reproducibility check, not correctness).

---

### G6. No checkpointing

**Problem:** Phase 4 is 12 GPU-hours. A crash at hour 11 loses everything.

**Repair:** Save chain state + diagnostics every 1000 iterations, resume from checkpoint on restart.

**Estimated time:** 0.2 day.

**Status:** Operational (doesn't affect interpretability).

---

## Part 5: Procedural Correctness Summary

**Blocking defect (must fix before Phase 4):**

| ID | Defect | Impact | Repair | Time |
|---|---|---|---|---|
| **A1** | Primary criterion measures wrong object | Pass and fail both uninterpretable | Replace with reference-sampler agreement | 0.3 day |

**Required clarifications (must decide before Phase 4):**

| ID | Clarification | Decision Required | Recommendation |
|---|---|---|---|
| **C1** | Coverage criterion math | Joint / marginal-with-failures / Bonferroni | Joint Mahalanobis (but demote to explanatory) |
| **C2** | Tolerance derivation + regime | float64-CPU test vs float32-TF32 production | Condition-number-derived + production regime |
| **C3** | Seed policy verification | Document + test frozen-seed requirement | One master ω, three tests (bitwise / involution / continuity) |

**Non-blocking gaps (good practice, defer to Phase 5+):**

| ID | Gap | Impact | Repair Time |
|---|---|---|---|
| **G1** | No exact-Kalman rung | Phase 4 failure uninterpretable | 0.5 day + 2 GPU-hr |
| **G2** | No volume-preservation test | Misses rare leapfrog bugs | 0.2 day |
| **G3** | No gradient correctness test | Misses wrong-derivative bugs | 0.1 day |
| **G4** | No N-convergence check | Can't verify value limit | Included in G1 |
| **G5** | No GPU determinism check | Reproducibility uncertain | 0.1 day |
| **G6** | No checkpointing | Crash loses work | 0.2 day |

**Total blocking + required work:** 0.3 + 0.3 + 0.5 = **1.1 days** (all CPU, no campaigns).

**Total if addressing non-blocking:** 1.1 + 1.3 = **2.4 days** + 2 GPU-hours.

---

## Part 6: What This Review Does NOT Predict

**Scientific outcomes (unknown until run):**

- ✗ Whether damped-force acceptance will be ≥ 0.2 (mixing question)
- ✗ Whether ESS/gradient will be acceptable (efficiency question)
- ✗ Whether the method works on nonlinear models (Tier A question)
- ✗ Whether finite-N bias is small enough for claims about π_∞ (value-quality question)

**What the procedurally correct experiment CAN answer:**

- ✓ Does surrogate-force HMC sample π_N^ω correctly? (Corollary 5.2)
- ✓ If yes: is mixing acceptable? (nomination for wider validation)
- ✓ If no: where does it fail? (acceptance, reversibility, determinism)

All three outcomes are interpretable under the corrected criterion.

---

## Part 7: Recommended Minimal Fix

**To make the current unified program procedurally correct with minimal disruption:**

### Fix 1: Replace Phase 4 primary criterion (0.3 day)

**Current (line 586):**
> Primary criterion (promotion): Arm 2 posterior 95% intervals cover true θ (all 5 parameters).

**Replacement:**
> Primary criterion (promotion): Arm 1 and Arm 2 posteriors agree within MCMC error. Measure via 2-Wasserstein distance W₂(P₁, P₂), acceptance threshold 2× Arm 1 MCMC SE. True-θ coverage (both arms) is explanatory only.

**Also update:**
- Phase 4 "Promotion vetoes" (line 591): remove "Arm 2 covers but Arm 1 does not", add "W₂(P₁, P₂) > threshold"
- Phase 4 "Explanatory diagnostics" (line 633): move coverage here
- Phase 4 "Non-claims" (line 639): add "True-θ coverage is not a correctness criterion"

### Fix 2: Document seed policy + add verification (0.5 day)

**Add to Phase 3 (Adapter Implementation):**

> Task 3.3: Seed policy verification
> 
> Requirements (Corollary 5.2 Remark line 980):
> - One trajectory-level master seed ω
> - ω frozen for entire HMC chain (may depend on initial θ₀, not on current θ)
> - No reseeding per step, per trajectory, or per adapter call
> 
> Tests:
> - T3.3a: Bitwise-identical repeat at same (θ, ω)
> - T3.3b: Involution (leapfrog reversibility)
> - T3.3c: Continuity (no O(1) jumps at nearby θ under same ω)
> 
> Deliverable: Seed-policy verification note, three tests passing.

### Fix 3: Decide and document (C1, C2 decisions)

**Add to Phase 0 (Planning):**

> Task 0.2: Tolerance derivation
> 
> Derive one tolerance for all checks (golden master, parity, Phase 2B unification):
> - Production regime: float32 TF32 on GPU
> - Inputs: condition number (from existing artifact), tuning-insensitivity band, backend ε = 1e-4
> - Formula: tol = max(κ × ε, tuning_band)
> 
> Deliverable: Tolerance derivation note, one constant for all checks.
> 
> Task 0.3: Coverage criterion (if used as diagnostic)
> 
> Decision: joint 95% Mahalanobis region (Option A).
> Rationale: Correct multivariate coverage, avoids 23% false-negative rate of independent marginals.
> Status: Diagnostic only (not primary criterion after Fix 1).

**Total minimal fix:** 1.1 days, makes Phase 4 interpretable.

---

## Part 8: What The Corrected Program Certifies

**If Phase 4 passes (Arm 1 ≈ Arm 2):**

Certified:
- ✓ Surrogate-force HMC samples π_N^ω correctly (Corollary 5.2)
- ✓ Acceptance ≥ 0.15 (chain is mixing)
- ✓ Method is viable on LGSSM d=3 T=50 at N=1008

Nominated (requires Phase 5):
- ? Robust across Tier A models (Austria SIR, KSC SV, Predator-Prey)
- ? ESS/gradient competitive with exact-force baseline

NOT certified:
- ✗ Samples from π_∞ (requires G4, N-convergence)
- ✗ True-θ coverage (diagnostic only, no correctness claim)
- ✗ Production-ready (requires Tier A + tuning campaign)

**If Phase 4 fails (Arm 1 ≠ Arm 2):**

Interpretable:
- ✓ Surrogate-force HMC does NOT sample π_N^ω correctly
- ✓ Mechanism is broken (not just "LEDH bias too large")
- ✓ Corollary 5.2 premises violated or implementation bug

Next action:
- Investigate: reversibility (T3.3b), determinism (T3.3a), seed policy (T3.3c)
- If all tests pass: mechanism failure, consider PaRIS or accept LEDH limitations
- If test fails: fix violation, re-run Phase 4

---

**END OF PROCEDURAL REVIEW**

**Summary:** One blocking defect (wrong estimand), three required clarifications (coverage, tolerance, seed policy), six non-blocking good-practice gaps. Minimal fix: 1.1 days, makes the program procedurally correct and interpretable regardless of scientific outcome.
