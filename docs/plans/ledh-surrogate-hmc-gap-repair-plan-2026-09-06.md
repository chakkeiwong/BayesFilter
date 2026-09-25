# LEDH Surrogate-Force HMC — Gap Repair Plan

**Date:** 2026-09-06  
**Authority:** Gap audit corrected 2026-09-06  
**Status:** DRAFT — requires user decisions on D1, A1, C3 before execution

---

## Executive summary

**19 gaps found** (5 blocking, 6 major, 6 minor, 2 verified non-gaps).  
**Repair sequence:** Resolve 6 user decisions → close 5 blockers → build Phase 2C test suite → close 6 major gaps → close 6 minor gaps → execute unified program.

**Total estimated cost:** 4–6 days (all CPU test/decision work, no GPU campaigns).

**Critical path:** User decisions (D1, A1, C3) block all subsequent work.

---

## Part 1: User decisions (BLOCKS ALL WORK)

### Decision 1: Coverage criterion (D1)

**Question:** Phase 4 criterion "$95\%$ marginal intervals cover true $\theta$ (all 5 parameters)" has 77% false-negative rate for a correct sampler. Which replacement?

**Options:**

**A. Joint 95% Mahalanobis ellipsoid (RECOMMENDED)**
```
T² = (θ_post - θ_true)ᵀ Σ_post⁻¹ (θ_post - θ_true)
Accept if T² < χ²₀.₉₅(5)
```
- One joint test, 95% coverage for correct sampler
- Natural for multivariate posterior
- Requires posterior covariance estimate

**B. Marginal with expected-failures**
```
Accept if ≥4 of 5 marginal 95% intervals cover
P(4 or 5 of 5) ≈ 0.95 under independence
```
- Easier to explain
- Robust to moderate posterior correlation
- Less power if failures cluster

**C. Bonferroni-corrected marginals**
```
Use 99% intervals per parameter (1 - 0.05/5)
Joint coverage ≈ 95%
```
- Most conservative
- Simple implementation
- Loses power

**Default if no choice:** Option A (joint Mahalanobis).

---

### Decision 2: Primary criterion (A1)

**Question:** Current Phase 4 criterion is "true-θ coverage." This confounds: (1) sampler correctness, (2) finite-N bias, (3) data property. Replace with reference-sampler agreement?

**Proposed change:**

**Primary criterion:** Posterior agreement between:
- Arm 1: long-run exact-force HMC on frozen $(ω, N)$ target
- Arm 2: damped-force HMC on **same** frozen $(ω, N)$ target

Measure via KL divergence, 2-Wasserstein, or Monte Carlo overlap. Acceptance threshold: agreement within reference MCMC error.

**Demotion:** True-θ coverage becomes **explanatory diagnostic** — we report it, but don't promote/reject based on it.

**What this changes:** Phase 4 now proves "surrogate-force HMC samples the finite-N pseudo-posterior correctly" instead of "the finite-N pseudo-posterior covers true θ."

**Rationale:** Corollary 5.2 targets $\pi_N^\omega(\theta) \propto \exp(-U_N^\omega(\theta))$, not the exact posterior. Testing coverage of true θ is testing the wrong object.

**Cost impact:** Adds Arm 1 (reference sampler), ~6 GPU-hours.

**Default if no choice:** Approve the change (reference-sampler agreement becomes primary).

---

### Decision 3: Tolerance policy and production regime (C3)

**Question:** Golden-master fixtures demand 10⁻¹² relative; cross-lane parity allows 5×10⁻⁴. Eight orders apart. Which tolerance, and which regime does Phase 4 certify?

**Proposed resolution:**

1. **Derive tolerance** from:
   - Condition number of reset Jacobian (amplifies perturbation)
   - Tuning-insensitivity band (largest drift that doesn't trigger retuning per Per-Scope Tuning Rule)
   - Backend arithmetic (float32 TF32 ε ≈ 10⁻⁴, float64 ε ≈ 10⁻¹⁶)

2. **Choose production regime:**
   - **Option A (RECOMMENDED):** Phase 4 runs in **float32 TF32 on GPU** (the actual production target). Tests in float64-CPU remain as logic/correctness checks. Accept that certificate covers production regime, not test regime.
   - **Option B:** Phase 4 runs in **float64 on GPU** (the tested regime). ~30% slower. Certificate covers what was tested, not what will be deployed.

3. **One tolerance everywhere:** The derived value applies to golden-master, parity, and Phase 2B Step 5 unification checks. No more incoherent thresholds.

**Default if no choice:** Option A (production regime, accept test/production gap) + condition-number-derived tolerance.

---

### Decision 4: 2× filter cost (A3)

**Question:** "Damped force" means value at $(λ_{\text{value}}, δ_{\text{value}})$, force at $(λ_{\text{force}}, δ_{\text{force}})$ — two filter evaluations per leapfrog step. Phase 3 budgeted 6–12 GPU-hours assuming 1× cost. With 2×, the budget is 12–24 GPU-hours. Approve?

**Rationale:** Remark 5.3 explicitly licenses this ("a damped or clipped variant of the recursive score"). Corollary 5.2 applies to **any** deterministic $F$, including the gradient of a different reset configuration.

**Default if no choice:** Approve as written. Update Phase 3 budget to 12–24 GPU-hours.

---

### Decision 5: One-seed policy (B4)

**Question:** Proposition 6 proves θ-dependent branching makes $\widehat{L}^N(\theta)$ discontinuous. Seed policy must be: one trajectory-level master seed $ω$, frozen for entire HMC chain. Seed may depend on **initial** $θ_0$ (part of context), not on **current** $θ$ during leapfrog. Approve?

**Default if no choice:** Approve as determined by Proposition 6.

---

### Decision 6: Independent review (F1)

**Question:** Governance policy recommends independent review for "publication-grade claims, unusually expensive campaigns." Send program + audit to a fresh independent agent for red-team review before Phase 2B?

**Cost:** ~0.3 day (reviewer) + 2–4 hours user adjudication time.

**Default if no choice:** Skip (user oversight is continuous; second-model review optional).

---

## Part 2: Blocking gap repairs (SERIAL, must complete before Phase 2B)

### Step 1: Premise verification (V1, V2) `[0.6 day]`

**V1. All seeds frozen**
- Instrument every TF random op in LEDH value and force paths
- Assert all resolve to slices of one master seed $ω$
- Test: evaluate $(U, F)$ twice at same $θ$ with same $ω$, assert bitwise-identical

**V2. Force history-independent**
- Involution test:
  ```
  (θ_fwd, p_fwd) = leapfrog(θ, p, L, ε)
  (θ_back, p_back) = leapfrog(θ_fwd, -p_fwd, L, ε)
  assert |θ_back - θ| < tol and |p_back - p| < tol
  ```
- Run on toy (Phase 2A) and LGSSM

**Deliverable:** Two tests passing, one instrumentation report.

---

### Step 2: Exact-Kalman surrogate-force rung (B1) `[0.5 day + 2 GPU-hours]`

**New Phase 2A.5** between Phase 2A (toy) and Phase 4 (LEDH T=50).

**Model:** Linear-Gaussian d=3 T=20, exact Kalman posterior available.

**Arms:**
1. Exact-force HMC: $F = -\nabla_θ U$ with $U = -\log p(y_{1:T} | θ)$ (exact Kalman log-likelihood)
2. Damped-force HMC: $F$ uses exact Kalman score with **frozen seed ≠ value seed** (simulates finite-particle noise)

**Acceptance:**
- Both arms: posterior agreement with exact Kalman (reference) within MCMC error
- Damped arm may have lower ESS or acceptance, but must sample correct target

**Also includes C1 (N-convergence):** Run $N \in \{50, 100, 200, 500, 1000\}$, plot $\|\mu_N - \mu_{\text{exact}}\|$ and $\|\Sigma_N - \Sigma_{\text{exact}}\|_F$ vs $1/\sqrt{N}$.

**Deliverable:** Phase 2A.5 result note, one experiment artifact.

---

### Step 3: Tolerance derivation (C3) `[0.3 day]`

**Inputs (from Decision 3):**
- Production regime: float32 TF32 on GPU (or float64 if user chose Option B)
- Condition-number estimate from existing LGSSM T=50 N=500 reset Jacobian

**Derivation:**
```
κ = condition_number(reset_Jacobian)
ε_backend = 1e-4  # float32 TF32
ε_amplified = κ * ε_backend
ε_tuning = tuning_insensitivity_band()  # from trust-region tuning artifact
tol_derived = max(ε_amplified, ε_tuning)
```

**Deliverable:** One markdown note with derivation, one tolerance constant for all checks.

---

### Step 4: Coverage criterion implementation (D1) `[0.2 day]`

**Implementation (using Decision 1 choice):**
- If joint Mahalanobis: compute $T^2$ statistic, compare to $\chi^2_{0.95}(5)$
- If marginal-with-failures: check ≥4 of 5 cover
- If Bonferroni: use 99% marginals

**Deliverable:** Updated Phase 4 criterion code, one test on synthetic data.

---

### Step 5: Reference-sampler baseline (A1) `[0.3 day]`

**Implementation (if Decision 2 approved):**
- Add Arm 1 to Phase 4: long-run exact-force HMC on frozen $(ω, N)$ target
- Posterior agreement metrics: 2-Wasserstein distance, KL via histogram, Monte Carlo overlap
- Acceptance threshold: agreement within 2× reference MCMC SE

**Deliverable:** Updated Phase 4 plan with Arm 1, agreement-metric code.

---

## Part 3: Test suite (Phase 2C) — AFTER blockers closed

**Timing:** After Steps 1–5, before Phase 2B (engine unification).

**Rationale:** Phase 2B will touch every surface where wiring/contract defects occurred (reset_policy dispatch, fixture wiring, adapter signatures). Build the safety net **before** the refactor, not after.

**Content:** Full 7-class suite from unified program Phase 2C (contract validation, wiring, fixture hashing, parameter resolution, run-manifest schema, policy compliance, document-code correspondence).

**Estimated cost:** 2–3 days (already in unified program).

**Deliverable:** Phase 2C complete, all tests green.

---

## Part 4: Major gap repairs (PARALLEL or AFTER Phase 2C)

### Step 6: HMC correctness tests (B5, B6) `[0.3 day]`

**B5. Volume-preservation test:**
```
states = sample_grid(n=1000, region=posterior_support)
volumes_before = estimate_volume(states)
states_after = [leapfrog(θ, p, L, ε) for (θ, p) in states]
volumes_after = estimate_volume(states_after)
assert |volumes_after - volumes_before| / volumes_before < 1e-6
```

**B6. Gradient correctness test:**
```
for i in range(dim):
    fd_i = (U(θ + ε*e_i) - U(θ - ε*e_i)) / (2*ε)
    auto_i = F(θ)[i]
    assert |fd_i - auto_i| < tol * (1 + |fd_i|)
```

Run on toy (Phase 2A) and LGSSM (Phase 2A.5).

**Deliverable:** Two tests passing.

---

### Step 7: Seed policy implementation (B4) `[0.1 day]`

**Policy (from Decision 5):**
- One trajectory-level master seed $ω$, frozen for entire HMC chain
- Seed may depend on initial $θ_0$, not on current $θ$

**Test:**
```
for θ in test_grid:
    u1, f1 = evaluate(θ, seed=12345)
    u2, f2 = evaluate(θ, seed=12345)
    assert u1 == u2 and f1 == f2  # bitwise-identical
    
    # discontinuity test: nearby θ should have O(ε) difference, not O(1)
    u_nearby = evaluate(θ + ε*direction, seed=12345)
    assert |u_nearby - u1| < C * ε  # not O(1) jump
```

**Deliverable:** Policy documented, test passing.

---

### Step 8: ω-ensemble plan (A2) `[0.1 day]`

**Not a code change — planning only.**

Add to Phase 5 (Tier A suite): repeat each model at 3–5 independent ω draws, report ω-variance alongside model-variance.

**Deliverable:** Updated Phase 5 plan with ω-ensemble design.

---

## Part 5: Minor gap repairs (PARALLEL or AFTER Phase 2C)

### Step 9: Operational hardening (E1, E2, E5) `[0.4 day]`

**E1. Checkpointing:**
```python
if iteration % 1000 == 0:
    save_checkpoint(chain_state, diagnostics, iteration)
if checkpoint_exists():
    chain_state, diagnostics, start_iter = load_checkpoint()
```

**E2. Device-level determinism test:**
```python
results1 = run_on_gpu(seed=12345, theta_grid=test_points)
results2 = run_on_gpu(seed=12345, theta_grid=test_points)
assert results1 == results2  # bitwise-identical
```

**E5. Memory-growth preflight:**
```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
for gpu in gpus:
    assert tf.config.experimental.get_memory_growth(gpu)
```

**Deliverable:** Three operational checks added to all GPU runners.

---

### Step 10: Precision regime note (E3) `[0.1 day]`

**If Decision 3 chose Option A (Phase 4 in production float32 TF32):**

Add to Phase 4 result note:
> **Certificate scope:** This validation ran in float32 TF32 on GPU (the production regime). Correctness tests in float64-CPU cover logic and wiring; they do not cover float32-specific numerical pathologies. N-convergence and reference-sampler agreement provide indirect evidence that finite-precision errors are within tolerance, but a float32-specific failure mode could exist outside tested conditions.

**If Decision 3 chose Option B (Phase 4 in float64):**

Add to Phase 4 result note:
> **Certificate scope:** This validation ran in float64 on GPU. Production deployment uses float32 TF32, which has ~30% higher throughput but 10⁴× larger unit roundoff. Results generalize under the assumption that the method is not sensitive to the 10⁴× change in accumulated rounding error.

**Deliverable:** Certificate-scope note added to Phase 4 plan.

---

## Part 6: Sequencing and critical path

```
USER DECISIONS (D1, A1, C3, A3, B4, F1)
  ↓
Step 1: V1, V2 premise checks (0.6 day)
  ↓
Step 2: B1 exact-Kalman rung (0.5 day + 2 GPU-hours)
  ↓
Step 3: C3 tolerance derivation (0.3 day) ← BLOCKS fixture generation
  ↓
Step 4: D1 coverage implementation (0.2 day)
  ↓
Step 5: A1 reference-sampler baseline (0.3 day)
  ↓
Phase 2C: Test suite (2-3 days) ← BLOCKS Phase 2B
  ↓
Phase 2B: Engine unification (6-9 days) ← from unified program
  ↓
[Step 6-10 in parallel or after Phase 2B]
  ↓
Phase 3: Damping derivation + calibration (1-2 days)
  ↓
Phase 4: LGSSM validation (1 day + 12-24 GPU-hours)
  ↓
Phase 5: Tier A suite (2 days + 48 GPU-hours)
```

**Critical path:** Decisions → Steps 1-5 → Phase 2C → Phase 2B → Phase 3 → Phase 4 → Phase 5.

**Parallelizable:** Steps 6-10 can run during or after Phase 2B.

---

## Part 7: Budget summary

| Item | Time | GPU-hours |
|---|---|---|
| User decisions | 1-2 hours | 0 |
| Steps 1-5 (blockers) | 1.9 days | 2 |
| Phase 2C (test suite) | 2-3 days | 0 |
| Steps 6-10 (major+minor) | 1.0 day | 0 |
| **Repair total** | **4.9-5.9 days** | **2** |
| Phase 2B (unification) | 6-9 days | 0 |
| Phase 3 (damping) | 1-2 days | 12-24 |
| Phase 4 (LGSSM) | 1 day | 12-24 |
| Phase 5 (Tier A) | 2 days | 48 |
| **Program total** | **14.9-19.9 days** | **74-98** |

**All repair work is CPU-only test/decision work (no campaigns).**

---

## Part 8: What blocks execution right now

**Cannot proceed without:**

1. **Decision D1** (coverage criterion) — blocks Phase 4 design
2. **Decision A1** (primary criterion) — blocks Phase 4 design and Arm 1 baseline
3. **Decision C3** (tolerance + regime) — blocks fixture generation for Phase 2B Step 3

**Can proceed in parallel once D1/A1/C3 resolved:**
- Steps 1-2 (premise checks, exact-Kalman rung)
- Phase 2C critical triad (enum validation, fixture hash, parameter resolution) — catches the four live blockers

**Recommendation:** User decides D1, A1, C3 now. I execute Steps 1-2 and Phase 2C critical triad in parallel (1.5 days). Then full Phase 2C (2-3 days), then Phase 2B (6-9 days), then Phase 3-5.

---

**END OF GAP REPAIR PLAN**
