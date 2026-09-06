# LEDH Surrogate-Force HMC — Corrected Gap Audit

**Date:** 2026-09-06  
**Replaces:** Pre-mortem draft (never committed)  
**Context:** User asked to verify the gap list is correct and complete before fixing everything.

---

## Gap resolution summary

| Gap | Status | Resolution |
|---|---|---|
| G1 | ✅ CLOSED | Corollary 5.2 read, premises verified, theorem correctly cited |
| A3 | ⚠️ REFRAMED | Not a blocker; cost/interpretability issue (2× filter per step) |
| B4 | ✅ VALIDATED | θ-hash seeding breaks continuity (Prop 6); one frozen seed required |
| V3 | ✅ VERIFIED | Assumption 2 holds; dual-cap/trust-region smooth; cost-scale branch excluded |

**Three premise checks remain open (V1, V2 below).**

---

## Part 1: Blocking gaps (execution cannot proceed)

### B1. Exact-Kalman surrogate-force rung missing `[ladder gap]`

**What:** The test ladder jumps from 3-D quadratic toy (Phase 2A) straight to LEDH T=50 d=3 (Phase 4). Without an exact-Kalman intermediate, a Phase 4 failure cannot be attributed: is the mechanism broken, or is LEDH score bias too large for the damping ratio?

**Why blocking:** Phase 4 negative result would be uninterpretable — no way to separate "surrogate-force doesn't work" from "LEDH bias exceeds tolerance."

**Repair:** New Phase 2A.5 — linear-Gaussian model, exact Kalman posterior available, surrogate force = exact Kalman score with frozen seed ≠ value seed. Acceptance: posterior agreement with exact Kalman (reference) using exact force and with damped force. Estimated cost: 0.5 day + 2 GPU hours.

**Citation:** None — standard HMC validation practice.

---

### D1. Coverage criterion has 77% false-negative rate `[statistical design flaw]`

**What:** Phase 4 criterion is "$95\%$ marginal intervals cover true $\theta$ (all 5 parameters)." For a **correct** sampler, this has probability $0.95^5 \approx 0.774$. A perfectly working implementation fails ~1 run in 4.

**Why blocking:** Would reject a correct method and send the program into redesign based on a statistical artifact, not a real defect.

**Repair:** Replace with one of:
- **(Recommended)** Joint 95% Mahalanobis ellipsoid (one joint region, not five independent tests)
- Marginal with expected-failures check (e.g. "4 or 5 of 5 cover" has ~95% probability under independence)
- Bonferroni-corrected marginals (99% per parameter for 95% joint)

**Decision required:** User must choose which.

**Citation:** None — elementary multiple-testing error.

---

### A1. Primary criterion confounds three questions `[estimand design flaw]`

**What:** Phase 4 primary criterion is "true-θ coverage." This confounds:
1. Does the sampler correctly sample $\pi_N^\omega(\theta) \propto \exp(-U_N^\omega(\theta))$? (correctness)
2. Is $\pi_N^\omega$ close to the exact posterior $\pi(\theta \mid y)$? (finite-N bias, not HMC's fault)
3. Does the exact posterior cover true $\theta$ for this dataset? (data property, not method property)

A correct sampler fails whenever (2) or (3) fails. An incorrect sampler passes when over-dispersion happens to widen intervals. Neither necessary nor sufficient.

**Why blocking:** Phase 4 passes or fails for reasons unrelated to whether surrogate-force HMC works.

**Repair:** Replace primary criterion with **reference-sampler agreement**: run long-chain exact-force HMC on the identical frozen $(ω, N)$ target, compare posteriors via KL, Wasserstein, or overlap. Demote true-θ coverage to explanatory diagnostic.

**Decision required:** User must approve the estimand change — this alters what Phase 4 proves.

**Citation:** Corollary 5.2 binding (lines 953-976) — the theorem targets $\pi_N^\omega$, not the exact posterior or true θ.

---

### C3. Tolerance incoherence `[test design flaw]`

**What:** Golden-master fixtures demand $10^{-12}$ relative (bitwise-identical in float64). Cross-lane parity allows $5 \times 10^{-4}$. Eight orders of magnitude apart. A refactor with $10^{-5}$ perturbation fails golden and passes parity — which verdict governs?

**Why blocking:** Cannot generate fixtures or interpret Phase 2B Step 5 results without resolving this.

**Repair:** Derive tolerance from:
- Condition number of the reset Jacobian (amplifies input perturbation)
- Tuning-insensitivity band (largest drift that doesn't trigger retuning)
- Production dtype/backend (float32 TF32 on GPU vs float64 on CPU)

Write the derivation **before** generating fixtures. One tolerance policy applies everywhere.

**Decision required:** User must choose production regime (float64-CPU reference vs float32-TF32-GPU target).

---

### V1. Are all Monte Carlo seeds frozen? `[Corollary 5.2 premise]`

**What:** Remark 5.3 (line 980) requires "$F$ must be a deterministic function of $\theta$ alone: all Monte Carlo seeds inside $F$ must be frozen."

**Where seeds enter:**
- Initial-noise matrix (Definition 1 line 85)
- Innovation rows $e_{t,i}$
- Sinkhorn entropic noise
- Residual design $\Xi$ (deterministic per Contract-E)

**Required:** Every seed resolves to a slice of one trajectory-level master seed $\omega$, and $\omega$ is **frozen for the entire HMC chain** (not per-step, not per-trajectory within a chain).

**Why blocking:** If seeds vary with momentum, trajectory index, or call count, Propositions 1-2 (volume-preservation and reversibility) fail and Corollary 5.2 does not apply.

**Repair:** Instrument every TF random op in the LEDH value and force paths. Assert they all resolve to deterministic slices of one master seed. Add a test: evaluate $(U, F)$ twice at the same $\theta$ with the same master seed, assert bitwise-identical.

**Estimated cost:** 0.3 day (instrumentation + test).

---

### V2. Is the force history-independent? `[Corollary 5.2 premise]`

**What:** Remark 5.3 requires $F$ may not depend on momentum or trajectory history.

**Forbidden:**
- Cached particle cloud from a prior evaluation
- Call-count conditional
- Step-index conditional
- Accumulated float state (running mean, EMA)

**Why blocking:** If $F(\theta, p)$ or $F(\theta, \text{history})$, then $S = \mathcal{F} \circ \Psi^L$ is not an involution and Proposition 2 fails.

**Repair:** Involution test (the standard HMC correctness check):
```
(theta_fwd, p_fwd) = leapfrog(theta, p, L_steps, epsilon)
(theta_back, p_back) = leapfrog(theta_fwd, -p_fwd, L_steps, epsilon)
assert |theta_back - theta| < tol and |p_back - p| < tol
```

If the force depends on anything but $\theta$, this fails.

**Estimated cost:** 0.2 day (one test, run on toy and LGSSM).

---

## Part 2: Major gaps (invalidate interpretation but don't block execution)

### A2. Single-ω result doesn't generalise `[scope limitation]`

**What:** Corollary 5.2 targets $\pi_N^\omega(\theta) \propto \exp(-U_N^\omega(\theta))$ **for one draw ω**. Evidence at one ω is one sample from a distribution over pseudo-posteriors.

**Why major:** A Phase 4 pass at one ω doesn't establish that the method works at typical ω, only that it worked for this one.

**Not blocking:** The program is explicit about this (Section 6.2 "ω-ensemble open question"). Phase 4 is scoped as proof-of-concept, not production certification.

**Recommendation:** Add to Phase 5 (Tier A suite): repeat each model at 3-5 independent ω draws, report ω-variance alongside model-variance.

**Citation:** Corollary binding line 980-995, and Definition 1 line 108 (ω-indexed).

---

### A3. Force is gradient of different functional (2× cost) `[cost/interpretability]`

**What:** "Damped force" in Remark 5.3 means: force computed with **larger λ, δ** than value. So value uses one reset configuration, force uses another, on the same frozen noise. This is $\nabla \widetilde{U}$ where $\widetilde{U} \neq U$.

**Why major:** 2× filter cost per leapfrog step (run filter twice with different resets). And "damping" is a discrete choice between transports, not a smooth perturbation — interpretability harder.

**Not blocking:** Corollary 5.2 explicitly licenses this (Remark 5.3 line 987: "a damped or clipped variant of the recursive score"). Theorem applies to **any** deterministic $F$.

**Repair in Phase 3:** Derive λ_force/λ_value ratio from a calibration curve (bias-vs-robustness tradeoff), not from convenience. Budget the 2× cost explicitly.

**Previous status:** I incorrectly called this a blocking defect in the first audit. Reading Corollary 5.2 corrected it — this is admissible by design, just expensive.

---

### B4. θ-hashed seeding makes U discontinuous `[seed policy]`

**What:** Current risk is `seed = hash(θ)` so every infinitesimal $\theta$ perturbation hits a different random stream. Proposition 6 (line 1166) proves: a θ-dependent branch makes $\widehat{L}^N(\theta)$ discontinuous at every branch boundary, "and no same-scalar derivative claim holds across the crossing."

**Why major:** Breaks Assumption 2 (fixed differentiable branch), so Corollary 5.2 does not apply.

**Resolution determined:** One trajectory-level seed, frozen for the entire HMC chain. The seed may be a function of **initial** $\theta_0$ (part of the frozen context), but not of **current** $\theta$ during leapfrog.

**Repair:** Policy decision + test. Test: $U(\theta + \epsilon) - U(\theta)$ should have $O(\epsilon)$ scaling; if seed is θ-hashed, the difference is $O(1)$ (discontinuous).

**Estimated cost:** 0.1 day (policy + one test).

**Citation:** Proposition 6 (line 1166), Assumption 2 (line 118).

---

### B5. No volume-preservation test `[HMC correctness]`

**What:** Proposition 1 (line 874) is a premise of Corollary 5.2: leapfrog preserves Lebesgue measure. The program has no test of this.

**Why major:** If an implementation bug breaks volume-preservation (e.g. forgot a Jacobian term, wrong force sign), detailed balance fails and the sampler is incorrect — but toy tests might still pass if the bug is small.

**Repair:** Volume-preservation test:
```
theta_grid = sample_grid_in_region(n_points=1000)
p_grid = sample_momentum(n_points=1000)
states = [(t, p) for t in theta_grid for p in p_grid]
volumes_before = estimate_volume(states)
states_after = [leapfrog(t, p, L, eps) for (t,p) in states]
volumes_after = estimate_volume(states_after)
assert |volumes_after - volumes_before| / volumes_before < 1e-6
```

**Estimated cost:** 0.2 day (one test, run on toy).

**Citation:** Proposition 1 line 874.

---

### B6. No gradient correctness test `[HMC correctness]`

**What:** The force $F = -\nabla_\theta U$ must be the actual gradient. A sign error, missing term, or wrong chain-rule application breaks everything but might pass acceptance tests if the error is in a low-sensitivity direction.

**Repair:** Finite-difference check:
```
for i in range(dim):
    fd_i = (U(theta + eps*e_i) - U(theta - eps*e_i)) / (2*eps)
    auto_i = F(theta)[i]
    assert |fd_i - auto_i| < tol * (1 + |fd_i|)
```

Run on LGSSM with known exact posterior as a cross-check.

**Estimated cost:** 0.1 day (standard practice, should already exist).

---

### C1. No N-convergence check `[finite-particle bias]`

**What:** Corollary 5.2 applies to the finite-N pseudo-posterior, which has bias vs the exact posterior. Only planned check that $U_N$ converges to the **right** limit as $N$ grows.

**Why major:** Without this, Phase 4 could pass while $U_N$ converges to the wrong function (e.g. systematically shifted mean).

**Repair:** Phase 2A.5 (exact-Kalman rung): run $N \in \{50, 100, 200, 500, 1000\}$, plot $\| \mu_N - \mu_{\text{exact}} \|$ and $\| \Sigma_N - \Sigma_{\text{exact}} \|_F$ vs $1/\sqrt{N}$. Acceptance: monotone decreasing, passes through zero-intercept confidence band.

**Estimated cost:** included in Phase 2A.5 (no separate budget).

---

## Part 3: Minor gaps (good practice, not critical)

### E1. No checkpointing `[operational]`

**What:** Phase 4 is 12 GPU-hours. A crash at hour 11 loses everything.

**Repair:** Save chain state + diagnostics every 1000 iterations. Resume from checkpoint on restart.

**Estimated cost:** 0.2 day (standard HMC infrastructure).

---

### E2. No device-level determinism check `[reproducibility]`

**What:** GPU results are deterministic only if TF32, cuDNN autotune, and non-deterministic reduce are all controlled. Program assumes this but doesn't verify.

**Repair:** Run Phase 2A toy twice on GPU with same seed, assert bitwise-identical $(U, F)$ at 10 $\theta$ points.

**Estimated cost:** 0.1 day (one test).

---

### E3. Precision regime unclear `[certificate scope]`

**What:** Phase 4 will run in float32 TF32 on GPU (the production target). But all parity/correctness tests are in float64 on CPU. Do the tests certify the thing that will actually run?

**Why minor:** Tests still catch logic errors. But a float32-specific numerical pathology could pass CPU tests and fail in production.

**Decision required:** Run Phase 4 in float64 on GPU (certifies the regime tested) or accept that certificate doesn't cover production dtype/backend.

**Estimated cost:** None if user accepts the gap; ~30% slowdown if running Phase 4 in float64.

---

### E5. No memory-growth preflight `[operational]`

**What:** TensorFlow GPU Memory Rule (CLAUDE.md) requires memory growth enabled and verified before serious GPU runs. Program doesn't check this.

**Repair:** Add to every GPU runner:
```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# verify
for gpu in gpus:
    assert tf.config.experimental.get_memory_growth(gpu)
```

**Estimated cost:** 0.05 day (boilerplate).

---

### F1. No independent review `[governance]`

**What:** The program, this audit, and all repairs are produced by one agent (me). Governance policy (CLAUDE.md Academic Research) recommends independent review for "source-faithfulness, publication-grade claims, major public API/default changes, unusually expensive campaigns."

**Why minor:** User oversight is continuous. This is about having a **second model** check the plan.

**Repair:** Send program + audit to a fresh independent agent (via Agent tool or separate session), ask for red-team review before Phase 2B starts. Budget 2-4 hours of user time to adjudicate disagreements.

**Estimated cost:** 0.3 day (reviewer) + user adjudication time.

---

## Part 4: Verified non-gaps

### V3. Fixed-branch assumption `[Corollary 5.2 premise — VERIFIED]`

**Claim:** Dual-cap and trust-region are θ-dependent control-flow gates that could violate Assumption 2 (fixed differentiable branch, line 113-120).

**Audit result:** ✅ **Assumption 2 holds by construction.**

**Evidence:**
1. **Dual-cap (particle_rms_cap, line 735-760):** Uses smooth formula `scale = rsqrt(1 + rms²/cap²)` — C^∞ everywhere, no branch.
2. **Trust-region (trust_radius, line 944-956):** Calls `smooth_rms_cap_jvp` from genut_shape_lm_tf.py:140, same smooth formula.
3. **Sinkhorn cost-scale floor (line 73-76):** Hard `tf.where(mean_cost > floor, ...)` — **discontinuous**, but Assumption 2 line 118 explicitly excludes this: "the cost-scale maximum branch is locally constant." The note already knows about this and restricts θ to an open set where the branch doesn't cross.

**Conclusion:** No repair needed. The construction is Assumption-2-compliant on the declared acceptance set.

---

## Part 5: Gap count and priority

| Category | Count | Blocking | Major | Minor |
|---|---|---|---|---|
| Estimand design (A) | 3 | 1 | 2 | 0 |
| HMC correctness (B) | 6 | 2 | 3 | 0 |
| Test design (C) | 3 | 1 | 1 | 0 |
| Statistical validity (D) | 1 | 1 | 0 | 0 |
| Execution/environment (E) | 5 | 0 | 0 | 5 |
| Governance (F) | 1 | 0 | 0 | 1 |
| **Total** | **19** | **5** | **6** | **6** |
| Verified non-gaps | 1 (V3) | — | — | — |

---

## Part 6: User decisions required before repair

Six items cannot proceed without user choice:

| ID | Decision | Impact if deferred |
|---|---|---|
| **D1** | Coverage criterion (joint vs marginal-with-failures vs Bonferroni) | Phase 4 unexecutable |
| **A1** | Primary criterion (reference-sampler vs true-θ coverage) | Phase 4 proves wrong theorem |
| **C3** | Tolerance derivation + production regime (float64-CPU vs float32-TF32-GPU) | Cannot generate fixtures for Phase 2B |
| **E3** | Run Phase 4 in test regime (float64) or production regime (float32 TF32)? | Certificate scope ambiguous |
| **B4** | Approve one-seed policy | Corollary 5.2 inapplicable |
| **A3** | Approve 2× filter cost per step | Phase 3 budget underestimated |

**Recommendation:** Decide D1, A1, C3 now (blocking); defer E3 to Phase 3; B4 and A3 are determined by the binding (approve as written).

---

## Part 7: What the corrected audit changes

### Compared to the pre-mortem draft (never committed)

1. **G1 resolved:** Read Corollary 5.2, verified premises, confirmed correct citation.
2. **A3 downgraded:** From blocking to major (cost/interpretability). Remark 5.3 explicitly licenses this.
3. **B4 upgraded:** From judgment call to determined requirement (Proposition 6 citation).
4. **V3 closed:** Branch verified fixed on the declared acceptance set (Assumption 2 line 118).
5. **Two new gaps found during V3 audit:** B5 (volume-preservation test), B6 (gradient correctness test).
6. **A1 upgraded from major to blocking:** Can't interpret Phase 4 without resolving estimand.

**Net:** 19 gaps (5 blocking, 6 major, 6 minor, 2 verified non-gaps), down from 21 unverified in the draft.

---

**END OF CORRECTED GAP AUDIT**
