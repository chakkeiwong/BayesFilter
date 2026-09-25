# Implementation Plan: Surrogate-Force HMC
## Three-Phase Validation Protocol

**Date:** August 30, 2026  
**Status:** Implementation-ready, reviewed against Codex feedback  
**Estimated effort:** 3–5 days total (Phase 0: 1–3 days, Phase 1: 1 day, Phase 2: 1 day)

---

## Summary

Implement surrogate-force HMC where:
- **Value:** Exact filter output with (λ=1e-5, δ=1e-5)
- **Force:** Damped analytical score with (λ=1e-3, δ=1e-3)
- **Same frozen noise** for both (determinism requirement)

The chain samples the executed finite-particle pseudo-posterior exactly. Score bias affects mixing, not correctness.

---

## Phase 0: Route Identity and Wiring Repair (PREREQUISITE)

**Purpose:** Verify we're testing what we think we're testing.

**Owner disposition:** This phase depends on the state of the existing score-discrepancy audit. Check `../requests/ledh_score_discrepancy_audit_response_*` for blockers.

### Tasks

1. **Resolve audit blockers**
   - [ ] Check audit status: `find docs/requests -name "*score_discrepancy*"`
   - [ ] Address any P0/P1 findings
   - [ ] Document resolution in audit response

2. **Verify route identity**
   - [ ] Confirm this is the canonical LEDH configuration
   - [ ] Match production: dtype=float64, seed policy, Sinkhorn params
   - [ ] No experimental branches active

3. **Analytical-JVP parity**
   - [ ] Run FD vs JVP on current route at multiple θ points
   - [ ] Residual should be <1e-6 (already passing per historical data)
   - [ ] If fails: diagnose before proceeding

4. **Self-consistency diagnostics**
   - [ ] Sinkhorn marginal TV: `||Kₑ diag(u) Kₑᵀ diag(v) - β||₁ < 1e-3`
   - [ ] Contract-E moment residual: actual vs target mean/cov
   - [ ] Cholesky condition number: log it, establish baseline
   - [ ] Dual-cap convergence: iterations, floor hits

5. **Documentation**
   - [ ] Record all baseline metrics in artifact
   - [ ] State configuration hash or commit

**Gate:** Phase 1 starts only after all checks pass and audit is resolved.

**Estimated time:** 1–3 days depending on audit state.

---

## Phase 1: Deterministic Mechanics Check (TOY POTENTIAL)

**Purpose:** Isolate surrogate-force mechanics from filter complexity.

### Fixture

**NOT LGSSM.** Use a simple quadratic potential:
```
U(θ) = 0.5 * θᵀ Σ⁻¹ θ
```
where Σ = diag([1, 4, 9]) (different scales to test anisotropy).

True posterior is N(0, Σ), no particle filter involved.

### Implementation

#### Dual-adapter structure

```python
class ToyPotentialAdapter:
    """Simple quadratic for mechanics testing."""
    def __init__(self, Sigma_inv, damping_scale=1.0):
        self.Sigma_inv = Sigma_inv
        self.damping_scale = damping_scale

    def log_prob_and_grad(self, theta):
        value = -0.5 * theta @ self.Sigma_inv @ theta
        grad_exact = -self.Sigma_inv @ theta
        grad_damped = grad_exact * self.damping_scale
        return value, grad_damped

class DualAdapterToy:
    def __init__(self):
        Sigma_inv = np.diag([1.0, 0.25, 1.0/9.0])
        self.exact = ToyPotentialAdapter(Sigma_inv, damping_scale=1.0)
        self.damped = ToyPotentialAdapter(Sigma_inv, damping_scale=0.1)

    def log_prob_and_grad(self, theta):
        value, _ = self.exact.log_prob_and_grad(theta)
        _, force = self.damped.log_prob_and_grad(theta)
        return value, force
```

### Tests

#### T1: Deterministic repeated calls
```python
theta0 = np.array([1.0, 2.0, 3.0])
v1, f1 = adapter.log_prob_and_grad(theta0)
v2, f2 = adapter.log_prob_and_grad(theta0)
assert np.allclose(v1, v2) and np.allclose(f1, f2)
```

#### T2: Endpoint energy equality
```python
# Run HMC for one trajectory
theta_start, p_start = initial_state()
theta_end, p_end = leapfrog_trajectory(theta_start, p_start, L=10, eps=0.1)

H_start = U(theta_start) + 0.5 * p_start.T @ M_inv @ p_start
H_end = U(theta_end) + 0.5 * p_end.T @ M_inv @ p_end

# Should be equal up to leapfrog discretization error
assert abs(H_start - H_end) < 0.1  # loose bound for toy
```

#### T3: Acceptance across damping ladder
```python
for damping in [1.0, 0.5, 0.1]:  # 1.0=exact, 0.1=heavily damped
    adapter = DualAdapterToy(damping_scale=damping)
    results = run_hmc(adapter, chains=2, warmup=500, samples=500)
    print(f"Damping={damping:.1f}: acceptance={results.acceptance:.3f}")
```

**Expected:**
- Damping=1.0: acceptance ≈ 0.7–0.8 (exact gradient, optimal)
- Damping=0.5: acceptance ≈ 0.5–0.6 (moderate degradation)
- Damping=0.1: acceptance ≈ 0.3–0.5 (heavy degradation, but still mixing)

**Veto:** If damping=0.1 gives acceptance <0.2, the force is too poor.

#### T4: Force-norm diagnostic
```python
theta = np.array([1.0, 2.0, 3.0])
_, f_exact = exact_adapter.log_prob_and_grad(theta)
_, f_damped = damped_adapter.log_prob_and_grad(theta)

print(f"||F_exact|| = {np.linalg.norm(f_exact):.4f}")
print(f"||F_damped|| = {np.linalg.norm(f_damped):.4f}")
assert np.linalg.norm(f_damped) < np.linalg.norm(f_exact)
```

#### T5: Posterior recovery
```python
# True posterior is N(0, Σ)
results = run_hmc(adapter, chains=4, warmup=1000, samples=2000)
mean_recovered = results.samples.mean(axis=0)
cov_recovered = np.cov(results.samples.T)

assert np.allclose(mean_recovered, 0, atol=0.1)
assert np.allclose(np.diag(cov_recovered), [1, 4, 9], rtol=0.2)
```

### Deliverables

- [ ] `toy_potential_surrogate_force.py` — standalone script
- [ ] Test results logged: all 5 tests pass
- [ ] Acceptance vs damping table
- [ ] Artifact with Phase 1 metrics

**Gate:** Proceed to Phase 2 only if:
- T1–T4 pass
- T5 acceptance at damping=0.1 is >0.3
- Posterior mean within 0.1 of true mean

**Estimated time:** 1 day.

---

## Phase 2: Short Diagnostic Arm on LGSSM (BOUNDED CLAIM)

**Purpose:** Test surrogate-force on the actual particle filter with real bias.

### Fixture

- **Model:** d=3 T=50 diagonal LGSSM
- **True θ:** [0.72, 0.55, 0.35, 0.35, 0.45]
- **Observation:** seed 81100 (same as historical)
- **Exact reference:** Kalman value=-145.434, score φ₁=-6.217

### Implementation

#### Dual-adapter for LEDH

```python
class DualAdapterLEDH:
    def __init__(self, model, observations, N, seed_base):
        self.model = model
        self.observations = observations
        self.N = N
        self.seed_base = seed_base

        # Two configurations: exact and damped
        self.config_exact = ResetConfig(
            epsilon=1.0, sinkhorn_steps=8,
            ridge=1e-5, damping=1e-5
        )
        self.config_damped = ResetConfig(
            epsilon=1.0, sinkhorn_steps=8,
            ridge=1e-3, damping=1e-3
        )

    def log_prob_and_grad(self, theta):
        # Generate frozen noise ONCE per theta
        rng = self._make_rng(theta)
        initial = rng.standard_normal((self.N, self.model.d))
        noises = rng.standard_normal((self.model.T, self.N, self.model.d))
        covs = np.stack([np.eye(self.model.d)] * self.N)

        # Value from exact config
        value = canonical_value_only(
            self.model, theta, initial, covs, noises,
            self.observations, self.config_exact
        )

        # Score from damped config (SAME noise)
        _, score = canonical_value_and_score(
            self.model, theta, initial, covs, noises,
            self.observations, self.config_damped
        )

        return value, score

    def _make_rng(self, theta):
        # Hash theta to get deterministic seed
        seed = hash((self.seed_base, tuple(theta))) % (2**31)
        return np.random.default_rng(seed)
```

**Critical:** The `_make_rng` must be deterministic in θ. Same θ → same noise → same (value, score).

### Three-Arm Comparison

**Arm A (baseline):** Exact score as force
- Config: λ=1e-5, δ=1e-5 for both value and score
- This is the current route

**Arm B (surrogate):** Damped score as force
- Value: λ=1e-5, δ=1e-5
- Score: λ=1e-3, δ=1e-3

**Arm C (fallback):** Intermediate damping
- Value: λ=1e-5, δ=1e-5
- Score: λ=1e-4, δ=1e-4
- Only run if Arm B fails

### HMC Configuration

```python
for arm_name, adapter in [("exact", arm_a), ("damped", arm_b)]:
    results = tfp.mcmc.sample_chain(
        num_results=1000,
        num_burnin_steps=1000,
        current_state=theta_init,
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=adapter.log_prob_and_grad,
            step_size=0.01,
            num_leapfrog_steps=10
        ),
        num_chains=4,
        trace_fn=lambda _, pkr: {
            'accept': pkr.is_accepted,
            'step_size': pkr.step_size,
        }
    )
    # Record metrics...
```

### Metrics

For each arm, measure:

1. **Acceptance rate**
   - Warmup phase: first 500 steps
   - Sampling phase: last 500 steps
   - By chain (detect divergences)

2. **ESS per gradient**
   ```python
   ess = tfp.mcmc.effective_sample_size(samples)
   ess_per_grad = ess / (warmup + sampling)
   ```
   Compute for each θ component, report min/mean/max

3. **Posterior coverage**
   ```python
   for p in range(5):
       mean = samples[:, :, p].mean()
       std = samples[:, :, p].std()
       ci_low, ci_high = mean - 1.96*std, mean + 1.96*std
       covers = ci_low <= theta_true[p] <= ci_high
   ```

4. **Posterior mean shift**
   ```python
   mean_exact = samples_arm_a.mean(axis=(0,1))
   mean_damped = samples_arm_b.mean(axis=(0,1))
   shift = np.linalg.norm(mean_damped - mean_exact)
   ```

5. **Diagnostics**
   - Rhat < 1.01 for all parameters
   - No divergences
   - Trace plots: visual mixing check

### Success Criteria

**Pass if ALL hold:**
- Acceptance in sampling phase: >0.3
- ESS/grad (min across components): >0.3 × Arm A baseline
- Posterior coverage: all 5 parameters covered by 95% CI
- Mean shift: ≤ 0.18 (= 2× the value bias of 0.09 on a log-like of ~145)
- Rhat < 1.01, no divergences

**Hard veto:**
- Mean shift > 0.18: pseudo-posterior too far from true posterior
- Acceptance < 0.2: force too poor, chain barely moving
- Any parameter's 95% CI excludes truth: bias problem

### What Can Be Claimed (If Passes)

✓ **Can claim:**
- "Surrogate-force HMC with damped score produces deterministic mechanics and acceptable mixing on the LGSSM diagnostic fixture."
- "The chain samples the executed pseudo-posterior, which differs from the true posterior by the measured value bias of 0.01–0.09%."
- "Acceptance and ESS/grad are within acceptable bounds relative to exact-score baseline."

✗ **Cannot claim:**
- "HMC-ready" (not tested on DSGE)
- "Removes score bias" (score bias still present, just moved out of correctness)
- "Exact posterior inference" (pseudo-posterior ≠ true posterior)
- Any statement about convergence on untested models

### Deliverables

- [ ] `surrogate_force_lgssm_three_arm.py` — runner
- [ ] Artifact with metrics table, trace plots, coverage report
- [ ] Decision: pass/fail against success criteria
- [ ] If fail: diagnosis (which criterion? intermediate damping?)

**Estimated time:** 1 day (implementation + 4 chains × 2K steps ≈ 2–3 hours wall).

---

## Phase 3: Extended Validation (FUTURE, NOT IN THIS PLAN)

If Phase 2 passes and you want to deploy on DSGE:

1. Run on a cheap DSGE fixture (d=5–10, T=50)
2. Cannot measure posterior coverage (no oracle), only mechanics:
   - Acceptance, ESS, Rhat, divergences
   - Trace plots, posterior predictive checks
3. Value-bias profiling: does the pseudo-posterior shift make sense?

**This is NOT part of the current 3–5 day plan.** Phase 2 establishes bounded mechanics; deployment is a separate decision.

---

## Implementation Checklist

### Code Components

- [ ] `DualAdapterToy` (Phase 1)
- [ ] `DualAdapterLEDH` (Phase 2)
- [ ] `_make_rng(theta)` — deterministic seed from θ hash
- [ ] Verification: same θ → same noise → same (value, score)
- [ ] TFP HMC wrapper using `reviewed_value_score_target_fn`
- [ ] Metrics collectors: acceptance, ESS, coverage, shift, Rhat

### Runners

- [ ] `toy_potential_surrogate_force.py` (Phase 1, ~150 lines)
- [ ] `surrogate_force_lgssm_three_arm.py` (Phase 2, ~300 lines)

### Artifacts

- [ ] Phase 0: route-identity artifact (baseline metrics)
- [ ] Phase 1: toy-potential artifact (5 tests + acceptance table)
- [ ] Phase 2: three-arm artifact (metrics table, plots, coverage)

### Documentation

- [ ] Update this plan with Phase 0 findings
- [ ] Record all vetoes/failures with diagnosis
- [ ] Final status: pass → bounded claim, fail → surrogate-force not viable

---

## Risk Registry

| Risk | Mitigation | Veto condition |
|---|---|---|
| Phase 0 blockers unknown | Check audit state first | Cannot proceed until resolved |
| Toy potential too simple | Test anisotropic Σ, check posterior recovery | Fail T5 → diagnose |
| Damping too aggressive | Three-arm with intermediate fallback | Acceptance <0.2 → try λ=1e-4 |
| Pseudo-posterior shift | Measure against 2× value bias | Shift >0.18 → investigate value bias |
| Single-path bias | LGSSM uses one obs path | Caveat in claims, flag for future |
| Rhat/divergence | Longer warmup, smaller step size | Rhat >1.05 → tune HMC params |

---

## Timeline

- **Day 1–3:** Phase 0 (route repair, depends on audit state)
- **Day 4:** Phase 1 (toy potential, all 5 tests)
- **Day 5:** Phase 2 (LGSSM three-arm, 2–3 hours run + analysis)

**Total: 3–5 days** (3 if Phase 0 is clean, 5 if audit needs work)

---

## Execution Decision

This plan is ready to execute. Should I:

1. **Proceed with implementation** — write the runners, execute Phase 1–2
2. **Revise plan first** — address any concerns you have
3. **Defer** — close out the investigation with the LaTeX doc only

Which path?
