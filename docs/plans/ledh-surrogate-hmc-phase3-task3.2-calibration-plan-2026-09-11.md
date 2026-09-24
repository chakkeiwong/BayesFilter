# LEDH Surrogate-Force HMC: Phase 3 Task 3.2 — Damping Calibration Plan

**Date:** 2026-09-11  
**Authority:** Phase 3 Task 3.2 of ledh-surrogate-hmc-unified-program-2026-09-06.md  
**Status:** DRAFT

---

## Question

What damping ratio maintains acceptable mixing (acceptance ≥ 0.15, ESS/grad ≥ 0.3× baseline) on LGSSM d=3 T=50?

---

## Method

### Test Model

**LGSSM d=3 T=50** with frozen observations (seed 81100, φ=[0.72, 0.55, 0.35], q=0.35, r=0.45, 3×3 observation matrix)

**Fixture:** `bayesfilter/highdim/ledh_canonical_neutra_targets_tf._lgssm_frozen_observations()`

**True θ:** 5 parameters (3 AR coefficients, 2 covariances)

### Damping Sweep

Test 4 damping ratios using dual-parameter adapter:

| Arm | Damping Ratio | reset_ridge | correction_lm_damping | Description |
|---|---|---|---|---|
| 0 | 1× | 1e-5 | 1e-2 | Baseline (exact = biased) |
| 1 | 10× | 1e-4 | 1e-1 | Mild damping |
| 2 | 100× | 1e-3 | 1.0 | Target (per program) |
| 3 | 1000× | 1e-2 | 10.0 | Extreme damping |

**Shared parameters:** All arms use identical settings except damping:
- N = 1008 particles
- substeps = 8
- reset_policy = "contract_e"
- reset_epsilon = 2.0
- Sinkhorn/balance steps = 8
- Dual-cap: diagonal 4 steps @ 0.2, pairwise 4 steps @ 0.02
- Trust radius = 0.5
- Annealed stages = 1

### HMC Configuration

**Per arm:**
- 2 chains × 500 warmup × 500 sampling
- Step size = 0.01 (conservative, same for all arms)
- Mass matrix = identity (no preconditioning)
- Seed = 97701

**Why short chains:** Calibration is a screening run, not a convergence study. 500 samples sufficient to measure acceptance and ESS trends.

### Primary Metrics

1. **Acceptance rate** (promotion criterion)
   - Target: ≥ 0.15
   - Below 0.15 → mixing broken, damping too aggressive

2. **ESS per gradient** (promotion criterion)
   - Target: ≥ 0.3× baseline (Arm 0)
   - Below 0.3× → efficiency loss too large

### Veto Diagnostics

- Any divergence (integration unstable)
- R-hat > 1.1 (chains not converged)
- Non-finite values (filter broke)

### Explanatory Diagnostics

- Mean score L2 norm per arm (how much does damping reduce force magnitude?)
- Score bias vs exact Kalman (if available)
- Runtime per arm

---

## Implementation

### File Structure

**Runner:** `docs/benchmarks/ledh_surrogate_hmc_phase3_damping_calibration.py`

**Artifact:** `docs/benchmarks/artifacts/ledh-damping-calibration-lgssm-t50-20260911/`
- `result.json` (acceptance, ESS, metrics per arm)
- `manifest.json` (git commit, conda env, seeds, wall time)
- `chains/` (optional: saved posterior samples if needed)

### Pseudocode

```python
# Load LGSSM fixture
observations = _lgssm_frozen_observations()
model, theta_true, init_states, init_cov, noises, obs = ...

# Shared parameters
shared_params = {...}

# For each damping ratio
for ratio in [1.0, 10.0, 100.0, 1000.0]:
    # Create dual-parameter target
    target = make_dual_parameter_target(
        model, init_states, init_cov, noises, obs,
        damping_ratio=ratio,
        **shared_params
    )
    
    # Wrap for TFP HMC
    def neg_log_prob(theta):
        value, _ = target(theta)
        return -value
    
    def grad_neg_log_prob(theta):
        _, score = target(theta)
        return -score
    
    # Run HMC
    samples, accept_rate, ess = run_tfp_hmc(
        neg_log_prob, grad_neg_log_prob,
        num_chains=2, num_warmup=500, num_samples=500,
        step_size=0.01
    )
    
    # Record results
    results[ratio] = {
        'acceptance_rate': accept_rate,
        'ess_per_param': ess,
        'ess_per_gradient': ess / (500 + 500),  # warmup + sampling
        'divergences': count_divergences,
        'rhat': compute_rhat(samples),
    }

# Select optimal damping
passing = [r for r in results if 
           results[r]['acceptance_rate'] >= 0.15 and
           results[r]['ess_per_gradient'] >= 0.3 * results[1.0]['ess_per_gradient']]

if passing:
    selected = max(passing)  # Coarsest passing damping
else:
    selected = None  # No damping passes criteria
```

---

## Success Criteria

**Primary:** At least one damping ratio passes both criteria:
- Acceptance ≥ 0.15
- ESS/grad ≥ 0.3× baseline

**If all pass:** Select coarsest (largest damping ratio)

**If none pass:** Report findings, recommend:
- Option A: Relax criteria (e.g., 0.1 acceptance floor)
- Option B: Abandon surrogate-force approach for this model
- Option C: Try different parameter (e.g., only damp `reset_ridge`, not `correction_lm_damping`)

---

## Exit Criterion

One artifact documenting:
- Acceptance and ESS/grad per damping ratio
- Selected damping (or "none pass")
- Decision for Phase 4: proceed with selected damping, or stop

---

## Budget

**Implementation:** 0.3 day (runner + artifact serialization)  
**Execution:** 4 GPU-hours (4 arms × 2 chains × 1000 steps × ~30s per 1000 steps)  
**Analysis:** 0.2 day (parse results, make decision)

**Total:** 0.5 day + 4 GPU-hours

---

## Risks

**Risk 1: All dampings fail acceptance floor**
- Likely if 100× is too aggressive for LGSSM T=50
- Mitigation: Try intermediate ratios (30×, 50×)

**Risk 2: Baseline (1×) already has low acceptance**
- Indicates tuning issue or model difficulty
- Check: Does exact HMC work on this fixture?

**Risk 3: Adapter implementation bug**
- Value or score computed incorrectly
- Mitigation: Add manual spot-check before full sweep

---

## Spot-Check Before Full Sweep

Before running 4-arm calibration, verify adapter works:

```python
# Load fixture
model, theta_true, init_states, init_cov, noises, obs = ...

# Create 1× target (exact = biased)
target_1x = make_dual_parameter_target(
    model, ..., damping_ratio=1.0, **shared_params
)

# Create 100× target
target_100x = make_dual_parameter_target(
    model, ..., damping_ratio=100.0, **shared_params
)

# Evaluate at theta_true
value_1x, score_1x = target_1x(theta_true)
value_100x, score_100x = target_100x(theta_true)

# Checks:
assert tf.math.is_finite(value_1x)
assert tf.math.is_finite(value_100x)
assert tf.reduce_all(tf.math.is_finite(score_1x))
assert tf.reduce_all(tf.math.is_finite(score_100x))

# Values should be identical (both use exact params for value)
np.testing.assert_allclose(value_1x, value_1x, rtol=1e-12)

# Scores should differ (100× uses coarser params)
score_diff = tf.norm(score_100x - score_1x).numpy()
print(f"Score L2 difference: {score_diff}")
assert score_diff > 1e-6, "Scores too similar, damping may not be working"

print("✓ Spot-check passed, adapter works correctly")
```

Run this before full calibration. If checks fail, debug adapter before proceeding.

---

## Status

**NEXT:** Implement runner and spot-check.
