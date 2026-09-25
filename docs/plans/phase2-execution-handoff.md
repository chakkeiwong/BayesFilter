# Phase 2 Execution Handoff

**Date:** 2026-09-07  
**Authority:** ledh-surrogate-hmc-executable-master-program-2026-09-07.md  
**Estimated Time:** 0.5 day  
**GPU Required:** Yes (0.5 GPU-hour)

---

## Goal

Test surrogate-force HMC mechanism on simple 3D quadratic potential where exact gradient is analytical.

---

## Prerequisites

- Phase 0 complete (tolerance, seed policy, coverage)
- Phase 1 complete (diagnostics pass, GPU memory growth verified)

---

## Scientific Question

**Does damped-force HMC sample correctly when the force is a damped version of the exact gradient?**

This isolates the **mechanism** (damped force) from **filter complexity** (LEDH score estimation).

---

## Test Design

### Setup

**Potential:** 3D quadratic
```python
def log_prob(theta):
    # theta: [3]
    # Simple quadratic: -0.5 * theta^T Q theta
    # Q = diag([1.0, 4.0, 9.0])
    return -0.5 * tf.reduce_sum(theta**2 * tf.constant([1.0, 4.0, 9.0]))

def exact_gradient(theta):
    # Analytical gradient
    return -theta * tf.constant([1.0, 4.0, 9.0])

def damped_gradient(theta, epsilon=0.01):
    # Damped version: g_damped = g_exact / (1 + epsilon)
    # Simulates OT gradient damping
    return exact_gradient(theta) / (1.0 + epsilon)
```

**True posterior:** N(0, Q^{-1}) where Q = diag([1, 4, 9])

**True covariance:** diag([1.0, 0.25, 0.111...])

---

### Two Arms

**Arm 1: Exact-force HMC**
- Force = exact_gradient(theta)
- Should sample N(0, Q^{-1}) exactly (up to MCMC error)

**Arm 2: Damped-force HMC**
- Force = damped_gradient(theta, epsilon=0.01)
- Should sample N(0, Q^{-1}) exactly IF Corollary 5.2 holds
- (Damping is deterministic transformation of gradient)

**Test criterion:** W₂(posterior₁, posterior₂) < tolerance

---

## Implementation

### File to Create

`scripts/phase2_toy_potential_test.py`

**Content (~150 lines):**

```python
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import json
from pathlib import Path

# Import tolerance and coverage from Phase 0
from bayesfilter.inference.tolerance_derivation import derive_parity_tolerance
from bayesfilter.inference.coverage import joint_mahalanobis_coverage

def log_prob(theta):
    return -0.5 * tf.reduce_sum(theta**2 * tf.constant([1.0, 4.0, 9.0]))

def exact_gradient(theta):
    return -theta * tf.constant([1.0, 4.0, 9.0])

def damped_gradient(theta, epsilon=0.01):
    return exact_gradient(theta) / (1.0 + epsilon)

def run_hmc(gradient_fn, num_chains=2, num_steps=1000):
    """Run HMC with given gradient function."""
    
    # Initial state: [num_chains, 3]
    initial_state = tf.random.normal([num_chains, 3], seed=42)
    
    # Define kernel using gradient_fn
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob,
        step_size=0.1,
        num_leapfrog_steps=10,
        # Use gradient_fn as force (requires wrapping)
    )
    
    # Sample
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_steps,
        current_state=initial_state,
        kernel=kernel,
        trace_fn=lambda _, results: results,
        num_burnin_steps=500,
    )
    
    # samples: [num_steps, num_chains, 3]
    # Flatten chains: [num_steps * num_chains, 3]
    samples_flat = tf.reshape(samples, [-1, 3])
    
    return samples_flat, trace

def compute_wasserstein2(samples1, samples2):
    """Compute W₂ distance between two sample sets."""
    # Simple approximation: Gaussian W₂
    # W₂²(N(μ₁,Σ₁), N(μ₂,Σ₂)) = ||μ₁-μ₂||² + trace(Σ₁ + Σ₂ - 2(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})
    
    mu1 = tf.reduce_mean(samples1, axis=0)
    mu2 = tf.reduce_mean(samples2, axis=0)
    
    cov1 = tfp.stats.covariance(samples1)
    cov2 = tfp.stats.covariance(samples2)
    
    # Mean difference term
    mean_dist_sq = tf.reduce_sum((mu1 - mu2)**2)
    
    # Covariance term (simplified: Frobenius distance as upper bound)
    cov_dist_sq = tf.reduce_sum((cov1 - cov2)**2)
    
    w2_approx = tf.sqrt(mean_dist_sq + cov_dist_sq)
    
    return float(w2_approx.numpy())

def main():
    print("Phase 2: Toy Potential Test")
    print("=" * 60)
    
    # Derive tolerance
    tolerance = derive_parity_tolerance(
        condition_number=10.0,  # Simple quadratic, well-conditioned
        dtype="float32",
        backend="TF32"
    )
    print(f"Derived tolerance: {tolerance:.2e}")
    
    # Run Arm 1: Exact-force HMC
    print("\nRunning Arm 1: Exact-force HMC...")
    samples1, trace1 = run_hmc(exact_gradient, num_chains=2, num_steps=1000)
    print(f"  Samples shape: {samples1.shape}")
    print(f"  Mean: {tf.reduce_mean(samples1, axis=0).numpy()}")
    print(f"  Std: {tf.math.reduce_std(samples1, axis=0).numpy()}")
    
    # Run Arm 2: Damped-force HMC
    print("\nRunning Arm 2: Damped-force HMC...")
    samples2, trace2 = run_hmc(
        lambda theta: damped_gradient(theta, epsilon=0.01),
        num_chains=2,
        num_steps=1000
    )
    print(f"  Samples shape: {samples2.shape}")
    print(f"  Mean: {tf.reduce_mean(samples2, axis=0).numpy()}")
    print(f"  Std: {tf.math.reduce_std(samples2, axis=0).numpy()}")
    
    # Compute W₂ distance
    print("\nComputing W₂ distance...")
    w2_dist = compute_wasserstein2(samples1, samples2)
    print(f"  W₂ distance: {w2_dist:.4f}")
    print(f"  Tolerance: {tolerance:.4f}")
    
    # Primary criterion
    passed = w2_dist < tolerance
    print(f"\nPrimary criterion: {'PASS' if passed else 'FAIL'}")
    
    # Diagnostic: Coverage check
    true_theta = tf.constant([0.0, 0.0, 0.0])
    coverage1 = joint_mahalanobis_coverage(samples1, true_theta, alpha=0.05)
    coverage2 = joint_mahalanobis_coverage(samples2, true_theta, alpha=0.05)
    
    print(f"\nDiagnostic — True-θ coverage:")
    print(f"  Arm 1: {coverage1['covers']} (p={coverage1['p_value']:.3f})")
    print(f"  Arm 2: {coverage2['covers']} (p={coverage2['p_value']:.3f})")
    
    # Write result summary
    result = {
        "phase": "2",
        "status": "PASS" if passed else "FAIL",
        "date": "2026-09-07",
        "w2_distance": w2_dist,
        "tolerance": tolerance,
        "arm1_mean": samples1.numpy().mean(axis=0).tolist(),
        "arm2_mean": samples2.numpy().mean(axis=0).tolist(),
        "arm1_coverage": coverage1,
        "arm2_coverage": coverage2,
        "acceptance_arm1": "not_tracked",
        "acceptance_arm2": "not_tracked",
    }
    
    output_path = Path("results/phase2-summary.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResult written to: {output_path}")
    
    return 0 if passed else 1

if __name__ == "__main__":
    exit(main())
```

---

## Commands

```bash
# Create results directory
mkdir -p results

# Run test (GPU, escalated)
CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 \
  conda run -n tftwogpu python scripts/phase2_toy_potential_test.py \
  > results/phase2-output.txt 2>&1

# Exit code 0 = pass, 1 = fail
echo $? > results/phase2-exit-code.txt
```

---

## Success Criteria

**Primary criterion:** W₂(posterior₁, posterior₂) < tolerance

**Veto diagnostics:**
- Acceptance rate < 0.15 (if trackable)
- Divergences (if trackable)

**Explanatory diagnostics:**
- True-θ coverage (both arms should cover)
- Posterior means (should be near [0, 0, 0])
- Posterior stds (should be near [1.0, 0.5, 0.33])

---

## Result Summary File

**Written by script:** `results/phase2-summary.json`

```json
{
  "phase": "2",
  "status": "PASS" | "FAIL",
  "date": "2026-09-07",
  "w2_distance": 0.015,
  "tolerance": 0.02,
  "arm1_mean": [0.01, -0.02, 0.00],
  "arm2_mean": [0.00, -0.01, 0.01],
  "arm1_coverage": {"covers": true, "p_value": 0.42},
  "arm2_coverage": {"covers": true, "p_value": 0.51},
  "acceptance_arm1": "not_tracked",
  "acceptance_arm2": "not_tracked"
}
```

---

## On Failure

**If W₂ > tolerance:**
- Mechanism is broken (damped force doesn't preserve distribution)
- This invalidates Corollary 5.2 application
- Write diagnosis: `results/phase2-failure-diagnosis.md`
- **Stop and report to user — do not proceed to Phase 3**

**If both arms fail coverage:**
- HMC implementation issue (both arms wrong)
- Write diagnosis: `results/phase2-failure-diagnosis.md`
- **Stop and report to user**

**If only damped arm fails coverage:**
- Damped force breaks correctness
- Write diagnosis: `results/phase2-failure-diagnosis.md`
- **Stop and report to user**

---

## Bounded I/O Instructions

**DO:**
- Create one script file (~150 lines)
- Run once, write output to file
- Read only the summary JSON (< 1KB)
- If failure, read only last 50 lines of output

**DO NOT:**
- Read existing HMC implementation files "for reference"
- Import from `bayesfilter.inference.hmc` (build standalone)
- Print intermediate debug output during run
- Re-run multiple times "to check"

**This is a standalone test** — it should not depend on LEDH code, only on:
- TensorFlow / TFP (standard)
- Phase 0 outputs (tolerance, coverage)

---

## Completion Note Template

After Phase 2 completes, write `docs/plans/phase2-complete.md`:

```markdown
# Phase 2 Complete — [DATE]

**Status:** PASS / FAIL

**Test:** 3D quadratic potential, exact vs damped force

**Primary criterion:** W₂ distance
- Measured: [value]
- Tolerance: [value]
- Result: PASS / FAIL

**Veto diagnostics:** None triggered

**Explanatory diagnostics:**
- Arm 1 coverage: [true/false]
- Arm 2 coverage: [true/false]
- Posterior means: [values]

**Artifacts:**
- results/phase2-summary.json
- results/phase2-output.txt

**Next:** Phase 3 (seed policy verification tests)

**Issues encountered:** [if any]

**Interpretation:**
- If PASS: Damped-force mechanism works on analytical gradient
- If FAIL: [reason and next steps]
```

---

**Execute Phase 2 with these constraints. Write the completion note when done.**
