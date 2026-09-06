# Phase 3 Execution Handoff

**Date:** 2026-09-07  
**Authority:** ledh-surrogate-hmc-executable-master-program-2026-09-07.md  
**Estimated Time:** 1 day  
**GPU Required:** No (CPU tests only)

---

## Goal

Build seed-policy verification tests (V1, V2, V3) and dual-adapter wrapper for exact-force vs damped-force HMC on same frozen ω.

---

## Prerequisites

- Phase 0 complete (seed policy documented)
- Phase 1 complete (diagnostics pass)
- Phase 2 complete (toy potential PASS)

---

## Tasks

### Task 3.1: Dual-Adapter Wrapper (0.4 day)

**Goal:** Create wrapper that runs exact-force and damped-force HMC on identical frozen ω.

**File to create:** `bayesfilter/inference/ledh_dual_force_adapter.py`

**Key requirements:**
1. Sample one master ω (particle noise) at initialization
2. Freeze it for all subsequent force evaluations
3. Provide two force functions:
   - `exact_force(theta)` → exact LEDH gradient (Contract E, zero damping)
   - `damped_force(theta)` → damped LEDH gradient (Contract E, epsilon damping)
4. Both use **identical** ω (same particles, same noise realization)

**Structure:**

```python
import tensorflow as tf
from bayesfilter.highdim import ledh_contract_e_streaming_tf

class LEDHDualForceAdapter:
    """
    Adapter for running exact-force and damped-force HMC on same frozen ω.
    
    Ensures Corollary 5.2 premise: deterministic force on fixed particle noise.
    """
    
    def __init__(
        self,
        model,  # LGSSM model
        observations,  # y_1:T
        N: int = 1008,  # particle count
        master_seed: int = 42,
        damping_epsilon: float = 0.01,
    ):
        """
        Initialize with frozen master ω.
        
        Args:
            model: State-space model (LGSSM)
            observations: Data y_1:T
            N: Particle count
            master_seed: Seed for master ω (frozen)
            damping_epsilon: Damping coefficient for damped force
        """
        self.model = model
        self.observations = observations
        self.N = N
        self.master_seed = master_seed
        self.damping_epsilon = damping_epsilon
        
        # Sample master ω (frozen)
        self._sample_master_omega()
        
        # Call count for V3 test
        self._call_count = 0
        
    def _sample_master_omega(self):
        """Sample and freeze master particle noise ω."""
        # Implementation: sample N particles at t=0
        # Store as self.master_omega
        pass
        
    def exact_force(self, theta: tf.Tensor) -> tf.Tensor:
        """
        Exact LEDH force (Contract E, zero damping).
        
        Uses self.master_omega (frozen).
        
        Args:
            theta: [param_dim]
            
        Returns:
            force: [param_dim]
        """
        self._call_count += 1
        
        # Run LEDH filter with theta and frozen omega
        # Return gradient of log p(y|theta)
        pass
        
    def damped_force(self, theta: tf.Tensor) -> tf.Tensor:
        """
        Damped LEDH force (Contract E, epsilon damping).
        
        Uses self.master_omega (frozen).
        
        Args:
            theta: [param_dim]
            
        Returns:
            force: [param_dim]
        """
        self._call_count += 1
        
        # Run LEDH filter with theta and frozen omega
        # Apply damping: force_damped = force_exact / (1 + epsilon)
        # Return damped gradient
        pass
        
    def reset_call_count(self):
        """Reset call count for V3 test."""
        self._call_count = 0
        
    def get_call_count(self) -> int:
        """Get current call count."""
        return self._call_count
```

**Success criterion:**
- File exists with docstrings
- Both force methods defined
- Frozen ω is used (not resampled per call)

**Bounded I/O:**
- Read only the Contract E force function signature (grep + 20 lines):
  ```bash
  grep -n "def.*contract_e.*gradient\|def.*contract_e.*force" \
    bayesfilter/highdim/ledh_contract_e_streaming_tf.py
  ```
- Do not read full LEDH implementation files

---

### Task 3.2: Test V1 — Determinism (0.2 day)

**Goal:** Verify same ω → bitwise identical trajectories.

**File to create:** `tests/inference/test_ledh_seed_policy.py`

**Test V1:**

```python
import tensorflow as tf
import numpy as np
from bayesfilter.inference.ledh_dual_force_adapter import LEDHDualForceAdapter

def test_v1_determinism():
    """
    V1: Same ω → bitwise identical trajectories.
    
    Run exact_force twice with same master_seed, assert outputs match bitwise.
    """
    # Create simple LGSSM model
    model = create_simple_lgssm()  # d=3, T=10
    observations = generate_observations(model)
    
    # Adapter 1
    adapter1 = LEDHDualForceAdapter(
        model, observations, N=100, master_seed=42
    )
    
    # Adapter 2 (same seed)
    adapter2 = LEDHDualForceAdapter(
        model, observations, N=100, master_seed=42
    )
    
    # Evaluate force at same theta
    theta = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    
    force1 = adapter1.exact_force(theta)
    force2 = adapter2.exact_force(theta)
    
    # Assert bitwise identical
    np.testing.assert_array_equal(
        force1.numpy(), force2.numpy(),
        err_msg="V1 FAIL: Same seed produced different forces"
    )
    
    print("V1 PASS: Determinism verified")
```

**Success criterion:**
- Test passes (bitwise equality)
- If fails → ω is not frozen or force depends on hidden state

---

### Task 3.3: Test V2 — Reversibility (0.2 day)

**Goal:** Verify involution (forward-backward-forward = identity).

**Test V2:**

```python
def test_v2_reversibility():
    """
    V2: Reversibility (involution test).
    
    Standard HMC correctness check: momentum flip reverses trajectory.
    """
    model = create_simple_lgssm()
    observations = generate_observations(model)
    
    adapter = LEDHDualForceAdapter(
        model, observations, N=100, master_seed=42
    )
    
    # Initial state
    theta0 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    momentum0 = tf.constant([0.5, -0.3, 0.2, 0.1, -0.4])
    
    # Forward N steps
    theta_fwd, momentum_fwd = leapfrog_steps(
        theta0, momentum0, adapter.exact_force, 
        step_size=0.01, num_steps=10
    )
    
    # Reverse momentum, backward N steps
    theta_bwd, momentum_bwd = leapfrog_steps(
        theta_fwd, -momentum_fwd, adapter.exact_force,
        step_size=0.01, num_steps=10
    )
    
    # Reverse momentum again
    theta_final = theta_bwd
    momentum_final = -momentum_bwd
    
    # Assert: (theta_final, momentum_final) ≈ (theta0, momentum0)
    np.testing.assert_allclose(
        theta_final.numpy(), theta0.numpy(),
        rtol=1e-6, atol=1e-8,
        err_msg="V2 FAIL: Reversibility violated (theta)"
    )
    
    np.testing.assert_allclose(
        momentum_final.numpy(), momentum0.numpy(),
        rtol=1e-6, atol=1e-8,
        err_msg="V2 FAIL: Reversibility violated (momentum)"
    )
    
    print("V2 PASS: Reversibility verified")

def leapfrog_steps(theta, momentum, force_fn, step_size, num_steps):
    """Standard leapfrog integrator."""
    # Implementation of leapfrog
    pass
```

**Success criterion:**
- Test passes (reversibility within numerical tolerance)
- If fails → force depends on trajectory history

---

### Task 3.4: Test V3 — No Call-Count Dependence (0.2 day)

**Goal:** Verify force(θ, call=N) = force(θ, call=N+1).

**Test V3:**

```python
def test_v3_no_call_count_dependence():
    """
    V3: No call-count dependence.
    
    Force at same θ should be identical regardless of how many times 
    force has been called previously.
    """
    model = create_simple_lgssm()
    observations = generate_observations(model)
    
    adapter = LEDHDualForceAdapter(
        model, observations, N=100, master_seed=42
    )
    
    theta = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Call 1
    adapter.reset_call_count()
    force_call1 = adapter.exact_force(theta)
    assert adapter.get_call_count() == 1
    
    # Call force at different θ 50 times (change internal state if any)
    for i in range(50):
        theta_other = tf.constant([1.1, 2.1, 3.1, 4.1, 5.1]) * float(i + 1)
        _ = adapter.exact_force(theta_other)
    
    # Call at original θ again (call #52)
    force_call52 = adapter.exact_force(theta)
    assert adapter.get_call_count() == 52
    
    # Assert: force(θ, call=1) = force(θ, call=52)
    np.testing.assert_allclose(
        force_call1.numpy(), force_call52.numpy(),
        rtol=1e-10, atol=1e-12,
        err_msg="V3 FAIL: Force depends on call count"
    )
    
    print("V3 PASS: No call-count dependence")
```

**Success criterion:**
- Test passes (forces identical)
- If fails → force has hidden state (call count, cached clouds, etc.)

---

## Commands

```bash
# Create test file with all three tests
# (tests/inference/test_ledh_seed_policy.py)

# Run tests (CPU-only, no GPU needed)
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
  conda run -n tftwogpu python -m pytest \
  tests/inference/test_ledh_seed_policy.py -v --tb=short \
  > results/phase3-seed-tests.txt 2>&1

# Check exit code
echo $? > results/phase3-exit-code.txt
```

---

## Success Criteria (Phase 3 Complete)

1. ✅ `LEDHDualForceAdapter` exists with frozen ω
2. ✅ Test V1 passes (determinism)
3. ✅ Test V2 passes (reversibility)
4. ✅ Test V3 passes (no call-count dependence)

**All three tests must pass. Any failure blocks Phase 4.**

---

## Result Summary File

**Write:** `results/phase3-summary.json`

```json
{
  "phase": "3",
  "status": "PASS" | "FAIL",
  "date": "2026-09-07",
  "tests_run": 3,
  "tests_passed": 3,
  "test_results": {
    "v1_determinism": "PASS" | "FAIL",
    "v2_reversibility": "PASS" | "FAIL",
    "v3_no_call_count_dependence": "PASS" | "FAIL"
  },
  "artifacts": [
    "bayesfilter/inference/ledh_dual_force_adapter.py",
    "tests/inference/test_ledh_seed_policy.py",
    "results/phase3-seed-tests.txt"
  ],
  "next_phase": "4a"
}
```

---

## On Failure

**If V1 fails (determinism):**
- ω is not frozen, or force has stochastic component
- Write diagnosis: `results/phase3-v1-failure-diagnosis.md`
- **Blocking defect — Corollary 5.2 premise violated**
- Stop and report to user

**If V2 fails (reversibility):**
- Force depends on trajectory history or cached state
- Write diagnosis: `results/phase3-v2-failure-diagnosis.md`
- **Blocking defect — HMC correctness violated**
- Stop and report to user

**If V3 fails (call-count dependence):**
- Force has hidden state (call count, cache, global counter)
- Write diagnosis: `results/phase3-v3-failure-diagnosis.md`
- **Blocking defect — determinism violated**
- Stop and report to user

**Do not proceed to Phase 4 if any test fails.**

---

## Bounded I/O Instructions

**DO:**
- Create two new files (adapter + tests)
- Use grep to find Contract E force signature (20 lines)
- Read only test output summary (last 50 lines)
- Write result summary JSON

**DO NOT:**
- Read full LEDH implementation files
- Read existing HMC test files "for examples"
- Print intermediate debug output during tests
- Re-run tests multiple times

**Minimal dependencies:**
- LEDHDualForceAdapter imports from `bayesfilter.highdim.ledh_contract_e_streaming_tf`
- Tests import from LEDHDualForceAdapter
- Tests use simple LGSSM (d=3, T=10, N=100)

---

## Helper Functions Needed

**In test file, add:**

```python
def create_simple_lgssm():
    """Create d=3 LGSSM for testing."""
    # Minimal LGSSM: x_{t+1} = x_t + w_t, y_t = x_t + v_t
    # w_t ~ N(0, Q), v_t ~ N(0, R)
    pass

def generate_observations(model, T=10, seed=42):
    """Generate T observations from model."""
    pass

def leapfrog_steps(theta, momentum, force_fn, step_size, num_steps):
    """Standard leapfrog integrator for V2 test."""
    pass
```

These are test utilities (~50 lines total).

---

## Completion Note Template

After Phase 3 completes, write `docs/plans/phase3-complete.md`:

```markdown
# Phase 3 Complete — [DATE]

**Status:** PASS / FAIL

**What was built:**
- LEDHDualForceAdapter (frozen ω)
- Test V1 (determinism)
- Test V2 (reversibility)
- Test V3 (no call-count dependence)

**Test results:**
- V1: PASS / FAIL
- V2: PASS / FAIL
- V3: PASS / FAIL

**Artifacts:**
- bayesfilter/inference/ledh_dual_force_adapter.py
- tests/inference/test_ledh_seed_policy.py
- results/phase3-summary.json

**Next:** Phase 4a (ultra-short LGSSM diagnostic, 2 GPU-hours)

**Issues encountered:** [if any]

**Interpretation:**
- If all PASS: Corollary 5.2 premises verified (determinism, reversibility)
- If any FAIL: [which premise violated, implications]
```

---

**Execute Phase 3 with these constraints. Write the completion note when done.**
