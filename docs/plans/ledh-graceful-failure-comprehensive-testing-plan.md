# LEDH Graceful Failure - Comprehensive Testing Master Program

**Date:** 2026-09-17  
**Branch:** surrogate-hmc  
**Purpose:** Validate graceful failure mechanism on actual state space models with parameter recovery

## Executive Summary

The current testing (Phases 4-5) validates the graceful failure mechanism on toy analytic targets and simplified UKF dynamics, but does NOT test on actual state space models (LGSSM, SV, predator-prey, SIR). This plan specifies comprehensive testing on real models with proper identifiability conditions and parameter recovery validation.

## Background: Why Current Testing Is Inadequate

### What Was Tested (Phases 4-5)
1. **Analytic pathological target**: Simple toy function returning -inf in invalid region
2. **Synthetic covariance matrices**: Hand-crafted matrices with known condition numbers
3. **Nonlinear UKF dynamics**: `f(x,θ) = x + θ·x²` - not a real state space model
4. **Custom gradient propagation**: Unit tests for -inf sentinel and zero gradient

### What Was NOT Tested
1. **Full LGSSM with proper identifiability**: No test validates that an identifiable LGSSM specification with stationary initial distribution works correctly
2. **Parameter recovery**: No test shows that HMC can actually recover known parameters after graceful failure is applied
3. **Real nonlinear models**: SV, predator-prey, SIR models untested
4. **Various dimensions and horizons**: Only small toy cases tested

### Critical Gap
The graceful failure mechanism sits inside the LEDH canonical score computation, which is called during HMC sampling for state space model parameter inference. If this mechanism breaks parameter recovery, the implementation fails its primary purpose even if unit tests pass.

## LGSSM Identifiability Theory

### Identifiability Conditions

For a linear Gaussian state space model to be identifiable:

```
State equation:     x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)
Observation equation: y_t = H x_t + v_t,    v_t ~ N(0, R)
Initial:           x_0 ~ N(m_0, P_0)
```

**Necessary conditions:**
1. **Observability**: The pair (F, H) must be observable
   - Observability matrix has full rank: rank([H; HF; HF²; ...; HF^{n-1}]) = n
   - Without observability, some state dimensions are invisible and parameters are unidentifiable

2. **Controllability**: The pair (F, Q^{1/2}) must be controllable (when Q > 0)
   - Ensures process noise affects all state dimensions
   - Without controllability, some dimensions evolve deterministically

3. **Sufficient excitation**: Observation noise R must be positive definite
   - Ensures observations provide information about the state

4. **Stationary initial distribution**: For parameter inference, use:
   ```
   P_0 = stationary covariance of state process
       = solution to: P_0 = F P_0 F' + Q
   ```
   - This eliminates arbitrary initialization effects on likelihood
   - Critical for fair parameter recovery tests

### Computing Stationary Distribution

For stable F (all eigenvalues |λ| < 1), the stationary covariance is:
```
vec(P_0) = (I - F ⊗ F)^{-1} vec(Q)
```

In TensorFlow:
```python
def stationary_covariance(F, Q):
    n = F.shape[0]
    kron_FF = tf.linalg.LinearOperatorKronecker(
        [tf.linalg.LinearOperatorFullMatrix(F),
         tf.linalg.LinearOperatorFullMatrix(F)]
    ).to_dense()
    I = tf.eye(n * n, dtype=F.dtype)
    vec_Q = tf.reshape(Q, [-1])
    vec_P0 = tf.linalg.solve(I - kron_FF, vec_Q[:, None])
    return tf.reshape(vec_P0, [n, n])
```

### Identifiable LGSSM Test Cases

**Case 1: Univariate AR(1) - (state_dim=1, T=10)**
```
F = [[φ]],  φ = 0.8
Q = [[σ²]],  σ² = 1.0
H = [[1.0]]
R = [[0.5]]
P_0 = σ² / (1 - φ²) = 1.0 / 0.36 = 2.778
```
- Fully observable (H = I)
- Fully controllable
- Simple scalar case for basic validation

**Case 2: Bivariate VAR(1) - (state_dim=2, T=20)**
```
F = [[0.7, 0.2],
     [0.1, 0.6]]
Q = [[1.0, 0.3],
     [0.3, 0.8]]
H = [[1.0, 0.0],
     [0.0, 1.0]]
R = [[0.5, 0.0],
     [0.0, 0.5]]
```
- Check observability: rank([H; HF]) = rank(4×2 matrix) should equal 2
- Compute P_0 from Lyapunov equation
- Tests interaction between state dimensions

**Case 3: Four-dimensional partially observed - (state_dim=4, T=50)**
```
F = 0.8 * I_4 + 0.1 * 1_4 1_4'  (stable with coupling)
Q = I_4
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]  (observe only first 2 states)
R = [[0.5, 0.0],
     [0.0, 0.5]]
```
- Partially observed system
- Tests UKF sigma point generation with higher dimension
- More opportunities for numerical instability

**Case 4: High-dimensional - (state_dim=20, T=120)**
```
F = sparse banded AR structure (tridiagonal)
Q = diagonal with varying scales
H = random projection to 10 dimensions
R = 0.5 * I_10
```
- Stress test for high dimensions
- Most likely to trigger Cholesky failures
- Critical for production readiness

## Stochastic Volatility (SV) Model Testing

### Model Specification

The `StochasticVolatilitySSM` in `models.py` implements:
```
State:        x_t = γ x_{t-1} + σ ε_t,     ε_t ~ N(0,1)
Observation:  y_t = β exp(x_t/2) η_t,      η_t ~ N(0,1)
Initial:      x_0 ~ N(0, σ²/(1-γ²))
Parameters:   θ = (Φ^{-1}(γ), log(β))  (unconstrained)
```

**Test Cases:**

**SV-1: Short horizon (T=20)**
- True parameters: γ = 0.95, β = 0.5, σ = 1.0
- Generate synthetic data from true parameters
- Test HMC parameter recovery with graceful failure enabled
- Expected: Should recover γ, β within reasonable posterior intervals

**SV-2: Long horizon (T=40)**
- Same parameters, longer trajectory
- More observations → tighter posteriors
- Higher chance of numerical issues during filtering

### Why SV Is Critical

1. **Exponential observation model**: `exp(x_t/2)` can cause numerical overflow
2. **Persistent state**: High γ (≈0.95) makes state highly correlated
3. **Real inference target**: SV is used in actual financial applications
4. **Nonlinear observation**: Tests UKF sigma point handling under nonlinearity

## Predator-Prey Model Testing

### Model Specification

The `PredatorPreySSM` in `models.py` implements Lotka-Volterra dynamics:
```
Prey:      dx/dt = rx(1 - x/K) - axy/(x+s)
Predator:  dy/dt = u·axy/(x+s) - vy

State: [prey, predator]
Parameters: θ = (r, K, a, s, u, v)  [6 parameters]
Discretization: RK4 with substeps
```

**Test Case: Default parameters**
- True: (r=0.6, K=114, a=25, s=0.3, u=0.5, v=0.5)
- T = 20 timesteps
- Initial: (50, 5) for prey/predator
- Process noise: 4.0 * I_2
- Observation noise: 4.0 * I_2

### Why Predator-Prey Is Critical

1. **Complex nonlinear dynamics**: Limit cycles possible
2. **Multiple parameters**: 6-dimensional parameter space
3. **Sensitive to initialization**: Phase-space trajectories matter
4. **Domain constraints**: Populations must stay positive
5. **Real ODE integration**: RK4 substeps test numerical stability

## SIR Epidemic Model Testing

### Model Specification

The `SpatialSIRSSM` in `models.py` implements spatial epidemic dynamics:
```
Susceptible: dS_j/dt = -κ_j S_j (I_j + ∑_{k∈N_j} I_k) / degree_j
Infectious:  dI_j/dt = κ_j S_j (I_j + ∑_{k∈N_j} I_k) / degree_j - ν_j I_j

State: [S_1, I_1, S_2, I_2, ..., S_J, I_J]  (2J dimensions)
Parameters: Fixed κ, ν (not inferred in first gate)
Discretization: RK4 with substeps
```

**Test Case: 3-compartment spatial network**
- J = 3 compartments (state_dim = 6)
- T = 30 timesteps
- Network: linear chain (1--2--3)
- κ = [0.5, 0.5, 0.5]
- ν = [0.2, 0.2, 0.2]
- Observe only infectious counts (I_1, I_2, I_3)

### Why SIR Is Critical

1. **Partially observed**: Susceptible counts unobserved
2. **Network structure**: Spatial coupling between compartments
3. **Domain constraints**: S, I must stay non-negative
4. **Public health relevance**: Real-world epidemic inference
5. **Mixed observability**: Some state dimensions observed, others latent

## Test Implementation Strategy

### Phase 1: LGSSM Identifiability and Stationary Initialization

**Objective:** Verify LGSSM specifications are identifiable and initial distributions are correct

**Files to create:**
- `tests/integration/test_lgssm_identifiable_specifications.py`
  - Test stationary covariance computation
  - Verify observability matrices have full rank
  - Check stability of F matrices (eigenvalues inside unit circle)

**Success criteria:**
- All 4 LGSSM cases have observable (F,H) pairs ✓
- Stationary P_0 computed correctly via Lyapunov equation ✓
- F matrices are stable (all |λ| < 1) ✓

### Phase 2: LGSSM Parameter Recovery Without Graceful Failure

**Objective:** Baseline - verify parameter recovery works on well-conditioned LGSSMs

**Files to create:**
- `tests/integration/test_lgssm_parameter_recovery_baseline.py`
  - For each of 4 LGSSM cases:
    1. Generate synthetic data from known parameters
    2. Run HMC to recover parameters (using LEDH score)
    3. Check posterior means are close to true values
    4. Verify no numerical failures occur

**Success criteria:**
- Parameter recovery works for cases 1-3 (dimensions 1, 2, 4)
- Case 4 (dim=20) may have issues - document behavior
- Baseline establishes what "correct" parameter recovery looks like

### Phase 3: LGSSM Graceful Failure Stress Test

**Objective:** Deliberately trigger Cholesky failures and verify graceful handling

**Files to create:**
- `tests/integration/test_lgssm_graceful_failure_stress.py`
  - Use pathological parameters that cause ill-conditioned covariances:
    * Very small process noise (Q → 0)
    * Very small observation noise (R → 0)
    * Near-singular initial covariance
  - Verify HMC continues without crashing
  - Verify -inf sentinel is returned for invalid proposals
  - Verify valid proposals are still accepted

**Success criteria:**
- HMC completes without NaN or crash ✓
- Invalid proposals return -inf with zero gradient ✓
- Valid proposals return finite log probability ✓
- MH acceptance rate is reasonable for valid region ✓

### Phase 4: SV Model Parameter Recovery

**Objective:** Verify graceful failure works on real SV inference

**Files to create:**
- `tests/integration/test_sv_parameter_recovery.py`
  - SV-1 (T=20): Generate data, recover (γ, β)
  - SV-2 (T=40): Generate data, recover (γ, β)
  - Use `StochasticVolatilitySSM` from `models.py`
  - Run HMC with LEDH canonical score
  - Check posterior intervals contain true parameters

**Success criteria:**
- HMC converges for both T=20 and T=40
- Posterior means within 2 posterior SD of true values
- No numerical failures or divergences
- ESS > 100 for both parameters

### Phase 5: Predator-Prey Parameter Recovery

**Objective:** Verify graceful failure works on complex nonlinear ODE model

**Files to create:**
- `tests/integration/test_predator_prey_parameter_recovery.py`
  - Use `PredatorPreySSM` from `models.py`
  - Generate data from true parameters
  - Recover subset of parameters (e.g., fix some, infer others)
  - Verify HMC handles RK4 integration numerics correctly

**Success criteria:**
- HMC completes without crash
- Recovered parameters in plausible range
- No divergences from numerical instability
- Graceful failure prevents NaN propagation from RK4 steps

### Phase 6: SIR Partial Observability

**Objective:** Verify graceful failure works when some state dimensions are unobserved

**Files to create:**
- `tests/integration/test_sir_partial_observability.py`
  - Use `SpatialSIRSSM` with 3 compartments
  - Observe only infectious counts (I_j), not susceptible (S_j)
  - Run particle filter or UKF-based filtering
  - Verify graceful failure handles unobserved dimensions

**Success criteria:**
- Filtering completes without crash
- Posterior state estimates are reasonable
- Graceful failure doesn't break partial observability
- Unobserved dimensions don't cause instability

### Phase 7: Comprehensive Test Suite

**Objective:** Run all models in one test suite, produce test report

**Files to create:**
- `tests/integration/test_ledh_graceful_failure_comprehensive.py`
  - Runs all LGSSM cases (4)
  - Runs all SV cases (2)
  - Runs predator-prey (1)
  - Runs SIR (1)
  - Total: 8 model configurations
  - Produces summary report with pass/fail for each

**Success criteria:**
- All 8 configurations complete
- At least 7/8 pass parameter recovery checks
- Comprehensive report generated in `docs/plans/artifacts/`

## Implementation Timeline

**Estimated effort:** 3-4 days

**Day 1:**
- Implement stationary covariance computation
- Create 4 identifiable LGSSM specifications
- Verify identifiability conditions (Phase 1)
- Baseline parameter recovery tests (Phase 2)

**Day 2:**
- LGSSM stress tests with pathological parameters (Phase 3)
- SV parameter recovery tests (Phase 4)

**Day 3:**
- Predator-prey parameter recovery (Phase 5)
- SIR partial observability (Phase 6)

**Day 4:**
- Comprehensive test suite (Phase 7)
- Generate test report
- Write hands-off memo to codex

## Expected Results

### What Success Looks Like

1. **All unit tests passing** (already achieved in Phase 4-5)
2. **LGSSM parameter recovery working** for dimensions 1, 2, 4, 20
3. **SV parameter recovery working** for T=20 and T=40
4. **Predator-prey inference completing** without numerical failures
5. **SIR partial observability handled** correctly
6. **Graceful failure mechanism** prevents crashes in all cases
7. **Parameter posteriors** are reasonable (contain true values)

### What Failure Might Look Like

1. **Parameter recovery broken**: Posteriors don't contain true values
   - Diagnosis: Graceful failure may be too aggressive (rejecting valid proposals)
   - Fix: Adjust safe_cholesky threshold or validity propagation logic

2. **Still getting NaN crashes**: Graceful failure incomplete
   - Diagnosis: Some code path not using safe_cholesky
   - Fix: Audit all Cholesky calls, add missing safe_cholesky wrappers

3. **HMC gets stuck**: Zero gradient prevents exploration
   - Diagnosis: Invalid proposals have zero gradient → no movement
   - Fix: This is expected behavior - MH rejection handles it

4. **Poor mixing**: Acceptance rate too low
   - Diagnosis: Too many proposals landing in invalid region
   - Fix: Adjust HMC step size, not a graceful failure issue

## Memo to Codex for Review

### Review Request

**Subject:** LEDH Graceful Failure - Comprehensive Testing Plan Review

**Background:**
The LEDH graceful failure implementation (Phases 1-5) successfully handles Cholesky decomposition failures by returning -inf sentinel values with zero gradients. Current testing validates the mechanism on toy functions and simplified dynamics, but does NOT test on actual state space models used in production.

**This Plan:**
Specifies comprehensive testing on real models (LGSSM, SV, predator-prey, SIR) with proper identifiability conditions, stationary initial distributions, and parameter recovery validation.

**Critical Review Points:**

1. **Identifiability conditions:** Are the specified LGSSM cases actually identifiable? Verify observability and controllability conditions.

2. **Stationary distribution:** Is the Lyapunov equation approach correct for computing stationary P_0? Check TensorFlow implementation.

3. **Parameter recovery criterion:** Is "posterior contains true parameter" sufficient, or should we use tighter criteria (e.g., posterior mean within 1 SD)?

4. **Model coverage:** Are LGSSM, SV, predator-prey, SIR sufficient, or should we add more models from the repo?

5. **Stress test design:** Does Phase 3 actually trigger Cholesky failures, or do we need more pathological cases?

6. **Timeline realism:** Is 3-4 days reasonable, or should we budget more time?

7. **Success criteria:** Are the pass/fail criteria clear and measurable?

**Review Deliverable:**
- Approve plan as-is, OR
- Request specific changes with technical justification, OR
- Identify missing test cases or risk scenarios

**Urgency:** Medium - implementation is "production ready" according to Phase 4-5 status, but this testing is required to validate that claim on actual models.

---

**Plan Author:** Claude Opus 5  
**Date:** 2026-09-17  
**Branch:** surrogate-hmc