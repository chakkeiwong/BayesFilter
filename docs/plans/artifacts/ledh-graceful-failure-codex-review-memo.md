# LEDH Graceful Failure - Codex Review Request

**To:** Codex  
**From:** Claude Opus 5  
**Date:** 2026-09-17  
**Subject:** Request for thorough review of LEDH graceful failure comprehensive testing plan  
**Branch:** surrogate-hmc  
**Priority:** Medium

---

## Executive Summary

The LEDH graceful failure implementation (Phases 1-5, completed 2026-09-16) successfully implements a mechanism to handle Cholesky decomposition failures by returning `-inf` sentinel values with zero gradients, allowing HMC to continue via Metropolis-Hastings rejection rather than crashing.

**Current Status:**
- ✓ Implementation complete (5 files modified, 5 new files)
- ✓ Unit tests passing (26/26)
- ✓ Documentation complete
- ✓ Declared "production ready"

**Critical Gap Identified:**
Testing was performed on toy analytic functions and simplified UKF dynamics, NOT on actual state space models (LGSSM, SV, predator-prey, SIR) that the system is designed to handle.

**This Review Request:**
Thorough technical review of the comprehensive testing plan (`ledh-graceful-failure-comprehensive-testing-plan.md`) before implementation begins.

---

## Background: What Was Done (Phases 1-5)

### Implementation (Phases 1-3)

**Three-layer strategy:**

1. **Layer 1: Safe Cholesky** (`ledh_numerical_safety_tf.py`)
   - Post-Cholesky NaN detection
   - Returns `(is_valid, chol)` tuple
   - Zero matrix on failure

2. **Layer 2: Validity Propagation**
   - Reset policy: combines validity flags from gap/target/injected sources
   - Canonical score: combines all validity flags with callback validity
   - Uses `tf.reduce_all()` to convert batch flags to scalars

3. **Layer 3: Sentinel Values**
   - Canonical score: `-inf` for value, `0.0` for gradient when invalid
   - Dual-parameter target: propagates `-inf` with zero custom gradient

**Design decisions:**
- `-inf` instead of NaN: Deterministic MH rejection via `exp(-inf) = 0`
- Post-Cholesky detection: No eigenvalue threshold to tune
- Zero gradient for invalid: Prevents NaN propagation in leapfrog

### Testing (Phases 4-5)

**What was tested:**

1. **Unit tests** (23 passing):
   - `test_ledh_numerical_safety_tf.py`: Safe Cholesky wrapper (12 tests)
   - `test_ledh_reset_validity.py`: Validity propagation (5 tests)
   - `test_dual_parameter_target_invalid.py`: Custom gradient (4 tests)
   - `test_ledh_canonical_score_ukf_tangent.py`: UKF integration (2 tests)

2. **Integration tests** (3 passing):
   - `test_hmc_graceful_failure_pathological_target`: HMC with -inf target
   - `test_dual_parameter_target_handles_neg_inf`: Gradient propagation
   - `test_metropolis_hastings_rejection_of_invalid`: MH rejection probability

**What was NOT tested:**
- Full LGSSM with proper identifiability conditions
- Parameter recovery on any real model
- Stochastic Volatility (SV) model
- Predator-prey model
- SIR epidemic model
- Any model with T > 10 timesteps
- High-dimensional state spaces (dim > 2)

---

## The Comprehensive Testing Plan

### Overview

The plan specifies 7 testing phases covering:
- 4 LGSSM configurations: (state_dim=1, T=10), (2, 20), (4, 50), (20, 120)
- 2 SV configurations: T=20, T=40
- 1 Predator-prey configuration: 6 parameters, T=20
- 1 SIR configuration: 3 compartments, T=30, partial observability

Total: 8 model configurations with parameter recovery validation

### Key Technical Components

**1. LGSSM Identifiability Theory**

Proposes identifiability conditions:
- Observability: rank([H; HF; HF²; ...; HF^{n-1}]) = n
- Controllability: (F, Q^{1/2}) controllable
- Stationary initialization: P_0 from Lyapunov equation

**Implementation:**
```python
def stationary_covariance(F, Q):
    n = F.shape[0]
    kron_FF = tf.linalg.LinearOperatorKronecker([...]).to_dense()
    I = tf.eye(n * n, dtype=F.dtype)
    vec_Q = tf.reshape(Q, [-1])
    vec_P0 = tf.linalg.solve(I - kron_FF, vec_Q[:, None])
    return tf.reshape(vec_P0, [n, n])
```

**2. Test Strategy**

Phase 1: Verify identifiability conditions hold for all 4 LGSSM cases
Phase 2: Baseline parameter recovery without failures
Phase 3: Stress test with pathological parameters (trigger failures)
Phase 4-6: Real model parameter recovery (SV, predator-prey, SIR)
Phase 7: Comprehensive suite with summary report

**3. Success Criteria**

Per model:
- HMC completes without crash ✓
- Posterior means within 2 SD of true parameters ✓
- No divergences or NaN propagation ✓
- ESS > 100 for all parameters ✓

Overall:
- At least 7/8 configurations pass
- Graceful failure prevents all crashes
- Parameter recovery still works correctly

---

## Critical Review Questions

### 1. Mathematical Correctness

**Q1.1:** Are the stated identifiability conditions (observability, controllability, stationary initialization) sufficient and necessary for LGSSM parameter identification?

**Q1.2:** Is the Lyapunov equation approach for stationary covariance correct?
```
vec(P_0) = (I - F ⊗ F)^{-1} vec(Q)
```
Should this be solved via `tf.linalg.solve` or a specialized Lyapunov solver?

**Q1.3:** For the 4 proposed LGSSM cases, verify:
- Case 1 (dim=1): Is φ=0.8, σ²=1.0, P_0=2.778 correct?
- Case 2 (dim=2): Does the proposed F matrix have stable eigenvalues?
- Case 3 (dim=4): Is the "0.8*I + 0.1*1_4*1_4'" formulation stable?
- Case 4 (dim=20): Is "sparse banded tridiagonal" sufficiently specified?

**Q1.4:** The plan claims stationary initialization "eliminates arbitrary initialization effects on likelihood". Is this claim justified? Under what conditions does likelihood depend on initial distribution parameterization?

### 2. Implementation Feasibility

**Q2.1:** TensorFlow Kronecker product for n=20 creates a 400×400 matrix. Is this computationally feasible, or should we use iterative Lyapunov solvers (e.g., `tf.linalg.solve_lyapunov` if available)?

**Q2.2:** The plan proposes observability matrix rank check. For n=20, this creates a 200×20 matrix. Should we use numerical rank with tolerance, or exact rank?

**Q2.3:** Phase 3 proposes "very small process noise (Q → 0)" to trigger failures. What specific values trigger Cholesky failure without making the model degenerate? Should we specify exact pathological parameter ranges?

**Q2.4:** Parameter recovery requires running HMC. What are the HMC hyperparameters (step size, num leapfrog steps, num chains, warmup length)? Should these be specified in the plan?

### 3. Test Coverage Completeness

**Q3.1:** The plan tests 4 models (LGSSM, SV, predator-prey, SIR). Are there other critical models in the repo that should be tested? The search found:
- `GeneralizedSVPriorMeanSSM`
- `ParameterizedZhaoCuiSIRSSM`
Should these be added?

**Q3.2:** The plan tests dimensions 1, 2, 4, 20 for LGSSM. Is there a critical dimension between 4 and 20 where numerical behavior changes (e.g., dim=10)?

**Q3.3:** The plan tests T=10, 20, 40, 50, 120. Is there a critical horizon length where failures become more likely? Should we add T=100 or T=200 stress tests?

**Q3.4:** The plan tests parameter recovery but not filtering accuracy. Should we also verify that filtered state estimates are correct when graceful failure is active?

### 4. Success Criteria Appropriateness

**Q4.1:** "Posterior means within 2 SD of true parameters" - is 2 SD reasonable? Should this be 1 SD for well-identified parameters, or 3 SD for weakly identified ones?

**Q4.2:** "ESS > 100" - is this sufficient? What if ESS = 101 but mixing is poor? Should we also check R-hat < 1.1?

**Q4.3:** "At least 7/8 configurations pass" - why allow 1 failure? Which configuration is expected to fail, and why is that acceptable?

**Q4.4:** The plan has no performance criteria. Should we verify that graceful failure overhead is < 5% of total computation time?

### 5. Risk Analysis

**Q5.1:** What happens if Phase 2 (baseline parameter recovery) fails? This would indicate LEDH canonical score is broken independent of graceful failure. Should we have a fallback plan?

**Q5.2:** What if graceful failure is too aggressive (rejecting valid proposals)? How would we diagnose this from test results? Should we log rejection rates?

**Q5.3:** The plan assumes HMC will encounter invalid proposals naturally during exploration. What if it doesn't? Should we deliberately initialize HMC near the invalid boundary to force encounters?

**Q5.4:** What if different models require different safe_cholesky thresholds? The current implementation uses post-Cholesky NaN detection (no threshold). Is this uniform approach robust across all model types?

### 6. Timeline and Resource Requirements

**Q6.1:** The plan estimates 3-4 days. This includes:
- Implementing stationary covariance computation
- Creating 8 test configurations
- Running HMC for parameter recovery (potentially hours per config)
- Generating and analyzing results
Is 3-4 days realistic? Should we budget 5-7 days?

**Q6.2:** What are the computational requirements? Should tests run on CPU or GPU? Do we need to reserve GPU time for longer HMC runs?

**Q6.3:** Phase 7 produces a "comprehensive report". What format? Should this be a LaTeX document, markdown, or structured JSON for automated checking?

### 7. Integration with Existing Codebase

**Q7.1:** The plan creates files in `tests/integration/`. Should these be added to the existing test suite, or kept separate as a validation artifact?

**Q7.2:** The predator-prey model has 6 parameters. Should we infer all 6, or fix some and infer others (as suggested in the plan)? If fixing some, which ones and why?

**Q7.3:** The SIR model has "fixed κ, ν (not inferred in first gate)". What parameters ARE inferred for SIR? Or is this purely a filtering test, not parameter inference?

**Q7.4:** Should the comprehensive test suite be run as part of pre-commit hooks, or only on-demand due to computational cost?

### 8. Documentation and Reproducibility

**Q8.1:** Should each test configuration save its synthetic data, HMC samples, and diagnostics for reproducibility?

**Q8.2:** Should we specify random seeds for all synthetic data generation?

**Q8.3:** The plan mentions "hands-off memo to codex". Is this memo the current document, or should there be a separate implementation memo after testing completes?

**Q8.4:** Should test results be archived in `docs/plans/artifacts/` with timestamps and git commits for provenance?

---

## Specific Technical Concerns

### Concern 1: Lyapunov Equation Solver

The proposed Kronecker-based approach:
```python
vec_P0 = tf.linalg.solve(I - kron_FF, vec_Q[:, None])
```

For n=20, this creates and solves a 400×400 linear system. Alternative approaches:
1. Use `scipy.linalg.solve_discrete_lyapunov` and convert to TensorFlow
2. Iterative solver for large n
3. Bartels-Stewart algorithm (standard in control theory)

**Recommendation needed:** Which approach should we use?

### Concern 2: Observability/Controllability Checks

The plan proposes rank checks on observability and controllability matrices. For floating-point matrices, rank is ambiguous without a tolerance. Standard approach is:
```python
def numerical_rank(A, tol=1e-10):
    s = tf.linalg.svd(A, compute_uv=False)
    return tf.reduce_sum(tf.cast(s > tol, tf.int32))
```

**Recommendation needed:** What tolerance should we use? 1e-10? Relative to largest singular value?

### Concern 3: Pathological Parameter Selection

Phase 3 proposes triggering Cholesky failures with "very small Q" or "very small R". But how small? If Q = 1e-20 * I, is the model still meaningful? Should we specify:
- Minimum meaningful noise level (e.g., Q > 1e-10 * I)
- Specific pathological cases that are physically plausible
- Gradient of pathology (slightly ill-conditioned → severely ill-conditioned)

**Recommendation needed:** Specify exact pathological parameter values.

### Concern 4: HMC Hyperparameter Selection

The plan is silent on HMC configuration. For parameter recovery to work, we need:
- Appropriate step size (too large → divergences, too small → poor mixing)
- Sufficient warmup (at least 500 iterations?)
- Sufficient samples (at least 1000 post-warmup?)
- Multiple chains (at least 4 for R-hat computation?)

**Recommendation needed:** Should these be specified in the plan, or left to individual test implementations?

### Concern 5: SIR Model Parameter Inference

The `SpatialSIRSSM` class has `parameter_dim() -> 0`, meaning it has no inferred parameters in the base implementation. The plan says "Fixed κ, ν (not inferred in first gate)". But then what is being tested?

**Options:**
1. This is a filtering test only (no parameter inference)
2. Use `ParameterizedZhaoCuiSIRSSM` instead (which may have parameters)
3. Create a parameterized wrapper around `SpatialSIRSSM`

**Recommendation needed:** Clarify what is being tested for SIR.

---

## Review Deliverables Requested

### Primary Deliverable: Technical Approval

**Option A: Approve as-is**
- Plan is technically sound
- No major gaps or errors identified
- Ready to proceed with implementation

**Option B: Approve with minor revisions**
- Specify required changes (list)
- Changes do not affect core testing strategy
- Can proceed after revisions

**Option C: Request major revisions**
- Fundamental issues with approach
- Cannot proceed without substantial rework
- Provide detailed technical justification

### Secondary Deliverable: Specific Answers

Please provide technical answers to:
1. Q1.2: Correct Lyapunov solver approach
2. Q2.3: Exact pathological parameter values
3. Q3.1: Additional models to test
4. Q5.3: Strategy to force invalid proposal encounters
5. Concern 5: SIR testing clarification

### Tertiary Deliverable: Risk Assessment

Classify each risk (Q5.1-Q5.4) as:
- **High:** Likely to occur, would block testing
- **Medium:** Possible, would require adaptation
- **Low:** Unlikely or easily mitigated

For high-risk items, provide mitigation strategy.

---

## Timeline for Review

**Requested completion:** Within 48 hours
**Implementation blocked:** Yes - cannot proceed without review
**Urgency justification:** Phase 4-5 claims "production ready", but this testing is required to validate that claim on actual models.

---

## Contact for Questions

**Author:** Claude Opus 5 (AI agent)  
**Supervisor:** User (chakwong)  
**Branch:** surrogate-hmc  
**Plan document:** `docs/plans/ledh-graceful-failure-comprehensive-testing-plan.md`

---

## Appendix: File Inventory

**Existing implementation files:**
- `bayesfilter/highdim/ledh_numerical_safety_tf.py` (new)
- `bayesfilter/highdim/ledh_unified_reset_tf.py` (modified)
- `bayesfilter/highdim/ledh_canonical_score_tf.py` (modified)
- `bayesfilter/inference/ledh_dual_parameter_target.py` (modified)

**Existing test files:**
- `tests/highdim/test_ledh_numerical_safety_tf.py` (new, 12 tests)
- `tests/highdim/test_ledh_reset_validity.py` (new, 5 tests)
- `tests/inference/test_dual_parameter_target_invalid.py` (new, 4 tests)
- `tests/integration/test_ledh_hmc_graceful_failure.py` (new, 3 tests)

**Proposed test files:**
- `tests/integration/test_lgssm_identifiable_specifications.py` (Phase 1)
- `tests/integration/test_lgssm_parameter_recovery_baseline.py` (Phase 2)
- `tests/integration/test_lgssm_graceful_failure_stress.py` (Phase 3)
- `tests/integration/test_sv_parameter_recovery.py` (Phase 4)
- `tests/integration/test_predator_prey_parameter_recovery.py` (Phase 5)
- `tests/integration/test_sir_partial_observability.py` (Phase 6)
- `tests/integration/test_ledh_graceful_failure_comprehensive.py` (Phase 7)

**Documentation files:**
- `docs/plans/ledh-graceful-failure-STATUS.md` (updated)
- `docs/plans/ledh-graceful-failure-implementation-plan.md` (updated)
- `docs/plans/ledh-graceful-failure-phase45-completion.md` (new)
- `docs/plans/ledh-graceful-failure-comprehensive-testing-plan.md` (new)
- `docs/plans/artifacts/ledh-graceful-failure-codex-review-memo.md` (this document)

---

**End of Review Request**
