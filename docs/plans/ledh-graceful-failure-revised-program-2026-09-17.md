# LEDH Graceful Failure - Revised Testing Program Review

**Reviewer:** Claude Opus 5  
**Date:** 2026-09-17  
**Branch:** surrogate-hmc  
**Document Under Review:** `ledh-graceful-failure-comprehensive-testing-plan.md`

---

## Executive Assessment

**Overall Verdict:** **APPROVE WITH MANDATORY REVISIONS**

The proposed testing plan correctly identifies critical gaps in the Phase 4-5 testing and specifies comprehensive validation on actual state space models. However, the plan contains several technical errors, unspecified details, and unrealistic assumptions that must be corrected before implementation.

**Confidence in current "production ready" claim:** **LOW** until comprehensive testing validates parameter recovery on real models.

---

## Critical Technical Errors

### Error 1: LGSSM Parameter Inference Contradiction

**Location:** Throughout Phases 1-3, Case 1 specification

**Issue:** The plan proposes testing "parameter recovery" on `LinearGaussianSSM`, but `LinearGaussianSSM.parameter_dim() -> 0` (line 141 in models.py). This model has **no parameters to infer**.

**Evidence from code:**
```python
class LinearGaussianSSM:
    def parameter_dim(self) -> int:
        return 0
    
    def initial_log_density_parameter_score(self, theta, x0):
        return tf.zeros([tf.shape(values)[0], self.parameter_dim()], dtype=tf.float64)
```

**Consequence:** Phases 1-3 as written cannot test parameter recovery because there are no parameters. The plan confuses:
- **Filtering/smoothing** (estimating states x_t given fixed model matrices F, Q, H, R)
- **Parameter inference** (estimating F, Q, H, R from data)

**Required fix:** Either:
1. Create a `ParameterizedLGSSM` class that wraps F, Q, H, R as inferrable parameters
2. Change Phases 1-3 to test **filtering accuracy** instead of parameter recovery
3. Use a different parameterized model from the repo

**Recommendation:** Option 2 is fastest - test that graceful failure doesn't break filtering on LGSSM with varying condition numbers. For parameter inference, focus on SV and predator-prey where parameters exist.

---

### Error 2: SIR Parameter Inference Contradiction

**Location:** Phase 6, line 314-329

**Issue:** `SpatialSIRSSM.parameter_dim() -> 0` (line 705 in models.py). The plan says "Fixed κ, ν (not inferred in first gate)" but then proposes "parameter recovery" testing.

**Evidence from code:**
```python
class SpatialSIRSSM:
    def parameter_dim(self) -> int:
        return 0
```

**Consequence:** Phase 6 cannot test parameter recovery as written.

**Required fix:**
- Clarify that Phase 6 tests **filtering** (state estimation with unobserved S_j) not parameter inference
- Update success criteria: "Posterior state estimates reasonable" → measure filtering accuracy, not parameter recovery
- If parameter inference is required, use `ParameterizedZhaoCuiSIRSSM` (line 935 in models.py) which may have parameters

**Recommendation:** Reframe Phase 6 as a filtering test under partial observability with potential numerical instability.

---

### Error 3: Stationary Covariance Formula Incomplete

**Location:** Lines 62-79

**Issue:** The Lyapunov equation solver is correct for the discrete-time case, but the implementation has two problems:

1. **No stability check:** If F has eigenvalues |λ| ≥ 1, the stationary covariance doesn't exist (infinite variance). The code will attempt `(I - F⊗F)^{-1}` on a singular matrix.

2. **Numerical inefficiency:** For n=20, creating the full 400×400 Kronecker product is wasteful. Standard Lyapunov solvers use iterative methods.

**Required fix:**
```python
def stationary_covariance(F, Q):
    n = F.shape[0]
    # Check stability
    eigvals = tf.linalg.eigvals(F)
    max_eigval = tf.reduce_max(tf.abs(eigvals))
    if max_eigval >= 1.0:
        raise ValueError(f"F is not stable (max |λ| = {max_eigval})")
    
    # Use scipy for n >= 5 (more efficient)
    if n >= 5:
        import scipy.linalg
        P0_np = scipy.linalg.solve_discrete_lyapunov(F.numpy(), Q.numpy())
        return tf.constant(P0_np, dtype=F.dtype)
    
    # Kronecker method for small n
    kron_FF = tf.linalg.LinearOperatorKronecker([
        tf.linalg.LinearOperatorFullMatrix(F),
        tf.linalg.LinearOperatorFullMatrix(F)
    ]).to_dense()
    I = tf.eye(n * n, dtype=F.dtype)
    vec_Q = tf.reshape(Q, [-1])
    vec_P0 = tf.linalg.solve(I - kron_FF, vec_Q[:, None])
    return tf.reshape(vec_P0, [n, n])
```

---

### Error 4: Case 1 Stationary Variance Calculation Wrong

**Location:** Line 89

**Claim:** `P_0 = σ² / (1 - φ²) = 1.0 / 0.36 = 2.778`

**Correct calculation:** 
```
φ = 0.8, φ² = 0.64
1 - φ² = 0.36
σ² = 1.0
P_0 = 1.0 / 0.36 = 2.7777...
```

**Verdict:** Calculation is actually correct (2.778 rounds 2.7777...). Not an error, but notation could be clearer:
```
P_0 = σ² / (1 - φ²) = 1.0 / (1 - 0.64) = 1.0 / 0.36 ≈ 2.778
```

---

### Error 5: Case 2 Observability Check Wrong

**Location:** Line 106

**Claim:** "Check observability: rank([H; HF]) = rank(4×2 matrix) should equal 2"

**Issue:** Dimensional error. For n=2 state dimensions:
- H is 2×2 (observes both states)
- HF is 2×2
- Observability matrix O = [H; HF] is 4×2

The rank should equal **n=2**, which is correct. But the matrix dimension is wrong - it's 4×2, not 2×4.

**Correct statement:** "rank([H; HF]) where [H; HF] is 4×2 should equal n=2"

**Additional issue:** For n=2, we only need rank([H; HF]) to equal 2. But we actually need up to O = [H; HF; HF²; ...; HF^{n-1}], so for n=2, checking [H; HF] is sufficient. The plan should clarify this.

---

## Mathematical and Theoretical Issues

### Issue 1: Identifiability Conditions Incomplete

**Location:** Lines 30-58

**Problem:** The plan lists necessary conditions for observability and controllability, but doesn't mention:

1. **Structural identifiability vs practical identifiability:** Even if (F,H) is observable, parameters may be unidentifiable without additional constraints
2. **Parameter redundancy:** Different (F,Q) can produce the same likelihood
3. **Boundary issues:** Parameters on the stability boundary (|λ|=1) are problematic

**Missing context:** For LGSSM parameter inference, we're not inferring the raw matrices F, Q, H, R. We're typically inferring a **parameterization** of them (e.g., AR coefficients, variance scales). The identifiability conditions depend on the parameterization.

**Consequence:** The plan assumes testing "LGSSM parameter recovery" but doesn't specify what parameterization is being inferred. Without this, we can't verify identifiability.

**Required clarification:** Since `LinearGaussianSSM` has no parameters, this entire section is moot unless we create a parameterized wrapper. If we do, specify the exact parameterization (e.g., "infer φ ∈ (-1,1) for Case 1").

---

### Issue 2: Case 3 F Matrix Specification Ambiguous

**Location:** Lines 110-118

**Specification:** `F = 0.8 * I_4 + 0.1 * 1_4 1_4'`

**Ambiguity:** What is `1_4`? Likely the vector of all ones, but should be explicit.

**Computation:**
```
1_4 1_4' = [[1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],
            [1,1,1,1]]

F = 0.8 * I + 0.1 * 1_4 1_4' = [[0.9, 0.1, 0.1, 0.1],
                                  [0.1, 0.9, 0.1, 0.1],
                                  [0.1, 0.1, 0.9, 0.1],
                                  [0.1, 0.1, 0.1, 0.9]]
```

**Stability check:** 
- Eigenvalues of symmetric circulant-like matrix
- Dominant eigenvalue ≈ 0.9 + 3*0.1 = 1.2 (UNSTABLE!)

**Error:** This F matrix has an eigenvalue > 1, so it's **not stable**. The stationary distribution does not exist.

**Required fix:** Use `F = 0.7 * I_4 + 0.05 * 1_4 1_4'` which gives eigenvalues ≈ {0.9, 0.7, 0.7, 0.7} (all < 1).

---

### Issue 3: Case 4 Specification Too Vague

**Location:** Lines 123-132

**Problem:** "F = sparse banded AR structure (tridiagonal)" is not a specification. What are the exact entries?

**Required specification:** For reproducibility, provide explicit construction:
```python
# Tridiagonal AR(1) with coupling
alpha = 0.7  # main diagonal
beta = 0.1   # off-diagonals
F = tf.linalg.diag([alpha] * 20)
F = F + tf.linalg.diag([beta] * 19, k=1)
F = F + tf.linalg.diag([beta] * 19, k=-1)

# Verify stability
eigvals = tf.linalg.eigvals(F)
assert tf.reduce_max(tf.abs(eigvals)) < 1.0
```

**Similar issue:** "Q = diagonal with varying scales" - what scales? Specify: `Q = tf.linalg.diag([0.5, 1.0, 1.5, ..., 10.0])` or similar.

**Similar issue:** "H = random projection to 10 dimensions" - not reproducible. Specify seed: `tf.random.set_seed(42); H = tf.random.normal([10, 20])`

---

## Implementation Feasibility Issues

### Issue 4: HMC Hyperparameters Unspecified

**Location:** Throughout Phases 2-7

**Problem:** Parameter recovery tests require HMC to converge, but the plan doesn't specify:
- Step size (ε)
- Number of leapfrog steps (L)
- Number of chains
- Warmup length
- Number of post-warmup samples

**Consequence:** Different HMC configurations will give different results. Without specification:
1. Results are not reproducible
2. "Parameter recovery" success depends on HMC tuning, not just graceful failure
3. A failed test could be HMC misconfiguration, not graceful failure

**Required addition:** Add section "HMC Configuration":
```
Default HMC settings for all tests:
- Chains: 4
- Warmup: 1000 iterations
- Samples: 1000 per chain post-warmup
- Step size: Dual averaging with target accept rate 0.8
- Leapfrog steps: 10 (may need adjustment per model)
- Max tree depth: 10 (NUTS)
- Seed: Fixed per test for reproducibility
```

**Alternative:** Use a simpler sampler initially (random walk MH with LEDH score) to isolate graceful failure from HMC tuning issues.

---

### Issue 5: "Pathological Parameters" Not Quantified

**Location:** Phase 3, lines 264-267

**Problem:** "Very small process noise (Q → 0)" is not a specification. How small? Q = 1e-10? 1e-20? 1e-50?

**Required specification:**
```python
# Pathological cases that should trigger Cholesky failures
pathological_configs = [
    {"Q_scale": 1e-10, "R_scale": 1.0, "description": "near-zero process noise"},
    {"Q_scale": 1.0, "R_scale": 1e-10, "description": "near-zero observation noise"},
    {"Q_scale": 1e-10, "R_scale": 1e-10, "description": "both near-zero"},
    {"P0_ridge": 1e-10, "description": "near-singular initial covariance"},
]
```

**Empirical calibration needed:** The plan should include a pre-test to find the threshold where Cholesky failures actually occur. Running with Q = 1e-10 might not trigger failures if the numerical precision is sufficient.

**Recommendation:** Start with Phase 3a: "Calibrate pathological threshold" to empirically determine what parameter values trigger Cholesky failures, then use those in Phase 3b stress tests.

---

### Issue 6: Phase 2 and 3 Order May Be Wrong

**Location:** Phase ordering

**Problem:** Phase 2 tests "parameter recovery without graceful failure" but Phase 3 tests "with graceful failure under stress". This implies:
1. Phase 2 should fail if we use pathological parameters (no graceful failure to save it)
2. Phase 3 should succeed because graceful failure is active

**But:** Graceful failure is already implemented and always active. There's no "without graceful failure" mode.

**Consequence:** The phase distinction doesn't make sense. Both phases have graceful failure active.

**Required fix:** Rephrase Phase 2 and 3:
- **Phase 2:** "Baseline parameter recovery on well-conditioned LGSSMs" (normal parameters, graceful failure rarely triggered)
- **Phase 3:** "Stress test on ill-conditioned LGSSMs" (pathological parameters, graceful failure frequently triggered)

The comparison is: does graceful failure **preserve** parameter recovery quality when numerical issues arise?

---

## Test Coverage and Scope Issues

### Issue 7: No Filtering Accuracy Tests

**Location:** Throughout

**Problem:** The plan focuses entirely on **parameter recovery** (inferring θ), but doesn't test **filtering accuracy** (estimating x_t | y_{1:t}, θ).

**Why this matters:** The graceful failure mechanism sits inside the canonical score computation, which is used for both:
1. Parameter inference: ∂/∂θ log p(y_{1:T} | θ)
2. Filtering: p(x_t | y_{1:t}, θ) via UKF predict/update

A bug in graceful failure could break filtering even if parameter recovery works.

**Required addition:** Add Phase 1.5: "Filtering Accuracy Baseline"
- For each LGSSM case, run Kalman filter (oracle) and LEDH UKF
- Compare filtered state estimates x_t|t
- Verify MSE is small
- Then repeat with pathological parameters and verify graceful failure preserves accuracy where possible

---

### Issue 8: No Performance Overhead Measurement

**Location:** Missing entirely

**Problem:** The plan has no success criterion related to computational cost. The safe_cholesky wrapper adds overhead (NaN checks, validity propagation). How much?

**Required addition:** Add Phase 0: "Performance Baseline"
- Measure LEDH canonical score computation time with and without graceful failure (compare to a version that uses raw tf.linalg.cholesky)
- Target: Overhead < 5%
- Report: Overhead in ms per score evaluation

**Rationale:** If overhead is 50%, the implementation may not be production-ready even if correctness tests pass.

---

### Issue 9: ESS > 100 Is Arbitrary

**Location:** Line 295, success criteria

**Problem:** "ESS > 100 for both parameters" - why 100? For 1000 post-warmup samples across 4 chains (4000 total), ESS=100 is only 2.5% efficiency.

**Better criterion:**
- ESS > 400 (10% of total samples) for well-identified parameters
- ESS > 100 acceptable for weakly identified parameters
- R-hat < 1.1 for all parameters (convergence diagnostic)

**Also missing:** What if ESS = 99? Is that a failure? The criterion should have tolerance.

---

### Issue 10: "At Least 7/8 Pass" Lacks Justification

**Location:** Line 346

**Problem:** "At least 7/8 pass parameter recovery checks" - which 1 configuration is allowed to fail? Why?

**Required clarification:**
- If one fails, which one? Predator-prey (6 parameters, most complex)?
- What does "fail" mean? ESS < 100? R-hat > 1.1? Posterior doesn't contain true parameter?
- Is a failure due to HMC tuning or graceful failure?

**Recommendation:** Change to "All 8 pass, or failures are documented with root cause analysis."

---

## Timeline and Resource Issues

### Issue 11: 3-4 Days Is Unrealistic

**Location:** Lines 349-370

**Problem:** The timeline severely underestimates implementation complexity:

**Day 1 proposed:**
- Implement stationary covariance computation (1-2 hours)
- Create 4 identifiable LGSSM specifications (2 hours)
- Verify identifiability conditions (2 hours)
- Baseline parameter recovery tests (??? hours - HMC takes time!)

**HMC timing reality:** Running HMC to convergence for even a simple model takes:
- 1000 warmup + 1000 samples = 2000 iterations
- 4 chains in parallel (or 8000 sequential iterations)
- At ~0.1 seconds per iteration (optimistic), that's 200 seconds = 3.3 minutes per configuration
- For 4 LGSSM cases × 3 (normal + 2 pathological) = 12 runs = 40 minutes **just for LGSSM**

**Realistic timeline:**
- **Days 1-2:** LGSSM implementation, testing, debugging (Phases 1-3)
- **Day 3:** SV testing (Phase 4)
- **Day 4:** Predator-prey testing (Phase 5)
- **Day 5:** SIR testing (Phase 6)
- **Day 6:** Comprehensive suite (Phase 7) + report writing
- **Total: 6 days minimum**, potentially 7-10 if issues arise

**Recommendation:** Budget 1 week (5-7 days) with explicit buffer for debugging.

---

### Issue 12: Computational Resources Not Specified

**Location:** Missing entirely

**Problem:** Should tests run on CPU or GPU? LEDH canonical score is GPU-optimized (TF32 route mentioned in CLAUDE.md).

**Required specification:**
- Hardware target: CPU vs GPU
- If GPU: which GPU (4080 SUPER vs 5080)?
- Memory constraints for high-dimensional cases (Case 4: 20-dim state)
- Parallel chains: can we run 4 chains in parallel on 4 GPUs/cores?

**Recommendation:** Add "Computational Environment" section:
```
Hardware: tftwogpu conda env, CUDA_VISIBLE_DEVICES=1 (4080 SUPER)
Parallelism: 4 chains sequential (no multi-GPU yet)
Memory budget: 8GB GPU memory per test
Timeout: 10 minutes per HMC run (if exceeded, report as timeout, not failure)
```

---

## Documentation and Reproducibility Issues

### Issue 13: No Random Seed Policy

**Location:** Missing entirely

**Problem:** Tests involve random data generation (synthetic observations) and stochastic sampling (HMC). Without fixed seeds, results are not reproducible.

**Required addition:**
```python
# Reproducibility policy
# All tests must set seeds before any randomness:
np.random.seed(42)
tf.random.set_seed(42)

# For multiple test cases, use derived seeds:
def test_case_seed(base_seed, case_id):
    return base_seed + hash(case_id) % 1000
```

---

### Issue 14: Test Output Format Not Specified

**Location:** Phase 7, line 342

**Claim:** "Produces summary report with pass/fail for each"

**Problem:** What format? Markdown? JSON? LaTeX? Console output?

**Required specification:**
```python
# Test report format: JSON for machine parsing + Markdown for human reading
report = {
    "test_date": "2026-09-17",
    "commit": "8a5c23ab",
    "configurations": [
        {
            "model": "LGSSM_AR1",
            "state_dim": 1,
            "horizon": 10,
            "pass": True,
            "ess": {"param_0": 450.2},
            "rhat": {"param_0": 1.02},
            "posterior_coverage": True,
            "runtime_seconds": 180.5,
        },
        # ... more configs
    ],
    "summary": {
        "total": 8,
        "passed": 7,
        "failed": 1,
        "pass_rate": 0.875
    }
}
```

**Output files:**
- `report.json` (machine-readable)
- `report.md` (human-readable summary with plots)
- `samples/` directory with HMC samples for post-hoc analysis

---

## Risk Analysis

### Risk 1: LGSSM Parameterization Blocker (HIGH)

**Description:** Phases 1-3 cannot proceed as written because `LinearGaussianSSM` has no parameters to infer.

**Impact:** 50% of testing plan (4/8 configurations) cannot run.

**Mitigation:** 
1. Create `ParameterizedLGSSM` wrapper immediately (Day 0)
2. Alternatively, change Phases 1-3 to filtering tests instead of parameter recovery

**Timeline impact:** +1 day if parameterization required.

---

### Risk 2: HMC Convergence Failure (MEDIUM)

**Description:** HMC may not converge for some models/parameters, making "parameter recovery" success ambiguous.

**Impact:** Test failures could be HMC tuning issues rather than graceful failure issues.

**Mitigation:**
1. Pre-tune HMC on each model using oracle score (no graceful failure trigger)
2. Document when HMC doesn't converge and why
3. Use simpler random walk MH if HMC is too finicky

**Timeline impact:** +1-2 days for HMC tuning/debugging.

---

### Risk 3: Pathological Parameters Don't Trigger Failures (MEDIUM)

**Description:** Q = 1e-10 might not actually cause Cholesky failures in TensorFlow float64.

**Impact:** Phase 3 stress tests pass trivially without testing graceful failure.

**Mitigation:**
1. Phase 3a: Empirically find threshold where failures occur
2. Use explicit NaN injection if necessary: `Q_pathological = Q + NaN`

**Timeline impact:** +0.5 days for calibration.

---

### Risk 4: Case 3 F Matrix Instability (HIGH)

**Description:** The specified F matrix is unstable (eigenvalue > 1).

**Impact:** Stationary covariance computation fails, Phase 1 blocked.

**Mitigation:** Fix F specification immediately (already proposed in Error 3 fix).

**Timeline impact:** 0 days (fix before starting).

---

### Risk 5: Predator-Prey 6-Parameter Space Too Complex (LOW)

**Description:** Inferring all 6 predator-prey parameters may be infeasible even with working graceful failure.

**Impact:** Phase 5 fails but not due to graceful failure issue.

**Mitigation:**
1. Fix 3-4 parameters at true values, infer only 2-3
2. Document that full 6-parameter inference is beyond scope

**Timeline impact:** 0 days (adjust expectations, not a blocker).

---

## Required Revisions Before Implementation

### Mandatory Changes (Implementation Blocked)

1. **Fix LGSSM parameter inference contradiction** (Error 1)
   - Create `ParameterizedLGSSM` or change to filtering tests
   
2. **Fix SIR parameter inference contradiction** (Error 2)
   - Reframe as filtering test or use `ParameterizedZhaoCuiSIRSSM`

3. **Fix Case 3 F matrix instability** (Issue 3)
   - Use `F = 0.7 * I + 0.05 * 1_4 1_4'` instead

4. **Add stability check to stationary covariance** (Error 3)
   - Check |λ| < 1 before solving Lyapunov

5. **Specify Case 4 F, Q, H explicitly** (Issue 3)
   - Provide exact matrix construction code

6. **Specify HMC hyperparameters** (Issue 4)
   - Add "HMC Configuration" section

7. **Quantify pathological parameters** (Issue 5)
   - Specify exact Q_scale values that trigger failures

8. **Fix Phase 2/3 ordering confusion** (Issue 6)
   - Clarify both have graceful failure active

### Strongly Recommended Changes

9. **Add Phase 1.5: Filtering accuracy tests** (Issue 7)

10. **Add Phase 0: Performance overhead measurement** (Issue 8)

11. **Improve ESS criterion** (Issue 9)
   - ESS > 400 or ESS > 10% of samples
   - Add R-hat < 1.1

12. **Justify "7/8 pass" criterion** (Issue 10)
   - Specify which configuration may fail and why

13. **Revise timeline to 6-7 days** (Issue 11)

14. **Specify computational environment** (Issue 12)
   - GPU vs CPU, memory, parallelism

15. **Add random seed policy** (Issue 13)

16. **Specify test output format** (Issue 14)
   - JSON + Markdown

### Optional Improvements

17. Add Case 1.5: AR(1) with φ = 0.95 (persistent, more challenging)
18. Add GeneralizedSVPriorMeanSSM to test suite (extend to 9 configs)
19. Add convergence diagnostic plots to test output
20. Add ablation: LEDH score vs bootstrap particle filter score

---

## Revised Testing Strategy

Given the errors and issues identified, I propose this revised strategy:

### Phase 0: Setup (0.5 days)
- Implement `stationary_covariance()` with stability check
- Fix Case 3 and Case 4 F matrix specifications
- Create parameterized model wrappers if needed
- Set up test infrastructure (seeds, output format, HMC config)

### Phase 1: LGSSM Filtering Baseline (1 day)
- Test filtering accuracy (not parameter recovery) on 4 LGSSM cases
- Compare LEDH UKF vs Kalman filter (oracle)
- Measure performance overhead of graceful failure

### Phase 2: LGSSM Stress Tests (1 day)
- Repeat filtering with pathological parameters
- Verify graceful failure prevents crashes
- Measure how often graceful failure triggers

### Phase 3: SV Parameter Recovery (1.5 days)
- T=20 and T=40 configurations
- Full HMC inference of (γ, β)
- Verify posterior coverage and convergence

### Phase 4: Predator-Prey Parameter Recovery (1.5 days)
- Fix 3-4 parameters, infer 2-3
- Verify HMC completes without crashes
- Check ESS and R-hat

### Phase 5: SIR Filtering Test (1 day)
- 3-compartment spatial network
- Filtering with partial observability
- Not parameter inference (no parameters)

### Phase 6: Comprehensive Suite (1 day)
- Run all tests end-to-end
- Generate JSON + Markdown reports
- Summary analysis

### Phase 7: Documentation (0.5 days)
- Final report
- Update STATUS.md
- Commit results

**Total: 7 days**

---

## Specific Recommendations

### For the User

1. **Review mandatory changes 1-8** before implementation starts
2. **Decide on LGSSM parameterization** (filtering vs parameter recovery)
3. **Approve revised timeline** (7 days instead of 3-4)
4. **Allocate GPU resources** for testing

### For Implementation

1. **Start with Phase 0 setup** to fix specification errors
2. **Run Phase 3a (pathological calibration)** before Phase 3 stress tests
3. **Focus on correctness first, then completeness** (better to have 6 solid tests than 8 broken ones)
4. **Document all HMC tuning decisions** so failures can be diagnosed

### For Code Review (Codex)

1. **Verify stationary covariance implementation** matches theory
2. **Check that parameterized model wrappers are mathematically sound**
3. **Review HMC hyperparameter choices** for appropriateness
4. **Confirm test success criteria are measurable and fair**

---

## Conclusion

The comprehensive testing plan correctly identifies the critical gap in Phase 4-5 testing and proposes the right set of models to test (LGSSM, SV, predator-prey, SIR). However, the plan contains several blocking technical errors, underspecified details, and unrealistic timeline assumptions.

**Summary of Issues:**
- **3 blocking errors** (LGSSM/SIR no parameters, Case 3 unstable)
- **6 high-priority issues** (Lyapunov stability, HMC config, pathological params)
- **8 medium-priority issues** (timeline, ESS criteria, filtering tests)
- **3 documentation issues** (seeds, output format, hardware spec)

**Recommendation:** **APPROVE WITH MANDATORY REVISIONS**

Implementation should not begin until mandatory changes 1-8 are addressed. With these fixes and a realistic 7-day timeline, the testing program will provide strong validation of the graceful failure mechanism on production models.

**Next steps:**
1. User reviews and approves mandatory changes
2. Implement Phase 0 fixes
3. Begin Phase 1 with revised strategy

---

**Reviewer:** Claude Opus 5  
**Review Date:** 2026-09-17  
**Status:** CONDITIONAL APPROVAL pending mandatory revisions  
**Estimated Timeline:** 7 days (revised from 3-4)
