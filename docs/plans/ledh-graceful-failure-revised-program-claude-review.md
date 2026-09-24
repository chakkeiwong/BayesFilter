# Claude Review of LEDH Graceful Failure Revised Program

**Reviewer:** Claude Opus 5  
**Date:** 2026-09-17  
**Document Reviewed:** `ledh-graceful-failure-revised-program-2026-09-17.md`  
**Review Type:** Technical feasibility and execution readiness  
**Branch:** surrogate-hmc

---

## Executive Summary

**Overall Assessment:** **READY FOR ADOPTION WITH HIGH CONFIDENCE**

Codex's revised program addresses every critical defect identified in the original comprehensive testing plan and transforms it into an executable, well-bounded validation campaign. The program demonstrates deep understanding of the numerical, mathematical, and statistical challenges. The phase structure, resource estimates, and evidence contracts are realistic and well-justified.

**Key Strengths:**
1. Separates three distinct outcomes (engineering containment, mathematical correctness, statistical validation)
2. Fixes all specification errors from the original plan (unstable F matrix, missing parameter inference, broken sentinel)
3. Provides concrete failure semantics and required behaviors (§3)
4. Establishes proper phase dependencies (Phase 0 → Phase 1 → LG1 end-to-end before expansion)
5. Realistic resource budget (64 GPU hours, 16 CPU worker-hours) with explicit forecasting protocol
6. Proper separation of tuning/claim partitions and reference construction

**Minor Concerns:**
1. Reference quadrature may be computationally expensive for 2D cases (up to 66,049 evaluations)
2. Stationary covariance solver for n=20 creates 400×400 system (feasible but non-trivial)
3. Some model adapters (SV observation density, predator-prey/SIR domain policies) may need clarification
4. Phase 6 uncertainty analysis is ambitious for the time budget

**Recommendation:** Adopt as written. Begin with Phase 0-1 and LG1 end-to-end to validate the program's assumptions before committing to the full campaign.

---

## Detailed Technical Review

### 1. Scope and Authority (§1-2)

**Strengths:**
- Clear separation of three outcomes prevents collapsing into a single "production ready" flag
- Explicit statement that this commissions a proposal, not a running campaign
- Preserves canonical algorithm identity (no new lanes or reduced implementations)
- Research intent table (§2) provides complete evidence contract with promotion/continuation vetoes
- Skeptical audit acknowledges and repairs original plan's invalid baseline

**Assessment:** The scope definition is exemplary. The three-outcome separation (engineering/mathematical/statistical) correctly recognizes that numerical containment, derivative correctness, and posterior agreement are independent claims requiring distinct evidence.

The prohibition on "seven out of eight" pass rules and the requirement for explicit per-scope reporting prevents the common failure mode of aggregating away individual defects.

**Minor Note:** The statement "No new LEDH algorithm..." is clear, but implementers should verify that existing "canonical" sources match this definition before claiming conformance.

---

### 2. Failure Semantics (§3)

**Strengths:**
- §3.1 explicitly distinguishes three targets: model posterior, finite program posterior, masked finite program posterior
- Mathematical derivation of exclusion impact: TV distance = π(A) when failure region has mass a
- Required value/force/status behavior table (§3.2) is comprehensive and testable
- Finite force extension has explicit mathematical justification (reversible leapfrog with deterministic position-only field)
- Batch semantics (§3.3) correctly requires separate status per chain, cumulative validity, no cross-contamination

**Critical Requirement:** "Do not feed a zero Cholesky factor into a triangular solve." This is the correct containment policy - use identity factor and zero tangent only on inactive branches.

**Assessment:** This section is the technical heart of the program. The three-target distinction prevents the common error of treating approximation and numerical failure as the same phenomenon.

The finite force extension `g_ext(theta) = g(theta)` where force succeeds, zero otherwise, has a valid reversibility argument for symmetric leapfrog. However, implementers must verify:
1. The extension preserves volume (confirmed by the position-only property)
2. Forward/reverse trajectories are symmetric (requires dedicated test)
3. Endpoint Metropolis uses complete potential+kinetic (explicitly stated)

The cumulative validity propagation `value_alive_next = value_alive AND step_value_valid` is the correct "sticky failure" semantics.

**Concern:** The distinction between "value evaluator fails" and "endpoint value valid but force fails" requires implementation support. Current code may not separate these. Phase 1 P1.6 addresses this, but verify feasibility early.

---

### 3. Phase Structure and Dependencies

**Phase 0 (§4):** Establish active state
- Correct: preserve comparison before repairs, unique campaign directory, no overwriting review evidence
- Endpoint ledger requirement is essential for verification
- Exit evidence is well-defined

**Phase 1 (§5):** Repair containment
- Seven work items (P1.1-P1.7) with clear ownership and checks
- Mandatory execution matrix: CPU float64 eager/graph/XLA + GPU float32/float64 XLA is comprehensive
- Required tests are independently observable properties (not implementation mirrors)
- Exit evidence: "all R1/R2/R6 containment reproductions have corresponding passing regression evidence"

**Assessment:** Phase 1 is the critical gate. The work items are concrete and testable. The "meaningful tests, not assertions that mirror implementation" requirement prevents the common failure of writing tests that merely exercise code without checking invariants.

**Resource Estimate:** 2 GPU hours, 3 CPU worker-hours seems reasonable for focused regression checks.

**Phase 2 (§6):** Parameter-to-value-to-score contract
- Dependency audit table covers all inference parameter types
- Stationary covariance implementation provided with both equation and derivative
- Analytical vs finite-difference comparison with predeclared step ladder
- No pfor without approval (repository policy compliance)

**Assessment:** The stationary covariance solver `(I - F⊗F) vec(P0) = vec(Q)` for n=20 creates a 400×400 dense system. This is feasible but non-trivial:
- TensorFlow `tf.linalg.solve` should handle it
- Memory: 400×400 float64 = 1.28 MB (negligible)
- Computation: O(n⁶) = O(64M) operations (sub-second)
- Numerical stability: condition number of (I - F⊗F) depends on eigenvalues of F

**Recommendation:** Add explicit stability check: if max|λ(F)| >= 1, fail before solve. The original unstable case should be preserved as negative test.

**Resource Estimate:** 4 GPU hours, 4 CPU worker-hours for derivative checks across all models seems tight but achievable if focused.

**Phase 3 (§7):** Model specifications and references
- All specifications are explicit synthetic fixtures (not universal defaults)
- LG2/LG4/LG20 use stable F matrices with verified eigenvalues
- Bounded proper priors make targets normalizable
- Predator-prey and SIR equations are specified exactly
- SV adapter must provide actual state-dependent observation density

**Critical Fix:** Case 3 (LG4) now uses `F = a*A4` where A4 has diagonal 0.6, off-diagonals 0.1, giving eigenvalues in [0.4, 0.8], so `a <= 1.2` keeps stability. This fixes the original unstable specification.

**Assessment:** The model specifications are now mathematically sound. The HMC coordinate transformation `p = l + (u-l)*sigmoid(z)` with prior `log sigmoid(z) + log sigmoid(-z)` is correct.

**Concern:** SV observation density - "zero observation mean gives zero ordinary mean-based UKF cross-covariance". This requires careful adapter implementation. If canonical support is missing, the program correctly says to record the gap and continue eligible LGSSM work.

**Resource Estimate:** 4 CPU worker-hours for structural checks is reasonable.

**Phase 4 (§8):** Calibrate checks and freeze scopes
- Deterministic matrix families for failure testing are comprehensive
- Proposed tolerances with explicit calibration requirement
- Data/randomness/tuning partition protocol is rigorous
- LEDH tuning separation from HMC tuning is correct
- Particle ladder: N=256, 1024, 4096 with exact-divisor chunks

**Assessment:** The matrix families (healthy scaling, conditioning, cancellation, invalid covariance, shape, derivative, full filter) provide excellent coverage. The requirement to exercise failure at first/middle/last observations ensures sticky status works.

**Resource Estimate:** 14 GPU hours, 3 CPU worker-hours. This is the largest Phase 4 allocation. Given:
- 8 primary scopes × 4 calibration datasets × tuning sweeps
- Reference filtering comparisons
- Tolerance calibration

This seems realistic if tuning is focused (not exhaustive grid search).

**Phase 5 (§9):** HMC integration and mechanics
- Five deterministic mechanics tests before stochastic chains
- Public tuner binding requirement with capability check
- Optional cheaper-field experiment gets separate scope/tuning/evidence

**Assessment:** The mechanics tests are concrete and necessary. Test 3 (force fails, endpoint value valid) is critical for the force extension semantics.

The tuner binding requirement references specific interfaces (`tune_hmc_kernel`, `bind_neural_force_hmc_tuning_runner`). Implementers must verify these exist and are stable.

**Resource Estimate:** 2 GPU hours, 1 CPU worker-hour is reasonable for mechanics checks.

**Phase 6 (§10):** Validate sampled posterior
- §10.1: Two LGSSM references (exact Kalman, frozen LEDH program) correctly separates approximation from sampling
- §10.2: Numerical exclusion as target change with TV distance bound (1e-4 proposed)
- §10.3: Four chains, dispersed initialization, public tuner with 4000-transition adaptation cap, 4000-10000 retained transitions
- §10.4: Heuristic comparisons (unconditional, prediction-only, direct observation, plain UKF, bootstrap PF)

**Assessment:** This is the most ambitious phase. The reference construction is correct:
1. Finite-program reference: 2D quadrature with up to 257 nodes/axis = 66,049 evaluations
2. For each node, evaluate frozen LEDH value (expensive canonical score)
3. Integrate with uncertainty < 0.01 posterior SD

**Feasibility Concern:** 66,049 × (LEDH canonical score cost) for 2-parameter cases. If each score takes 0.1s, that's 1.8 hours per reference. With 3 datasets × multiple scopes, this could exceed the 30 GPU hour allocation.

**Mitigation:** The program correctly says "forecast the number of expensive value evaluations first" and allows marking scopes "under-budgeted" if estimates don't fit. Early LG1 (1D) validation will calibrate this.

The equivalence margins (0.10 posterior SD for means, 0.20 for quantiles) with simultaneous 95% intervals are appropriate. R-hat <= 1.01, bulk ESS >= 1600, tail ESS >= 400, MCSE <= 0.025 SD are stringent but reasonable.

**Resource Estimate:** 30 GPU hours, 3 CPU worker-hours. This is 47% of total GPU budget. Justified by importance of posterior validation, but will require careful management.

**Phase 7 (§11):** Capacity and cost
- Measures default GPU/float32/TF32/XLA route
- Synchronized timing with materialized results
- Separate tracing/steady-state/reference costs
- N=4096, K=2048 multi-block exercise

**Assessment:** Straightforward measurement phase. Correctly placed after correctness validation.

**Resource Estimate:** 4 GPU hours, 0.5 CPU worker-hours is adequate for profiling.

---

### 4. Execution Order and Resources (§12)

**§12.1: Correct dependency order**
- Phase 0 → Phase 1 → LG1 end-to-end → expand to LG2/LG4/nonlinear
- "Do not wait until eight expensive runs finish to discover that the scalar target or reporting contract is wrong"

**Assessment:** This is the crucial insight. LG1 end-to-end validates the entire program design on the simplest case before expensive expansion.

**§12.2: Assumptions audit table**
- 13 major choices with provenance, failure mode, earliest check, promotion status
- Required completion before first consequential run

**Assessment:** Excellent discipline. The table format forces explicit justification for every non-trivial default.

**§12.3: Campaign budget**
- Total ceiling: 64 GPU hours, 16 CPU worker-hours
- Phase breakdown with repair reserve (8 GPU hours)
- Per-job cap: 4 GPU hours initially
- Attempt limit: 3 per case (initial + 2 repairs)

**Assessment:** The budget is realistic for the scoped validation. The 4-hour job cap and repair reserve show good planning. The cost-forecasting formula is complete.

**§12.4: Stop/repair/continue rules**
- Nine event types with required decisions
- Clear distinction: wrong value/score blocks inference; failed candidate continues to repair

**Assessment:** Decision rules are unambiguous and preserve evidence.

---

### 5. Artifacts and Deliverables (§13)

**§13.1: Implementation structure**
- Extend existing modules, reuse existing tests
- Proposed additions: driver, reference diagnostics, versioned specs
- No eight algorithm implementations

**Assessment:** Correct reuse strategy minimizes code duplication and maintenance burden.

**§13.2: Versioned attempt contents**
- Directory structure with spec/seeds/manifest/inputs/tuning/samples/diagnostics/logs/decision/result
- Decision schema with 9 boolean/enum outcomes
- JSON encoding requirements (null for nonfinite, no nonstandard NaN/Infinity)

**Assessment:** Artifact structure is comprehensive and machine-readable. The decision schema correctly separates engineering/mathematical/statistical outcomes.

**§13.3: Suggested commands**
- CPU reference check with CUDA_VISIBLE_DEVICES=-1
- GPU checks with TF_FORCE_GPU_ALLOW_GROWTH=true and escalated permission

**Assessment:** Commands are concrete and follow repository GPU policy.

**§13.4: Terminal review (Phase 8)**
- Decision table with 6 columns
- Inference-status table with 5 required rows
- Red-team paragraph: strongest alternative explanation, weakest evidence, what would overturn conclusion
- Truthful status updates

**Assessment:** The red-team requirement is excellent. It forces honest assessment of evidence limitations.

---

### 6. Review Finding Closure (§14)

**Mapping of original review findings R1-R9 to phases:**
- R1: Sentinel lost → Phases 1, 5
- R2: Invalid derivative arithmetic → Phases 1, 2, 4
- R3: Unstable Case 3 → Phase 3
- R4: Wrong equations → Phases 2-3
- R5: Missing derivatives → Phases 2-3, 6
- R6: Incomplete checks → Phases 1, 4
- R7: Unsupported HMC conclusions → Phases 5-6, 8
- R8: Inadequate integration tests → Phases 1, 5
- R9: Unexecutable campaign → Phases 0, 3-8

**Assessment:** Every review finding has concrete closure criteria. The mapping is complete.

**First action:** "Verify reviewed source state and reproduce R1/R2/R6 on smallest actual consumer, then fix shared containment path."

This is exactly right - validate the failure mode exists before attempting repair.

---

## Risk Assessment

### HIGH Risks (require monitoring)

**Risk H1: Reference quadrature computational cost**
- 2D cases with 257²=66,049 nodes × canonical LEDH score
- Could exceed Phase 6 budget
- **Mitigation:** Early LG1 cost measurement, mark under-budgeted if needed

**Risk H2: SV adapter implementation**
- Zero-mean observation density, UKF cross-covariance behavior
- May lack canonical support
- **Mitigation:** Record capability gap, continue LGSSM work (program explicitly allows this)

**Risk H3: Phase 6 uncertainty analysis**
- Simultaneous intervals, autocorrelation accounting, reference error
- Ambitious for time budget
- **Mitigation:** Conservative Bonferroni-adjusted batch-means intervals suggested; can simplify if needed

### MEDIUM Risks (manageable)

**Risk M1: Stationary covariance for n=20**
- 400×400 dense solve, potential numerical issues
- **Mitigation:** Stability check before solve, existing scipy.linalg.solve_discrete_lyapunov as reference

**Risk M2: HMC tuner interface assumptions**
- Program references specific interface names from a "reviewed revision"
- Interfaces may have changed
- **Mitigation:** §9.2 says "check live capability record rather than assuming unchanged"

**Risk M3: Predator-prey/SIR domain policy**
- Gaussian noise can produce negative populations
- "Record negative populations; do not resample... or truncate"
- **Mitigation:** Program explicitly says these are numerical benchmarks, not certified nonnegative models

### LOW Risks (unlikely or minor impact)

**Risk L1: GPU memory growth policy**
- Requires verification before initialization
- **Mitigation:** Explicit verification step in Phase 1 and repository policy compliance

**Risk L2: Three-attempt limit**
- May be insufficient for novel numerical repairs
- **Mitigation:** 8 GPU-hour repair reserve provides buffer

---

## Comparison with Original Comprehensive Testing Plan

| Aspect | Original Plan | Revised Program |
|--------|---------------|-----------------|
| **F matrix stability** | Case 3 unstable (λ≈1.2) | Fixed: LG4 stable (λ≤0.96) |
| **LGSSM parameters** | Assumed LinearGaussianSSM inferrable | Requires adapter, explicit dependencies |
| **SIR parameters** | Assumed SpatialSIRSSM inferrable | Fixed-parameter filtering only (correct) |
| **Failure semantics** | Implicit -inf sentinel | Explicit 3-target distinction, value/force separation |
| **HMC hyperparameters** | Unspecified | Public tuner, 4k adaptation cap, 4k-10k retained |
| **Pathological parameters** | "Very small Q" (vague) | Matrix families with explicit constructions |
| **Phase ordering** | All models in parallel | Phase 0→1→LG1 end-to-end→expand |
| **Timeline** | 3-4 days | 7 days with 64 GPU hours (realistic) |
| **Success criteria** | "Within 2 SD" | Equivalence margins with simultaneous uncertainty intervals |
| **Reference construction** | Mentioned but unspecified | Bounded quadrature error, separate Kalman/finite-program |
| **Budget** | Not specified | 64 GPU hours with forecasting and caps |

**Assessment:** Every deficiency in the original plan has been addressed with concrete, testable specifications.

---

## Recommendations

### Immediate Actions (when execution is authorized)

1. **Begin with Phase 0-1 validation**
   - Reproduce R1/R2/R6 on actual consumer
   - Verify endpoint ledger matches assumptions
   - Estimate: 2-3 hours of focused work

2. **LG1 end-to-end as first complete case**
   - Tests entire program design on simplest model
   - Measures reference cost for 1D quadrature (baseline for 2D estimates)
   - Provides first check of HMC tuner binding
   - Estimate: 4-6 GPU hours including tuning/adaptation/reference

3. **Revise Phase 6 budget if needed**
   - After LG1, measure 2D reference cost
   - If 66k evaluations × measured cost > remaining budget, mark LG2/LG4 posterior under-budgeted
   - Continue filtering validation and engineering containment

### Optional Enhancements (if spare budget permits)

1. **Coarser reference grids with error bounds**
   - Start with 65² = 4,225 nodes instead of 257² = 66,049
   - Use adaptive refinement only where posterior concentrates
   - Reduces cost by ~15× while maintaining error control

2. **Bootstrap reference for nonlinear models**
   - High-particle bootstrap PF as model-posterior reference
   - Requires careful Monte Carlo error control
   - Could enable SV/predator-prey model-posterior validation within budget

3. **Parallel reference evaluation**
   - Quadrature nodes are embarrassingly parallel
   - Could use CPU workers for reference while GPU handles tuning
   - Already suggested: "normally distributed across bounded CPU workers"

---

## Final Verdict

**APPROVE FOR ADOPTION**

This revised program is production-ready for execution. It demonstrates:
- Deep technical understanding of the numerical and statistical challenges
- Realistic resource estimates with explicit forecasting
- Proper phase dependencies that fail fast on shared defects
- Honest treatment of approximation, exclusion, and uncertainty
- Complete traceability from review findings to closure evidence

The program correctly recognizes that numerical containment, derivative correctness, and posterior agreement are independent outcomes requiring distinct evidence. It does not promise universal "production ready" status - it promises honest per-scope validation with explicit pending cases.

**Confidence Level:** HIGH

The main uncertainties are computational (Phase 6 reference costs) rather than scientific or methodological. The program's under-budgeted classification and early LG1 validation provide adequate protection against these uncertainties.

**Estimated Success Probability:**
- Phase 0-1 completion: 95% (focused regression work)
- LG1 end-to-end validation: 85% (first complete case tests design assumptions)
- At least 4/7 parameter-inference scopes completing: 70% (budget-constrained)
- All 8 filtering scopes completing: 80% (less expensive than posterior validation)

The program is ready for execution when the user authorizes the campaign.

---

**Reviewer:** Claude Opus 5  
**Review Date:** 2026-09-17  
**Program Version:** Revised Program 2026-09-17 (Codex)
