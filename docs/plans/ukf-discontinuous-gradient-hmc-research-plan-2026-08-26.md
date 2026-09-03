# UKF Discontinuous Gradient HMC Research Plan

**Date:** 2026-08-26  
**Status:** Proposed  
**Lead:** Research Team  
**Target Model:** DSGE NAWM II class (T=120, d=100, ~12,000 latent dimensions)  
**Problem:** UKF variance collapse at ZLB creates parameter-space gradient discontinuities that break HMC leapfrog integration

## Executive Summary

This plan addresses the UKF filtering collapse problem documented in §7 of `docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex`. When observations pin state variables to the zero lower bound, UKF filtering variance collapses, sigma points cluster, and small parameter perturbations shift the entire collapsed cloud across steep softplus boundaries. The result is a **continuous-kink target** (ΔU = 0, but ∇U jumps) that causes O(ε·Δg) leapfrog energy errors and chain stagnation.

The research program has five phases:

1. **Measurement and characterization** (1 week): Diagnose boundary geometry, measure gradient jump magnitudes, establish ground truth
2. **Event-aware integration pilot** (3 weeks): Implement Tran-Kleppe GRHMC for continuous-kink targets, test on toy model
3. **UKF variance inflation** (2 weeks): Test inflation schedules, validate scientific fidelity
4. **Neural surrogate exploration** (4 weeks): Train implicit boundary surrogates, compare gradient quality
5. **Production candidate selection** (1 week): Compare viable candidates, select default policy

**Total Duration:** 11 weeks (2.75 months)  
**Compute Budget:** 200 GPU-hours for phases 1-3, 800 GPU-hours for phase 4  
**Success Criterion:** Achieve R̂ < 1.01, ESS/grad > 0.1, sampling divergence rate < 1% on NAWM II posterior with T=120, d=100

---

## Phase 0: Prerequisites (Complete)

**Status:** ✓ Complete  
**Artifact:** `docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex` §7

Comprehensive survey section documenting:
- UKF variance collapse mechanism (§7.1-7.2)
- Continuous-kink vs value-jump geometry (§7.3-7.4)
- 5-step measurement protocol (§7.5)
- Why particle filters and DHMC don't solve this (§7.6, §7.8)
- Candidate solutions by geometry type (§7.7)
- Open research questions (§7.9)

---

## Phase 1: Measurement and Characterization

**Duration:** 1 week  
**Compute Budget:** 20 GPU-hours  
**Objective:** Diagnose boundary geometry, measure gradient discontinuity magnitudes, establish baseline HMC failure modes

### 1.1 Implement 5-Step Diagnostic Protocol

**Target:** `bayesfilter/hardbound/ukf_boundary_diagnostic.py`

Implement the measurement protocol from survey §7.5:

1. **Filtering variance time series:** Plot σ²ᵢ,ₜ(θ) for constrained states across t=1..T
   - Identify collapse episodes (σ² < threshold)
   - Measure cluster radius in state space
   
2. **Sigma-point trajectories:** Visualize χ̃ᵢ,ₜ trajectories for collapsed episodes
   - Count sigma points within ε-ball of softplus kink
   - Measure nearest-neighbor distances
   
3. **Parameter-space gradient samples:** For parameters θ₁, θ₂ affecting constrained states:
   - Sample ∇θ log p(y | θ) on dense grid near posterior mode
   - Compute finite-difference approximation: [log p(θ + δ) - log p(θ)] / δ
   - Identify discontinuity surfaces where |FD - AD| > tolerance
   
4. **Leapfrog energy error:** Run HMC with various step sizes ε
   - Track |ΔH| per leapfrog step
   - Correlate energy errors with proximity to identified discontinuity surfaces
   - Measure ε-threshold below which |ΔH| < acceptable bound
   
5. **Chain mixing metrics:** Run baseline HMC for 10,000 draws
   - R̂ by parameter (target < 1.01)
   - ESS/iteration (target > 0.01)
   - ESS/gradient evaluation (target > 0.1)
   - Divergence rate (target < 1%)
   - Acceptance rate by proposal distance

**Baseline Model:** Start with toy 2D softplus-constrained Gaussian
```python
# Toy model: x ~ N(μ, Σ), y_t = softplus(x_1) + ε_t
# Parameter: θ = μ_1, controls how close x_1 is to kink at 0
# When θ ≈ 0, small changes shift probability mass across steep softplus region
```

**Promotion Criteria:**
- All 5 diagnostic steps execute without error
- Gradient discontinuity surfaces identified with spatial resolution < 0.1 sd
- Leapfrog energy error shows predicted O(ε·Δg) scaling
- Baseline HMC fails: R̂ > 1.05 or divergence rate > 5% or ESS/grad < 0.01

**Veto Criteria:**
- Cannot identify discontinuity surfaces (suggests wrong problem diagnosis)
- Energy error does not correlate with boundary proximity (suggests different failure mode)
- Baseline HMC succeeds (problem doesn't exist on toy model)

**Artifacts:**
- `docs/plans/artifacts/phase1_toy_boundary_diagnostic.npz` (diagnostic arrays)
- `docs/plans/artifacts/phase1_toy_baseline_hmc_trace.npz` (baseline chain)
- `docs/plans/artifacts/phase1_measurement_result.md` (measurements, plots, interpretation)

---

### 1.2 Scale to Hardbound Test Case

**Target Model:** G2_3 fixture from `bayesfilter/hardbound/model_tf.py`
- 8 state variables, 6 structural + 3 log-scale parameters
- T=40 horizon
- 2 states near ZLB (level, slope)
- Target dimension: 337 = 8×40 + 9 parameters

**Tasks:**
- Rerun 5-step diagnostic on G2_3
- Compare gradient discontinuity pattern to toy model
- Measure Δg (gradient jump magnitude) for parameters controlling ZLB proximity
- Identify which parameters exhibit worst discontinuities
- Establish step-size threshold εₘₐₓ above which HMC fails

**Promotion Criteria:**
- Gradient discontinuities confirmed on realistic model
- Δg magnitude measured (expected 10² - 10⁴ depending on parameter)
- Baseline HMC fails with ε > εₘₐₓ (expected εₘₐₓ ≈ 10⁻⁴ to 10⁻⁵)
- Problem severity quantified for production target scale

**Veto Criteria:**
- Baseline HMC succeeds at reasonable step size (ε ≥ 10⁻³)
- No correlation between collapse episodes and gradient jumps

**Artifacts:**
- `docs/plans/artifacts/phase1_g2_boundary_diagnostic.npz`
- `docs/plans/artifacts/phase1_g2_baseline_hmc_trace.npz`
- `docs/plans/artifacts/phase1_characterization_report.md`

**Compute:** 15 GPU-hours (5 for toy, 10 for G2_3)

---

## Phase 2: Event-Aware Integration Pilot

**Duration:** 3 weeks  
**Compute Budget:** 60 GPU-hours  
**Objective:** Implement and test Tran-Kleppe GRHMC (2025) event-aware integration for continuous-kink targets

### 2.1 Literature Integration

**Source:** Tran & Kleppe (2025) "Gradient-based MCMC for models with discontinuous likelihoods"

**Tasks:**
1. Obtain and store paper: `docs/.localresources/tran_kleppe_2025_gradient_rhmc.pdf`
2. Extract algorithmic details:
   - Event detection: how to identify when leapfrog crosses discontinuity surface
   - Step truncation: shrink ε to land exactly at boundary
   - Gradient switching: which gradient to use after crossing
   - Momentum handling: reflection, refraction, or continuation
3. Identify assumptions and applicability conditions
4. Map paper notation to BayesFilter conventions

**Promotion Criteria:**
- Paper obtained and stored locally
- Algorithm pseudocode extracted and verified against paper
- Applicability to continuous-kink targets (ΔU=0 case) confirmed
- Implementation plan written with exact equations

**Veto Criteria:**
- Paper assumes value jumps only (ΔU ≠ 0), cannot handle continuous kinks
- Method requires dense Hessian or other infeasible computation
- Method fundamentally incompatible with UKF autodiff

**Artifacts:**
- `docs/.localresources/tran_kleppe_2025_gradient_rhmc.pdf`
- `docs/plans/artifacts/phase2_tran_kleppe_algorithm_extraction.md`
- `docs/plans/artifacts/phase2_implementation_spec.md`

---

### 2.2 Implement Event-Aware Leapfrog

**Target:** `bayesfilter/inference/event_aware_hmc.py`

**Core Components:**

1. **Boundary implicit function:** Given parameter θ, compute implicit function F(θ) = 0 defining discontinuity surface
   - For UKF: F(θ) relates to collapsed variance or sigma-point cluster condition
   - May require numerical approximation or learned surrogate

2. **Event detection:** During leapfrog θ → θ', test if trajectory crosses F=0
   - Bracket crossing: sign(F(θ)) ≠ sign(F(θ'))
   - Bisection or root-finding to locate crossing point θ*

3. **Step truncation:** Replace full step ε with ε* that lands exactly at θ*
   - Complete leapfrog up to boundary
   - Record which parameters crossed, gradient before/after

4. **Gradient switching:** Compute ∇U(θ*⁻) and ∇U(θ*⁺)
   - Use appropriate gradient for continuation
   - Handle momentum according to Tran-Kleppe prescription

5. **Continuation:** Resume leapfrog with remaining budget ε - ε*

**Implementation Strategy:**
- Start with 1D toy model (analytic boundary F(θ) = θ - θ_kink)
- Test event detection, truncation, gradient switching independently
- Verify energy conservation: |ΔH| ≤ O(ε³) even across discontinuity
- Scale to 2D toy model, then G2_3

**Promotion Criteria:**
- Event detection finds boundary crossings with error < 10⁻⁶
- Truncated leapfrog reproduces hand-calculated trajectory
- Energy error across boundary: |ΔH| ≤ 10⁻⁴ for ε = 10⁻³ (vs baseline |ΔH| ≈ ε·Δg ≈ 10⁻¹)
- Works on 2D toy model with analytic boundary

**Veto Criteria:**
- Cannot detect boundary crossings reliably (false positive rate > 10%)
- Truncation introduces worse energy error than baseline
- Computational cost per leapfrog step > 10× baseline

**Artifacts:**
- `bayesfilter/inference/event_aware_hmc.py`
- `tests/inference/test_event_aware_leapfrog.py` (unit tests)
- `docs/plans/artifacts/phase2_event_aware_implementation_result.md`

**Compute:** 20 GPU-hours

---

### 2.3 Event-Aware HMC on G2_3

**Objective:** Run full event-aware HMC chains on G2_3 hardbound test case

**Experiment Design:**
- 4 chains, 10,000 draws each, 5,000 warmup
- Step size: ε = 10⁻³ (baseline fails at this ε)
- Mass matrix: diagonal, estimated during warmup
- Boundary function: empirical discontinuity surface from Phase 1
- Compare to baseline HMC at ε = 10⁻⁴ (slow but stable reference)

**Metrics:**
- R̂ < 1.01 (promotion criterion)
- ESS/grad > 0.1 (promotion criterion)
- Divergence rate < 1% (promotion criterion)
- Acceptance rate 60-90% (diagnostic)
- Wall time vs baseline HMC
- Event detection rate (how often boundary is crossed)

**Promotion Criteria:**
- All three primary metrics pass
- At least 10× faster than baseline HMC at safe step size
- No catastrophic failures (NaN, crash, infinite loop)

**Veto Criteria:**
- R̂ > 1.05 (method doesn't solve mixing problem)
- ESS/grad < 0.01 (no better than baseline)
- Boundary detection cost dominates: wall time > baseline despite larger ε

**Continuation Veto:**
- Method works but <2× speedup (not worth production complexity)
- Requires model-specific boundary function tuning (not general enough)

**Artifacts:**
- `docs/plans/artifacts/phase2_event_aware_g2_trace.npz`
- `docs/plans/artifacts/phase2_event_aware_g2_result.md`
- Decision table: promote to Phase 5 comparison, continue to Phase 3 inflation, or abandon

**Compute:** 40 GPU-hours

---

## Phase 3: UKF Variance Inflation

**Duration:** 2 weeks  
**Compute Budget:** 40 GPU-hours  
**Objective:** Test whether artificially inflating UKF filtering variance prevents collapse and smooths gradients

### 3.1 Inflation Schedule Design

**Mechanism:** Add process noise or measurement noise to prevent σ² → 0

**Candidate Schedules:**

1. **Additive process inflation:** Qₜ → Qₜ + δI, tuned δ
   - Pros: Simple, preserves UKF update structure
   - Cons: Changes filtering target, may degrade filter accuracy

2. **Multiplicative variance floor:** σ²ₜ → max(σ²ₜ, σ²_min)
   - Pros: Only affects collapsed regions
   - Cons: Non-differentiable clipping breaks autodiff

3. **Softplus variance floor:** σ²ₜ → σ²ₜ + (1/β) log(1 + exp(β σ²_min))
   - Pros: Smooth, differentiable, tunable steepness
   - Cons: Always inflates, even when not needed

4. **Adaptive inflation:** Detect collapse (σ² < threshold), increase Q_t dynamically
   - Pros: Only inflates when necessary
   - Cons: Adaptive rule may create new discontinuities

**Implementation:**
- Implement all 4 schedules in `bayesfilter/highdim/ukf_inflation.py`
- Test on toy model: does inflation prevent gradient discontinuity?
- Measure inflation-vs-accuracy tradeoff

**Promotion Criteria:**
- At least one schedule eliminates gradient discontinuities (|∇U jump| < 10% of baseline)
- Inflated filter still tracks true state (RMSE < 2× optimal UKF)
- Autodiff works (no NaN gradients)

**Veto Criteria:**
- All schedules fail to smooth gradient (jumps remain > 50% of baseline)
- Inflation required to smooth gradient degrades filter accuracy catastrophically (RMSE > 10× optimal)

**Artifacts:**
- `bayesfilter/highdim/ukf_inflation.py`
- `docs/plans/artifacts/phase3_inflation_schedule_comparison.md`
- `docs/plans/artifacts/phase3_inflation_toy_result.npz`

**Compute:** 15 GPU-hours

---

### 3.2 Scientific Validity Check

**Critical Question:** Does variance inflation preserve the posterior target?

**Answer:** NO, inflation changes the filtering model, which changes p(y|θ)

**Implication:** Inflated UKF produces a surrogate posterior, not the target posterior

**Required Evidence:**
1. **Bias quantification:** Compare inflated posterior to ground truth (particle filter or grid)
   - Measure KL divergence, Wasserstein distance, or moment mismatch
   - Acceptable bias threshold: <5% marginal sd shift, <0.1 correlation change

2. **Inflation-bias curve:** Sweep δ or σ²_min, measure gradient smoothness vs posterior bias
   - Find Pareto frontier: minimum inflation that achieves acceptable smoothness
   - Document tradeoff explicitly

3. **Scientific interpretation:** State clearly in docs and code:
   - "Inflated UKF produces approximate posterior for computational tractability"
   - "Bias is [measured value]; acceptable for [stated reason]"
   - "Not a ground-truth estimator; use for exploration, not final inference"

**Promotion Criteria:**
- Bias quantified and documented
- Pareto-optimal inflation level identified
- Bias is acceptable for stated use case (e.g., MCMC exploration, not publication-grade inference)

**Veto Criteria:**
- Cannot quantify bias (no reference available)
- Minimum inflation for smooth gradients produces unacceptable bias (>10% sd shift)

**Continuation Veto:**
- Bias is acceptable, but method is scientifically inferior to event-aware HMC or surrogates
- Cannot defend inflated UKF for production use

**Artifacts:**
- `docs/plans/artifacts/phase3_inflation_bias_quantification.md`
- `docs/plans/artifacts/phase3_inflation_validity_assessment.md`
- Decision: promote to Phase 5 if bias acceptable, otherwise archive as "explored, not viable"

**Compute:** 25 GPU-hours (expensive reference comparison)

---

## Phase 4: Neural Surrogate Exploration

**Duration:** 4 weeks  
**Compute Budget:** 800 GPU-hours  
**Objective:** Train neural network to predict log p(y|θ) with smooth gradients, use as HMC target

### 4.1 Surrogate Architecture Design

**Input:** θ ∈ ℝ^d (structural + scale parameters)  
**Output:** log p̂(y | θ) ∈ ℝ (scalar)  
**Requirement:** Smooth ∇_θ log p̂ everywhere, even near discontinuity surface

**Candidate Architectures:**

1. **Deep MLP with smooth activations**
   - 5-10 layers, width 128-512
   - Activation: SoftPlus, GELU, or Swish (all have smooth derivatives)
   - Batch normalization or layer normalization
   - Lipschitz regularization to bound ‖∇²_θ log p̂‖

2. **Polynomial feature expansion + MLP**
   - Explicit low-order polynomial features (θ², θ³, cross-terms)
   - Shallower network (2-3 layers)
   - Better extrapolation than deep MLP

3. **Gaussian Process surrogate**
   - RBF or Matérn kernel (infinitely differentiable)
   - Exact posterior mean/variance
   - Cons: Scales poorly (O(n³) training, O(n) prediction)
   - Viable only for d ≤ 20, n ≤ 10,000 training points

4. **Neural spline flow**
   - Directly learn p(θ | y) as normalizing flow
   - Smooth by construction
   - Cons: Requires sampling from flow, more complex training

**Initial Choice:** Start with MLP (option 1), add polynomial features if needed

**Promotion Criteria:**
- Architecture defined with explicit layer counts, widths, activations
- Surrogate produces finite, smooth gradients on test set
- No NaN or Inf values in forward/backward pass

**Veto Criteria:**
- No architecture achieves smooth gradients (all have AD failures)
- Training is unstable (loss doesn't decrease, gradients explode)

**Artifacts:**
- `bayesfilter/inference/neural_surrogate.py`
- `docs/plans/artifacts/phase4_surrogate_architecture_spec.md`

---

### 4.2 Training Data Generation

**Objective:** Generate (θ, log p(y|θ)) pairs covering posterior support and discontinuity regions

**Sampling Strategy:**

1. **Posterior samples:** Draw from approximate posterior (Laplace, variational Bayes, or cheap MCMC)
   - 10,000 samples covering typical set
   - Ensures surrogate is accurate where posterior mass lives

2. **Boundary-enriched samples:** Oversample near identified discontinuity surfaces
   - Use Phase 1 diagnostic to locate discontinuity regions
   - Sample 5,000 additional points within ±2sd of boundary
   - Ensures surrogate learns to smooth the problematic region

3. **Space-filling design:** Latin hypercube or Sobol sequence
   - 5,000 points for extrapolation robustness
   - Prevents surrogate from failing outside posterior support

**Total:** 20,000 training points

**Evaluation:** For each θ, compute log p(y|θ) via UKF (exact, but discontinuous)

**Data Splits:**
- Train: 16,000 (80%)
- Validation: 2,000 (10%)
- Test: 2,000 (10%)
- Stratify by distance to discontinuity surface

**Promotion Criteria:**
- 20,000 evaluations complete without failure
- Training set covers posterior support (min/max within ±3sd of mode)
- Boundary region has sufficient samples (≥5,000 within ±1sd)

**Veto Criteria:**
- UKF evaluations fail on >5% of sampled points (suggests sampling outside valid domain)
- Training set does not cover discontinuity region

**Artifacts:**
- `docs/plans/artifacts/phase4_training_data.npz` (θ_train, log_p_train, splits)
- `docs/plans/artifacts/phase4_data_generation_report.md`

**Compute:** 200 GPU-hours (20,000 UKF evaluations on G2_3)

---

### 4.3 Surrogate Training and Validation

**Objective:** Train surrogate to minimize prediction error, especially gradient error near boundaries

**Loss Function:**

L(ψ) = MSE_value + λ_grad · MSE_gradient + λ_reg · Regularization

Where:
- MSE_value = 𝔼[(log p̂(θ; ψ) - log p(θ))²]
- MSE_gradient = 𝔼[‖∇log p̂(θ; ψ) - ∇log p(θ)‖²]
- Regularization = ‖ψ‖² or Lipschitz penalty on ∇²log p̂

**Gradient Ground Truth:** Use autodiff on exact UKF log-likelihood

**Weighting:** Oversample boundary regions in loss (importance weights)

**Training Protocol:**
- Optimizer: Adam with learning rate 10⁻⁴
- Batch size: 256
- Max epochs: 1,000 with early stopping (patience=50)
- Learning rate schedule: reduce on plateau
- Hyperparameter search: λ_grad ∈ {0.1, 1.0, 10.0}, λ_reg ∈ {10⁻⁴, 10⁻³, 10⁻²}

**Validation Metrics:**
- Value RMSE: 𝔼[(log p̂ - log p)²]^(1/2) < 0.1 (promotion)
- Gradient RMSE: 𝔼[‖∇log p̂ - ∇log p‖²]^(1/2) < 1.0 (promotion)
- Gradient smoothness: max |∇log p̂(θ + δ) - ∇log p̂(θ)| / δ < M (promotion)
- Gradient jump reduction: max Δ‖∇log p̂‖ / max Δ‖∇log p‖ < 0.1 (promotion)

**Promotion Criteria:**
- All 4 validation metrics pass
- No NaN/Inf in gradients on test set
- Surrogate is faster than exact UKF (eval time < 0.1× UKF)

**Veto Criteria:**
- Cannot achieve value RMSE < 0.5 (too inaccurate)
- Cannot achieve gradient smoothness (surrogate gradients still jump)
- Training time > 100 GPU-hours (too expensive)

**Artifacts:**
- `docs/plans/artifacts/phase4_surrogate_trained_model.pt` (PyTorch checkpoint)
- `docs/plans/artifacts/phase4_training_curves.png`
- `docs/plans/artifacts/phase4_validation_report.md`

**Compute:** 400 GPU-hours (hyperparameter search × training runs)

---

### 4.4 Surrogate HMC on G2_3

**Objective:** Run HMC using surrogate likelihood instead of exact UKF

**Experiment Design:**
- 4 chains, 10,000 draws each, 5,000 warmup
- Target: log p̂(y | θ) from trained surrogate
- Step size: ε = 10⁻² (100× larger than baseline, should be stable with smooth surrogate)
- Mass matrix: diagonal, estimated during warmup
- Compare to ground truth: either Phase 2 event-aware HMC or very long baseline at ε=10⁻⁴

**Metrics:**
- Posterior agreement: KL divergence, Wasserstein distance, or moment comparison
  - Promotion: KL < 0.1 nats, marginal sd agreement < 10%
- Sampling efficiency: ESS/grad > 1.0 (much higher than exact UKF due to larger ε)
- Divergence rate < 0.1%
- Wall time vs exact UKF HMC

**Promotion Criteria:**
- Posterior agreement within acceptable error (KL < 0.1 nats)
- At least 50× faster than baseline HMC (due to larger ε and cheaper evals)
- Divergence rate < 1%

**Veto Criteria:**
- Posterior disagreement > 20% (surrogate bias too large)
- No speedup over baseline HMC
- High divergence rate (>5%) despite smooth gradients

**Continuation Veto:**
- Method works but requires expensive retraining for each new model/dataset
- Surrogate accuracy degrades outside training distribution (not robust)

**Artifacts:**
- `docs/plans/artifacts/phase4_surrogate_hmc_trace.npz`
- `docs/plans/artifacts/phase4_surrogate_posterior_comparison.png`
- `docs/plans/artifacts/phase4_surrogate_hmc_result.md`

**Compute:** 200 GPU-hours (includes ground-truth reference chains)

---

## Phase 5: Production Candidate Selection

**Duration:** 1 week  
**Compute Budget:** 80 GPU-hours  
**Objective:** Compare all viable candidates from Phases 2-4, select default production policy

### 5.1 Head-to-Head Comparison

**Models:**
- G2_3 (d=8, T=40, dimension=337) - primary test case
- Scaled G2_3 (d=8, T=80, dimension=649) - scalability test
- (If available) NAWM II proxy (d=20-30 subset, T=40)

**Candidates:**
- Baseline HMC (ε=10⁻⁴, exact UKF)
- Event-aware HMC (ε=10⁻³, exact UKF with boundary detection) [if Phase 2 passes]
- Inflated UKF HMC (ε=10⁻³, inflated variance) [if Phase 3 passes]
- Surrogate HMC (ε=10⁻², surrogate likelihood) [if Phase 4 passes]

**Design:**
- 8 chains × 20,000 draws per candidate per model
- 10,000 warmup, 10,000 retained
- Same random seeds across candidates
- Same target ESS ≈ 10,000 effective draws

**Primary Metrics:**
1. **Correctness:** R̂ < 1.01 (hard veto if fails)
2. **Efficiency:** ESS/grad (higher is better)
3. **Speed:** Wall time to achieve ESS = 10,000
4. **Robustness:** Divergence rate < 1%

**Secondary Metrics:**
- Posterior agreement with reference (if available)
- Implementation complexity
- Model-specificity (does it require per-model tuning?)
- Scalability (how does wall time grow with T, d?)

**Decision Criteria:**

| Outcome | Decision |
|---------|----------|
| One candidate dominates all metrics | Adopt as default |
| Surrogate fastest but approximate | Adopt for exploration, keep exact for final inference |
| Event-aware best exact method | Adopt as default, document boundary detection requirement |
| Inflation fastest, acceptable bias | Adopt with documented limitations |
| All candidates fail robustness | Escalate to research leadership |
| Baseline HMC fastest (problem not real at scale) | Archive project, no production change needed |

**Artifacts:**
- `docs/plans/artifacts/phase5_head_to_head_comparison_table.md`
- `docs/plans/artifacts/phase5_trace_diagnostics_all_candidates.npz`
- `docs/plans/artifacts/phase5_decision_report.md`

**Compute:** 80 GPU-hours

---

### 5.2 Production Integration Plan

**Objective:** Document integration path for selected candidate(s)

**For Event-Aware HMC:**
- Merge `bayesfilter/inference/event_aware_hmc.py` to main
- Update `bayesfilter/inference/neutra_hmc.py` to use event-aware leapfrog
- Add boundary detection to model API: `model.discontinuity_surfaces(theta)`
- Document per-model boundary function requirement
- Add to regression test suite

**For Inflated UKF:**
- Merge `bayesfilter/highdim/ukf_inflation.py`
- Add inflation schedule to model config
- Document bias and scientific limitations
- Mark as "exploration-grade, not publication-grade"
- Add to experimental feature docs

**For Surrogate HMC:**
- Create `bayesfilter/inference/surrogate_training/` module
- Document training data requirements (20K evals, boundary enrichment)
- Provide training script template
- Add surrogate-posterior diagnostic tools
- Mark as "fast approximate posterior, validate against exact for final inference"

**For All:**
- Update CLAUDE.md production-algorithm policy
- Add to NeuTra HMC route ledger
- Document tuning scope requirements (per-model or general?)
- Write tutorial notebook: `notebooks/hardbound_hmc_discontinuous_gradients.ipynb`

**Artifacts:**
- `docs/plans/artifacts/phase5_integration_plan.md`
- `docs/plans/artifacts/phase5_production_policy_update.md`

---

## Risk Register

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Tran-Kleppe method requires dense Hessian | Medium | High | Check paper assumptions in Phase 2.1; have inflation and surrogate as fallbacks |
| Boundary detection is too expensive | Medium | High | Profile boundary detection; set cost ceiling at 2× baseline leapfrog |
| Inflation bias unacceptable | Medium | Medium | Quantify bias early in Phase 3.2; have event-aware and surrogate as fallbacks |
| Surrogate training unstable | Low | Medium | Use robust architecture (MLP), extensive hyperparameter search |
| All candidates fail | Low | High | Escalate to research leadership; consider particle filter despite biased gradients |
| Problem doesn't exist at NAWM II scale | Low | Medium | Would be good news; archive as "not needed" |

### Resource Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU budget insufficient | Low | Medium | Phases 1-3 use only 120 GPU-hours; Phase 4 is largest consumer; can reduce training data or hyperparameter search |
| Phase 4 training exceeds budget | Medium | Low | Cap training at 400 GPU-hours; accept lower validation scores if needed |
| 11-week timeline too aggressive | Medium | Low | Phases are independent; can extend Phase 4 if needed |

### Scientific Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Wrong problem diagnosis | Low | High | Phase 1 measurement validates diagnosis; if wrong, stop and reassess |
| Continuous-kink vs value-jump confusion | Low | High | Phase 1 diagnostics distinguish; survey §7.3 clarifies |
| Reference posterior unavailable | Medium | Medium | Use Phase 2 event-aware HMC (slow but correct) as reference for Phase 3-4 |
| Selected method doesn't scale to NAWM II | Medium | High | Phase 5 includes scalability test; document scale limitations if needed |

---

## Success Criteria (Summary)

**Minimum Success:** 
- Understand and measure the problem (Phase 1 complete)
- At least one candidate method achieves R̂ < 1.01, ESS/grad > 0.1, divergence < 1% on G2_3
- Production integration plan documented

**Target Success:**
- Event-aware HMC or surrogate HMC works on G2_3 and scales to T=80
- At least 10× speedup over baseline HMC
- Method generalizes across hardbound models without per-model tuning

**Stretch Success:**
- Method works on full NAWM II (T=120, d=100)
- Surrogate HMC achieves 100× speedup with acceptable posterior approximation
- Published artifact: working implementation + paper draft on discontinuous-gradient HMC

---

## Stopping Conditions

**Stop and Escalate If:**
1. Phase 1 diagnostic fails to identify gradient discontinuities (wrong problem diagnosis)
2. All three candidate methods (event-aware, inflation, surrogate) fail promotion criteria
3. Compute budget exceeded by >50% with no viable candidate
4. Selected method fails scalability test in Phase 5

**Stop and Archive If:**
1. Baseline HMC succeeds on all test cases (problem doesn't exist at tested scale)
2. Problem only appears at toy scale, disappears on realistic models

**Stop and Simplify If:**
1. Smaller step size (ε=10⁻⁵) solves problem with acceptable wall time
2. Different mass matrix (e.g., learned dense or LEDH preconditioner) eliminates stagnation

---

## Budget Summary

| Phase | Duration | GPU-Hours | Critical Path? |
|-------|----------|-----------|----------------|
| 0 | Complete | 0 | N/A |
| 1 | 1 week | 20 | Yes |
| 2 | 3 weeks | 60 | Yes |
| 3 | 2 weeks | 40 | Parallel with Phase 2.3 |
| 4 | 4 weeks | 800 | Parallel with Phases 2-3 |
| 5 | 1 week | 80 | Yes |
| **Total** | **11 weeks** | **1,000 GPU-hours** | **11 weeks** |

**Note:** Phases 2 and 3 can overlap partially (Phase 3 starts after Phase 2.2). Phase 4 is independent and can run in parallel with Phases 2-3 if resources allow.

---

## Appendix: Open Research Questions (from Survey §7.9)

This plan addresses the first four questions. The fifth (regime-conditional HMC) is deferred pending integer-solver work.

1. **Can event-aware integration be made practical?** → Phase 2
   - Boundary detection cost vs accuracy tradeoff
   - Scalability to d=337 and beyond
   - Required model API (boundary function specification)

2. **Is the marginal likelihood Lipschitz continuous?** → Phase 1
   - Measure Lipschitz constant near discontinuity surface
   - Relate to leapfrog energy error bound
   - Inform step-size selection

3. **Can neural surrogates learn implicit boundaries?** → Phase 4
   - Architecture requirements for smooth gradients
   - Training data requirements (boundary enrichment)
   - Posterior approximation quality

4. **Does UKF variance inflation preserve scientific validity?** → Phase 3
   - Inflation-bias tradeoff measurement
   - When is bias acceptable vs fatal?
   - Comparison to exact posterior

5. **When is regime-conditional HMC competitive?** → Future work
   - Requires integer OccBin/PyDSGE solver integration
   - Separate research program for value-jump case

---

## References

1. Tran & Kleppe (2025). "Gradient-based MCMC for models with discontinuous likelihoods." *To be obtained in Phase 2.1*

2. Nishimura & Dunson (2020). "Recycling Intermediate Steps to Improve Hamiltonian Monte Carlo." *Bayesian Analysis* 15(4): 1087-1115. [Event detection methods]

3. Betancourt (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." *arXiv:1701.02434*. [HMC foundations, energy conservation]

4. Pakman & Paninski (2014). "Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussians." *JMLR* 15: 2099-2148. [Reflection HMC, boundary handling]

5. BayesFilter survey: `docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex` §7 [Problem documentation]

6. BayesFilter hardbound test case: `bayesfilter/hardbound/model_tf.py` G2_3 fixture

---

**Plan Status:** Proposed, pending review  
**Next Action:** Review plan for scientific validity, resource feasibility, and alignment with survey §7  
**Approval Required:** Research leadership sign-off before Phase 1 execution  
**Estimated Start Date:** 2026-08-27 (pending approval)