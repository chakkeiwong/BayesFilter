# LEDH Analytical Score Discrepancy on LGSSM: Request for Mathematical and Implementation Audit

## Executive Summary

The LEDH-PFPF-OT canonical score lane produces analytical scores for a 3-dimensional linear Gaussian state-space model (dlgssm) that differ from the exact Kalman gradient with **mean error -0.46±0.33** on an exact score of 7.30 (**6.24% relative error**). With 6 seeds the error is **not statistically separated from zero** at 2 SE (|mean|/SE = 1.39), but the magnitude of the discrepancy and its sign asymmetry (4 of 6 negative, though p=0.688 by sign test) warrant investigation. We request a thorough audit of the mathematical derivation, implementation correctness, parameter tuning, and algorithmic structure to identify whether this is sampling noise that collapses with more seeds, a finite-sample bias that requires higher particle count, or a structural approximation in the tangent propagation.

**Key finding:** The value lane and score lane agree on log-likelihood estimation (paired gap -0.38±0.25 log-units, not separated from zero at 2 SE), but the analytical score shows larger variance (SE 0.33 on mean 7.30, ~4.5% relative SE) than might be expected from a well-resolved estimator in d=3 with N=1008 particles.

## Problem Statement

### The Model (dlgssm)
- **State dimension:** 3
- **Observation dimension:** 3  
- **Transition:** diagonal, `x_{t+1} = Φ x_t + w_t` where `Φ = diag(0.9, 0.8, 0.7)`, `w_t ~ N(0, 0.6² I₃)`
- **Observation:** linear, `y_t = H x_t + v_t` where `H` is the 3×3 matrix `[[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]]` and `v_t ~ N(0, 0.8² I₃)`
- **Parameter vector:** `θ = [φ₁, φ₂, φ₃, q_scale, obs_scale] = [0.9, 0.8, 0.7, 0.6, 0.8]`
- **Horizon:** T=50 time steps

### Exact Reference
Since the model is linear Gaussian, the **Kalman filter gives the exact log-likelihood** and **central differences on it give the exact score** (up to difference truncation, verified by Richardson extrapolation at step sizes h=1e-5 and h/2: agreement to 4.3e-9).

### Score Lane Implementation
- Entry point: `canonical_value_and_analytical_score` in `bayesfilter/highdim/ledh_canonical_score_tf.py`
- Method: **forward-mode automatic differentiation** via hand-coded Jacobian-vector products (JVPs)
- Structure: threads both primal and tangent state through the filter in a single forward pass
- Reset: Sinkhorn + Contract-E OT reset with analytical tangent (`sinkhorn_contract_e_reset_with_tangent`)
- Trust region: dual-cap correction via `higher_moment_shape_jvp` (diagonal + pairwise caps)
- Resampling: annealed within-step telescope with systematic resampling; tangent convention treats **realized indices as fixed** (piecewise constant in θ)

## Test Results

### Lane Parity (6 seeds, ε=1.0, sk=8, flow=12, N=1008)

**Log-likelihood errors vs exact Kalman:**
```
Value lane:  mean -0.27 ± 0.20 log-units  (n=6)
Score lane:  mean -0.65 ± 0.09 log-units  (n=6)
Paired gap:  mean -0.38 ± 0.25 log-units  (not separated from zero at 2 SE)
```

**Analytical score error vs exact Kalman score:**
```
Mean error:      -0.46 ± 0.33
Relative error:   6.24% of |exact score| = 7.30
Range:           -1.52 to +0.30 (4 of 6 seeds negative, sign test p=0.688)
|mean|/SE:       1.39 (not separated from zero at 2 SE threshold)
```

**Interpretation:** Both lanes estimate the same log-likelihood within noise. The score error is not statistically established as bias with 6 seeds, but its magnitude (SE ~4.5% of the exact score) is larger than expected for N=1008 in d=3, and warrants investigation of whether it's sampling noise, finite-sample bias, or a structural approximation error.

### Marginal Convergence (value lane only; score lane does not surface this diagnostic)

- **Value lane pilot grid:** 72 configurations, 54 survived marginal-convergence veto (TV error ≤ 1e-4)
- **Rejected:** all 18 ε=0.5 configurations (TV errors 7e-4 to 1.0e-3)
- **Selected config:** ε=1.0, sk=8, flow=12, N=1008
  - Worst marginal TV error: 3.4e-5 (well under tolerance)
  - Worst ESS min fraction: 0.052 (roughly 52 of 1008 particles)

**Note:** The score lane does not return marginal diagnostics in its result dict, so we cannot verify whether its Sinkhorn marginals converge. The implementation calls `_restore_cloud_primal` which computes `marginal_valid` and `post_quotient_column_tv_error`, but `canonical_value_and_analytical_score` discards those fields before returning.

### Control Configuration Used
```
epsilon:              1.0
sinkhorn_steps:       8
balance_steps:        8
flow_substeps:        12
particle_count:       1008
reset_ridge:          1.0e-5
correction_steps:     1     (diagonal trust-region cap)
pairwise_steps:       1     (pairwise trust-region cap)
correction_lm_damping:         1.0e-2
correction_lm_scale_floor:     1.0e-4
correction_trust_radius:       0.5
annealed_stages:      1
score_direction:      0      (derivative w.r.t. φ₁)
```

These controls were selected by the R2-TUNE value-lane pilot as producing the lowest mean absolute error against exact Kalman log-likelihood among configurations whose Sinkhorn marginals converged.

## Hypotheses to Investigate

### H1: Fixed-Ancestry Tangent Bias
The score implementation treats resampling indices as constants in the tangent:
```python
# From ledh_canonical_score_tf.py comments:
"Tangent convention: realized resampling indices are FIXED 
 (piecewise-constant in theta almost everywhere)"
```

This is exact only where index selection is locally insensitive to θ. The value-lane pilot measured **ESS min fractions down to 0.045**, meaning highly concentrated weights (~45 of 1008 particles carrying the weight). This is precisely where the approximation is worst.

**Test:** The error being a systematic offset (mean -0.46, same sign in 4 of 6 seeds) rather than symmetric scatter is consistent with bias. However, **6 seeds do not establish this statistically** (|mean|/SE = 1.39, sign test p=0.688). The hypothesis remains viable but unproven.

**Diagnostic:** Compare the score error at different ESS regimes (if we can control ESS independently of other factors).

### H2: Sinkhorn Marginal Non-Convergence in the Score Lane
The value lane rejects ε=0.5 due to marginal non-convergence (TV > 1e-4), and ε=1.0 passes with TV=3.4e-5. But the score lane does not surface this diagnostic, so we cannot verify whether:
1. The score lane's Sinkhorn marginals actually converge at ε=1.0
2. The tangent propagation through an **almost-converged** but not fully-converged Sinkhorn introduces error

**Test:** The score lane needs to return `marginal_valid` and `post_quotient_column_tv_error` so we can check this. This is a Class-A observability promotion (same as what was done for the value lane).

**Diagnostic:** If ε=1.0 is on the edge of convergence, try ε=2.0 or ε=4.0 and see if score error decreases.

### H3: Tangent Through Trust-Region Correction Is Wrong or Missing
The score lane calls `higher_moment_shape_jvp` with `correction_steps=1, pairwise_steps=1`. This implements the dual-cap trust-region correction. The function name includes `_jvp`, suggesting it computes tangents, but:

**Questions:**
1. Is the tangent through the diagonal LM correction correctly implemented?
2. Is the tangent through the pairwise cap correctly implemented?  
3. Is the tangent through the coordinate-wise cap correctly implemented?
4. Do these tangents compose correctly when both `correction_steps` and `pairwise_steps` are active?

**Test:** Run with `correction_steps=0, pairwise_steps=0` (no trust region) and see if score error decreases. If it does, the error is in the trust-region tangent. If it doesn't, the error is upstream.

### H4: Flow Discretization Too Coarse for Tangent Convergence
The filter uses `flow_substeps=12`, meaning 12 Euler steps per dynamical flow. The value-lane pilot selected this for log-likelihood accuracy, but **tangent convergence may require finer discretization than primal convergence**.

**Test:** Run at flow_substeps ∈ {12, 24, 48, 96} and check whether score error falls as ~1/substeps or ~1/substeps².

**Diagnostic:** If error is O(1/substeps), the tangent has first-order truncation and 12 steps is too coarse. If error does not fall with substeps, discretization is not the issue.

### H5: Particle Count Insufficient for Gradient Noise
N=1008 in d=3 is not a small-sample regime, but the score lane propagates tangent state through stochastic ancestry, which could amplify variance.

**Test:** Run at N ∈ {504, 1008, 2016, 4032} and check whether score error SE falls as ~1/sqrt(N).

**Diagnostic:** If the error **mean** (not just SE) falls with N, the estimator is biased at finite N and we need a larger particle count. If only the SE falls but the mean stays at -0.46, the bias is structural, not statistical.

### H6: Implementation Bug in Tangent Propagation
Possible locations:
1. The tangent through `ukf_update_with_parameter_tangent` (lines 395-410)
2. The tangent through `sinkhorn_contract_e_reset_with_tangent` (lines 422-434)
3. The tangent through `higher_moment_shape_jvp` (lines 435-461)
4. The tangent through `_flow_substeps_with_tangent` (called at line 472+)
5. Composition of `d_step_weights` from `d_logits` (lines 418-421)

**Test:** Unit tests for each tangent function against finite differences. Particularly:
- `sinkhorn_contract_e_reset_with_tangent` against finite-difference on `_restore_cloud_primal`
- `higher_moment_shape_jvp` against finite-difference on `higher_moment_shape`
- `ukf_update_with_parameter_tangent` against the exact Kalman update derivative (available in closed form)

### H7: Mathematical Derivation Error in LaTeX
The implementation claims to follow the derivation in `docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex` (2407 lines). Possible issues:
1. The fixed-ancestry convention is stated but its bias not quantified
2. The tangent through the Sinkhorn iterations may be wrong
3. The tangent through the trust-region caps may be wrong
4. The composition of tangents across the UKF update, flow, resampling, and reset may be wrong

**Request:** Audit the LaTeX derivation for:
- Correctness of the Sinkhorn tangent (does it account for the implicit-function theorem correctly?)
- Correctness of the trust-region tangent (does it handle the constrained optimization correctly?)
- Whether the stated tangent convention (fixed ancestry) is an approximation or exact
- Whether any other approximations are stated that would explain a 6% bias

## Requested Audit Scope

### 1. Mathematical Derivation (LaTeX)
- **Primary source:** `docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex` (2407 lines)
  - **Note:** Chapter title mentions "Custom Gradients" and discusses VJP (vector-Jacobian products, reverse-mode) extensively (108 mentions), but also discusses JVP (Jacobian-vector products, forward-mode, 36 mentions). The **actual implementation uses forward-mode (JVP)** via unrolled Sinkhorn iterations, not reverse-mode VJP. Clarify whether the chapter's VJP derivation has a corresponding JVP derivation, and whether the implementation correctly implements whichever mode is intended.
- **Supporting:** `docs/chapters/ch09_kalman_score.tex` (178 lines, exact Kalman score reference)
- **Questions:**
  - Is the Sinkhorn tangent derivation correct?
  - Is the trust-region tangent derivation correct?
  - Is the fixed-ancestry approximation stated, and what is its expected bias?
  - Are there any other approximations that could explain 6% error?

### 2. Implementation Correctness (Code)
- **Primary source:** `bayesfilter/highdim/ledh_canonical_score_tf.py`
- **Supporting:**
  - `bayesfilter/highdim/ledh_canonical_reset_score_tf.py` (Sinkhorn+Contract-E tangent)
  - `bayesfilter/highdim/higher_moment_contract_e.py` (trust-region tangent)
- **Questions:**
  - Does the code implement the LaTeX derivation faithfully?
  - Are there sign errors, missing terms, or wrong Jacobian chains?
  - Do the tangent functions compose correctly?
  - Is the `d_step_weights` calculation correct (lines 418-421)?

### 3. Parameter Tuning
- **Questions:**
  - Is ε=1.0, sk=8 on the edge of Sinkhorn convergence, and could that cause tangent error?
  - Is flow_substeps=12 too coarse for tangent accuracy?
  - Is N=1008 insufficient for the score lane despite being adequate for the value lane?
  - Is the trust-region configuration (damping=1e-2, radius=0.5) introducing tangent error?

### 4. Algorithmic Structure
- **Questions:**
  - Is the fixed-ancestry tangent convention (treating resampling indices as constants) the primary error source?
  - Can this convention be replaced with a differentiable resampling scheme?
  - Are there known pathologies of forward-mode AD through particle filters that would explain this?

## Artifacts Available

1. **Value-lane pilot results:** `docs/benchmarks/r2_tuning_pilot_dlgssm.json` (216 cells, marginal veto, exact Kalman comparison)
2. **Lane parity results:** `docs/benchmarks/r2_lane_parity_dlgssm.json` (6 seeds, value vs score vs exact)
3. **LaTeX source:** `docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex` (2407 lines)
4. **Score implementation:** `bayesfilter/highdim/ledh_canonical_score_tf.py` (617 lines)
5. **Reset tangent:** `bayesfilter/highdim/ledh_canonical_reset_score_tf.py` (208 lines)
6. **Trust-region tangent:** `bayesfilter/highdim/higher_moment_contract_e.py` (1678 lines)

## Existing Test Coverage

The repository has extensive score-lane test coverage (19 test files in `tests/highdim/` matching "score"):
- `test_ledh_canonical_score_ukf_tangent.py` — UKF update tangent correctness
- `test_ledh_canonical_score_step.py` — per-step score correctness
- `test_ledh_canonical_score_recursion.py` — full recursion
- `test_ledh_canonical_score_full.py` — end-to-end score tests
- `test_ledh_lgssm_score_phase2_contract.py` — LGSSM-specific score contract

These tests may verify correctness against finite differences at the step level but may not catch a systematic bias that emerges only at horizon T=50 and specific ESS regimes. The request is to audit whether the **mathematical derivation** contains an approximation that explains the observed 6% bias, not just whether the code implements its intended formula.

## Requested Deliverable

A prioritized list of hypotheses with:
1. **Root-cause likelihood** (high/medium/low)
2. **Diagnostic test** to confirm or rule out
3. **Remediation path** if confirmed
4. **Specific LaTeX equation or code line** where the issue is located

Focus on hypotheses that explain the observed SE of 0.33 (~4.5% of the exact score 7.30) in d=3 with N=1008 particles. At this resolution, if the estimator were unbiased and variance-optimal, the SE should be smaller. Investigate whether this is:
- **High variance** from the fixed-ancestry tangent convention or other structural choices
- **Finite-sample bias** that requires larger N to collapse
- **Systematic error** in the tangent propagation (implementation bug or mathematical approximation)
- **Parameter tuning** (epsilon, flow_substeps, trust-region) insufficient for tangent accuracy despite being adequate for value accuracy

---

**Repository:** `/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild`  
**Commit:** `e3292331` (R2-TUNE dlgssm pilot: marginal-convergence diagnostic + grid veto)  
**Date:** 2026-08-28
