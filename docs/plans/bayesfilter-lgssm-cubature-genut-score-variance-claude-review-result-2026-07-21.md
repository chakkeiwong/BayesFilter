# Claude Review Result: LGSSM Cubature/GenUT Recursive Score

Date: 2026-07-21

Request reviewed: `docs/plans/bayesfilter-lgssm-cubature-genut-score-variance-claude-review-request-2026-07-21.md`

Additional files inspected beyond the exact scope:

- `bayesfilter/linear/experimental_batched_kalman_tf.py`

## Executive Verdict

The implemented **finite value program** and its **manual forward-sensitivity score** are mathematically consistent with the finite branch that the code actually executes, provided the recorded Cholesky-positive branch and the active `maximum` branch are unchanged. The implementation does **not** compute the exact Kalman likelihood or exact Kalman score; it computes a finite particle/filter scalar and the total derivative of that same finite scalar on the checked branch. That distinction is stated correctly in the LaTeX chapter.

The largest issue I found is in the **reporting layer**, not the core recurrence: the matched-comparison artifact presents **HMC-chain-scaled relative errors** as if they were the primary “score intervals” for the recursive **physical-parameter** score. The raw physical-score intervals are present in the JSON, but the headline tables and prose mix these objects. That is **wrong relative to the stated review target** of the physical-parameter score.

On variance, the wide `T=50` intervals are primarily a **Monte Carlo path-variance problem**: 50 increments of pathwise score noise, repeated post-increment equal-weight resets, and finite-cloud approximation error accumulate. The evidence does **not** point to Sinkhorn residual failure, and it does **not** point to GenUT-vs-Cubature design differences in this Gaussian `d=3` scope, because Gaussian GenUT is exactly the cubature design here and the deterministic Cubature arm still has wide `T=50` score intervals.

Because the core finite-program math is mostly sound but the score-reporting object is mislabeled and the statistical interpretation needs tightening, my terminal verdict is:

`VERDICT: REVISE`

## Findings

### 1. The matched comparison reports HMC-scaled relative score errors as if they were physical-parameter score intervals

**Severity:** high  
**Classification:** wrong relative to the stated target

**Anchors**

- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:975-993`
- `docs/benchmarks/run_lgssm_recursive_score_matched_comparison.py:74-87`
- `docs/plans/bayesfilter-lgssm-recursive-score-matched-three-horizon-comparison-result-2026-07-21.md:52-96`

**What the code computes**

`_evaluate_method` first forms the raw physical-parameter score error

- `score_error = candidate_score64 - kalman_score`

but the field named `relative_error` is **not** the relative error of that raw physical score. It first multiplies both particle and Kalman scores by

- `hmc_chain = [1-phi1^2, 1-phi2^2, 1-phi3^2, q_scale, r_scale]`

and then computes

- `(particle_hmc_score - kalman_hmc_score) / abs(kalman_hmc_score)`.

The matched-comparison summary then builds the displayed “Kalman-relative errors” from `row["relative_error"]`, while separately storing raw physical-score intervals under `physical_score_intervals` and `physical_score_error_intervals`.

**Consequence**

The displayed score-relative tables are about **HMC-chain-scaled coordinates**, not the raw physical score requested in the review brief. This matters most at `T=50`, where the prose highlights wide `phi3`, `q_scale`, and `r_scale` relative intervals. Those are real as HMC-scaled relative errors, but they should not be described as the primary physical-score agreement object.

For example, at `T=50` in the JSON:

- Contract E Gaussian raw `phi3` score-error interval is approximately `0.107 [-0.076, 0.291]`, while the relative table reports `+35.5% [-25.0%, 96.1%]`.
- Cubature raw `phi3` score-error interval is approximately `0.011 [-0.289, 0.311]`, while the relative table reports `+3.56% [-95.6%, 102.8%]`.

The relative explosion is mostly denominator instability because the Kalman raw `phi3` score is small (`0.302...`), not because the absolute raw score error is comparably huge.

### 2. Same-scalar derivative correctness is supported only on a fixed differentiable branch, not as a global differentiability statement

**Severity:** medium  
**Classification:** correct for the checked fixed branch; unsupported as a global claim

**Anchors**

- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:245-255`
- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:304-331`
- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:649-804`
- `bayesfilter/highdim/ledh_contract_e_reset_tf.py:263-344`
- `tests/test_lgssm_cubature_genut_fp32.py:104-171`

**What is correct**

The manual JVP path differentiates the same finite unrolled computations that the value path executes:

- finite-cost normalization with `cost_scale = maximum(mean_cost, 1e-3)`;
- finite unrolled Sinkhorn iterations with denominator `+ 1e-7`;
- barycentric map;
- Contract E-Chol reset using the reused forward factorization;
- recursive log-normalizer accumulation;
- uniform-weight reset after each restore.

The small-fixture test `test_recursive_score_matches_same_value_central_difference` supports same-scalar derivative correctness for the composite route on that fixture, and the Kalman analytic score is separately validated against value and finite differences.

**Boundary that remains**

This is still a **fixed-branch** statement, not a global differentiability theorem. The finite program has branch-sensitive points at least at:

- `maximum(mean_cost, 1e-3)`;
- Cholesky existence / SPD boundary for `gap + ridge I`, `target_cov + ridge I`, and `injected_cov + ridge I`;
- any future branch change induced by floating-point tie or loss of SPD.

At those boundaries, the implementation picks one branch. That is fine for “derivative of the executed branch,” but stronger language would overclaim.

**Consequence**

The current route supports: “manual forward sensitivity equals the derivative of the executed finite scalar on the checked branch.”  
It does **not** support: “the route is globally differentiable everywhere” or “the score is the exact posterior score.”

### 3. The interval construction is a valid conservative six-coordinate screen, but the narration should not read it as a global simultaneous guarantee across all tables

**Severity:** medium  
**Classification:** correct but over-broadly narratable if not qualified

**Anchors**

- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:57-59`
- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:1045-1059`
- `docs/benchmarks/run_lgssm_recursive_score_matched_comparison.py:57-71`
- `docs/plans/bayesfilter-lgssm-recursive-score-matched-three-horizon-comparison-plan-2026-07-21.md:36-47`

**What the constant means**

`CRITICAL_VALUE = 3.036283222821165` is the `t` critical value for `df = 15` and tail probability corresponding to Bonferroni over **6 coordinates** at familywise `0.05`:

- `t_{15, 1 - 0.05/(2*6)} = 3.036283222821165`.

So the code is implementing a conservative six-coordinate simultaneous interval screen for one summary table containing

- value,
- `phi1`, `phi2`, `phi3`, `q_scale`, `r_scale`.

That part is mathematically coherent.

**What it does not cover**

It is **not** a simultaneous guarantee across the full report containing multiple horizons, multiple methods, and paired-delta tables. Bonferroni over six coordinates does not automatically extend to all reported families.

**Consequence**

The within-table interpretation is fine. The cross-table interpretation must stay modest: these are conservative per-summary screens, not a single global multiplicity correction for everything printed in the campaign.

## Mathematical Audit

### 1. Cubature points, replication, mean, and population covariance

**Classification:** correct

**Anchors**

- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:116-121`
- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:1503-1543`
- `tests/test_lgssm_cubature_genut_fp32.py:35-39`

The design is

- base rows `sqrt(d) [+e_a, -e_a]`, `a = 1, ..., d`,
- replicated `M = N / (2d)` times.

With population-covariance convention `1/N`, the mean is zero because each `+sqrt(d)e_a` is paired with `-sqrt(d)e_a`. The covariance is

`(1/N) Xi^T Xi = (1/(2dM)) * M * (2d I_d) = I_d`.

This matches the code exactly and is also covered by the test.

### 2. Gaussian GenUT specialization and alias to Cubature

**Classification:** correct

**Anchors**

- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:124-139`
- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:1684-1773`
- `docs/benchmarks/run_lgssm_recursive_score_matched_comparison.py:175-183`
- `tests/test_lgssm_cubature_genut_fp32.py:42-47`

For Gaussian GenUT in whitened coordinates, the chapter specializes to `s_a = 0`, `k_a = 3`. Then the axis construction gives

- `u_a = v_a = sqrt(3)`,
- `b_a = c_a = 1/6`,
- `w_0 = 1 - 3*(1/6 + 1/6) = 0` when `d = 3`.

The code sets `u = sqrt(3)` and uses only the six noncentral points because the central weight is zero. Since `STATE_DIM = 3`, `sqrt(3) = sqrt(d)`, so the six GenUT points are exactly the six cubature points. The equality here is exact for the executed `d=3` Gaussian specialization; it is not a general GenUT theorem for other dimensions or non-Gaussian moments.

### 3. Sinkhorn scaling, marginals, barycentric map, and JVP

**Classification:** correct for the finite unrolled route on the active branch

**Anchors**

- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:189-225`
- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:227-331`
- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:1566-1580`
- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:1901-1973`

The finite route is:

1. cost `C_ij = ||x_i - x_j||^2`;
2. normalized scale `s = maximum(mean(C), 1e-3)`;
3. kernel `K_ij = exp(-C_ij / (s * epsilon))`;
4. finite Sinkhorn updates
   - `left = uniform / (K right + 1e-7)`
   - `right = weights / (K^T left + 1e-7)`;
5. coupling `Pi = diag(left) K diag(right)`;
6. implementation matrix `Gamma = N Pi`;
7. barycentric map `Y_i^+ = sum_j Gamma_ij x_j`.

That agrees with the chapter’s convention `Gamma 1 = 1`, `Gamma^T 1 = N alpha` up to the recorded finite residuals.

The JVP differentiates:

- the cost;
- the mean-cost normalization and its active `maximum` branch;
- the kernel exponential;
- every finite Sinkhorn iteration;
- the barycentric matrix-vector product.

The `1e-7` denominator floor is included as a constant additive term in the differentiated scalar, so its derivative is correctly zero. The `maximum` is piecewise differentiable; the implementation correctly returns zero tangent on the floored branch. That is a branch-conditional derivative, not a global smooth derivative.

### 4. Contract E target moments, injected moments, cross terms, ridge, and covariance identity

**Classification:** correct

**Anchors**

- `bayesfilter/highdim/ledh_contract_e_reset_tf.py:18-166`
- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:334-423`
- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:1584-1669`

The reset computes:

- weighted target moments `(mu_w, Sigma_w)` from the source cloud with current normalized weights;
- equal-weight transported moments `(bar Y^+, Sigma_+)` from the barycentric cloud;
- gap `G = sym(Sigma_w - Sigma_+)`;
- injected cloud `Y_tilde = Y^+ + Xi B^T` with `B = chol(G + ridge I)`;
- realized injected covariance `Sigma_tilde` computed from the actual injected cloud;
- restoration affine map `A = L_w L_tilde^{-1}` via triangular solve.

The important identity in code is the **ridged** one:

`A (Sigma_tilde + ridge I) A^T = Sigma_w + ridge I`.

Therefore the raw covariance satisfies

`Sigma_out - Sigma_w = ridge (I - A A^T)`.

This is exactly the identity stated in the chapter and exactly the identity implemented in the forward core. The code correctly uses the **realized** injected covariance, so the cross terms between the transported cloud and injected residuals are included. It does **not** pretend that `B B^T` alone is the full covariance correction.

### 5. Recursive particle value and forward sensitivity recurrence

**Classification:** correct for the executed finite scalar on the checked branch

**Anchors**

- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:565-646`
- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:649-804`

The recurrence is internally consistent:

- stationary initial standard deviation is `q / sqrt(1 - phi^2)`;
- its derivative is propagated into the initial particles;
- transition update is `x_t = phi ⊙ x_{t-1} + q * noise_t`;
- observation log likelihood is the Gaussian scalar used in the value path;
- normalized weight derivative is the derivative of the log-sum-exp normalization;
- reset JVP is applied after the increment;
- weights are then reset to uniform and weight tangents reset to zero, matching the value path.

The score accumulator is therefore the sum of the per-time derivatives of the same `increment = logsumexp(log_weights)` scalar used by the value path.

### 6. Kalman analytic score and stationary initial covariance derivative

**Classification:** correct

**Anchors**

- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:437-562`
- `bayesfilter/linear/experimental_batched_kalman_tf.py:295-602`
- `tests/test_lgssm_cubature_genut_fp32.py:152-171`
- `tests/test_experimental_batched_linear_kalman_tf.py:303-322`

The Kalman oracle includes the derivative of the stationary initial covariance

`diag(q^2 / (1 - phi^2))`.

For each `phi_a`, the diagonal derivative is

`2 q^2 phi_a / (1 - phi_a^2)^2`,

and for `q_scale` it is

`2 q / (1 - phi^2)`

componentwise. That matches the code. The batched Kalman score kernel is also independently checked against tape-Jacobian reference in the additional inspected test.

### 7. Nondifferentiabilities and branch dependence

**Classification:** partly correct, partly not checked globally

**Anchors**

- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:200-201`
- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py:245-249`
- `bayesfilter/highdim/ledh_contract_e_reset_tf.py:66-76`
- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:1653-1670`
- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:1916-1973`

- `maximum(mean_cost, 1e-3)`: piecewise differentiable; code differentiates the active branch only.
- `+ 1e-7` denominator floor: differentiable as written because it is a constant additive term.
- Cholesky calls: differentiable on the SPD branch, undefined at loss of SPD.
- finite Sinkhorn truncation: differentiable as a finite unrolled map, not the exact OT optimizer derivative.
- TensorFlow `while_loop`: not a conceptual problem here; it is just the execution vehicle for the finite recurrence.

So the correct claim is: **same-scalar derivative on the realized differentiable branch**. Any stronger branch-free claim would be unsupported.

### 8. LaTeX propositions versus executed finite algorithm

**Classification:** correct

**Anchors**

- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:1653-1671`
- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:1792-1830`
- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:1901-2058`

The chapter is appropriately careful. It says:

- current increment unchanged because reset is post-increment;
- future likelihood generally changes because the carried cloud changes;
- derivative claim is the total derivative of the **finite scalar actually executed** on a fixed branch;
- it is **not** an exact filtering likelihood/variance/score theorem.

I did not find a material overclaim in the cited LaTeX sections.

## Implementation Audit

### Core program correctness

The value path `_particle_value` and score path `_particle_value_score_recursive` match each other structurally at every materially differentiable step. The reset is post-increment in both. Both paths reset weights to uniform after the equal-weight restore. This is the main same-scalar contract, and it is implemented consistently.

### Tests that support the implementation

Strongest supporting tests:

- `tests/test_lgssm_cubature_genut_fp32.py:35-47`: cubature and Gaussian-GenUT design identities;
- `tests/test_lgssm_cubature_genut_fp32.py:104-171`: same-scalar FD check for the recursive score on a small fixture, and Kalman parity;
- `tests/test_experimental_batched_linear_kalman_tf.py:303-322`: analytic Kalman score matches autodiff Jacobian.

### What is still missing programmatically

I did **not** see a direct unit test that isolates:

- `_sinkhorn_barycentric_jvp` against autodiff / central difference on a small differentiable fixture;
- `_restore_cloud_jvp` or `_contract_e_chol_cloud_jvp_from_forward_core` against autodiff / central difference on a small SPD fixture.

The end-to-end small recursive FD test is good evidence, but it leaves less localization if a future regression hits the OT or reset JVP specifically.

## Score-Variance Diagnosis

### What is driving the large `T=50` variance

1. **Time accumulation of pathwise derivative noise.**  
   The score is a sum of 50 increment tangents. Even if each increment is individually stable, their Monte Carlo variation accumulates with horizon.

2. **Repeated reset dependence.**  
   Each future increment depends on the previously restored equal-weight cloud. Small finite-cloud differences therefore propagate forward through transition, weighting, OT, and reset, rather than washing out after one step.

3. **Process-noise randomness dominates residual-design randomness by `T=50`.**  
   Cubature is deterministic in the residual design here, yet its `T=50` raw score intervals for `phi3`, `q_scale`, and `r_scale` are still wide. That means the main problem is not “random Contract E residual design only.” The process/initial noise path and finite-cloud filter evolution are the main variance source.

4. **Relative-error denominators magnify weak coordinates.**  
   The worst visual blow-up is `phi3` at `T=50` because the Kalman raw score is only about `0.302`. A moderate absolute error then becomes a very large relative error. This is a reporting instability, not proof of catastrophic score failure.

5. **`q_scale` and `r_scale` are intrinsically noisy score coordinates.**  
   They enter the transition-noise and observation-noise scales directly, so each time step contributes quadratic-innovation and log-scale terms. At longer horizons this can produce broad raw score intervals even when the mean remains near the Kalman reference.

### What is *not* the main problem

- **Not Sinkhorn residual failure.** Residual gates are tiny and pass comfortably. Example row residuals are around `1e-7` to `1e-5` in the artifact.
- **Not GenUT/Cubature mismatch.** In this exact Gaussian `d=3` scope, GenUT is bitwise Cubature.
- **Not evidence of derivative-route inconsistency by itself.** Wide Monte Carlo intervals do not contradict the same-scalar FD evidence.

## Variance-Reduction Options

Ranked from most justified for this research question to least justified.

| Rank | Remedy | Preserves derivative of the current per-run finite scalar? | Changes probability measure / reset target / scientific quantity? | Expected effect | Transferability / compatibility | Cheapest discriminating experiment |
|---|---|---|---|---|---|---|
| 1 | **More paired seeds and/or average across independent particle clouds before comparing methods** | **Per run:** yes. **Campaign estimator:** no, because you average multiple runs. | No change to route family or reset rule. | Variance down roughly like `1/K` in SE for `K` independent replications; memory linear if parallelized. | High transfer. Fully compatible with float32/TF32 and no-autodiff score path. Preferable to changing the reset rule if the goal is precision rather than a new algorithm. | Extend the current campaign to 32 or 64 paired seeds, or keep 16 seeds but use 2 independent clouds per seed. Pass if raw physical-score error intervals for `T=50` contract materially; fail if widths barely change, implying heavier-tail path effects. |
| 2 | **Antithetic initial and process innovations** | No, because the reported scalar becomes the average of `Z` and `-Z` runs, not the current single-cloud scalar. | No reset-target change; Monte Carlo coupling changes. | Often large variance reduction for symmetric-noise models; 2x work, little extra memory if sequential. | Good transfer to nonlinear Gaussian-noise models; compatible with float32/TF32 and manual score path. | For each of 16 seeds, evaluate `Z` and `-Z`, average scores, compare raw `T=50` physical-score interval widths to the current baseline. Pass if `phi3`, `q_scale`, `r_scale` widths shrink materially without creating replay/finite failures. |
| 3 | **Per-time score decomposition with common-path paired differences** | Yes for the decomposition itself; yes for each per-run scalar. | No target change. | Best for diagnosing where variance enters; paired per-time differences can strongly reduce method-comparison variance even if single-arm variance remains high. | High transfer. Compatible with existing recursive path because the score is already a sum of increment tangents. | Log/store per-time increment score contributions and paired Cubature-minus-Contract-E differences. Pass if a small subset of late times explains most variance; fail if variance is diffuse across all times. |
| 4 | **Score control variates / baseline subtraction using predictive-moment surrogates** | Usually no for the current raw scalar unless used only as a post-processing diagnostic estimator with added-back mean. | Does not need to change reset target, but it changes the comparison estimator. | Can reduce variance substantially if the baseline tracks score fluctuations; modest extra compute. | Potentially transferable if the baseline uses cloud moments or local Gaussian surrogates, not LGSSM-only Kalman formulas. Compatible with no-autodiff path if the baseline derivative is analytic or separately computed. | Build a post-processing control variate from cloud predictive mean/covariance or local linearized observation statistics. Pass if estimated variance of raw score error drops materially with negligible bias signal; fail if covariance with the baseline is weak. |
| 5 | **Increase `N` or replace one `N=1008` cloud by `K` independent `N=1008` clouds and average** | No: this changes the finite scalar from the current `N=1008` single-cloud object. | No scientific-target change in route family, but it is a different finite estimator. | Strong variance reduction; memory/runtime cost can be high. | High transfer. Fully compatible with float32/TF32 and no-autodiff route. | Run a short `N` ladder or `K`-cloud average on `T=50` only. Pass if SE shrinks near `1/sqrt(N)` or `1/sqrt(K)`; fail if costs explode before useful contraction. |
| 6 | **Stronger Sinkhorn convergence controls** (more steps, better warm starts, epsilon schedule) | No: changing Sinkhorn controls changes the finite scalar. | Changes the finite route, but not the scientific object class. | Likely small variance impact here because current row/column residuals are already tiny; may reduce bias more than variance. | Transferable and compatible, but low priority given the current residuals. | Repeat only `T=50` with more Sinkhorn steps. Pass only if raw score intervals shrink beyond noise expected from seed-to-seed fluctuation *and* residuals tighten further; otherwise reject as a primary fix. |
| 7 | **Residual-design replication / orthogonalized or antithetic residual blocks** | No for Contract E Gaussian; Cubature already deterministic here. | No target-class change, but does change the finite design path. | Could reduce design randomness in Contract E Gaussian, but limited because Cubature still has wide `T=50` intervals. | Moderate transfer. Compatible with float32/TF32 and no-autodiff. | Compare current Contract E Gaussian with an antithetic or orthogonal residual-design variant at fixed process noise. Pass only if Contract E variance contracts materially relative to current Contract E while Cubature remains unchanged. |
| 8 | **Rao-Blackwellization / conditional Gaussian treatment where valid** | No. | Yes, this changes the estimator/object, and in LGSSM it collapses toward a model-specific exact treatment. | Could slash variance dramatically in LGSSM, but this is mostly a diagnostic oracle, not a transferable repair. | Poor transfer to the intended nonlinear high-dimensional target unless a genuine conditionally Gaussian decomposition exists there. | Use only as a diagnostic ceiling, not as the recommended repair. Pass/fail question is not “better score” but “how much variance headroom exists relative to the exact conditional treatment?” |

### Preferred interpretation

If the goal is **better scientific precision without changing the route family**, I would prioritize:

1. more paired seeds and/or independent-cloud averaging;
2. antithetic initial/process innovations;
3. per-time decomposition plus common-path paired differences;
4. transfer-minded control variates.

I would **not** make “change the reset rule” the first variance remedy, because that changes the candidate algorithm when the present evidence is more consistent with ordinary Monte Carlo noise accumulation.

## Statistical and Evidence Audit

### 1. What the 16-seed intervals do support

- Hard-valid and replay checks passed for all 96 rows.
- Within a six-coordinate family for one summary table, the interval construction is a conservative Bonferroni/t screen.
- For the raw physical score at `T=50`, both arms remain statistically compatible with the Kalman reference under this 16-seed screen because every raw physical-score error interval still contains zero.
- For paired Cubature-versus-Contract-E comparisons, every paired absolute-error interval contains zero.

### 2. What the 16-seed intervals do **not** support

- No ranking of the two methods on score accuracy.
- No claim that Cubature and Contract E are mathematically identical.
- No claim that the score route equals the exact Kalman score pointwise.
- No transfer to nonlinear filtering.

### 3. Near-zero denominators

This matters most for raw `phi3` at `T=50`, because the Kalman raw physical score is approximately `0.302`. Relative error there is highly unstable and should be treated as descriptive only. The raw physical-score error intervals are the safer primary object for that coordinate.

### 4. Multiplicity

The critical value is coherent for six coordinates in one table. It is conservative because Bonferroni does not require independence. But it is not a correction across every horizon/method/paired table in the whole report. The report should say that explicitly if “simultaneous” is used prominently.

## Minimal Repair/Test Plan

1. **Repair the reporting object.**  
   In the matched comparison artifact, either:
   - relabel the current relative-error tables as **HMC-chain-scaled relative error**, or
   - add separate raw physical-score relative error tables if that is the intended headline object.

2. **Promote raw physical-score error intervals to primary status for this review question.**  
   The review brief asks about the recursive physical-parameter score, so the primary score-agreement table should be built from `score_error` / `physical_score_error_intervals`, not from HMC-scaled relative errors.

3. **Add isolated JVP regression tests.**  
   Add small-fixture tests for:
   - `_sinkhorn_barycentric_jvp` versus autodiff or central differences on a differentiable non-floored case;
   - `_restore_cloud_jvp` or `_contract_e_chol_cloud_jvp_from_forward_core` versus autodiff or central differences on an SPD fixture.

4. **Add per-time score decomposition logging for `T=50`.**  
   This is the cheapest next diagnostic for the variance question.

5. **Run one small transfer-minded variance experiment.**  
   My preferred first trial is antithetic initial/process innovations or 2-cloud averaging at `T=50`, evaluated on raw physical-score error intervals. That directly tests whether ordinary Monte Carlo coupling is the dominant variance source.

## Nonclaims and Transfer Limits

- The exact GenUT-equals-Cubature result here depends on the executed Gaussian specialization and `STATE_DIM = 3`. It does **not** transfer to general non-Gaussian or other-dimensional settings.
- Agreement or near-agreement with the Kalman oracle in this LGSSM does **not** prove exact nonlinear filtering validity.
- A same-scalar derivative result for this finite branch does **not** prove exact posterior-score correctness.
- Rao-Blackwellized or exact conditional Gaussian repairs are mostly LGSSM-only diagnostics unless a genuine analogous decomposition exists in the nonlinear target.
- The main transferable lesson from this diagnostic is the **derivative contract** and the **variance-accumulation mechanism**, not any claim of method-wide superiority.

VERDICT: REVISE
