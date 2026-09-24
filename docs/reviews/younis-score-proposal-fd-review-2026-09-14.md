# Review: Younis KDM score master program and finite-difference additions

**Reviewer:** Claude Opus 5  
**Date:** 2026-09-14  
**Review scope:** Master program, Phase 4B/4C finite-difference design, SGQF proposal integration, experimental architecture  
**Documents reviewed:**
- `docs/plans/younis-kdm-score-master-program-2026-09-14.md`
- `docs/plans/younis-kdm-score-master-program-claude-review-handoff-2026-09-14.md`

**Review status:** COMPLETE — Stage 4 bounded review of Phase 4C finite-difference ladder

---

## Executive summary

The master program defines a systematic investigation of marginal-likelihood score estimation for particle filters using LEDH, KDM, and proposal-quality improvements. The program correctly distinguishes exact model scores, finite-program derivatives, KDM expectation gradients, and unnormalised estimates as non-interchangeable targets. The phase structure, registry architecture, and heuristic-dominance gates are sound scientific practice.

**Phase 4C (fixed symmetric directional finite-difference ladder)** contains one critical mathematical confusion that invalidates its stated pass/fail criterion. The phase correctly distinguishes deterministic stencil truncation order from stochastic particle-score MSE in its prose, but the calibration test conflates them: it requires "observed order compatible with the claimed stencil" by fitting slopes on particle-score replicates, then uses slope failure as a "diagnostic veto for the ladder." This is wrong.

**VERDICT: REVISE**

Phase 4C requires repair before execution. The remaining program structure, target definitions, and phase dependencies are sound for a bounded first-tranche execution on linear-Gaussian models.

---

## Critical finding: Phase 4C stencil-order test confuses deterministic and stochastic targets

### Severity: BLOCK (invalidates Phase 4C calibration test as written)

**Source:** Master program lines 806–831, especially calibration test parts 1–2 and the diagnostic-veto clause at line 827.

### The defect

Phase 4C states:

> "Fit the slope of log absolute error versus log h and verify the expected order before selecting a ladder."

and later:

> "Failure of the order test is a diagnostic veto for the ladder, not evidence that the underlying score method is wrong."

This test design conflates two distinct questions:

1. **Deterministic stencil algebra:** Does the stencil `D_h` satisfy `D_h[f] = f'(x) + O(h^p)` for a smooth deterministic function `f`?

2. **Stochastic particle-score MSE:** Does `E[(D_h[L_N] - s)²]` exhibit a particular power-law scaling in `h` when `L_N(theta)` is a noisy finite-particle log likelihood?

The first question has a definite mathematical answer controlled by Taylor remainder bounds. The second question depends on the interplay of three error sources with different `h`-dependence:

\[
E[D_h[L_N] - s]^2 
= \underbrace{E[D_h[\ell]]^2}_{O(h^{2p}) \text{ truncation}}
+ \underbrace{E[D_h[b_N]]^2}_{\text{stencil applied to log bias}}
+ \underbrace{c^T \Sigma_N(h) c / h^2}_{\text{particle noise amplified by } 1/h^2}
\]

where `b_N(theta) = E[L_N(theta)] - ell(theta)` is the finite-particle log bias and `Sigma_N(h)` is the covariance matrix of log-likelihood evaluations at the perturbed points.

**The particle-noise term grows as `h` shrinks**, because dividing by `h` amplifies the noise. Even with perfect common random numbers, `Sigma_N(h)` approaches a positive-definite limit as `h -> 0` (unless the particle program is pathwise-differentiable almost surely, which discrete resampling prevents). Thus the MSE curve is **U-shaped**: truncation dominates at large `h`, particle noise dominates at small `h`, and the minimum occurs at an intermediate `h*` that balances the two.

The observed slope on particle-score replicates can be:
- positive (truncation-dominated region),
- negative (noise-dominated region),
- approximately zero (near the minimum), or
- chaotic (roundoff or near-zero `h` divide).

A slope of 2 or 4 proves neither that the stencil is correct nor that it is wrong. It proves only that you sampled the truncation-dominated part of the U-curve at the chosen `h`, `N`, and horizon.

### Consequence

**Phase 4C as written will reject valid stencils and accept invalid stencils based on accidents of the `h`-`N` configuration.**

A second-order stencil tested at small `h` and moderate `N` will show slope ≈ −2 (noise dominance) and be vetoed, even though its algebra is correct. A fourth-order stencil tested at large `h` will show slope ≈ 2 (because `ell^(5)` or `ell^(6)` terms dominate) and pass, even though the implementation could have the wrong weights.

### Repair

**Separate the two tests completely:**

1. **Deterministic stencil correctness test (mechanics):** Use smooth analytic test functions `f(x) = exp(ax) + bx^5`, exact Kalman log-likelihoods with no particle noise, or another deterministic oracle. Verify the stencil weights, observed order, and error constant over `h` in the pre-roundoff range `[1e-8, 1e-2]`. This is a unit test, not an experiment. It establishes that `D_h` computes what it claims.

2. **Particle-score MSE calibration (experiment):** On stochastic particle runs, measure MSE vs `(h, N)` jointly. Fit or report the U-curve minimum `h*`, the achieved MSE at that minimum, and the replicate variance. Do not require any particular slope. The curve shape is data, not a pass/fail criterion. A steep noise amplification or a flat minimum are both scientific findings, not test failures.

The ladder selection uses the second test: choose `h` near the empirical `h*(N)` for the declared `N`, or choose `N` large enough that `h*` enters the range where the stencil's truncation order is visible. Report both the selected configuration and the MSE it achieved.

**Promotion criterion:** The finite-difference score is viable if its MSE at the selected `(h, N)` is competitive with other score estimators at matched compute. The stencil order and observed slope are explanatory diagnostics, not vetoes.

---

## Secondary findings

### 1. Phase 4C prose correctly explains the confusion, then contradicts itself

**Severity:** MAJOR (internal contradiction)

**Source:** Lines 723–742 (the bias decomposition and "Common random numbers... do not remove the log-normalisation bias, finite-particle bias, or finite-difference truncation bias") versus lines 806–831 (calibration test requires observed order and uses slope failure as a veto).

The prose at lines 723–742 is **correct**: it states that finite differences inherit `b_N`, that common random numbers reduce variance but not bias, and that `epsilon -> 0` may differentiate a discontinuous program. This is the right warning.

The calibration test at lines 806–827 then **contradicts this warning** by requiring "observed order compatible with the claimed stencil" and using "failure of the order test" as a veto. If `b_N` and noise dominate, the observed order will not match the stencil order, and the veto will fire inappropriately.

**Repair:** Rewrite lines 806–831 to align with lines 723–742. The prose already contains the correct understanding; the test specification needs to implement it.

---

### 2. Master program correctly distinguishes targets and tracks

**Severity:** None (this is praise)

**Source:** Lines 192–266 (target definitions), lines 26–52 (regular vs degenerate tracks)

The program explicitly defines:
- exact model score `s(theta) = grad log Z(theta)`,
- finite-program derivative `grad log Z_hat_N(theta; xi)`,
- KDM expectation gradient `grad E_m_phi[F(z, phi)]`,
- unnormalised pair/ratio estimates,

and forbids merging them into one unlabeled leaderboard (line 1000). This is exemplary scientific practice. The regular-transition and degenerate-transition track separation (lines 26–52) is also correct: ambient Gaussian KDM proposals and regular-HMM smoothing formulas cannot be applied to singular DSGE transitions without a separate support derivation.

**No repair needed.** This target discipline should be preserved throughout execution.

---

### 3. Phase structure and dependencies are sound

**Severity:** None (praise)

**Source:** Lines 949–980 (phase gates), lines 1022–1038 (first executable tranche)

The phase gates correctly sequence:
- Phase 0: oracle/target registry before any comparison,
- Phase 1: fixed-program derivative and call-chain tests,
- Phase 2: proposal numerator/denominator separation,
- Phase 3: estimator-level bias/variance on common clouds,
- Phase 4: normalisation-effect classification,
- Phases 5–6: regular and degenerate branches (disjoint),
- Phases 7–9: heuristic dominance, tuning, and audit.

The first tranche (lines 1022–1038) correctly starts on linear-Gaussian models where exact scores exist. This is the right order.

**No repair needed.**

---

### 4. SGQF integration claim is not checked in this review

**Severity:** UNCHECKED

**Source:** Handoff lines 126–128, LaTeX lines 1456–1524 (cited by handoff)

The handoff identifies an "unsupported implementation implication": an eager standalone SGQF filter exists, but whether it is wired as an integrated LEDH moment provider for claim-bearing score calculations is not established.

**This review did not trace that call chain.** The handoff Stage 3 protocol (lines 234–283) requests specific SGQF implementation inspection, including batch dimensions, device, XLA support, persistent component state, and analytical sensitivity. That inspection was deferred per the bounded-review instruction to start with Phase 4C only.

**Status:** The SGQF proposal-quality arm (Phase 2, lines 641–660 of master) and its integration into LEDH score recursions remain **NOT CHECKED** by this review. A separate Stage 3 review is needed before promoting SGQF-guided LEDH to a claim-bearing arm.

---

### 5. Heuristic-dominance gate is correctly specified

**Severity:** None (praise)

**Source:** Lines 899–912 (Phase 7), master program origin note in global policy (CLAUDE.md)

Phase 7 requires constructing cheap practitioner baselines (Kalman, EKF/UKF, bootstrap PF) and evaluating complex methods against them conditionally on salient regimes. A loss to any heuristic in any salient regime is a promotion veto. The baselines are falsification checks, not tuning targets.

This matches the Heuristic Dominance Gate policy in CLAUDE.md and is the correct asymmetric burden: heuristics are weak adversaries, so beating them proves little, but losing to them proves the method is not ready.

**No repair needed.**

---

### 6. Literature gap addendum is thorough

**Severity:** None (praise)

**Source:** Lines 1040–1096 (literature coverage addendum)

The addendum identifies:
- Nemeth–Fearnhead–Mihaylova KDE Rao–Blackwell score,
- PaRIS backward-draw parameter and Fearnhead–Wyncoll–Tawn linear-cost smoother,
- Jacob–Lindsten–Schön coupled conditional-particle debiasing,
- Malik–Pitt continuous-likelihood particle filters,
- Ionides et al. iterated filtering,
- twisted PF change-of-measure correction,
- multilevel PF telescoping,
- variational objectives (FIVO/AESMC/VSMC) as distinct from observed-data scores.

Each is correctly classified by its target, support assumptions, and role (comparator, proposal quality, or target-boundary marker). The addendum notes that none of these close the degenerate-DSGE gap.

**No repair needed.** This is a model literature review for a score-estimation study.

---

## Unchecked components (deferred per handoff protocol)

The following remain **NOT CHECKED** by this bounded Stage 4 review:

1. **Stage 1 (KDM/IWSG target derivations):** IWSG identity, mixture score, support, differentiation under integral, bandwidth-dependent KDM target, control-variate centering, unnormalised pair estimates. Handoff lines 136–189.

2. **Stage 2 (proposal preservation, twisting, iAPF):** Model-corrected weights, evaluable proposal densities, twisted normalizer cancellation, iAPF pilot/fitting/iteration, OT role separation. Handoff lines 191–232.

3. **Stage 3 (SGQF moment lifecycle and integration):** Full time-indexed recursion, process vs filtered covariance, signed weights vs sampling mixture, call-chain audit from claim-bearing LEDH consumer to SGQF provider. Handoff lines 234–283.

4. **Stage 5 (test coverage and experiment fairness):** Oracle registry, coupled evaluator, proposal adapters, tuning/selection, baseline ladder, material defaults audit. Handoff lines 345–405.

5. **Stage 6 (implementation and source-faithfulness):** Call-chain audit of KDM integrated implementation, SGQF integration, source-faithfulness to Younis–Sudderth papers, author-code comparison. Handoff lines 406–456.

6. **Stage 7 (manuscript readability and MathDevMCP audit):** LaTeX equations, symbol definitions, proposition ledger. Handoff lines 457–489.

**Recommendation:** Repair Phase 4C (Finding 1), then proceed with Stage 1 (KDM/IWSG) as the next bounded review before executing the first tranche.

---

## Decision table

| Decision dimension | Status | Rationale |
|---|---|---|
| **Mathematical correctness of Phase 4C calibration test** | REVISE REQUIRED | Test conflates deterministic stencil order with stochastic MSE slope; will produce false vetoes and false passes. |
| **Target definitions and track separation** | AGREE | Exact model score, finite-program derivative, KDM expectation gradient, and degenerate vs regular tracks are correctly distinguished. |
| **Phase dependencies and gate structure** | AGREE | Sequential gating from oracle registry through heuristic dominance to audit is sound. First tranche on linear-Gaussian is correct. |
| **Heuristic-dominance gate** | AGREE | Cheap practitioner baselines, conditional evaluation, loss-to-heuristic veto, and no-tuning rule are correct. |
| **Literature coverage** | AGREE | Addendum correctly identifies missing named methods and their target/support classes. |
| **SGQF integration** | NOT CHECKED | Call chain from LEDH score consumer to SGQF provider not traced in this review. |
| **KDM/IWSG derivations** | NOT CHECKED | Deferred to Stage 1 bounded review. |
| **Proposal preservation and twisting** | NOT CHECKED | Deferred to Stage 2 bounded review. |
| **Implementation audit** | NOT CHECKED | Deferred to Stage 6 bounded review. |

---

## Inference-status table

| Inference claim | Evidence status | Next evidence needed |
|---|---|---|
| **Phase 4C test is valid** | HARD VETO | Deterministic and stochastic tests must be separated; slope-based veto removed. |
| **Master program structure is sound** | SUPPORTED | No change needed to registry architecture, phase sequence, or track separation. |
| **Finite-difference scores improve LEDH** | NO EVIDENCE YET | Phase 4C must be repaired and executed; MSE at selected `(h, N)` must beat current analytical score. |
| **SGQF integration is complete** | UNCHECKED | Stage 3 call-chain audit and wiring test required. |
| **KDM expectation gradient is correctly implemented** | UNCHECKED | Stage 1 derivation review and Stage 6 call-chain audit required. |
| **Heuristic baselines exist** | NOT ESTABLISHED | Kalman, EKF/UKF, bootstrap PF oracles and endpoints must be implemented before Phase 7. |
| **Stochastic ranking of score estimators** | NO EVIDENCE YET | Requires Phase 8 replication with paired uncertainty on oracle-bearing models. |

---

## Repair sequence (prioritized)

### 1. Repair Phase 4C calibration test (MANDATORY before execution)

**What to change:**

In master program lines 806–831, replace the calibration test with:

**Part 1 (deterministic stencil correctness):**
- Use smooth analytic functions or exact Kalman scores (no particle noise).
- Verify stencil weights via finite-precision identity tests.
- Fit log-error vs log-h slope over `h in [1e-8, 1e-2]` and verify order = 2, 4, or 4 for the three stencils.
- This is a unit test of `D_h` algebra. Pass/fail: observed order matches claimed order ± 0.5 over the pre-roundoff range.

**Part 2 (particle-score MSE calibration):**
- On stochastic particle runs at fixed `N`, measure `MSE(h)` over the ladder `h in {h0, h0/2, h0/4, h0/8}`.
- Fit or report the U-curve and its minimum `h*`.
- Record replicate variance, plus/minus covariance, ESS, and cost at each `h`.
- Select `h` by minimizing MSE, not by matching a slope.
- **No slope-based veto.** The curve shape is data.

**Part 3 (validation):**
- Freeze selected `(h, N)` on disjoint validation data.
- Report MSE and uncertainty.

**Part 4 (directional reconstruction):**
- For `d`-parameter score, use coordinate directions or full-rank `V`.
- Reconstruct score via `(V^T)^{-1} d` and record condition number.
- Rank deficiency is a hard failure.

**Promotion criterion:** The finite-difference score arm is viable if its MSE at selected `(h, N)` competes with other score estimators at matched compute on the heuristic-dominance table.

**Artifact:** Record all perturbed points, stencil weights, U-curve data, selected configuration, and MSE uncertainty. The observed slope is an explanatory diagnostic, not a gate.

### 2. Execute Stage 1 bounded review (KDM/IWSG derivations)

Before executing the first tranche, review:
- IWSG identity and mixture score (handoff lines 136–189),
- unnormalised pair/ratio estimates,
- control-variate centering and bias,
- bandwidth-dependent KDM target vs marginal score.

This review checks whether the KDM estimator computes the target it claims.

### 3. Execute Stage 3 bounded review (SGQF integration)

Before promoting SGQF-guided proposals to claim-bearing status, trace:
- call chain from LEDH score consumer to SGQF provider,
- batch dimensions, device, XLA support,
- persistent component state through resampling,
- analytical sensitivity through OT reset.

This review checks whether SGQF moments actually reach the LEDH score recursion.

### 4. Implement oracle and baseline ladder (Phase 0, Phase 7 preparation)

Before Phase 1 execution, implement:
- exact Kalman score for linear-Gaussian models,
- bootstrap PF score (Fisher recursion),
- UKF score (if UKF covariance is used in proposals),
- oracle-bearing nonlinear model (e.g., univariate/bivariate nonlinear Gaussian).

These are required comparators for the first tranche and heuristic-dominance table.

### 5. Execute first tranche on linear-Gaussian model (Phase 0–4, regular track)

After Repairs 1–4:
- target/oracle registry,
- fixed-program derivative tests,
- bootstrap, UKF, LEDH, Fisher, and repaired FD estimators,
- common-cloud estimator comparison (Phase 3),
- full-filter proposal comparison (Phase 2),
- repaired Phase 4C directional FD ladder,
- heuristic-dominance table,
- decision note.

This establishes whether any estimator improves over the current LEDH analytical score on an oracle-bearing model.

---

## Strongest alternative explanation

If finite-difference scores appear to "improve" over the current analytical score in early experiments:

**Alternative 1 (analytical score is wrong):** The current LEDH analytical total-derivative composition omits a term, computes a derivative incorrectly, or fails to account for OT reset. The finite-difference score is accidentally closer to the true score because it uses the complete particle program. **Test:** Independent autodiff/GradientTape should agree with finite differences and disagree with the analytical score at the same term.

**Alternative 2 (finite differences are biased downward):** The U-curve minimum occurs at large `h` where truncation bias partially cancels log-normalisation bias `b_N`, producing low MSE by accident rather than correctness. **Test:** Compare finite-difference MSE against exact Kalman score on linear-Gaussian models. If FD "wins" on oracle-free models but loses on Kalman models, the win is a bias-cancellation artifact.

**Alternative 3 (comparison is unfair):** The analytical score uses one particle count `N_a`, the FD score uses another `N_f`, and the comparison is by compute rather than by `N`. If `N_f > N_a`, the FD score improvement is Monte Carlo variance reduction, not score-method improvement. **Test:** Compare at fixed `N` and at fixed compute separately.

**Alternative 4 (the correct target is unbiased, FD is biased, and MSE improved anyway):** If the underlying statistical procedure (HMC, VI, or another consumer) is robust to score bias in a particular direction, a biased-but-low-variance score can outperform an unbiased-but-high-variance score in downstream performance. This is a valid finding, but it must be reported as "biased score that improves downstream task," not "better model score."

---

## What is not established by this program (non-claims)

Even after successful execution of the regular-track first tranche, the following remain unestablished:

1. **Degenerate-DSGE score estimation:** Phase 6 is a derivation prerequisite. No result from Phases 0–5 transfers to DSGE without a separate support-aware identity and estimator.

2. **Default or HMC-facing recommendation:** Phase 9 audit (call chain, TFP/XLA, tuning artifact, source-faithfulness, GPU compliance) is required before any candidate can replace the canonical LEDH analytical score in HMC consumers.

3. **High-dimensional or long-horizon scalability:** The first tranche uses modest `d`, `T`, and `N`. Scaling evidence requires Phase 7–8 experiments with declared compute caps.

4. **Unbiased score estimation:** KDM expectation gradients, finite-program derivatives, and finite-difference estimates are distinct targets with their own bias/variance decomposition. An unbiased KDM gradient is not an unbiased model score. A low-MSE biased estimator is still biased.

5. **SGQF moment accuracy guarantees:** Sparse-grid quadrature provides moment approximations. Whether those approximations help proposal quality and whether improved proposals reduce score MSE are empirical questions, not guarantees.

6. **Score improvement on oracle-free models:** Heuristic dominance and MSE reduction on oracle-bearing models are necessary but not sufficient. Validation on oracle-free models (where the "true score" is unknown) requires either a proved bias bound or calibrated diagnostics that predict score error on held-out oracle cases.

---

## VERDICT: REVISE

**Scope of verdict:** This verdict applies to Phase 4C and its readiness for execution. The broader master program structure, target definitions, and phase dependencies are sound and receive local AGREE verdicts as documented above.

**Mandatory repair before Phase 4C execution:** Separate deterministic stencil-correctness tests from stochastic MSE calibration. Remove slope-based vetoes from particle-score experiments. Select `h` by minimizing observed MSE, not by matching a truncation-order slope.

**Next review:** After repairing Phase 4C, proceed to Stage 1 bounded review (KDM/IWSG derivations, handoff lines 136–189) before executing the first tranche.

**Reviewer availability:** This review is complete for its bounded scope. Further stages remain unreviewed and require separate bounded reviews per the handoff protocol.

---

## Appendix: Phase 4C repair sketch (concrete replacement text)

### Recommended replacement for master program lines 806–831

**Calibration test structure:**

The calibration has four independent parts. Parts 1 and 4 are mechanics tests (deterministic, no particle noise). Parts 2 and 3 are stochastic experiments (particle noise, MSE calibration).

**Part 1: Deterministic stencil-correctness test (unit test)**

Verify the stencil algebra on smooth deterministic test functions. Use either:
- analytic functions `f(x) = exp(a*x) + b*x^5` with exact derivatives, or
- linear-Gaussian models with exact Kalman log-likelihood and scores (no particle approximation).

Evaluate `D_2(h)`, `D_4(h)`, and the eleven-point cubic-fit derivative at `h in {2^(-30), 2^(-27), ..., 2^(-6)}` (pre-roundoff range). Fit `log|error| = log(C) + p*log(h)` and verify `p in [1.5, 2.5]` for `D_2`, `p in [3.5, 4.5]` for `D_4` and cubic fit. Record the error constant `C` and leading-order coefficient from the fit.

**Pass criterion:** Observed order matches claimed order within 0.5. Failure indicates implementation error, not particle noise.

**Artifact:** Stencil weights, test functions, observed orders, error constants, and residual plots.

**Part 2: Particle-score MSE calibration (stochastic experiment)**

On oracle-bearing models with finite-particle approximation, measure MSE of each directional finite-difference estimator as a function of `(h, N)`.

For fixed `N in {128, 256, 512}` and `h in {h0, h0/2, h0/4, h0/8}` with `h0` chosen in parameter units to keep perturbed models in the support:
- Generate `R >= 30` independent replicate particle runs at each `(h, N)`.
- Compute exact oracle directional score `s_true`.
- Compute `MSE(h, N) = mean over replicates of (D_h - s_true)^2`.
- Estimate replicate variance and 95% uncertainty interval on MSE.
- Record plus/minus likelihood covariance under common random numbers.
- Record ESS, weight tail diagnostics, non-finite counts, and wall time.

Plot `MSE(h)` at each `N`. The curve is U-shaped: truncation dominates at large `h`, particle noise dominates at small `h`. Identify the empirical minimum `h*(N)` for each `N`.

**No slope-based veto.** The shape of the U-curve is data. A steep noise amplification or a flat minimum are both scientific findings.

**Part 3: Ladder selection and validation**

On the calibration partition, select `(h_sel, N_sel)` by one of:
- minimize `MSE(h, N_sel)` at declared `N_sel`, or
- minimize compute-normalized MSE `MSE(h, N) * cost(h, N)`, or
- balance truncation and noise via `h*(N)`.

Record the selection criterion and the achieved MSE.

On a disjoint validation partition, rerun the selected configuration and verify that MSE remains within the calibration uncertainty. A large validation-calibration gap indicates overfitting or insufficient calibration replicates.

**Part 4: Full-rank directional reconstruction**

For a `d`-parameter score, construct a direction matrix `V` (shape `d x d` or `d x m` with `m >= d`). Use either:
- coordinate directions `V = I_d`, or
- random orthonormal directions from QR decomposition, or
- adaptive directions from previous parameter draws.

Compute directional estimates `v_i^T s` for each direction `v_i`. Reconstruct the full score via least-squares: `s_recon = (V^T V)^{-1} V^T d_obs`, where `d_obs` is the vector of directional estimates.

Compute and record the condition number `kappa(V^T V)`. A condition number above `1e6` indicates near-rank-deficiency; above `1e12` indicates numerical rank loss and is a hard test failure.

**Promotion criterion**

The finite-difference directional score arm is viable for downstream comparison if:
1. Part 1 (deterministic stencil test) passes for the claimed stencil order.
2. Part 2 (MSE calibration) completes without non-finite values or support violations at the selected `(h, N)`.
3. Part 3 (validation) confirms the calibration MSE within uncertainty.
4. Part 4 (directional reconstruction) achieves condition number below `1e12`.

The viable arm then enters Phase 7 (heuristic dominance) and Phase 8 (full stochastic comparison with uncertainty). Its promotion depends on MSE competitive with other score estimators at matched compute, not on a particular U-curve slope.

**Diagnostic role of slopes**

The observed `MSE(h)` slope in log-log space is an explanatory diagnostic:
- Slope ≈ `+2p` (truncation-dominated): truncation error dominates; larger `N` or smaller `h` may help.
- Slope ≈ `-2` (noise-dominated): particle noise amplified by `1/h^2` dominates; larger `N` or larger `h` required.
- Slope ≈ `0` (minimum): near-optimal balance; further `h` tuning has limited benefit.

Record slopes in the artifact for interpretation, but do not use them as pass/fail gates.

---

**End of review.**

