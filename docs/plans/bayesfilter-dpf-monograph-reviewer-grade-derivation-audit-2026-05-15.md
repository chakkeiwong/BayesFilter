# Phase P11 result: mathematical derivation audit

## Date

2026-05-15

## Governing plan

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p11-derivation-audit-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p10-notation-claim-consolidation-result-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`

## Branch and worktree classification

- Branch: `main`.
- Divergence: `main...origin/main [ahead 2]`.
- In-lane dirty DPF chapter files present before P11:
  - `docs/chapters/ch19_particle_filters.tex`;
  - `docs/chapters/ch19b_dpf_literature_survey.tex`;
  - `docs/chapters/ch19c_dpf_implementation_literature.tex`;
  - `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`;
  - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`;
  - `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`;
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`.
- In-lane reviewer-grade planning/result artifacts remain untracked.
- Out-of-lane dirty files remain present and were not edited:
  - `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`;
  - `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
  - `experiments/controlled_dpf_baseline/README.md`;
  - `experiments/controlled_dpf_baseline/fixtures/README.md`.

## Allowed write set used

- This P11 derivation-audit result artifact only.

No student-baseline file, controlled-baseline experiment file, chapter file, git
history operation, deletion, push, merge, rebase, reset, or staging operation was
performed in P11.

## Tool status

MathDevMCP was available for label lookup and diagnostic extraction, but its
formal proof backend was limited:

- available: `sympy`;
- unavailable: `latexml`, `pandoc`, `lean`, `sage`, `lean_dojo`.

Therefore MathDevMCP results in this phase are parser/diagnostic evidence, not
machine proof certificates.  Probabilistic filtering, measure-theoretic
transport, HMC target-contract, and optimization-KKT claims were audited
manually against the local derivations and recorded as manual checks.

ResearchAssistant was queried again for DPF/particle-flow/OT/Sinkhorn/HMC
material.  It returned no local paper summaries.  Source support therefore
remains the P2 status: bibliography-spine provenance plus local derivation, not
ResearchAssistant-reviewed source evidence.

## Derivation-obligation register

| ID | Chapter | Equation/claim | Assumptions | Derivation location | Source support | Audit method | Result | Required repair |
|---|---|---|---|---|---|---|---|---|
| P11-O01 | `ch19_particle_filters.tex` | Nonlinear filtering recursion and marginal-likelihood factorization, `eq:bf-pf-predictive-law`--`eq:bf-pf-marginal-factorization` | Dominated transition/observation densities; fixed `theta`; finite predictive normalizers; state-space conditional independence | `sec:bf-pf-ssm-recursion`, especially the Chapman--Kolmogorov set identity and Bayes-rule paragraph | Locally derived; SMC bibliography-spine per P2 | Manual derivation check; MathDevMCP attempt on `eq:bf-pf-filtering-law` failed with a tool execution error | Passed. Symbols are defined locally and exactness is bounded to model identities. | None. |
| P11-O02 | `ch19_particle_filters.tex` | SIS recursive weight formula, `eq:bf-pf-sis-recursion` | Proposal factorization; proposal support covers target support; finite/integrable weights | `sec:bf-pf-sis`, from trajectory proposal/target ratios | Locally derived; SMC bibliography-spine per P2 | Manual ratio check; MathDevMCP returned unverified/inconclusive because notation exceeds bounded algebraic backend | Passed. The recursion follows by dividing the trajectory target/proposal ratio at time `t` by its `t-1` counterpart. | None. |
| P11-O03 | `ch19_particle_filters.tex` | Bootstrap likelihood-estimator status, `prop:bf-pf-bootstrap-likelihood-status` | Bootstrap propagation; conditionally unbiased resampling; dominated/integrable likelihood factors; standard Feynman--Kac particle system | `sec:bf-pf-likelihood-estimator`, proposition and proof | Locally derived; Doucet/Andrieu bibliography-spine per P2 | Manual conditional-expectation/tower-property audit; MathDevMCP returned diagnostic-only/unverified | Passed with non-formal status. The proposition states likelihood unbiasedness only and explicitly blocks log-likelihood, score, and pathwise-gradient overclaims. | None. |
| P11-O04 | `ch19b_dpf_literature_survey.tex` | Homotopy derivative and normalizer role, `eq:bf-pff-log-homotopy`--`eq:bf-pff-log-homotopy-derivative` | Positive likelihood where log is used; finite normalizer; differentiability in pseudo-time and dominated interchange | `sec:bf-pff-homotopy`, including centered log-likelihood identity | Locally derived; particle-flow bibliography-spine per P2 | Manual calculus audit | Passed. The normalizer derivative is translated into an expectation under `pi_{t,lambda}` and its omission is explicitly identified as a law change. | None. |
| P11-O05 | `ch19b_dpf_literature_survey.tex` | Continuity and log-density transport equations, `eq:bf-pff-continuity-equation`--`eq:bf-pff-flow-pde` | Density differentiable in pseudo-time; continuously differentiable in state; positive density for log form; boundary flux vanishes or boundary condition handled | `sec:bf-pff-continuity` | Locally derived; particle-flow bibliography-spine per P2 | Manual weak-form and integration-by-parts audit | Passed. The text states the boundary/regularity assumptions and emphasizes non-uniqueness of the velocity field. | None. |
| P11-O06 | `ch19b_dpf_literature_survey.tex` | EDH covariance and affine coefficients, `eq:bf-pff-homotopy-covariance`--`eq:bf-pff-edh-ode` | Gaussian closure; positive-definite covariance; linear-Gaussian observation within the closure; stable/invertible precision matrices | `sec:bf-pff-edh` | Locally derived; Daum-Huang/PF-flow bibliography-spine per P2 | Manual matrix-calculus audit; MathDevMCP flagged inverse/solve conditioning obligations but could not certify | Passed. Matrix inverse derivative, affine mean/covariance matching, and approximation boundary are local. | None. Implementation-side solve residual and conditioning diagnostics remain P9/P12/P13 evidence requirements, not chapter repairs. |
| P11-O07 | `ch19b_dpf_literature_survey.tex` | Linear-Gaussian recovery of mean and covariance, `eq:bf-pff-kalman-endpoint`--`eq:bf-pff-kalman-mean-endpoint` | Truly linear-Gaussian observation model; Gaussian predictive law; exact or tolerance-controlled affine ODE integration | `sec:bf-pff-linear-gaussian-recovery` | Locally derived; Kalman/flow source role through chapter context | Manual information-form Kalman audit | Passed. Exactness is restricted to the special case and not exported to nonlinear filtering. | None. |
| P11-O08 | `ch19b_dpf_literature_survey.tex` | LEDH local information vector, `eq:bf-pff-local-jacobian`--`eq:bf-pff-ledh-ode` | Local differentiability of observation map; stable particle-local Jacobian; local linear Gaussian approximation; positive-definite local precision | `sec:bf-pff-ledh` | Locally derived; particle-flow bibliography-spine per P2 | Manual completion-of-squares/local-linearization audit | Passed. The information vector is derived from the shifted local observation equation and LEDH is framed as approximate closure. | None. |
| P11-O09 | `ch19c_dpf_implementation_literature.tex` | PF-PF change-of-variables density, `eq:bf-pfpf-postflow-density` | Differentiable bijective flow on region of interest; nonzero Jacobian determinant; pre-flow proposal density defined | `sec:bf-pfpf-change-of-variables` | Locally derived; PF-PF bibliography-spine per P2 | Manual density-transform audit | Passed. Forward/inverse determinant conventions are explicit. | None. |
| P11-O10 | `ch19c_dpf_implementation_literature.tex` | PF-PF corrected weight, `eq:bf-pfpf-generic-weight` and `eq:bf-pfpf-weight` | Intended one-step target named; transported proposal density correct; flow bijection; finite densities; ancestor conditioning acknowledged | `sec:bf-pfpf-corrected-weights` | Locally derived; PF-PF/SMC bibliography-spine per P2 | Manual importance-ratio audit | Passed. The correction restores a one-step proposal-to-target density ratio, while finite-particle and full-recursion limits are stated. | None. |
| P11-O11 | `ch19c_dpf_implementation_literature.tex` | Flow Jacobian ODE, `eq:bf-pfpf-jacobian-ode` | Differentiable flow field; differentiability with respect to initial condition; square state dimension | `sec:bf-pfpf-logdet` | Locally derived | Manual chain-rule audit | Passed. The ODE follows from differentiating the flow equation with respect to the initial state. | None. |
| P11-O12 | `ch19c_dpf_implementation_literature.tex` | Log-determinant trace identity, `eq:bf-pfpf-logdet-ode` and `eq:bf-pfpf-logdet-affine` | Invertible Jacobian along path; differentiable flow; valid log-absolute determinant; affine case has `n_x x n_x` drift matrix | `sec:bf-pfpf-logdet` | Locally derived | Manual Jacobi-formula audit; MathDevMCP flagged logdet-domain and inverse/solve checks but could not certify | Passed. Trace cyclicity justifies the simplification and sign conventions are explicit. | None. |
| P11-O13 | `ch32_diff_resampling_neural_ot.tex` | Soft-resampling mean preservation and nonlinear bias, `eq:bf-dr-soft-mean-preserved`--`eq:bf-dr-soft-test-bias` | Normalized nonnegative weights; ancestor draw from weights; `alpha in [0,1]`; twice differentiable scalar test function near interpolation segments | `sec:bf-dr-soft-resampling` | Locally derived; differentiable-resampling bibliography-spine per P2 | Manual expectation and Taylor-expansion audit | Passed. Mean preservation is limited to affine summaries; nonlinear bias is explicitly first-order with remainder. | None. |
| P11-O14 | `ch32_diff_resampling_neural_ot.tex` | OT coupling constraints, `eq:bf-dr-coupling-set` and `eq:bf-dr-unregularized-ot` | Nonnegative weights; matching total mass; finite cost matrix; chosen source/target support convention | `sec:bf-dr-transport-view` | Locally derived; OT bibliography-spine per P2 | Manual dimension/marginal audit | Passed. The text separates exactness for the transport projection from categorical resampling. | None. |
| P11-O15 | `ch32_diff_resampling_neural_ot.tex` | EOT first-order conditions and Sinkhorn scaling, `eq:bf-dr-eot-primal`--`eq:bf-dr-sinkhorn-updates` | Positive marginals; finite cost; `epsilon>0`; positive optimizer for log derivative; fixed entropy convention | `sec:bf-dr-entropic-ot` | Locally derived; EOT/Sinkhorn bibliography-spine per P2 | Manual KKT/scaling audit; MathDevMCP returned diagnostic-only/inconclusive | Passed. Entropy sign convention, stationarity, Gibbs factorization, scaling equations, and finite-iteration caveat are local. | None. |
| P11-O16 | `ch32_diff_resampling_neural_ot.tex` | Barycentric projection dimensions, `eq:bf-dr-barycentric-map` | Target marginal `u_j=1/N`; column mass positive; particles in `R^{n_x}`; coupling dimensions `N x N` | `sec:bf-dr-entropic-ot` | Locally derived | Manual dimensional audit | Passed. The `N` factor is explained as conditional source weights for each output particle. | None. |
| P11-O17 | `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` | Learned teacher/student residual hierarchy, `eq:bf-learned-ot-student-map`--`eq:bf-learned-ot-hmc-gradient-residual` and `tab:bf-learned-ot-residual-hierarchy` | Named teacher variant; declared training distribution; student family fixed; residual metric matched to claim | `sec:bf-learned-ot-teacher` and `sec:bf-learned-ot-residuals` | Locally derived/defined; learned-OT and set-architecture bibliography-spine per P2 | Manual object-definition and claim-boundary audit | Passed. Residual layers are defined as different objects and do not imply posterior or HMC correctness. | None. |
| P11-O18 | `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` | Permutation-equivariance equation, `eq:bf-learned-ot-equivariance` and `eq:bf-learned-ot-invariance` | Particle labels arbitrary; permutation matrix `P`; output ordering convention declared | `sec:bf-learned-ot-symmetry` | Locally defined; DeepSets/Set Transformer bibliography-spine per P2 | Manual symmetry audit | Passed. Symmetry is framed as necessary but not target-sufficient. | None. |
| P11-O19 | `ch19e_dpf_hmc_target_suitability.tex` | HMC value-gradient contract, `eq:bf-dpf-hmc-contract` | Differentiable scalar on sampler-valid region; gradient vector in same unconstrained coordinates; accept/reject value tied to same scalar unless separate proposal proof supplied | `sec:bf-dpf-hmc-contract` | Locally defined; HMC/pseudo-marginal bibliography-spine per P2 | Manual target-contract audit; MathDevMCP flagged dimension/formalization limits | Passed. The equation is a minimum consistency contract, not a claim of exact filtering likelihood. | None. |
| P11-O20 | `ch19e_dpf_hmc_target_suitability.tex` | Rung-by-rung target-status classification, `sec:bf-dpf-hmc-rung-analysis` and `tab:bf-dpf-hmc-rungs` | Scalar value and gradient path named for each rung; correction/randomness policy declared before promotion | `sec:bf-dpf-hmc-rung-analysis`, `tab:bf-dpf-hmc-rungs`, and `sec:bf-dpf-hmc-recommendation` | Locally synthesized; HMC/PMCMC/surrogate bibliography-spine per P2 | Manual claim-ledger audit against P1/P2/P8 | Passed. EDH/LEDH, PF-PF, soft, OT/EOT, and learned OT are classified conservatively; banking and production claims remain blocked. | None. |

## Repairs and weakenings

No chapter repair was required in P11.  The audit found tool-limited formal
verification, not silent mathematical failure.  The chapters already contain:

- local assumptions near each load-bearing formula;
- exact/approximate/relaxed/surrogate status boundaries;
- source roles treated as provenance rather than proof;
- implementation diagnostics tied to the equations.

The only remaining caution is evidentiary: several obligations are manually
audited rather than formally certified.  This is acceptable for the P11 gate
because the plan allows manual derivation checks and requires tool limitations
to be recorded honestly.

## Checks run

- `git branch --show-current`.
- `git status --short --branch`.
- `git log --oneline --decorate -n 8`.
- Direct label inventory with `rg -F '\label{'` across the DPF chapter block.
- Direct equation/proposition inventory with `rg` over equation, align, and
  proposition environments.
- Overclaim search across the DPF chapter block for:
  - `exact`;
  - `unbiased`;
  - `consistent`;
  - `validated`;
  - `valid`;
  - `robust`;
  - `HMC-ready`;
  - `production`;
  - `optimal`;
  - `guarantee`;
  - `solves`;
  - `proves`;
  - `suitable`;
  - `credible`;
  - `first serious`;
  - `bank`;
  - `governance`;
  - `ResearchAssistant`;
  - `reviewed`.
- ResearchAssistant query:
  - `differentiable particle filter particle flow PF-PF entropic optimal transport Sinkhorn HMC pseudo marginal`;
  - result: no local paper summaries.
- MathDevMCP:
  - `doctor`;
  - `audit_derivation_v2_label` on `eq:bf-pf-filtering-law`, which failed with a tool execution error;
  - `audit_derivation_v2_label` on `eq:bf-pf-sis-recursion`, diagnostic-only/unverified;
  - `audit_derivation_v2_label` on `prop:bf-pf-bootstrap-likelihood-status`, diagnostic-only/unverified;
  - `audit_derivation_v2_label` on `eq:bf-pff-homotopy-cov-derivative`, diagnostic-only/unverified;
  - `audit_derivation_v2_label` on `eq:bf-pfpf-logdet-ode`, diagnostic-only/unverified;
  - `audit_derivation_v2_label` on `eq:bf-dr-eot-stationarity`, diagnostic-only/unverified;
  - `audit_derivation_v2_label` on `eq:bf-dpf-hmc-contract`, diagnostic-only/unverified.

## Overclaim disposition

The overclaim search found many instances of terms such as `exact`, `unbiased`,
`valid`, `credible`, `bank`, `production`, and `governance`.  Manual review found
these uses bounded by local assumptions, target-status language, or explicit
non-claim language.  Notable dispositions:

- `exact` is used for model identities, named special cases, exact regularized
  or transport objects, and explicitly blocked where nonlinear/generic exactness
  would be false.
- `unbiased` is limited to likelihood-estimator status, not log likelihood,
  score, pathwise gradients, or HMC validity.
- `valid` appears mainly in support/target-contract warnings and does not
  promote relaxed or learned rungs to original-posterior correctness.
- `credible`, `bank`, `production`, and `governance` appear inside evidence
  ladders and blocked-promotion language, not as achieved claims.
- `first serious` is absent from the chapter block; the surviving language is
  the weaker "first rung with explicit proposal-correction interpretation" or
  "research candidate" language.
- `ResearchAssistant` and `reviewed` are not used in the chapters as source
  promotions.

## Veto diagnostic review

- Central equations remain unchecked: no.  All 20 required obligations were
  audited and recorded above.
- Failed obligations silently ignored: no.  No chapter obligation failed; tool
  limitations are recorded.
- MathDevMCP limitations omitted: no.  Backend availability and diagnostic-only
  status are recorded.
- Reviewer-critical claims left as "obvious": no.  Each central formula has
  assumptions, local derivation, and claim-status boundaries.
- Repairs create notation inconsistency: no chapter repairs were made.
- Student-baseline files edited or staged: no.
- Source citations accepted as proof without local notation translation: no.
  Source status remains bibliography-spine plus local derivation.
- MathDevMCP/manual/blocked status omitted for any load-bearing obligation: no.

## Exit gate

Passed for manual derivation audit only.

Every load-bearing obligation required by P11 has passed manual audit, been
classified with source-support status, and had tool limitations recorded where
formal certification was unavailable.  This is not a machine-formal proof
certificate, not a ResearchAssistant-reviewed source audit, and not an
implementation-validation result.  The next phase is P12:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p12-hostile-reader-audit-plan-2026-05-14.md`.
