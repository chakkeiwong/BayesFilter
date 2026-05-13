# BayesFilter V1 Structural Sigma-point Score Plan For Default Model C

## Date

2026-05-13

## Governing Master Program

This subplan closes R1 in:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

R1 is now the Chapter 18b structural sigma-point score implementation, not a
separate fixed-null SVD derivative theory.

## Motivation

Default Model C has a deterministic phase coordinate with zero variance.  The
current collapsed smooth SVD score path blocks this law through the active-floor
gate.  That block is useful, but it is not the final design answer.

Chapter 18b already states the correct structural law: sigma points should be
placed on the pre-transition uncertainty
\[
  A_t=(x_{t-1},\varepsilon_t),
\]
or an equivalent parameterization, and the state should then be completed
pointwise by
\[
  x_t=F_\theta(A_t).
\]
For default Model C, the deterministic phase coordinate is not an independent
post-transition noise coordinate.  It is part of the structural map.

## Core Hypothesis

H-S1:
The Chapter 18b structural sigma-point recursion provides the correct score
path for default Model C.  It should allow analytic first-order scores of the
implemented structural sigma-point likelihood without adding an artificial
phase nugget or differentiating arbitrary null-space eigenvectors from a
collapsed full-state SVD.

## Phase S0: Derivation From Chapter 18b With MathDevMCP

Goal:
- extract the Chapter 18b equations and propositions that define the
  structural sigma-point law;
- derive the first-order score recursion in the notation used by
  `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py`;
- write a concrete document block in the monograph, either in Chapter 18b or
  a Chapter 18 cross-reference section, depending on where it fits best.

Required source anchors:

- `prop:bf-structural-ukf-pushforward`;
- `eq:bf-structural-ukf-prop-map`;
- `eq:bf-structural-ukf-prop-factorization`;
- `eq:bf-structural-ukf-prop-pushforward`;
- `eq:bf-structural-ukf-prop-deterministic-completion`;
- `sec:bf-structural-versus-numerical-degeneracy`;
- `eq:bf-structural-ukf-moment-objects`;
- `eq:bf-structural-ukf-update-objects`;
- `eq:bf-structural-ukf-loglik-object`.

MathDevMCP tasks:
- use `extract_latex_context` or `latex_label_lookup` for the labels above;
- use `derive_label_step` or `check_proof_obligation` for the key pushforward
  and score-chain equations when practical;
- use `compare_label_code` or `audit_implementation_label` later after code is
  written.

Deliverable:
- a derivation note or chapter subsection stating:
  \[
    A_t=(x_{t-1},\varepsilon_t),\qquad
    X_t^{(j)}=F_\theta(A_t^{(j)}),\qquad
    Z_t^{(j)}=h_\theta(X_t^{(j)}),
  \]
  the moment equations, the innovation likelihood contribution, and the
  first-order derivative recursion.

Concrete derivation targets:
- define the pre-transition Gaussian approximation
  \(A_t\sim N(\mu_{a,t},P_{a,t})\), including the filtered state block,
  innovation block, and any declared cross-covariance block;
- define the chosen sigma rule by canonical points \(\xi_j\), weights
  \(w_j^{(m)},w_j^{(c)}\), covariance factor \(L_{a,t}\), and points
  \(A_t^{(j)}=\mu_{a,t}+L_{a,t}\xi_j\);
- show that deterministic coordinates enter only through
  \(X_t^{(j)}=F_\theta(A_t^{(j)})\), not as independent zero-variance columns in
  a collapsed post-transition covariance factor;
- derive the implemented moment objects
  \(\hat x_t,\hat z_t,P_{xx,t},S_t,P_{xz,t}\) and update
  \((\hat x_{t|t},P_{t|t})\) from the structural point cloud;
- derive first derivatives of each object with respect to a scalar parameter
  \(\theta_r\):
  \(dA_t^{(j)},dX_t^{(j)},dZ_t^{(j)},d\hat z_t,dS_t,dP_{xz,t},dv_t,dK_t\),
  \(d\hat x_{t|t}\), and \(dP_{t|t}\);
- derive the likelihood score contribution
  \[
    d\ell_t
    =
    -\frac12\left[
      \operatorname{tr}(S_t^{-1}dS_t)
      -2(d\hat z_t)^\top S_t^{-1}v_t
      -v_t^\top S_t^{-1}(dS_t)S_t^{-1}v_t
    \right],
  \]
  with the sign convention checked against
  \(v_t=y_t-\hat z_t\);
- state the rank/floor rule for the factor of \(P_{a,t}\): structural
  deterministic completion is not regularized by adding a fake phase nugget;
  any floor diagnostic belongs to the stochastic pre-transition covariance
  factor actually used by the implemented sigma rule.

Chapter placement:
- Chapter 18b remains the source for the structural filtering law;
- the derivative subsection should preferably live in
  `docs/chapters/ch18_svd_sigma_point.tex` with explicit references back to
  Chapter 18b, unless the derivation reveals that Chapter 18b itself needs a
  short clarifying cross-reference;
- if Chapter 18b must be edited, record that lane-opening reason in this
  V1 reset memo before making the change.

Stop rule:
- stop if the derivation requires changing the structural law in Chapter 18b
  rather than merely implementing it.

## Phase S1: Update The Master Plan From The Derivation

Status:
- completed after S0.

Goal:
- revise this subplan and the V1 master with the exact derivation result;
- name whether default Model C should become score-ready or remain blocked.

Deliverable:
- update `docs/plans/bayesfilter-v1-master-program-2026-05-13.md`;
- update the V1 reset memo with interpretation and whether implementation is
  justified.

Continuation criterion:
- continue only if the derivation gives a concrete code contract.

Concrete code contract required before S2:
- parameter vector layout and derivative-provider callback signature for
  default Model C;
- tensor shapes for \(\mu_{a,t},P_{a,t},L_{a,t},A_t^{(j)},X_t^{(j)},Z_t^{(j)}\);
- exact list of diagnostics returned by the value path and by the score path;
- finite-difference target: the same structural sigma-point likelihood, not a
  collapsed full-state likelihood;
- pass/fail tolerances for SVD cubature, SVD-UKF, and SVD-CUT4.

S0 derivation outcome:
- Chapter 18 now contains
  `sec:bf-svd-sp-structural-fixed-support-score`;
- the implementation should preserve the declared sigma-rule dimension
  \(n_x+n_\varepsilon\);
- structural zero directions should be represented by zero factor columns and
  zero factor derivatives;
- continuation to S2 is justified only for a branch that checks fixed null
  support and blocks a parameter-moving null block, a positive placement floor,
  or weak active spectral gaps.

## Phase S2: Implement The Structural Score Path

Goal:
- implement the derivative path according to the derivation, not according to a
  collapsed full-state SVD workaround.

Likely implementation targets:

- `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py`;
- `bayesfilter/testing/nonlinear_models_tf.py`;
- `tests/test_nonlinear_sigma_point_scores_tf.py`;
- `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py`.

Expected design:
- place sigma points on the pre-transition Gaussian approximation for
  \((x_{t-1},\varepsilon_t)\), respecting the declared structural integration
  space;
- preserve the backend's declared sigma rule dimension
  \(n_x+n_\varepsilon\), so SVD-CUT4 remains the documented CUT4-G rule even
  when default Model C has a rank-two stochastic support inside a
  three-dimensional pre-transition variable;
- represent structurally null directions by zero factor columns and zero
  factor derivatives, after checking that the null block is fixed rather than
  a hidden parameter-dependent floor branch;
- propagate each point through `model.transition`;
- complete deterministic identities pointwise;
- differentiate the structural map and moment closure directly;
- avoid adding a fake nugget to deterministic coordinates.

Implementation boundary:
- production code may use TensorFlow and TensorFlow Probability only;
- testing oracles may use NumPy only when they are clearly testing-only and do
  not enter production imports;
- no GPU/XLA specialization, HMC target, or Hessian path is part of this phase.

Stop rule:
- stop if implementation would require changing MacroFinance, DSGE, Chapter
  18b structural claims, or the shared monograph reset memo.

## Phase S3: Code-Document Consistency Audit

Goal:
- verify that the implementation follows the derivation.

Tool-aided audit:
- use MathDevMCP `compare_label_code` or `audit_implementation_label` against
  the new derivation label;
- manually audit tensor dimensions, branch conditions, diagnostics, and
  derivative-target metadata.

Required diagnostics:
- deterministic residual;
- support residual;
- active floor count;
- spectral-gap diagnostics;
- derivative provider name;
- derivative method;
- whether the derivative target is the structural implemented law.

Stop rule:
- stop if code and document disagree on the sigma-point variable.

## Phase S4: Test Against Default Model C

Goal:
- test default Model C with zero phase variance, not only the smooth
  nondegenerate phase variant.

Tests:
- analytic score versus centered finite differences of the same implemented
  structural sigma-point likelihood;
- all three backends where the derivation supports them:
  SVD cubature, SVD-UKF, and SVD-CUT4;
- branch summaries over a small default-Model-C parameter box;
- regression test that no artificial phase nugget is introduced.
- regression test that the score diagnostics report the sigma-point variable as
  pre-transition/structural rather than collapsed full-state.

Pass criterion:
- scores match finite differences within documented tolerance;
- deterministic residual remains zero to tolerance;
- default Model C has zero placement floor count when the only zero eigenvalue
  is the structurally fixed phase direction and `placement_floor=0`;
- diagnostics make clear that the score path is structural, not collapsed SVD
  regularization.

Fallback:
- if default Model C remains blocked, record the mathematical reason and keep
  HMC/GPU/Hessian blocked.

## Phase S5: Finish, Document, And Update Artifacts

Goal:
- close the phase cleanly and update the V1 control documents.

Deliverables:
- result artifact under `docs/plans`;
- updated V1 reset memo;
- updated `docs/source_map.yml`;
- any chapter updates needed after implementation;
- focused nonlinear score/branch tests;
- full default CPU suite if code changed.

Completion rule:
- commit only V1-lane files;
- leave shared monograph reset memo and structural/DSGE/MacroFinance lanes
  untouched.

## Non-goals

- no HMC implementation in this phase;
- no GPU/XLA benchmark in this phase;
- no nonlinear Hessian implementation in this phase;
- no MacroFinance or DSGE switch-over;
- no independent fixed-null SVD theory separate from Chapter 18b.
