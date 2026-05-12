# BayesFilter V1 Nonlinear Derivative-Provider Gap Closure Plan

## Date

2026-05-13

## Motivation

The previous nonlinear filtering passes established value-filter evidence for
SVD cubature, SVD-UKF, and SVD-CUT4 and certified analytic first-order scores
only on a smooth affine fixture.  The main remaining nonlinear gap is that
Models B--C have value diagnostics but no explicit structural derivative
providers, so analytic score, GPU/XLA, and HMC claims remain blocked.

This pass tests the next hypothesis in the V1 nonlinear lane:

\[
  \nabla_\theta \ell_{\mathrm{analytic}}(\theta)
  \approx
  \nabla_\theta \ell_{\mathrm{FD}}(\theta)
\]

for the implemented SVD sigma-point likelihood on smooth no-floor branches.

## Lane Boundary

Owned files for this pass:

- `bayesfilter/testing/nonlinear_models_tf.py`
- `bayesfilter/testing/__init__.py`
- `tests/test_nonlinear_sigma_point_scores_tf.py`
- `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py`
- `docs/chapters/ch28_nonlinear_ssm_validation.tex`
- `docs/plans/bayesfilter-v1-*.md`
- `docs/source_map.yml`

Do not edit or stage:

- MacroFinance source;
- DSGE source;
- Chapter 18b;
- structural SVD/SGU lane plans;
- the shared monograph reset memo;
- unrelated `Zone.Identifier` files or local images.

## Hypotheses

H-D1:
Model B can receive an explicit first-derivative provider for
\(\theta=(\rho,\sigma,\beta)\) without changing the production structural
callback contract.

H-D2:
For Model B, SVD cubature, SVD-UKF, and SVD-CUT4 analytic scores match centered
finite differences of the same implemented value filters on the selected
smooth parameter point.

H-D3:
Model C can receive an explicit first-derivative provider for
\(\theta=(\sigma_u,\sigma_y,P_{0,x})\).  On a nondegenerate phase-state testing
variant, its analytic scores match finite differences.

H-D4:
The default Model C fixture, whose phase coordinate has zero initial variance,
should remain blocked by the current smooth simple-spectrum/no-active-floor SVD
score branch.  This is not a formula failure; it is evidence that a future
fixed-null-direction derivative branch is needed before scoring the default
deterministic-phase law.

H-D5:
Hessian, GPU/XLA, and nonlinear HMC work remain unjustified until H-D1--H-D4
are resolved and score branch diagnostics are finite on target boxes.

## Tightened Execution Plan

### Phase D0: Baseline And Drift Check

Plan:
- run the current focused nonlinear score/value/diagnostic baseline on CPU;
- confirm dirty files from other lanes remain unstaged.

Primary criterion:
- focused nonlinear tests pass before edits.

Veto diagnostics:
- fail if existing score tests are broken;
- fail if the plan would require MacroFinance, DSGE, Chapter 18b, or shared
  reset-memo edits.

### Phase D1: Model B Derivative Provider

Plan:
- add `make_nonlinear_accumulation_first_derivatives_tf`;
- parameter vector is \((\rho,\sigma,\beta)\);
- keep observation variance, initial law, innovation variance, and \(\alpha\)
  fixed.

Mathematical contract:
\[
  m_t=\rho m_{t-1}+\sigma\varepsilon_t,\qquad
  k_t=\alpha k_{t-1}+\beta\tanh(m_t),\qquad
  y_t=m_t+k_t+u_t .
\]

The provider must return:
\[
  F_x =
  \begin{bmatrix}
  \rho & 0\\
  \beta(1-\tanh^2 m_t)\rho & \alpha
  \end{bmatrix},
  \qquad
  F_\varepsilon =
  \begin{bmatrix}
  \sigma\\
  \beta(1-\tanh^2 m_t)\sigma
  \end{bmatrix},
\]
\[
  F_\rho =
  \begin{bmatrix}
  m_{t-1}\\
  \beta(1-\tanh^2 m_t)m_{t-1}
  \end{bmatrix},\quad
  F_\sigma =
  \begin{bmatrix}
  \varepsilon_t\\
  \beta(1-\tanh^2 m_t)\varepsilon_t
  \end{bmatrix},\quad
  F_\beta =
  \begin{bmatrix}
  0\\
  \tanh(m_t)
  \end{bmatrix},
\]
and \(H_x=[1,1]\), \(H_\theta=0\).

Primary criterion:
- analytic scores match finite differences for all three backends.

Veto diagnostics:
- active floors, weak spectral gaps, nonfinite values, or deterministic
  residuals above tolerance.

### Phase D2: Model C Derivative Provider

Plan:
- add `make_univariate_nonlinear_growth_first_derivatives_tf`;
- add an optional `initial_phase_variance` fixture argument defaulting to zero;
- keep the default value fixture unchanged;
- use a strictly positive `initial_phase_variance` only in score tests to
  isolate nonlinear-map derivative correctness on the current smooth branch.

Mathematical contract:
\[
  x_t =
  \frac12 x_{t-1}
  +\frac{25x_{t-1}}{1+x_{t-1}^2}
  +8\cos(1.2\tau_{t-1})
  +\sigma_u\varepsilon_t,
  \qquad
  \tau_t=\tau_{t-1}+1,
\]
\[
  y_t=\frac{x_t^2}{20}+u_t,\qquad
  u_t\sim N(0,\sigma_y^2).
\]

The provider must return:
\[
  F_x =
  \begin{bmatrix}
  \frac12+25\frac{1-x^2}{(1+x^2)^2} & -9.6\sin(1.2\tau)\\
  0 & 1
  \end{bmatrix},
  \qquad
  F_\varepsilon =
  \begin{bmatrix}
  \sigma_u\\0
  \end{bmatrix},
\]
\[
  F_{\sigma_u}=(\varepsilon_t,0)',\qquad
  \partial_{\sigma_y}R=2\sigma_y,\qquad
  \partial_{P_{0,x}}P_0=\mathrm{diag}(1,0),
\]
and \(H_x=[x/10,0]\).

Primary criterion:
- analytic scores match finite differences on a nondegenerate phase-state
  testing variant for all three backends.

Veto diagnostics:
- if the default zero phase-variance fixture passes without a fixed-null
  derivative policy, stop and audit because the branch label would be
  misleading;
- if the smooth variant has active floors or weak spectral gaps, keep Model C
  score blocked and document why.

### Phase D3: Branch Summaries And Documentation

Plan:
- extend score branch summaries from affine-only to Model B and smooth Model C;
- add an explicit default-Model-C blocked-branch test;
- update Chapter 28 so the reader sees exactly which score claims are now
  certified and which remain blocked.

Primary criterion:
- branch summaries show finite scores for Model B and smooth Model C on small
  parameter boxes;
- default Model C remains explicitly blocked by active floor under the current
  smooth SVD branch.

### Phase D4: Final Audit, Hygiene, And Result

Plan:
- run focused nonlinear tests;
- run the full default CPU suite if focused gates pass;
- run `py_compile`, `git diff --check`, YAML parsing, and production NumPy
  scan;
- write a result artifact and update only the V1 reset memo and source map.

Primary criterion:
- tests and hygiene checks pass.

Continuation rule:
- continue automatically only if the primary criterion and veto diagnostics
  pass;
- otherwise stop with the failing hypothesis and ask for direction.

## Deliberate Non-Goals

- no Hessian implementation in this pass;
- no nonlinear HMC claim in this pass;
- no GPU/XLA benchmark claim in this pass;
- no MacroFinance or DSGE switch-over;
- no change to Chapter 18b or structural SVD/SGU plan files.
