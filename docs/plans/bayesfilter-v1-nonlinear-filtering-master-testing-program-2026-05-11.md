# Master Program: BayesFilter v1 Nonlinear Filtering Test And Validation

## Date

2026-05-11

## Purpose

This master program turns the Chapter 18 SVD sigma-point derivative derivation
and the current nonlinear-filtering implementation into a controlled testing
program.  The immediate target is not DSGE or MacroFinance switch-over.  The
target is a BayesFilter-local nonlinear model suite that can tell us, with
audit-grade evidence, whether SVD cubature, SVD-UKF, and SVD-CUT4 are correct,
effective, differentiable, and eventually usable inside HMC.

## Lane

Allowed write lane for this program:

```text
bayesfilter/nonlinear/*
bayesfilter/testing/*
tests/test_*nonlinear*
tests/test_*sigma*
tests/test_svd_cut*
docs/benchmarks/*
docs/chapters/ch16_sigma_point_filters.tex
docs/chapters/ch17_square_root_sigma_point.tex
docs/chapters/ch18_svd_sigma_point.tex
docs/chapters/ch28_nonlinear_ssm_validation.tex
docs/plans/bayesfilter-v1-*
docs/source_map.yml
pytest.ini
```

Protected out-of-lane files unless explicitly requested:

```text
docs/chapters/ch18b_structural_deterministic_dynamics.tex
docs/plans/bayesfilter-structural-*
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
/home/chakwong/MacroFinance/*
/home/chakwong/python/*
```

## Current Status

Chapter 18 already contains the analytic SVD sigma-point derivative scaffold:

- generic SVD sigma-point cloud derivatives:
  `eq:bf-svd-sp-point-first`, `eq:bf-svd-sp-point-second`;
- map and moment derivatives:
  `eq:bf-svd-sp-map-first` through `eq:bf-svd-sp-cov-second`;
- likelihood score:
  `eq:bf-svd-sp-score`;
- SVD factor reconstruction identities:
  `eq:bf-svd-factor-reconstruction-first`,
  `eq:bf-svd-factor-reconstruction-second`;
- SVD-CUT point, moment, score, and Hessian derivatives:
  `eq:bf-svd-cut-point-first` through `eq:bf-svd-cut-hessian`.

MathDevMCP lookup and audit found these labels and provenance in
`docs/chapters/ch18_svd_sigma_point.tex`.  The automated audit status is not a
complete theorem certificate: many obligations are marked unverified or require
manual formalization, mainly because the matrix dimensions, positive-definite
solve assumptions, and split proof obligations must be stated explicitly.  The
program therefore treats Chapter 18 as the source derivation and requires
implementation-level verification before public derivative promotion.

Implementation status:

- SVD cubature and SVD-UKF are value-only TF filters.
- SVD-CUT4 is value-only as a filter result.
- The current `tf_svd_cut4_score_hessian` is a TensorFlow `GradientTape`
  smooth-branch diagnostic of the implemented regularized law, not the manual
  analytic recursion from Chapter 18.
- HMC requires a stable score path, not an analytic Hessian.  Hessian remains
  a diagnostic and optimization/Laplace extension until a concrete need appears.

## Program Goals

1. Verify the Chapter 18 analytic derivative equations at the level needed for
   implementation.
2. Remove `GradientTape` from the production SVD-CUT4 derivative path; keep any
   autodiff reference under `bayesfilter.testing` or benchmark diagnostics.
3. Implement score-first nonlinear derivative paths for SVD cubature, SVD-UKF,
   and SVD-CUT4 only after the derivative contract is explicitly mapped to code.
4. Keep Hessian implementation optional and diagnostic unless a downstream
   target needs Newton, Laplace, Riemannian HMC, or curvature certification.
5. Build a literature-grounded nonlinear SSM benchmark suite and document the
   chosen models in LaTeX.
6. Test correctness, approximation quality, branch stability, compiled parity,
   and runtime behavior under controlled CPU and optional GPU/XLA tiers.

## Literature Search Program

The literature task is to choose a small set of benchmark nonlinear SSMs with
known stress properties, not to maximize model count.  Each candidate must have:

- a primary or standard reference;
- an explicit state transition, observation equation, and noise law;
- a reference likelihood/filtering method suitable for tests;
- a reason it stresses a BayesFilter feature;
- a LaTeX model block before implementation is promoted.

Seed sources to verify and cite:

| Source | Why It Matters | Candidate Models |
| --- | --- | --- |
| Gordon, Salmond, and Smith (1993), "Novel approach to nonlinear/non-Gaussian Bayesian state estimation", DOI `10.1049/ip-f-2.1993.0015` | Bootstrap filter benchmark source; includes bearings-only tracking. | Bearings-only/radar tracking; nonlinear observation with additive angular noise. |
| Kitagawa (1996), "Monte Carlo Filter and Smoother for Non-Gaussian Nonlinear State Space Models", DOI `10.1080/10618600.1996.10474692` | Standard Monte Carlo filtering/smoothing nonlinear SSM source. | Univariate nonlinear growth model; stochastic volatility candidates. |
| Arulampalam, Maskell, Gordon, and Clapp (2002), DOI `10.1109/78.978374` | Tutorial benchmark source for online nonlinear/non-Gaussian tracking. | Nonlinear tracking examples and particle-filter reference policies. |
| Kantas, Doucet, Singh, Maciejowski, and Chopin (2015), DOI `10.1214/14-STS511` | Parameter-estimation and SSM benchmark survey. | Stochastic-volatility and parameter-learning benchmarks; score/HMC relevance. |
| Nemeth, Fearnhead, and Mihaylova (2016), DOI `10.1080/10618600.2015.1093492` | Score and observed-information particle approximations. | External diagnostic reference for score/Hessian claims, not a v1 implementation dependency. |

Search steps:

1. Use local research-assistant first.  If the local index has no matching
   paper, use web search and record URLs, DOI, and whether the full equations
   were verified from a primary source.
2. Extract the exact benchmark equations into a short source note under
   `docs/plans/bayesfilter-v1-*`.
3. Add accepted models to `docs/chapters/ch28_nonlinear_ssm_validation.tex`.
4. Add cross-references from Chapter 18 derivative-validation targets to the
   accepted model suite.
5. Do not add a benchmark to tests until the LaTeX block states the reference
   law and the test oracle.

## Candidate Nonlinear Model Suite

### Model A: Affine Gaussian Structural Oracle

Purpose:
- exact collapse test for SVD cubature, SVD-UKF, and SVD-CUT4.

Law:
\[
  x_t=(m_t,\ell_t)^\top,\qquad
  m_t=0.35m_{t-1}-0.10\ell_{t-1}+0.25\varepsilon_t,\qquad
  \ell_t=m_{t-1},
\]
\[
  y_t=m_t+\eta_t,\qquad
  \varepsilon_t\sim N(0,1),\qquad
  \eta_t\sim N(0,0.15^2).
\]

Reference:
- exact linear Gaussian Kalman likelihood, score, and Hessian after structural
  conversion.

Tests:
- value parity for SVD cubature, SVD-UKF, and SVD-CUT4;
- score parity after analytic score implementation;
- deterministic-residual and rank-deficient support diagnostics.

### Model B: BayesFilter Nonlinear Accumulation

Purpose:
- smooth structural nonlinear transition with deterministic completion.

Law:
\[
  m_t=\rho m_{t-1}+\sigma\varepsilon_t,\qquad
  k_t=\alpha k_{t-1}+\beta\tanh(m_t),
\]
\[
  y_t=m_t+k_t+\eta_t.
\]

Reference:
- one-step dense Gauss-Hermite quadrature;
- multi-step high-accuracy SMC or dense quadrature for small horizons.

Tests:
- finite value and score;
- structural deterministic identity;
- sigma-point approximation error versus dense reference;
- UKF versus cubature versus CUT4 comparison.

### Model C: Univariate Nonlinear Growth Model

Purpose:
- classic nonlinear transition and quadratic observation benchmark.

Candidate law to verify from the primary source before implementation:
\[
  x_t=\frac{x_{t-1}}{2}
      +\frac{25x_{t-1}}{1+x_{t-1}^2}
      +8\cos(1.2t)
      +\sigma_x\varepsilon_t,
\]
\[
  y_t=\frac{x_t^2}{20}+\sigma_y\eta_t.
\]

Reference:
- dense one-dimensional quadrature for short horizons;
- bootstrap particle filter with fixed seed and large particle count for
  longer horizons.

Tests:
- nonlinear approximation accuracy;
- multimodality stress;
- finite score away from spectral branch changes.

### Model D: Bearings-only Tracking

Purpose:
- nonlinear observation geometry and angle wrapping.

Candidate law:
- nearly constant velocity planar target with Gaussian process noise;
- observation is bearing angle
  \[
    y_t=\operatorname{atan2}(p_{y,t}-s_y,p_{x,t}-s_x)+\eta_t.
  \]

Reference:
- high-particle SMC;
- published benchmark settings from the tracking literature.

Tests:
- nonlinear observation moment approximation;
- angle residual policy;
- support diagnostics under weak observability.

### Model E: Radar Range-bearing Tracking

Purpose:
- multidimensional nonlinear observation with different scales.

Law:
\[
  y_t =
  \begin{bmatrix}
    \sqrt{p_{x,t}^2+p_{y,t}^2}\\
    \operatorname{atan2}(p_{y,t},p_{x,t})
  \end{bmatrix}
  +\eta_t.
\]

Reference:
- EKF/UKF/Cubature comparison and high-particle SMC.

Tests:
- cross-covariance and Kalman-gain correctness;
- scaling and conditioning of observation covariance;
- XLA/GPU point-axis behavior for moderate state dimension.

### Model F: Stochastic Volatility

Purpose:
- economically relevant nonlinear/non-Gaussian observation density.

Candidate law:
\[
  x_t=\mu+\phi(x_{t-1}-\mu)+\sigma_x\varepsilon_t,\qquad
  y_t=\beta\exp(x_t/2)\eta_t.
\]

Reference:
- particle filter or specialized likelihood approximation.

Status:
- not a first v1 sigma-point fixture unless the nonlinear observation-noise
  interface is extended.  Current TF sigma-point filters assume additive
  observation covariance after `observe(state)`.

## Derivative Verification And Implementation Program

### Phase 0: Baseline And Lane Audit

Run:

```bash
git status --short --branch
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  -p no:cacheprovider
```

Stop if out-of-lane files are required.

### Phase 1: Derivation-to-code Traceability

Actions:
- create a traceability table mapping each Chapter 18 equation to a planned
  tensor operation;
- split score-only requirements from Hessian requirements;
- state positive-definite solve assumptions for each likelihood contribution;
- record the spectral branch assumptions: no active hard floors, separated
  eigenvalues, fixed ordering/sign policy, and implemented covariance target.

Closure:
- every analytic score tensor has a source equation, shape, and test oracle.

### Phase 2: Move Autodiff Derivatives Out Of Production

Actions:
- move or duplicate current `GradientTape` SVD-CUT4 score/Hessian into
  `bayesfilter.testing` as an oracle;
- remove the public production export if no analytic replacement exists yet;
- keep branch-gated autodiff tests as diagnostic reference tests.

Closure:
- production nonlinear exports no longer advertise raw-tape SVD-CUT4
  derivatives as a production backend.

### Phase 3: Analytic Score-first Implementation

Actions:
- implement first derivatives of sigma-point placement, propagation, moments,
  innovations, and Gaussian contribution;
- support SVD cubature, SVD-UKF, and SVD-CUT4 through the same fixed-rule
  derivative core;
- do not implement Hessian in this phase unless score tests pass and a concrete
  Hessian consumer is named.

Closure:
- score matches finite differences and the testing autodiff oracle on smooth
  separated-spectrum branches.

### Phase 4: Hessian Decision Gate

Decision:
- implement Hessian only for diagnostics, Newton/Laplace workflows, or a
  curvature-validation release target.

Required evidence before Hessian implementation:
- score residuals stable across models A-C;
- factor first-derivative reconstruction residuals are small;
- branch diagnostics do not show frequent weak gaps in target regions;
- Hessian memory and compile cost are acceptable on a small shape.

Closure:
- either an explicit Hessian implementation plan exists, or Hessian remains
  deferred with rationale.

### Phase 5: Literature Benchmark Documentation

Actions:
- verify primary equations and parameter settings;
- write accepted models into
  `docs/chapters/ch28_nonlinear_ssm_validation.tex`;
- add a short table in Chapter 18 linking derivative-validation tests to model
  suite rows.

Closure:
- each implemented nonlinear benchmark has a LaTeX model block, reference, and
  oracle.

### Phase 6: Test Suite Build-out

Core tests:
- `tests/test_nonlinear_benchmark_models_tf.py`;
- `tests/test_nonlinear_sigma_point_values_tf.py`;
- `tests/test_nonlinear_sigma_point_scores_tf.py`;
- `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py`;
- `tests/test_nonlinear_reference_oracles.py`.

Minimum checks:
- affine Gaussian recovery;
- nonlinear one-step dense quadrature recovery;
- finite score and finite branch diagnostics;
- no production NumPy in TF implementation modules;
- eager/graph parity;
- opt-in extended CPU sweeps for branch frequency and approximation error.

### Phase 7: Effectiveness Benchmarks

Artifacts:
- JSON/Markdown benchmark files under `docs/benchmarks`;
- fixed seeds and model parameter records;
- approximation error tables versus dense quadrature or SMC;
- runtime and point-count tables for cubature, UKF, and CUT4.

Metrics:
- log-likelihood error;
- filtered mean RMSE versus reference;
- covariance calibration;
- branch blocker rates;
- runtime and memory;
- compiled/eager parity.

### Phase 8: HMC Readiness Gate

HMC requires:
- analytic score or approved custom-gradient score;
- finite value and score on parameter boxes;
- branch diagnostics recorded at every proposal region;
- target-specific multi-chain diagnostics;
- no convergence claim until R-hat, ESS, divergence, and posterior recovery
  thresholds are explicitly passed.

SVD-CUT HMC remains blocked until this phase passes.

## CI And Runtime Tiers

Tier 1 fast local:
- model constructors, rule moments, import surface.

Tier 2 focused nonlinear regression:
- affine value recovery;
- one-step nonlinear dense-reference tests;
- score finite-difference checks on tiny smooth branches.

Tier 3 extended CPU:
- branch sweeps, approximation-error grids, longer horizons.

Tier 4 optional GPU/XLA:
- escalated only;
- fixed-shape point-axis benchmarks;
- CPU/GPU matching-shape comparison.

Tier 5 HMC:
- opt-in only;
- target-specific claims only.

## Done Definition

The nonlinear filtering implementation is v1-testable when:

- the Chapter 18 derivative equations have a code traceability table;
- raw `GradientTape` SVD-CUT derivatives are no longer production-promoted;
- SVD cubature, SVD-UKF, and SVD-CUT4 value tests pass on affine and nonlinear
  fixtures;
- score-only analytic derivatives pass finite-difference and oracle checks on
  smooth branches;
- Hessian is either implemented with evidence or explicitly deferred;
- accepted literature benchmarks are written in LaTeX;
- the nonlinear test suite records correctness, approximation quality, branch
  stability, and runtime evidence without requiring external projects.

## Independent Audit Notes

The plan is intentionally score-first.  Requiring Hessian before HMC would
block useful nonlinear filtering work on the most fragile and expensive part of
the derivative stack.  The Hessian remains mathematically derived in Chapter
18, but implementation should wait until score correctness, spectral-factor
reconstruction, and model-suite coverage are stable.

The literature suite should stay small.  A three-rung suite is enough for the
first execution pass: affine Gaussian oracle, BayesFilter nonlinear
accumulation, and the verified univariate nonlinear growth model.  Bearings
and radar tracking can follow once angle residual and reference-particle
policies are documented.
