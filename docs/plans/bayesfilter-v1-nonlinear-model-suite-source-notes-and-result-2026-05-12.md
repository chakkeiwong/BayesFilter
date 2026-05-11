# Result: BayesFilter V1 Nonlinear Model Suite Documentation And Testing Tools

## Date

2026-05-12

## Parent Subplan

```text
docs/plans/bayesfilter-v1-nonlinear-model-suite-documentation-and-testing-tools-plan-2026-05-12.md
```

## Phase M0: Lane Recovery And Source Inventory

Status: passed.

Evidence:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  -p no:cacheprovider
```

Result:

```text
20 passed, 2 warnings
```

Interpretation:
- the existing SVD sigma-point, SVD-CUT, CUT rule, and smooth-branch
  derivative-oracle baselines are healthy before the model-suite pass;
- warnings are from TensorFlow Probability's `distutils` version checks and do
  not affect this phase.

Next phase justified?
- Yes.  Baseline health and lane boundaries allow source-note work to proceed.

## Phase M1: Literature Verification Notes

Status: passed with a local-index limitation.

Research-assistant status:
- local paper-summary search returned no records for Kitagawa (1996), Gordon,
  Salmond, and Smith (1993), Arulampalam et al. (2002), or Chopin and
  Papaspiliopoulos (2020);
- DOI/web metadata was therefore used for this first-pass source note.

### Model A: Affine Gaussian Structural Oracle

Source status:
- BayesFilter-local synthetic oracle, not a literature benchmark.

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

Oracle:
- exact linear Gaussian Kalman likelihood after structural-to-LGSSM
  conversion.

Reason included:
- exact recovery, singular structural process covariance, deterministic
  completion, and support diagnostics.

### Model B: BayesFilter Nonlinear Accumulation

Source status:
- BayesFilter-local synthetic nonlinear structural oracle.

Law:
\[
  m_t=\rho m_{t-1}+\sigma\varepsilon_t,\qquad
  k_t=\alpha k_{t-1}+\beta\tanh(m_t),
\]
\[
  y_t=m_t+k_t+\eta_t.
\]

Default parameters:
\[
  \rho=0.70,\quad \sigma=0.25,\quad \alpha=0.55,\quad
  \beta=0.80,\quad \sigma_\eta=0.30.
\]

Oracle:
- dense Gaussian moment-projection quadrature for one-step tests;
- not the exact nonlinear likelihood.

Reason included:
- smooth nonlinear deterministic completion without external timing issues.

### Model C: Univariate Nonlinear Growth Model

Source status:
- standard nonlinear SSM benchmark associated with Kitagawa (1996) and reused
  in later particle-filter and nonlinear-filter examples;
- DOI metadata verified for Kitagawa (1996):
  `10.1080/10618600.1996.10474692`;
- the common benchmark equation was verified from publicly indexed secondary
  examples that reproduce the law.

Canonical law:
\[
  x_t = \frac{x_{t-1}}{2}
      + \frac{25x_{t-1}}{1+x_{t-1}^2}
      + 8\cos(1.2(t-1))+\sigma_x\varepsilon_t,
\]
\[
  y_t=\frac{x_t^2}{20}+\sigma_y\eta_t.
\]

Testing embedding:
- the current structural TF model does not pass an explicit time index into
  `transition_fn`;
- BayesFilter V1 therefore uses an autonomous phase coordinate
  \(\tau_t=\tau_{t-1}+1\) and transition
  \[
    x_t = \frac{x_{t-1}}{2}
      + \frac{25x_{t-1}}{1+x_{t-1}^2}
      + 8\cos(1.2\tau_{t-1})+\sigma_x\varepsilon_t.
  \]

Default parameters:
\[
  x_1\sim N(0,0.2),\qquad \tau_1=1,\qquad
  \sigma_x=1,\qquad \sigma_y=1.
\]

Oracle:
- dense Gaussian moment-projection quadrature for one-step tests;
- not the exact nonlinear likelihood.

Reason included:
- strong nonlinear transition and quadratic observation stress the sigma-point
  moment approximation.

### Deferred Models D-F

Bearings-only tracking:
- Gordon, Salmond, and Smith (1993), DOI `10.1049/ip-f-2.1993.0015`, is a
  bootstrap-filter reference and includes a bearings-only simulation example;
- deferred because angle residual and wrapping policy must be explicit before
  BayesFilter tests should claim correctness.

Radar range-bearing tracking:
- deferred until bearings residual policy and mixed-scale observation
  diagnostics are available.

Stochastic volatility:
- deferred because the current sigma-point value filters assume additive
  observation covariance after `observe(state)`.

Next phase justified?
- Yes.  Models A-C have sufficient first-pass documentation support for
  Chapter 28 and testing fixtures, with Model C clearly labeled as an
  autonomous testing embedding.

## Phase M2: Chapter 28 Documentation

Status: passed.

Files changed:
- `docs/chapters/ch28_nonlinear_ssm_validation.tex`;
- `docs/references.bib`.

Result:
- Chapter 28 now documents Models A-C with equations, default parameters,
  stress purpose, and oracle status;
- Model C is explicitly tied to the Kitagawa nonlinear growth benchmark while
  the BayesFilter V1 fixture is labeled as an autonomous phase-state embedding;
- Models D-F are listed as deferred with concrete reasons;
- the text states that dense quadrature is a one-step Gaussian
  moment-projection oracle, not an exact nonlinear likelihood.

Interpretation:
- the documentation now supports reusable test implementation without drifting
  into gradients, HMC, GPU/XLA, or external-project coupling.

Next phase justified?
- Yes.  Proceed to M3 testing fixtures and reference oracles.

## Phase M3: General Testing Tools

Status: implemented, pending M4 test gate.

Files changed:
- `bayesfilter/testing/nonlinear_models_tf.py`;
- `bayesfilter/testing/__init__.py`;
- `tests/test_nonlinear_benchmark_models_tf.py`;
- `tests/test_nonlinear_reference_oracles.py`;
- `tests/test_nonlinear_sigma_point_values_tf.py`.

Result:
- added reusable TF builders for Models A-C;
- added fixed observation fixtures;
- added a dense one-step Gaussian moment-projection reference under
  `bayesfilter.testing`;
- added constructor/law tests, oracle tests, and value-filter tests.

Interpretation:
- implementation stayed in the testing lane;
- no production nonlinear filter API was changed;
- NumPy appears only in testing/reference code and tests, not in production
  `bayesfilter/nonlinear/*`.

Next phase justified?
- Yes, subject to M4 tests passing.

## Phase M4: Value And Oracle Tests

Status: passed.

Commands:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  -p no:cacheprovider
```

Result:

```text
11 passed, 2 warnings
```

Focused regression command:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  -p no:cacheprovider
```

Result:

```text
31 passed, 2 warnings
```

Additional checks:
- `py_compile` passed for the new fixture and tests;
- `git diff --check` passed for touched files;
- production NumPy scan found no NumPy imports under `bayesfilter/nonlinear`;
- NumPy usage is confined to tests and the testing-only dense quadrature
  oracle.

Interpretation:
- Models A-C are constructible from reusable testing helpers;
- Model A collapses to the exact Kalman reference for SVD cubature, SVD-UKF,
  and SVD-CUT4;
- Models B-C return finite values and zero deterministic residuals for all
  three nonlinear value backends;
- the dense projection oracle is suitable for one-step moment checks in the
  next analytic-score and benchmark phases.

Next phase justified?
- Yes.  Proceed to M5 provenance, reset memo, and lane-only commit.

## Phase M5: Provenance And Reset-Memo Update

Status: complete pending commit.

Result:
- registered plan/result provenance in `docs/source_map.yml`;
- updated only the V1 lane reset memo;
- did not stage the shared monograph reset memo, structural plans, sidecar
  files, local image artifacts, MacroFinance, or DSGE files.

Next phase justified?
- Yes.  The next phase should be the analytic-gradient subplan, beginning with
  Chapter 18 traceability and moving raw SVD-CUT4 `GradientTape` derivatives to
  testing-oracle status.
