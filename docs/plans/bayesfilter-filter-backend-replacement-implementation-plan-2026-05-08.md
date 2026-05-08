# BayesFilter TF/TFP Filter Backend Replacement Implementation Plan

Date: 2026-05-08

## Purpose

Implement the linear filter, nonlinear structural filter, and SVD-CUT filter
properly inside BayesFilter so that `/home/chakwong/python` and
`/home/chakwong/MacroFinance` can replace their local filtering algorithms with
BayesFilter backends.

The implementation target is now explicit: production BayesFilter filtering
code must be TensorFlow and TensorFlow Probability only.  NumPy may appear in
tests, historical comparison notes, or client-side legacy fixtures, but it
should not be a BayesFilter implementation backend or a required oracle for
production correctness.

## Current Starting Point

BayesFilter currently contains a committed NumPy-heavy foundation:
- `bayesfilter/filters/kalman.py`: NumPy covariance, solve, and SVD/eigen value
  backends.
- `bayesfilter/linear/types.py`: NumPy-backed LGSSM containers.
- `bayesfilter/linear/kalman_derivatives_numpy.py`: NumPy solve-form analytic
  Kalman score and Hessian.
- `bayesfilter/filters/sigma_points.py`: NumPy structural sigma-point backend.
- `bayesfilter/filters/particles.py`: NumPy particle filter.
- `bayesfilter/results.py` and `bayesfilter/backends.py`: NumPy-backed result
  freezing and spectral diagnostics.

That code is useful as historical context, but it is not the desired
implementation direction.  The next implementation work should pivot away from
it rather than extending it.  The intended end state is:
- TF tensor contracts for linear, nonlinear, and structural models;
- TFP-based Gaussian log probability or equivalent TF linear algebra with the
  same semantics;
- `tf.linalg` factorization, solves, eig/SVD, and masking conventions;
- `tf.GradientTape` and/or explicit TF analytic derivative recursions for score
  and Hessian;
- XLA-compatible static-shape paths where HMC or batched likelihood evaluation
  needs compilation;
- diagnostics that report regularization, implemented covariance, branch
  decisions, and derivative smoothness without converting tensors through
  `.numpy()`.

## Source Material To Reuse

### From `/home/chakwong/MacroFinance`

Use the TensorFlow/TFP files as the primary implementation sources.  The
implementation strategy is not to rederive these from scratch; it is to port
MacroFinance's already-tested TF algorithms into BayesFilter contracts, then
scrub MacroFinance-specific economics, debug-only NumPy conversions, and
application-specific HMC wrappers.

Highest-priority donor functions:
- `inference/hmc.py::tf_kalman_log_likelihood`
  - dense TF Cholesky Kalman value recursion.
- `inference/hmc.py::tf_lgssm_log_likelihood_backend`
  - backend dispatch convention for `tf_cholesky`, `tf_direct_qr`, and
    `tf_svd`.
- `filters/tf_masked_kalman.py::tf_masked_kalman_log_likelihood`
  - static-shape dummy-row masking convention.
- `filters/tf_svd_kalman.py::tf_svd_kalman_loglik`
  - TF SVD dense value backend.
- `filters/tf_svd_kalman.py::tf_svd_masked_kalman_loglik`
  - TF SVD masked value backend.
- `filters/tf_differentiated_kalman.py::tf_differentiated_kalman_loglik_grad`
  and `tf_differentiated_kalman_loglik_grad_graph`
  - first-order analytic TF recursion for HMC leapfrog and MAP gradients.
- `filters/tf_differentiated_kalman.py::tf_differentiated_kalman_loglik`
  and `tf_differentiated_kalman_loglik_grad_hessian_graph`
  - second-order analytic TF recursion.
- `filters/tf_solve_differentiated_kalman.py::tf_solve_differentiated_kalman_loglik`
  - solve-form TF analytic derivative recursion.
- `filters/tf_qr_sqrt_differentiated_kalman.py::tf_qr_sqrt_kalman_loglik`
  and `tf_qr_sqrt_masked_kalman_loglik`
  - TF QR/square-root value and masked robustness pattern.
- `filters/tf_qr_sqrt_differentiated_kalman.py::tf_qr_sqrt_differentiated_kalman_loglik`
  - TF QR/square-root analytic score/Hessian pattern.
- `filters/tf_sqrt_differentiated_kalman.py::tf_sqrt_differentiated_kalman_loglik`
  - simpler TF square-root trace/debug pattern.

Current donor caveat:
- several MacroFinance TF files still import NumPy for constants or eager trace
  conversion.  BayesFilter ports should replace those with `math`/TF constants
  and keep any `.numpy()` conversion in explicit debug extraction helpers, not
  in production filtering paths.

Other TensorFlow files:
- `filters/tf_differentiated_kalman.py`
  - TensorFlow first/second-order analytic derivative backends and graph
    variants.
- `filters/tf_masked_kalman.py`
  - TensorFlow masked likelihood convention.
- `filters/tf_svd_kalman.py`
  - TensorFlow SVD linear value backend, singular-value floor diagnostics, and
    masked SVD convention.
- `filters/tf_qr_sqrt_differentiated_kalman.py`
  - TF square-root derivative and masked robustness patterns where useful.

Use these only as mathematical and API reference, not as NumPy ports:
- `domain/types.py`
- `filters/solve_differentiated_kalman.py`
- `filters/differentiated_kalman.py`
- QR/square-root NumPy files.

Primary reusable tests:
- `tests/test_tf_kalman.py`
- `tests/test_filter_backend_parity.py`
- `tests/test_tf_masked_kalman.py`
- `tests/test_tf_svd_kalman.py`
- `tests/test_tf_qr_sqrt_differentiated_kalman.py`
- `tests/test_tf_qr_derivative_identities.py`
- `tests/test_one_country_analytic_backend_parity.py`
- `tests/test_one_country_hmc_analytic_gradient_hessian.py`
- `tests/test_perf_tfp_analytic_filter_speed.py`

Test reuse policy:
- transplant tests by first replacing NumPy-reference assertions with TF-only
  closed-form, TFP distribution, or MacroFinance TF-provider assertions;
- keep test-side NumPy only for array construction and `np.testing` during the
  migration window;
- do not let a BayesFilter production module import a MacroFinance test helper
  or a NumPy oracle.

### From `/home/chakwong/python`

Use the DSGE codebase as the primary donor for nonlinear UKF, SVD sigma-point,
structural DSGE metadata, XLA gates, and CUT experiments.  This code should be
consolidated with the MacroFinance linear TF/TFP code before new BayesFilter
implementation work begins.

Highest-priority donor functions and classes:
- `src/dsge_hmc/filters/_quadrature.py`
  - `QuadratureRule`, `CubatureRule`, and `UnscentedRule`.
- `src/dsge_hmc/filters/_svd_core.py`
  - `svd_clamp`, `svd_factorize`, `svd_predict_linear`,
    `svd_predict_sigma`, `svd_update`, and `svd_update_sigma`.
- `src/dsge_hmc/filters/_svd_filters.py`
  - `SVDKalmanFilter`, `SVDAugmentedKF`, and `SVDSigmaPointFilter`.
  - Generic SSM path via the `ssm=` constructor.
  - DSGE legacy path and mixed full-state approximation blocker.
  - Diagnostics modes: first batch, stats, eigbatch summary, and failure code.
- `src/dsge_hmc/filters/CUTSRUKF.py`
  - `ut_sigma_points`, `cut4g_sigma_points`, and
    `SquareRootSigmaPointFilter` as experimental CUT/UT donor material.
- `src/dsge_hmc/models/structural_metadata.py`
  - structural metadata helpers for stochastic/deterministic state
    partitions and eta-support validation.

Primary reusable tests:
- `tests/contracts/test_filter_contracts.py`;
- `tests/contracts/test_svd_lgssm_reference.py`;
- `tests/contracts/test_svd_nonlinear_ssm_reference.py`;
- `tests/contracts/test_svd_generic_nonlinear_ssm.py`;
- `tests/contracts/test_svd_generic_ssm_xla.py`;
- `tests/contracts/test_nk_svd_gradient_finiteness_gate.py`;
- `tests/contracts/test_nk_svd_xla_gates.py`;
- `tests/contracts/test_structural_dsge_partition.py`;
- `tests/contracts/test_dsge_structural_completion_residuals.py`;
- `tests/contracts/test_dsge_strong_structural_residual_gates.py`;
- `tests/numerics/test_svd_filters.py`;
- `tests/CUTSRUKF_test.py`;
- extended HMC/XLA tests under `tests/extended/test_svd_*` after local
  BayesFilter unit gates pass.

Current donor caveat:
- `_svd_core.py` currently imports the DSGE custom `robust_eigh` op
  unconditionally.  BayesFilter should define an eigensolver interface with a
  default `tf.linalg.eigh` implementation and an optional robust-eigh plugin.
- `_svd_filters.py` carries DSGE-specific debug capture hooks and environment
  variables.  BayesFilter should retain the diagnostic ideas, but not the
  client-specific pyfunc capture machinery in production paths.
- `CUTSRUKF.py` is explicitly marked experimental/deprecated in the DSGE tree.
  Use it as CUT4-G and square-root pattern donor material, not as a direct
  production drop-in.

Reuse policy:
- Port generic TF algorithms and contracts into BayesFilter notation.
- Do not port DSGE economic solution logic into BayesFilter.
- Keep custom MKL or robust-eigh ops optional.  The required default is
  TensorFlow/TFP, not NumPy.
- Keep legacy full-state mixed-DSGE paths blocked unless explicitly labeled as
  approximation-only.

## Target Package Shape

Add or expand these modules:

```text
bayesfilter/types_tf.py
bayesfilter/results_tf.py
bayesfilter/diagnostics.py
bayesfilter/linear/types_tf.py
bayesfilter/linear/kalman_tf.py
bayesfilter/linear/kalman_svd_tf.py
bayesfilter/linear/kalman_derivatives_tf.py
bayesfilter/nonlinear/protocols_tf.py
bayesfilter/nonlinear/quadrature_tf.py
bayesfilter/nonlinear/svd_factor_tf.py
bayesfilter/nonlinear/sigma_point_tf.py
bayesfilter/nonlinear/svd_cut_tf.py
bayesfilter/nonlinear/svd_cut_derivatives_tf.py
bayesfilter/adapters/macrofinance.py
bayesfilter/adapters/dsge.py
```

Existing NumPy modules should be isolated as legacy/reference modules until
removed or replaced:

```text
bayesfilter/filters/kalman.py
bayesfilter/filters/sigma_points.py
bayesfilter/filters/particles.py
bayesfilter/linear/kalman_derivatives_numpy.py
```

Compatibility re-exports may remain temporarily, but new production-facing
imports should point to TF/TFP modules.

## Unified Public Contracts

### Linear Gaussian State Space

BayesFilter should own a TF-first LGSSM contract:

```python
TFLinearGaussianStateSpace(
    initial_mean,              # tf.Tensor [n]
    initial_covariance,        # tf.Tensor [n, n]
    transition_offset,         # tf.Tensor [n] or [T, n]
    transition_matrix,         # tf.Tensor [n, n] or [T, n, n]
    transition_covariance,     # tf.Tensor [n, n] or [T, n, n]
    observation_offset,        # tf.Tensor [m] or [T, m]
    observation_matrix,        # tf.Tensor [m, n] or [T, m, n]
    observation_covariance,    # tf.Tensor [m, m] or [T, m, m]
    mask=None,
    partition=None,
)
```

The derivative contract should use TF tensors:

```python
TFLinearGaussianStateSpaceDerivatives(
    d_initial_mean,
    d_initial_covariance,
    d_transition_offset,
    d_transition_matrix,
    d_transition_covariance,
    d_observation_offset,
    d_observation_matrix,
    d_observation_covariance,
    d2_initial_mean,
    d2_initial_covariance,
    d2_transition_offset,
    d2_transition_matrix,
    d2_transition_covariance,
    d2_observation_offset,
    d2_observation_matrix,
    d2_observation_covariance,
)
```

Return objects should preserve tensors:

```python
TFFilterValueResult(log_likelihood, filtered_means, filtered_covariances, metadata, diagnostics)
TFFilterDerivativeResult(log_likelihood, score, hessian, metadata, diagnostics, trace)
```

No result container should call `.numpy()` in the implementation path.

### Nonlinear Structural State Space

BayesFilter should define a TF structural nonlinear protocol:

```python
partition
initial_mean(theta)
initial_cov(theta)
innovation_mean(theta)        # default tf.zeros
innovation_cov(theta)
observation_cov(theta)
transition(previous_state, innovation, theta, *, time_index=None)
transition_points(previous_points, innovation_points, theta, *, time_index=None)
observe(state_points, theta, *, time_index=None)
```

Optional derivative provider:

```python
transition_derivatives(...)
observe_derivatives(...)
factor_derivatives(...)
```

The filter must treat deterministic coordinates as completed by the structural
transition.  The default integration space for mixed structural models is the
declared innovation space, not the full state.

### Quadrature Rules

Implement TF-native rules:
- `TFCubatureRule(dim)`
- `TFUnscentedRule(dim, alpha=1.0, beta=2.0, kappa=None)`
- `TFCUT4GRule(dim)` with `dim >= 3`, `2 dim + 2**dim` points, positive
  weights, and TF moment-test helpers.

Expose standardized offsets and placed points:

```python
rule.standard_offsets_and_weights(dtype)
rule.place(mean, factor)
```

This makes SVD-CUT explicit: `factor` may be Cholesky, QR, SVD/eigen, or a
rectangular structural factor, but all implementations must use TF tensors.

## Revised Implementation Phases

### Phase 0: Cross-Repo Consolidation and TF/TFP Baseline Gate

Plan:
- preserve unrelated dirty files and local PDFs;
- inventory MacroFinance TF/TFP donor modules for linear value, masking,
  SVD, QR/square-root, analytic gradient/Hessian, HMC, and performance tests;
- inventory DSGE donor modules for UKF, SVD sigma-point filtering, structural
  metadata, generic nonlinear SSMs, XLA gates, and CUT4-G experiments;
- produce a consolidation map that assigns one BayesFilter owner module for
  each family:
  - dense linear Kalman;
  - masked linear Kalman;
  - SVD linear Kalman;
  - analytic linear score/Hessian;
  - QR/square-root robustness;
  - cubature/UKF quadrature;
  - generic SVD sigma-point filter;
  - structural DSGE partition and deterministic completion;
  - CUT4-G/SVD-CUT value and derivative work;
- run existing BayesFilter tests as a historical baseline;
- run a CPU-only TF/TFP import and version probe with `CUDA_VISIBLE_DEVICES=-1`;
- run GPU probes only with escalated permissions per `AGENTS.md`;
- record that the committed NumPy foundation is legacy and not the production
  target.

Exit criteria:
- current baseline is known;
- TensorFlow and TensorFlow Probability availability is recorded;
- MacroFinance and DSGE donor maps are recorded with tests to transplant;
- conflicts between donor implementations are resolved before porting;
- no new implementation work depends on NumPy.

### Phase 1: TF Result, Diagnostics, and Tensor Contracts

Plan:
- add TF result containers that preserve tensors;
- add a shared regularization diagnostics schema:
  - jitter;
  - eigenvalue/singular-value floors;
  - PSD projection residual;
  - implemented covariance `P_star`;
  - branch/fallback label;
  - derivative target: pre-regularized law vs implemented/floored law;
- add source checks that production modules do not import NumPy.

Tests:
- tensor shape validation;
- no `.numpy()` calls in production TF modules;
- diagnostics schema consistency;
- legacy NumPy modules remain isolated and are not imported by TF public
  front doors.

### Phase 2: TF Dense Linear Kalman Value Backend

Plan:
- port MacroFinance `inference/hmc.py::tf_kalman_log_likelihood` into
  `bayesfilter/linear/kalman_tf.py`;
- port MacroFinance `inference/hmc.py::tf_lgssm_log_likelihood_backend` into a
  BayesFilter backend selector after removing MacroFinance-specific wrappers;
- use TF linear algebra and TFP Gaussian log probability where appropriate;
- port `filters/tf_masked_kalman.py::tf_masked_kalman_log_likelihood` for
  static-shape missing-data masks;
- keep singular process covariance accepted when the predictive observation
  covariance is well-conditioned or explicitly regularized.

Tests:
- adapt MacroFinance `tests/test_tf_kalman.py`;
- adapt MacroFinance `tests/test_tf_masked_kalman.py` without requiring a
  NumPy masked oracle;
- scalar and multivariate LGSSM likelihood identities;
- TFP `MultivariateNormalTriL` or equivalent log-probability checks for one-step
  prediction errors;
- singular process covariance;
- masked observation rows;
- no NumPy import in implementation module;
- optional XLA compile on small static shapes.

### Phase 3: TF Analytic Score and Hessian for Linear Filters

Plan:
- port MacroFinance `filters/tf_qr_sqrt_differentiated_kalman.py` as the
  production derivative route, because MacroFinance already contains the clean
  QR/square-root first- and second-order factor derivative implementation and
  its identity tests;
- keep MacroFinance `filters/tf_differentiated_kalman.py` covariance-form and
  `filters/tf_solve_differentiated_kalman.py` solve-form derivatives under
  `bayesfilter.testing` as reference/debug backends, not production front
  doors;
- use those testing references to localize failures in QR derivative work:
  covariance-form for continuity with MacroFinance HMC conventions, solve-form
  for transparent Cholesky-solve algebra and trace diagnostics;
- support first and second derivatives of initial moments, transition matrices,
  transition covariances, observation matrices, and observation covariances;
- expose both eager TF and `tf.function` graph paths;
- keep the derivative target explicit when regularization is active.

Tests:
- adapt MacroFinance `tests/test_tf_qr_sqrt_differentiated_kalman.py`;
- adapt MacroFinance `tests/test_tf_qr_derivative_identities.py`;
- adapt MacroFinance `tests/test_filter_backend_parity.py`;
- adapt MacroFinance `tests/test_one_country_analytic_backend_parity.py`;
- keep BayesFilter-local solve/covariance derivative references in
  `bayesfilter.testing` and compare them against autodiff on controlled small
  LGSSMs;
- finite-difference validation using TF-only perturbations;
- Hessian symmetry;
- parameter-dependent initial condition policies;
- MacroFinance one-country parity through TF provider wrappers;
- static-shape graph reuse.

### Phase 4: TF SVD Linear Value and Derivative Gates

Plan:
- port MacroFinance `filters/tf_svd_kalman.py` value and masked value backends;
- use `tf.linalg.eigh` or `tf.linalg.svd` after choosing the more stable
  TensorFlow primitive for symmetric PSD covariance objects;
- report raw spectrum, floored spectrum, floor counts, and implemented
  covariance;
- add derivative status labels:
  - smooth spectrum;
  - repeated/near-repeated spectrum blocked;
  - floor branch derivative;
  - regularized value only.

Tests:
- adapt MacroFinance `tests/test_tf_svd_kalman.py`;
- equality with dense TF backend on well-conditioned cases;
- rank-deficient process covariance;
- near-singular innovation covariance;
- gradient and Hessian blocked near spectral branch events unless explicitly
  certified.

### Phase 5: TF Structural Nonlinear Protocol

Plan:
- replace the NumPy structural reference path with TF structural protocols;
- validate state partitions, innovation dimensions, deterministic completion,
  and observation dimensions;
- make structural UKF/Cubature rules injectable as TF quadrature rules.

Tests:
- deterministic identities hold pointwise;
- low-dimensional nonlinear examples match dense analytic or high-order
  quadrature references coded in TF;
- full-state mixed structural filtering remains blocked unless labeled.

### Phase 6: Generic TF SVD Sigma-Point Filter

Plan:
- port generic SSM SVD sigma-point infrastructure from `/home/chakwong/python`;
- separate structural integration space from legacy full-state paths;
- implement default TF eigensolver and optional robust-eigh plugin boundary;
- carry failure codes and first-failing-batch diagnostics as tensors or
  serializable metadata.

Tests:
- scalar LGSSM exact-Kalman parity;
- finite gradients on smooth fixtures;
- nonlinear observation comparison against TF reference quadrature;
- XLA static-shape gates.

### Phase 7: TF CUT4-G and SVD-CUT Value Backend

Plan:
- implement `TFCUT4GRule`;
- add TF moment checks through degree five;
- combine CUT4-G with SVD placement and the generic TF sigma-point filter;
- report point count and memory/shape diagnostics.

Tests:
- positive weights and weight sum;
- covariance and fourth-moment identities;
- affine exactness under TF SVD/eigen placement;
- rank-deficient covariance placement diagnostics.

### Phase 8: TF Analytic Gradient and Hessian of SVD-CUT

Split this phase:
- 8A: fixed-rule moment derivatives with supplied factor derivatives;
- 8B: SVD/eigen factor first derivative away from branch events;
- 8C: SVD/eigen factor second derivative away from branch events;
- 8D: full SVD-CUT score/Hessian integration with branch diagnostics.

Tests:
- TF finite-difference score/Hessian on small fixtures;
- `tf.GradientTape` parity where the spectral branch is smooth;
- explicit blocked labels near repeated singular/eigen values or active floors;
- Hessian symmetry and implemented-law metadata.

### Phase 9: GPU/XLA Benchmark and Compilation Gates

Plan:
- add JSON benchmark schema with compile time, first-call time, steady-state
  time, device, dtype, `q`, point count, state dimension, observation dimension,
  parameter count, time length, and batch/chain count;
- run CPU-only by default;
- run GPU/XLA only with escalated sandbox permissions.

Exit criteria:
- CUT point-count claims are dimension-bounded;
- no blanket claim that `2q + 2**q` is harmless on every GPU workload.

### Phase 10: MacroFinance Switch-Over

Plan:
- after BayesFilter ports pass local tests, replace MacroFinance filter calls
  one TF backend at a time:
  - dense linear value;
  - analytic score/Hessian;
  - masked backend;
  - SVD value/diagnostic backend;
  - optional SVD derivative backend only after branch gates pass.

Exit criteria:
- MacroFinance model construction remains in MacroFinance;
- filtering comes from BayesFilter TF/TFP modules;
- HMC targets use BayesFilter TF tensors without NumPy conversion.

### Phase 11: `/python` DSGE Switch-Over

Plan:
- replace local generic SVD SSM filter with BayesFilter TF backend;
- replace DSGE SVD sigma-point path only after structural metadata and
  deterministic completion are explicit;
- keep mixed full-state approximation blocked unless labeled.

Exit criteria:
- `/home/chakwong/python` filtering imports point to BayesFilter TF modules;
- local SVD/CUT files become wrappers or legacy compatibility modules.

### Phase 12: Deprecate or Remove BayesFilter NumPy Implementation Modules

Plan:
- remove public exports of NumPy implementation backends;
- either delete or move legacy NumPy files under a clearly non-production
  archival namespace;
- update tests to use TF/TFP fixtures;
- keep only test-side NumPy use if it is genuinely needed for simple array
  assertions and is not imported by production code.

Exit criteria:
- production `bayesfilter` filtering imports do not depend on NumPy;
- source checks enforce the TF/TFP-only policy.

## Acceptance Test Matrix

Required BayesFilter gates before client switch-over:

```text
tf_import_and_tfp_import_cpu_only             TF/TFP dependency gate
production_modules_no_numpy_imports          implementation policy
tf_result_metadata_schema_consistent         diagnostics contract
linear_tf_value                              exact Kalman identities
linear_tf_masked                             missing data convention
linear_tf_score_hessian                      TF finite-difference parity
linear_tf_svd_value                          dense parity and floor metadata
linear_tf_svd_derivative_gate                branch/floor labels
structural_sigma_tf                          identity and law tests
svd_sigma_tf_lgssm                           exact Kalman parity and finite gradient
svd_sigma_tf_nonlinear                       reference quadrature comparison
cut4_rule_tf                                 degree-five moment identities
svd_cut_value_tf                             SVD placement and rank-deficient tests
svd_cut_score_hessian_tf                     smooth-region finite-difference parity
xla_static_shape                             no retracing and compiled parity
gpu_optional_escalated                       GPU parity and throughput report
macrofinance_compat_tf                       one-country TF provider parity
dsge_compat_tf                               generic SSM and structural DSGE gates
legacy_numpy_exports_removed_or_blocked      final cleanup gate
```

## Risks and Controls

- Risk: NumPy reference code quietly remains the real implementation.
  Control: source checks and public import gates distinguish TF production
  modules from legacy files.
- Risk: TensorFlow dependency policy is implicit.
  Control: Phase 0 records TF/TFP availability and Phase 1 makes TF modules the
  public implementation path.
- Risk: SVD value robustness is mistaken for derivative readiness.
  Control: backend metadata separates value, derivative, compiled, HMC, and
  branch status.
- Risk: full-state sigma-point filtering hides deterministic structural
  coordinates.
  Control: default structural integration space is the innovation block; mixed
  full-state models require explicit approximation labels.
- Risk: CUT point count is treated as solved by GPU/XLA.
  Control: benchmark point-count scaling and declare valid `q` ranges.
- Risk: client switch-over breaks HMC targets.
  Control: switch one backend at a time and require value/gradient/Hessian
  parity before replacing client imports.

## Immediate Next Actions

1. Record the TF/TFP-only pivot in the reset memo.
2. Audit committed NumPy implementation files and classify each as:
   - replace with TF;
   - remove after TF parity;
   - retain only as test/legacy fixture.
3. Add a MacroFinance donor map to the plan and reset memo so implementation
   starts from tested TF/TFP code rather than a new derivation.
4. Implement TF result containers, diagnostics schema, and no-NumPy source
   checks.
5. Port dense TF linear Kalman value backend from MacroFinance.
6. Port TF masked and SVD value backends from MacroFinance.
7. Port TF analytic score/Hessian backend from MacroFinance.
8. Only then resume nonlinear SVD sigma-point and SVD-CUT implementation.

## Stop Rules

Stop and ask for direction if:
- TensorFlow or TensorFlow Probability is unavailable in the intended
  environment;
- a required backend cannot be made XLA-safe without changing public model
  contracts;
- replacing NumPy public exports would break client code before a TF wrapper is
  available;
- SVD-CUT derivative tests fail away from spectral branch events;
- a proposed client switch-over would silently change likelihood targets.
