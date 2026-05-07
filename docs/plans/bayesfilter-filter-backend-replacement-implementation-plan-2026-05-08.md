# BayesFilter Filter Backend Replacement Implementation Plan

Date: 2026-05-08

## Purpose

Implement the linear filter, nonlinear structural filter, and SVD-CUT filter
properly inside BayesFilter so that `/home/chakwong/python` and
`/home/chakwong/MacroFinance` can replace their local filtering algorithms with
BayesFilter backends.

The goal is not a cosmetic port.  BayesFilter should become the shared
filtering library with:
- exact linear Gaussian likelihood, score, and Hessian;
- masked and large-panel linear Gaussian support;
- structural nonlinear sigma-point filtering over declared stochastic
  integration coordinates;
- SVD/CUT variants with explicit regularization, derivative, and XLA/GPU
  contracts;
- client adapters that let MacroFinance and DSGE projects keep model
  construction while delegating filtering.

## Current Starting Point

BayesFilter already has:
- `bayesfilter/filters/kalman.py`: NumPy covariance-form Kalman value backend,
  singular process covariance accepted, Joseph update used.
- `bayesfilter/filters/sigma_points.py`: NumPy structural SVD/eigen cubature
  reference backend over `(previous_state, innovation)`.
- `bayesfilter/structural.py`: state partition and structural filter metadata.
- `bayesfilter/adapters/macrofinance.py`: MacroFinance-shaped LGSSM adapters
  and derivative delegation wrappers.
- `bayesfilter/adapters/dsge.py`: fail-closed structural metadata gate for
  DSGE-shaped models.
- Tests for degenerate Kalman, structural metadata, structural sigma-point
  examples, MacroFinance adapter metadata, and backend readiness gates.

BayesFilter does not yet have:
- first-class derivative dataclasses for linear Gaussian score/Hessian;
- production linear analytic score/Hessian copied from MacroFinance;
- TensorFlow/XLA linear backends;
- masked derivative backends;
- generic nonlinear TensorFlow SVD sigma-point backend;
- CUT4-G and SVD-CUT rule implementations;
- analytic SVD-CUT score/Hessian implementation;
- parity suites that allow client projects to switch imports to BayesFilter.

## Source Material To Reuse

### From `/home/chakwong/MacroFinance`

Primary reusable implementation sources:
- `domain/types.py`
  - `LinearGaussianStateSpace`
  - `LinearGaussianStateSpaceDerivatives`
  - `RunConfig` and backend labels
- `filters/solve_differentiated_kalman.py`
  - solve-form NumPy log likelihood, score, Hessian;
  - trace rows with innovation, solve, derivative, gain, and covariance
    diagnostics.
- `filters/differentiated_kalman.py`
  - covariance-form analytic derivative reference.
- `filters/tf_differentiated_kalman.py`
  - TensorFlow first/second-order analytic derivative backends and graph
    variants.
- `filters/tf_masked_kalman.py`
  - TensorFlow masked likelihood convention.
- `filters/tf_svd_kalman.py`
  - TensorFlow SVD linear value backend, singular-value floor diagnostics,
    masked SVD convention.
- `filters/qr_sqrt_differentiated_kalman.py`,
  `filters/masked_qr_sqrt_differentiated_kalman.py`,
  `filters/tf_qr_sqrt_differentiated_kalman.py`
  - QR/square-root derivative and masked robustness patterns.

Primary reusable tests:
- `tests/test_differentiated_kalman.py`
- `tests/test_filter_backend_parity.py`
- `tests/test_tf_masked_kalman.py`
- `tests/test_tf_svd_kalman.py`
- `tests/test_filter_backend_derivative_trace.py`
- `tests/test_factor_derivative_identities.py`
- `tests/test_kalman_backend_selector.py`
- `tests/test_one_country_analytic_backend_parity.py`
- `tests/test_one_country_hmc_analytic_gradient_hessian.py`

Reuse policy:
- Port generic algorithms and tests into BayesFilter notation.
- Keep MacroFinance economics, yield-curve providers, and one-country/cross-
  currency construction in MacroFinance.
- BayesFilter should expose protocols so MacroFinance providers can call
  BayesFilter without importing BayesFilter internals into model construction.

### From `/home/chakwong/python`

Primary reusable implementation sources:
- `src/dsge_hmc/filters/_quadrature.py`
  - `QuadratureRule`, `CubatureRule`, `UnscentedRule`.
- `src/dsge_hmc/filters/_svd_core.py`
  - SVD factorization, clamping, sigma predict/update primitives, conditioning
    guards, diagnostic payloads.
- `src/dsge_hmc/filters/_svd_filters.py`
  - generic SSM SVD sigma-point path;
  - DSGE legacy full-state guard;
  - XLA-safe debug-gating ideas;
  - failure-code and first-batch diagnostics.
- `src/dsge_hmc/filters/CUTSRUKF.py`
  - experimental CUT4-G point generation;
  - positive weights;
  - QR square-root CUT update pattern;
  - TensorFlow/XLA batching demonstration.
- `src/dsge_hmc/models/structural_metadata.py`
  - DSGE structural metadata and eta-support validation helpers.

Primary reusable tests:
- `tests/contracts/test_svd_lgssm_reference.py`
- `tests/contracts/test_svd_nonlinear_ssm_reference.py`
- `tests/contracts/test_svd_generic_nonlinear_ssm.py`
- `tests/contracts/test_svd_generic_ssm_xla.py`
- `tests/contracts/test_nk_svd_xla_gates.py`
- `tests/contracts/test_nk_svd_gradient_finiteness_gate.py`
- `tests/contracts/test_structural_dsge_partition.py`
- `tests/contracts/test_filter_contracts.py`
- `tests/numerics/test_svd_filters.py`
- `tests/CUTSRUKF_test.py`

Reuse policy:
- Port generic TensorFlow SVD sigma-point infrastructure into BayesFilter.
- Do not port DSGE economic solution logic into BayesFilter.
- Keep custom MKL ops optional.  BayesFilter should support a default
  `tf.linalg.eigh`/NumPy path and an optional robust-eigh plugin path.
- Keep legacy full-state mixed-DSGE paths blocked unless explicitly labeled as
  approximation-only.

## Target Package Shape

Add or expand these modules:

```text
bayesfilter/types.py
bayesfilter/results.py
bayesfilter/diagnostics.py
bayesfilter/linear/types.py
bayesfilter/linear/kalman_numpy.py
bayesfilter/linear/kalman_derivatives_numpy.py
bayesfilter/linear/kalman_tf.py
bayesfilter/linear/kalman_svd_tf.py
bayesfilter/nonlinear/protocols.py
bayesfilter/nonlinear/quadrature.py
bayesfilter/nonlinear/svd_factor.py
bayesfilter/nonlinear/sigma_point_numpy.py
bayesfilter/nonlinear/sigma_point_tf.py
bayesfilter/nonlinear/svd_cut.py
bayesfilter/nonlinear/svd_cut_derivatives.py
bayesfilter/adapters/macrofinance.py
bayesfilter/adapters/dsge.py
```

Keep compatibility re-exports in:

```text
bayesfilter/filters/kalman.py
bayesfilter/filters/sigma_points.py
bayesfilter/filters/__init__.py
```

until client projects are switched.

## Unified Public Contracts

### Linear Gaussian State Space

BayesFilter should own a generic type equivalent to MacroFinance's:

```python
LinearGaussianStateSpace(
    initial_mean,
    initial_covariance,
    transition_offset,
    transition_matrix,
    transition_covariance,
    observation_offset,
    observation_matrix,
    observation_covariance,
    partition=None,
)
```

Add:

```python
LinearGaussianStateSpaceDerivatives(
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

Return:

```python
FilterValueResult(log_likelihood, filtered_means, filtered_covariances, metadata, diagnostics)
FilterDerivativeResult(log_likelihood, score, hessian, metadata, diagnostics, trace)
```

### Nonlinear Structural State Space

BayesFilter should define a structural nonlinear protocol:

```python
partition
initial_mean(theta)
initial_cov(theta)
innovation_mean(theta)        # default zero
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
transition.  The default integration space for mixed structural models is
`innovation`, not full state.

### Quadrature Rules

Implement:
- `CubatureRule(dim)`
- `UnscentedRule(dim, alpha=1.0, beta=2.0, kappa=None)`
- `CUT4GRule(dim)` with `dim >= 3`, `2 dim + 2**dim` points, positive weights,
  and moment-test helpers.

Expose both standardized offsets and placed points:

```python
rule.standard_offsets_and_weights()
rule.place(mean, factor)
```

This makes SVD-CUT explicit: `factor` may be Cholesky, QR, SVD/eigen, or a
rectangular structural factor.

## Implementation Phases

### Phase 0: Repository Hygiene and Baseline

Plan:
- preserve existing untracked PDFs/templates and unrelated SGU/source-map work;
- create a branch or commit only BayesFilter implementation-plan files;
- run the current BayesFilter unit tests.

Execute:
- record current test baseline with `pytest -q`;
- record optional dependency status: NumPy only, TensorFlow available, GPU
  probe only with escalated permissions per local policy.

Audit:
- no implementation phase starts if the current test suite fails for reasons
  unrelated to the planned changes.

Exit criteria:
- clean baseline report;
- unrelated working-tree changes identified and excluded from commits.

### Phase 1: Core Types and Result Objects

Plan:
- move generic LGSSM and derivative dataclasses into BayesFilter-owned modules;
- add result and diagnostics dataclasses;
- preserve backwards imports from `bayesfilter.filters`.

Borrow:
- `MacroFinance/domain/types.py` dataclass shape.

Tests:
- conversion tests from current `tests/test_macrofinance_adapter.py`;
- shape validation tests for derivative tensors;
- immutability/read-only diagnostics tests.

Exit criteria:
- MacroFinance-style fake providers convert to BayesFilter types;
- current Kalman value tests still pass.

### Phase 2: Exact Linear Gaussian Value Backends

Plan:
- keep and harden NumPy covariance/Joseph backend;
- add solve-form backend;
- add masked observation convention;
- add SVD linear value backend with singular/floor diagnostics;
- expose backend selector.

Borrow:
- `MacroFinance/filters/kalman.py`;
- `MacroFinance/filters/tf_svd_kalman.py` value-side SVD diagnostics;
- current BayesFilter `kalman.py` missing-data and Joseph update behavior.

Tests:
- affine Gaussian parity across covariance, solve, SVD, masked variants;
- singular process covariance acceptance;
- missing-row prediction-only behavior;
- floor diagnostics for near-singular innovation covariance;
- backend selector tests from MacroFinance.

Exit criteria:
- all linear value backends agree on well-conditioned LGSSMs;
- SVD backend reports implemented/floored covariance diagnostics;
- no silent regularization without metadata.

### Phase 3: Linear Analytic Score and Hessian

Plan:
- port solve-form analytic score/Hessian into
  `bayesfilter/linear/kalman_derivatives_numpy.py`;
- optionally port covariance-form derivative reference for parity;
- add trace rows and finite-difference oracle helpers;
- add masked derivative support only after dense derivative parity passes.

Borrow:
- `MacroFinance/filters/solve_differentiated_kalman.py`;
- `MacroFinance/filters/differentiated_kalman.py`;
- tests from `test_differentiated_kalman.py`,
  `test_filter_backend_derivative_trace.py`.

Tests:
- one- and multi-parameter synthetic LGSSM finite-difference score/Hessian;
- MacroFinance one-country fixture via adapter, if importable;
- Hessian symmetry and trace diagnostics;
- stationary/initial-condition derivative policy tests.

Exit criteria:
- score and Hessian match finite differences on BayesFilter fixtures;
- MacroFinance provider can call BayesFilter derivative backend and match its
  previous backend on the one-country fixture.

### Phase 4: TensorFlow Linear Backends and XLA Gates

Plan:
- port TensorFlow dense value/score/Hessian backends;
- port graph variants with static-shape discipline;
- port masked TensorFlow likelihood and derivative support;
- define CPU/GPU parity tests and XLA tests as opt-in.

Borrow:
- `MacroFinance/filters/tf_differentiated_kalman.py`;
- `MacroFinance/filters/tf_masked_kalman.py`;
- `MacroFinance/tests/test_filter_backend_parity.py`;
- `MacroFinance/tests/test_tf_masked_kalman.py`.

Tests:
- TF vs NumPy value/score/Hessian parity;
- graph function reuse for same shapes;
- source checks that hot graph paths do not use `.numpy()` or Python trace-only
  operations;
- GPU parity only under escalated sandbox permissions.

Exit criteria:
- MacroFinance HMC target can call BayesFilter TF analytic backend without
  changing model-provider code;
- XLA readiness is labeled separately from correctness.

### Phase 5: Nonlinear Structural Protocol and NumPy Reference Filter

Plan:
- upgrade current `StructuralSVDSigmaPointFilter` into a clearer reference
  backend:
  - structural protocol validation;
  - integration-space selection;
  - deterministic completion metadata;
  - quadrature rule injection;
  - diagnostics for prediction/update eigenvalues and support dimension.
- preserve cubature and UKF behavior.

Borrow:
- BayesFilter current `sigma_points.py`;
- `/python/src/dsge_hmc/models/structural_metadata.py`;
- `/python/tests/contracts/test_structural_dsge_partition.py`.

Tests:
- worked structural UKF example;
- deterministic identities hold pointwise;
- dense quadrature comparison on low-dimensional nonlinear fixture;
- fail-closed mixed-model full-state integration unless labeled.

Exit criteria:
- nonlinear NumPy reference is stable enough to be the oracle for TF/SVD/CUT
  ports.

### Phase 6: Generic TensorFlow SVD Sigma-Point Filter

Plan:
- port generic SSM TensorFlow SVD sigma-point infrastructure;
- separate generic structural path from DSGE legacy path;
- implement optional robust-eigh backend interface:
  - default `tf.linalg.eigh`;
  - optional dsge_hmc robust-eigh adapter when installed;
  - diagnostics report backend.
- port conditioning guards and failure codes.

Borrow:
- `/python/src/dsge_hmc/filters/_svd_core.py`;
- `/python/src/dsge_hmc/filters/_svd_filters.py`;
- `/python/tests/contracts/test_svd_lgssm_reference.py`;
- `/python/tests/contracts/test_svd_nonlinear_ssm_reference.py`;
- `/python/tests/contracts/test_svd_generic_nonlinear_ssm.py`.

Tests:
- SVD sigma-point scalar LGSSM matches exact Kalman likelihood;
- gradient finite and matches finite difference on scalar LGSSM;
- nonlinear observation tracks dense quadrature within declared tolerance;
- boundary/extreme proposals return bounded fail-closed values;
- diagnostics shapes and finite failure codes.

Exit criteria:
- `/python` generic SVD SSM tests can be rewritten to import BayesFilter and
  pass without local filter classes.

### Phase 7: CUT4-G and SVD-CUT Value Backend

Plan:
- implement `CUT4GRule` in `bayesfilter/nonlinear/quadrature.py`;
- add moment-test helper for monomials through degree five;
- implement SVD-CUT value path by combining `CUT4GRule` with the SVD
  sigma-point backend;
- add structural low-dimensional integration support.

Borrow:
- `/python/src/dsge_hmc/filters/CUTSRUKF.py` point construction;
- docs Chapter 16 CUT4-G proof;
- docs Chapter 18 SVD-CUT affine exactness section.

Tests:
- weights positive and sum to one;
- covariance and fourth-moment identities:
  - `E[z_i^2]=1`;
  - `E[z_i^4]=3`;
  - `E[z_i^2 z_j^2]=1`;
- affine polynomial exactness after SVD/eigen placement;
- comparison with UT/Cubature on nonlinear low-dimensional fixtures;
- rank-deficient covariance placement uses `P_star = C C^T` diagnostics.

Exit criteria:
- CUT4-G is a selectable quadrature rule for the nonlinear SVD sigma-point
  backend;
- SVD-CUT value backend is documented as approximation-only unless model
  structure gives exactness.

### Phase 8: Analytic Gradient and Hessian of SVD-CUT

Plan:
- implement derivative objects for fixed quadrature offsets:
  - point derivatives;
  - map derivatives supplied by provider or AD wrapper;
  - weighted mean/covariance derivatives;
  - innovation derivatives;
  - solve-form likelihood score/Hessian;
  - factor derivative contract.
- start with NumPy analytic derivatives on small fixtures;
- add TensorFlow autodiff/custom-gradient parity only after NumPy passes.

Borrow:
- docs Chapter 18 SVD-CUT derivation;
- MacroFinance Chapter 10 solve-form Hessian implementation pattern;
- BayesFilter `certify_spectral_derivative_region`.

Tests:
- direct finite-difference score/Hessian of the same SVD-CUT scalar likelihood;
- factor reconstruction identities for `dot C`, `ddot C`;
- branch/floor tests labeled as branch derivatives;
- comparison to raw TensorFlow autodiff on simple spectra;
- blocked HMC label near repeated eigenvalues unless non-spectral custom
  gradient certification passes.

Exit criteria:
- analytic SVD-CUT score/Hessian matches finite differences away from spectral
  branch events;
- derivative result declares whether it is smooth, branch derivative,
  regularized, or blocked.

### Phase 9: GPU/XLA Benchmark and Compilation Gates

Plan:
- add benchmark scripts for:
  - point-count scaling;
  - chains/batch scaling;
  - CUT4 vs UT vs Cubature;
  - gradient and Hessian memory/time;
  - compile time and steady-state time.
- run CPU baseline by default;
- run GPU/XLA only with escalated sandbox permissions.

Tests:
- small static-shape XLA compile;
- CPU/compiled parity;
- optional GPU parity and throughput;
- no dynamic point-count retracing.

Exit criteria:
- claims are dimension-bounded:
  - `q <= q_max` for CUT4-G on target hardware;
  - compile time and memory reported;
  - no blanket claim that `2q+2^q` is harmless.

### Phase 10: MacroFinance Switch-Over

Plan:
- add a MacroFinance compatibility layer:
  - `from bayesfilter.adapters.macrofinance import ...`;
  - backend selector maps old names to BayesFilter backends.
- in MacroFinance, replace local filter calls one backend at a time:
  - value Kalman;
  - solve analytic score/Hessian;
  - TF analytic score/Hessian;
  - masked/large-scale backend;
  - optional SVD value backend.

Tests:
- MacroFinance one-country parity;
- cross-currency derivative gates still use MacroFinance providers but
  BayesFilter filtering;
- HMC readiness smoke uses BayesFilter backend;
- no regression in existing MacroFinance tests.

Exit criteria:
- MacroFinance local filter modules become thin compatibility wrappers or are
  deprecated with tests pointing to BayesFilter.

### Phase 11: `/python` DSGE Switch-Over

Plan:
- add a DSGE compatibility adapter:
  - consumes model `solve(theta)` output;
  - extracts structural metadata;
  - validates eta support;
  - builds BayesFilter nonlinear structural protocol objects.
- replace local generic SVD SSM filter first;
- replace DSGE SVD sigma-point path only after structural metadata and
  deterministic completion are explicit.

Tests:
- generic scalar LGSSM and nonlinear SSM tests import BayesFilter;
- NK SVD XLA gates import BayesFilter;
- mixed DSGE full-state approximation remains blocked unless labeled;
- structural stochastic-block filter passes deterministic identity tests.

Exit criteria:
- `/python/src/dsge_hmc/filters/_svd_core.py`,
  `_quadrature.py`, `_svd_filters.py`, and `CUTSRUKF.py` can be marked legacy
  or wrappers over BayesFilter.

### Phase 12: Production Readiness and Deprecation

Plan:
- define status labels for every backend:
  - `value_reference`;
  - `score_hessian_reference`;
  - `compiled_value`;
  - `compiled_gradient`;
  - `hmc_candidate`;
  - `production_gated`;
  - `approximation_only`;
  - `blocked`.
- write migration docs and examples.
- add versioned deprecation notices in client projects.

Exit criteria:
- BayesFilter is the source of truth for filtering algorithms;
- client projects own only model construction, parameter transforms, and
  application-specific inference wrappers.

## Acceptance Test Matrix

Required BayesFilter tests before client switch-over:

```text
linear_value_numpy               exact Kalman parity
linear_masked_numpy              missing data convention
linear_score_hessian_numpy       finite-difference parity
linear_tf_value                  NumPy parity
linear_tf_score_hessian          NumPy analytic parity
linear_tf_masked                 masked NumPy parity
linear_svd_value                 Cholesky/solve parity on well-conditioned cases
structural_sigma_numpy           dense quadrature and identity tests
svd_sigma_tf_lgssm               exact Kalman parity and finite gradient
svd_sigma_tf_nonlinear           dense quadrature comparison and diagnostics
cut4_rule                        degree-five moment identities
svd_cut_value                    SVD placement and rank-deficient covariance tests
svd_cut_score_hessian            finite-difference parity away from branch events
xla_static_shape                 no retracing and compiled parity
gpu_optional                     escalated GPU parity and throughput report
macrofinance_compat              one-country and derivative provider parity
dsge_compat                      generic SSM and structural DSGE gates
```

## Risks and Controls

- Risk: copying client code imports application-specific assumptions.
  Control: port only generic algorithms and tests; keep provider construction in
  clients.
- Risk: SVD value robustness is mistaken for derivative readiness.
  Control: backend metadata separates value, derivative, compiled, and HMC
  status.
- Risk: full-state sigma-point filtering hides deterministic structural
  coordinates.
  Control: default structural integration space is `innovation`; full-state
  mixed models require explicit approximation labels.
- Risk: CUT point count is treated as solved by GPU/XLA.
  Control: benchmark point-count scaling and declare valid `q` ranges.
- Risk: optional custom eigensolver creates dependency lock-in.
  Control: define an eigensolver protocol with default NumPy/TF implementation
  and optional robust plugin.
- Risk: client switch-over breaks HMC targets.
  Control: switch one backend at a time and require value/gradient/Hessian
  parity before replacing client imports.

## Immediate Next Actions

1. Run current BayesFilter `pytest -q` and record baseline.
2. Create Phase 1 PR/commit with core types and result objects.
3. Port MacroFinance solve-form analytic Kalman into BayesFilter.
4. Port MacroFinance derivative parity tests using tiny synthetic fixtures.
5. Only after dense linear score/Hessian passes, start the TensorFlow and
   nonlinear SVD sigma-point phases.

## Stop Rules

Stop and ask for direction if:
- current BayesFilter baseline tests fail for unrelated reasons;
- MacroFinance or `/python` source APIs have changed enough that direct reuse is
  no longer safe;
- TensorFlow dependency decisions require adding package metadata not currently
  present in BayesFilter;
- SVD-CUT derivative tests fail away from spectral branch events;
- a proposed client switch-over would reduce current client test coverage or
  silently change likelihood targets.
