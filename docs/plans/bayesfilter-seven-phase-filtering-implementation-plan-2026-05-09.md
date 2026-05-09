# Plan: Seven-phase BayesFilter filtering implementation program

## Date

2026-05-09

## Purpose

This plan makes the next filtering implementation decisions explicit.  It is
the execution-ready successor to the current filtering closure notes:

```text
docs/plans/bayesfilter-filtering-goals-gaps-hypotheses-closure-plan-2026-05-09.md
docs/plans/bayesfilter-filtering-gap-closure-execution-plan-2026-05-09.md
docs/plans/bayesfilter-filtering-goals-gaps-hypotheses-closure-audit-2026-05-09.md
```

The immediate decision is to run MacroFinance linear compatibility first, then
move to SVD/eigen value filtering, structural nonlinear protocols,
SVD sigma-point filters, CUT/SVD-CUT value filters, SVD-CUT derivatives, and
finally compiled/GPU/HMC/client switch-over gates.

The plan keeps the current project rule:

```text
Production BayesFilter filtering implementation is TensorFlow / TensorFlow
Probability first.  NumPy is allowed in tests and reference/testing modules,
but production TF modules must not import NumPy or call .numpy().
```

## Program Motivation

BayesFilter is meant to become the shared filtering backend for MacroFinance
and the DSGE codebase.  That replacement should not start from the most exotic
filter.  The safest route is to prove the linear TF spine in the nearest real
client first, then build singular-covariance and nonlinear layers on top of
that verified core.

The current state after Phase 1A is:

- dense and masked TF Cholesky value filters exist;
- dense and masked TF QR/square-root value filters exist;
- dense TF QR analytic score/Hessian exists;
- masked TF QR analytic score/Hessian now exists;
- solve-form and covariance-form derivative references are kept under
  `bayesfilter.testing`;
- production TF linear modules pass the no-NumPy/no-`.numpy()` hygiene check.

The remaining risk is not only mathematical correctness.  It is interface and
replacement risk: MacroFinance and the DSGE client may use conventions,
parameter layouts, missing-observation semantics, and regularization policies
that are easy to miss in abstract unit tests.  Therefore the next phase should
be client compatibility, not a new numerical backend.

## Global Invariants

Every phase must preserve these invariants.

1. Production filtering code remains TensorFlow/TFP only.
2. Reference NumPy, solve-form, and covariance-form code stays in testing or
   reference namespaces.
3. Diagnostics must distinguish the model law from the implemented numerical
   law whenever regularization, spectral floors, PSD projection, or fallback
   branches are active.
4. SGU production filtering remains blocked until the DSGE client supplies a
   causal, local, residual-closing filtering target.
5. GPU/CUDA detection, XLA-GPU tests, and GPU benchmarks require escalated
   sandbox permissions and explicit device/shape artifacts.
6. Client switch-over happens one backend at a time, with rollback boundaries.
7. Each phase ends with a reset-memo update containing results,
   interpretation, and whether the next phase is justified.

## Phase 1B: MacroFinance linear compatibility

### Motivation

MacroFinance is the closest real consumer for the completed linear spine.
Before adding SVD, structural nonlinear, UKF, CUT, or HMC layers, BayesFilter
should prove that the current QR value and derivative backends reproduce the
MacroFinance linear filtering contract.

This phase answers:

- Can BayesFilter replace the MacroFinance linear Kalman likelihood path?
- Do MacroFinance and BayesFilter agree on parameter derivative tensor
  ordering?
- Are masked observations, jitter, initial covariance, and state conventions
  identical?
- Is time-varying derivative support required now, or can it be deferred?

### Implementation Details

Actions:

- Inspect MacroFinance TF/TFP linear filtering code and tests without editing
  MacroFinance.
- Identify the smallest static LGSSM fixtures with:
  - dense observations;
  - masked observations;
  - score/Hessian parameters;
  - at least one nonzero transition derivative and one nonzero covariance or
    observation-noise derivative.
- Build BayesFilter-side compatibility tests under `tests/` or a dedicated
  `tests/compat/` namespace.
- Use MacroFinance outputs as fixtures or reference calls only if this can be
  done without adding MacroFinance as a production dependency.
- Compare:
  - dense QR value;
  - masked QR value;
  - dense QR score/Hessian;
  - masked QR score/Hessian;
  - metadata and diagnostic fields.
- Decide whether to implement time-varying LGSSM derivatives now:
  - if MacroFinance uses only static matrices in first switch-over fixtures,
    defer time-varying derivatives;
  - if MacroFinance requires time-indexed matrices/covariances, split a new
    Phase 1C before SVD work.

Expected BayesFilter files:

```text
tests/test_macrofinance_linear_compat_tf.py
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
```

Potential supporting files, only if needed:

```text
bayesfilter/testing/macrofinance_fixtures.py
tests/fixtures/macrofinance_linear_*.json
```

### Tests

Required:

- BayesFilter dense QR likelihood equals MacroFinance dense likelihood on the
  static fixture.
- BayesFilter masked QR likelihood equals MacroFinance masked likelihood under
  the static dummy-row convention.
- BayesFilter dense QR score and Hessian equal MacroFinance derivative
  references within documented tolerances.
- BayesFilter masked QR score and Hessian equal MacroFinance derivative
  references or TensorFlow autodiff for the same fixture.
- Production source hygiene remains clean:
  `rg -n "import numpy|from numpy|\\.numpy\\(" bayesfilter/...` returns no
  production matches.

### Exit Gate

Pass if:

- compatibility tests pass;
- any convention difference is documented and either resolved or explicitly
  declared unsupported;
- the time-varying derivative decision is recorded.

Stop if:

- MacroFinance conventions cannot be reconstructed without editing the client;
- results disagree and the difference cannot be localized to a convention;
- the first switch-over requires time-varying derivatives and no safe
  subphase is created.

## Phase 2: SVD/eigen linear value backend

### Motivation

The reviewer discussion and structural DSGE use case both require careful
handling of singular and near-singular covariance matrices.  QR/Cholesky paths
are appropriate for regular covariance factors, but BayesFilter also needs a
linear value backend whose diagnostics make the implemented singular or
regularized law explicit.

This phase is about value filtering only.  It should not claim SVD
score/Hessian readiness.

### Implementation Details

Actions:

- Define a production TF SVD/eigen covariance factor helper, preferably using
  `tf.linalg.eigh` for symmetric PSD covariance matrices.
- Implement a clear regularization policy:
  - raw eigenvalues;
  - floor threshold;
  - floor count;
  - effective rank;
  - implemented covariance;
  - PSD projection residual;
  - branch label.
- Add dense and masked linear value filters using the implemented covariance
  factor.
- Return diagnostics that state whether the value is for:
  - the raw model covariance;
  - a floored covariance;
  - a projected covariance;
  - a fallback branch.
- Keep derivative metadata blocked or value-only:

```text
differentiability_status = "value_only"
derivative_target = "implemented_regularized_law"
```

Expected BayesFilter files:

```text
bayesfilter/linear/svd_factor_tf.py
bayesfilter/linear/kalman_svd_tf.py
tests/test_linear_svd_factor_tf.py
tests/test_linear_kalman_svd_tf.py
```

### Tests

Required:

- Well-conditioned SVD/eigen value matches Cholesky and QR likelihood.
- Masked SVD/eigen value matches masked QR when covariance is regular.
- Rank-deficient covariance returns finite likelihood when the observed law is
  supported by the implemented covariance.
- Diagnostics report effective rank, floor count, implemented covariance, and
  PSD residual.
- Active floor tests show explicitly that the factor reconstructs the
  implemented/floored covariance, not necessarily the pre-regularized
  covariance.
- No derivative backend is exported for SVD/eigen in this phase.

### Exit Gate

Pass if:

- regular-case parity passes;
- singular/near-singular diagnostics are explicit;
- derivative claims remain blocked.

Stop if:

- regularization semantics are ambiguous;
- tests cannot distinguish raw law from implemented law;
- production code would require NumPy.

## Phase 3: TF structural nonlinear protocol

### Motivation

UKF, cubature, SVD sigma-point, and CUT filters need a common structural model
interface before the DSGE client can switch over.  BayesFilter should own the
generic protocol, while DSGE economics remain in the DSGE repository.

This phase should make deterministic completion a first-class contract rather
than a hidden singular covariance trick.

### Implementation Details

Actions:

- Define TF structural state-space protocols for:
  - initial state law;
  - transition map;
  - observation map;
  - process innovation covariance;
  - observation noise covariance;
  - deterministic completion;
  - declared stochastic integration space.
- Add metadata fields for:
  - exogenous block;
  - endogenous block;
  - deterministic-completion block;
  - approximation label;
  - collapsed full-state singular covariance, if explicitly selected.
- Implement toy nonlinear fixtures:
  - scalar nonlinear transition;
  - affine LGSSM embedded in structural protocol;
  - deterministic identity completion.
- Avoid importing DSGE client code into production BayesFilter modules.

Expected BayesFilter files:

```text
bayesfilter/structural_tf.py
bayesfilter/nonlinear/protocols_tf.py
tests/test_structural_protocols_tf.py
tests/test_structural_affine_lgssm_controls_tf.py
```

### Tests

Required:

- Structural affine LGSSM reduces to the linear Kalman value on affine
  controls.
- Deterministic completion holds pointwise on all propagated test points.
- Declared innovation-space integration and collapsed full-state integration
  are distinguishable in metadata.
- Diagnostics expose deterministic and stochastic blocks separately.

### Exit Gate

Pass if:

- protocol tests pass without DSGE imports;
- affine controls reproduce the linear backend;
- deterministic completion is pointwise, not merely covariance-level.

Stop if:

- the interface forces DSGE-specific economics into BayesFilter;
- deterministic coordinates can only be represented by hidden singular
  covariance metadata.

## Phase 4: SVD sigma-point value filters

### Motivation

Once structural protocols exist, BayesFilter can implement generic sigma-point
filters.  The SVD/eigen factor from Phase 2 should make sigma-point placement
work for singular or near-singular covariance matrices while reporting the
implemented law.

This phase includes UKF/cubature value filters and SVD placement.  It should
not include CUT4-G or analytic derivatives yet.

### Implementation Details

Actions:

- Implement TF sigma-point rule objects for:
  - UKF;
  - spherical-radial cubature;
  - optional user-provided point rule.
- Implement SVD/eigen sigma-point placement:
  - factor covariance in the declared stochastic integration space;
  - place sigma points only in active directions unless a full-state rule is
    explicitly requested;
  - record rank, floor, and implemented covariance diagnostics.
- Implement generic nonlinear prediction/update value recursion:
  - propagate sigma points through transition map;
  - compute predicted mean/covariance/cross-covariance;
  - propagate observation sigma points;
  - compute innovation covariance and update.
- Support deterministic completion pointwise through the structural protocol.

Expected BayesFilter files:

```text
bayesfilter/nonlinear/sigma_points_tf.py
bayesfilter/nonlinear/unscented_tf.py
bayesfilter/nonlinear/cubature_tf.py
tests/test_sigma_points_tf.py
tests/test_unscented_filter_tf.py
tests/test_cubature_filter_tf.py
```

### Tests

Required:

- Sigma-point weights sum to one and reproduce required first/second moments
  on regular covariance.
- Rank-deficient placement uses the active support and reports rank.
- Affine LGSSM sigma-point filter matches linear Kalman value within tolerance.
- Deterministic completion holds pointwise for propagated points.
- Same-shape graph reuse passes for fixed dimensions.

### Exit Gate

Pass if:

- UKF/cubature value gates pass on regular and rank-deficient controls;
- diagnostics identify the implemented covariance law;
- structural protocol integration works without client-specific code.

Stop if:

- point placement silently leaves the intended support;
- affine exactness fails and cannot be localized;
- graph behavior requires dynamic Python-side shape changes.

## Phase 5: CUT/SVD-CUT value filters

### Motivation

CUT4-G can improve moment accuracy relative to standard UKF/cubature rules, but
its point count grows quickly.  It should be introduced only after the generic
sigma-point value recursion is stable.

This phase adds value filtering only.  Derivatives come later.

### Implementation Details

Actions:

- Port the generic conjugate unscented transform rule from the DSGE and
  MacroFinance source material where appropriate, without DSGE economics.
- Implement CUT4-G rule generation in TF:
  - central point if required by the rule;
  - axis points;
  - conjugate points;
  - weights;
  - static point-count metadata.
- Combine CUT4-G with SVD/eigen placement:
  - regular covariance;
  - rank-deficient covariance;
  - floored implemented covariance.
- Add approximation labels:

```text
approximation_label = "CUT4-G"
point_count = ...
integration_rank = ...
```

Expected BayesFilter files:

```text
bayesfilter/nonlinear/cut_tf.py
bayesfilter/nonlinear/svd_cut_tf.py
tests/test_cut_rule_tf.py
tests/test_svd_cut_filter_tf.py
```

### Tests

Required:

- CUT rule weights and moments match the documented identities.
- CUT value filter matches linear Kalman on affine LGSSM controls.
- SVD-CUT rank-deficient placement remains on the implemented support.
- Point-count and memory-shape diagnostics are reported.
- CPU static-shape tests pass before any GPU claim.

### Exit Gate

Pass if:

- CUT/SVD-CUT value filters pass moment, affine, and rank-deficient tests;
- point-count diagnostics are explicit;
- no derivative claim is made.

Stop if:

- point count becomes impractical for target dimensions without a benchmark;
- CUT moment identities cannot be reproduced;
- SVD placement and CUT rule conventions conflict.

## Phase 6: analytic gradients and Hessians for SVD-CUT

### Motivation

Analytic SVD-CUT derivatives are useful for HMC and optimization, but they are
the most delicate part of the program.  They should be derived only after
value filters and implemented-law diagnostics are stable.

The derivative target must be the implemented law.  If a floor, PSD projection,
or fallback branch changes the covariance, derivatives of the raw law are not
being certified.

### Implementation Details

Actions:

- Split derivative work into two layers:
  - fixed quadrature-rule derivatives;
  - covariance-factor/SVD/eigen derivatives.
- For fixed rule derivatives, derive:
  - sigma-point locations;
  - propagated transition points;
  - predicted moments;
  - observation moments;
  - innovation covariance;
  - Kalman-like update;
  - likelihood contribution;
  - score and Hessian recursions.
- For SVD/eigen factor derivatives, require:
  - separated spectra;
  - inactive floor branches;
  - stable eigenvector sign/order convention or sign-invariant covariance
    reconstruction;
  - branch diagnostics.
- Add fail-closed labels when derivative assumptions are violated:

```text
blocked_repeated_spectrum
blocked_active_floor
blocked_fallback_branch
blocked_unstable_eigenvector_order
```

Expected BayesFilter files:

```text
bayesfilter/nonlinear/svd_cut_derivatives_tf.py
tests/test_svd_cut_derivatives_tf.py
docs/chapters/... analytic derivative sections, if documentation is updated
```

### Tests

Required:

- Finite-difference parity on smooth low-dimensional fixtures.
- TensorFlow autodiff parity on fixed smooth branches.
- Hessian symmetry.
- Implemented-law derivative target metadata.
- Fail-closed behavior near repeated spectra and active floors.

### Exit Gate

Pass if:

- derivative tests pass on explicitly smooth branches;
- blocked branch labels fire where expected;
- no raw-law derivative claim is made under active regularization.

Stop if:

- SVD/eigen second derivatives are unstable under realistic tolerances;
- branch diagnostics cannot identify invalid derivative regions;
- Hessian symmetry cannot be maintained.

## Phase 7: compiled/GPU/HMC gates and client switch-over

### Motivation

Performance and HMC readiness should certify fixed model/backend pairs, not
abstract filter names.  GPU/XLA can make large point sets practical, but only
after eager correctness, derivative correctness, and static-shape compile
parity are already established.

This phase also controls actual client replacement.

### Implementation Details

Actions:

- Add compile parity tests for each promoted backend:
  - eager value vs compiled value;
  - eager score/Hessian vs compiled score/Hessian where available.
- Add benchmark harnesses recording:
  - CPU/GPU device;
  - dtype;
  - state dimension;
  - observation dimension;
  - time length;
  - rank;
  - point count;
  - compile time;
  - first-call time;
  - steady-state time;
  - peak memory if available.
- Run GPU/XLA only with escalated sandbox permissions.
- Define client switch-over order:
  1. MacroFinance linear value and QR derivatives.
  2. MacroFinance SVD value if singular linear fixtures require it.
  3. DSGE generic nonlinear value after structural/SVD sigma-point gates.
  4. DSGE SVD-CUT value after CUT gates.
  5. HMC only after derivative and compiled gates for the exact target model.
- Keep SGU blocked until the DSGE client provides:

```text
sgu_causal_filtering_target_passed
```

Expected BayesFilter files:

```text
tests/test_compiled_filter_parity_tf.py
docs/benchmarks/benchmark_linear_filters_tf.py
docs/benchmarks/benchmark_sigma_point_filters_tf.py
docs/plans/client-switch-over-*.md
```

### Tests

Required:

- CPU eager/compiled parity for promoted value backends.
- CPU eager/compiled parity for promoted derivative backends.
- Escalated GPU/XLA benchmark artifacts for any GPU performance claim.
- MacroFinance switch-over parity tests before changing MacroFinance defaults.
- DSGE switch-over parity tests before changing DSGE defaults.

### Exit Gate

Pass if:

- compiled parity is established for the exact backend/model pair;
- GPU claims have escalated benchmark artifacts;
- client switch-over has a rollback plan and parity tests.

Stop if:

- compiled and eager values disagree;
- GPU performance claims cannot be reproduced under escalated runs;
- HMC is requested before value, derivative, and compiled parity are all
  available for the target model.

## Recommended Immediate Next Step

Start Phase 1B.

Concrete first action:

```text
Audit MacroFinance TF/TFP linear filtering fixtures and identify the smallest
static dense/masked likelihood and score/Hessian examples that BayesFilter can
reproduce without adding MacroFinance as a production dependency.
```

The first Phase 1B output should be a short compatibility audit:

```text
docs/plans/bayesfilter-phase1b-macrofinance-linear-compat-audit-2026-05-09.md
```

That audit should decide whether time-varying LGSSM derivatives are required
before Phase 2 begins.
