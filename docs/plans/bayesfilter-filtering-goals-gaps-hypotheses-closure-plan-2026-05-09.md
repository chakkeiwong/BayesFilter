# Plan: BayesFilter filtering goals, gaps, hypotheses, and closure path

## Date

2026-05-09

## Purpose

This plan consolidates the current BayesFilter filtering implementation status,
the remaining gaps, the hypotheses that could close those gaps, and an
execution order for finishing the filtering backend replacement program.

It is deliberately split into two lanes:

1. **Generic BayesFilter backend completion.**  This lane should make
   BayesFilter a TF/TFP-first filtering library that can replace duplicated
   filtering code in MacroFinance and the DSGE client.
2. **Model-specific structural evidence.**  This lane decides whether SGU and
   other DSGE models have production predictive filtering targets.  These
   economics stay in the owning client repo; BayesFilter records only generic
   metadata, diagnostics, and filtering backends.

The key rule is simple: generic TF/TFP filtering work can continue, but
model-specific downstream promotion needs a value target first.  Derivatives,
compiled parity, HMC, and client switch-over must not outrun value-level
evidence.

## Current Achieved Goals

### TF/TFP Linear Filtering Spine

Implemented and tested in BayesFilter:

- TF result and diagnostics containers that preserve tensors.
- TF linear Gaussian state-space contracts.
- Dense Cholesky Kalman value backend.
- Static dummy-row masked Cholesky Kalman value backend.
- QR/square-root factor helpers:
  - positive-diagonal thin QR;
  - QR first and second derivatives;
  - Cholesky first and second derivatives;
  - horizontal-stack factor reconstruction helpers.
- QR/square-root dense value backend.
- QR/square-root static masked value backend.
- Dense QR/square-root analytic score and Hessian backend.
- Solve-form and covariance-form TF derivative references under
  `bayesfilter.testing`, intentionally not production exports.

Latest local evidence:

```text
CUDA_VISIBLE_DEVICES=-1 pytest -q
106 passed, 2 warnings
```

Production TF modules currently satisfy the no-NumPy/no-`.numpy()` policy.

### Structural DSGE Evidence

Current allowed SGU evidence:

```text
sgu_state_identity_completion_passed
sgu_two_slice_projection_diagnostic_passed
sgu_causal_control_anchor_residual_closure_diagnostic_passed
```

Current active SGU blockers:

```text
blocked_sgu_causal_projection_residual_not_closed
blocked_sgu_causal_anchor_projection_nonlocal
blocked_sgu_two_slice_projection_not_filter_transition
blocked_sgu_projection_is_new_target_not_gate_b
blocked_sgu_expectation_averaging_does_not_rescue_quadratic_gate
blocked_sgu_quadratic_policy_not_better_than_linear_residual
blocked_nonlinear_equilibrium_manifold_residual
```

Interpretation:

- SGU deterministic state-identity completion passes for the selected support
  equations.
- Two-slice projection closes residuals but is not a predictive filtering
  transition.
- Causal current-control anchors can close residuals, but the best tested
  anchor exceeds the locality gate through future marginal-utility movement.
- No BayesFilter SGU backend/filter code change is currently justified.

## Remaining Gaps And Hypotheses

### Gap 1: Complete Linear Production Spine

BayesFilter has dense/masked Cholesky value and dense/masked QR value, plus
dense QR score/Hessian.  It does not yet have masked QR analytic derivatives or
time-varying QR derivative support.

Hypotheses:

- H1a: masked QR analytic derivatives can reuse the static dummy-row convention
  from the value backend and reduce exactly to dense QR when masks are all
  true.
- H1b: masked all-missing periods should contribute zero score and Hessian
  contribution while advancing prediction derivatives.
- H1c: time-varying LGSSM derivatives are not needed for the first
  MacroFinance switch-over unless the client exposes time-indexed matrices or
  covariances.

### Gap 2: SVD Linear Value And Regularization Diagnostics

SVD/eigen value filtering is still not ported to the TF production path.

Hypotheses:

- H2a: a TF eig/SVD linear value backend can match Cholesky and QR on
  well-conditioned cases.
- H2b: rank-deficient and near-singular cases should report the implemented or
  floored covariance, not silently claim the pre-regularized law.
- H2c: derivative readiness for SVD/eigen factors should be blocked or labeled
  near repeated spectra, active floors, and fallback branches.

### Gap 3: Generic TF Structural Protocols

The legacy structural sigma-point and particle paths are NumPy/reference
oriented.  BayesFilter needs TF-native structural protocols before DSGE
filtering can be consolidated.

Hypotheses:

- H3a: a generic TF structural protocol can represent deterministic completion
  pointwise without encoding DSGE economics.
- H3b: the default structural integration space should be the declared
  innovation space, not a collapsed full-state singular covariance, unless the
  model explicitly requests an approximation label.
- H3c: low-dimensional nonlinear fixtures can provide exact or high-order TF
  references for protocol validation before DSGE switch-over.

### Gap 4: Quadrature, UKF, Cubature, And CUT4-G

BayesFilter needs TF-native quadrature rules and a generic sigma-point filter.

Hypotheses:

- H4a: Cubature and UKF rules can be ported first and validated against scalar
  and affine LGSSM controls.
- H4b: CUT4-G can provide higher-order moment accuracy, but its point count
  `2q + 2**q` must be treated as dimension-bounded, not automatically harmless.
- H4c: XLA/GPU can make large point clouds practical only after static-shape
  memory and compile-time benchmarks pass.

### Gap 5: Generic SVD Sigma-Point And SVD-CUT Value Backend

The DSGE client has SVD sigma-point and experimental CUT/SRUKF code, but
BayesFilter does not yet own a generic TF implementation.

Hypotheses:

- H5a: the generic `ssm=` path from the DSGE client can be ported without DSGE
  economics if BayesFilter owns the protocol and diagnostics.
- H5b: SVD placement should use default `tf.linalg.eigh` with an optional
  robust-eigh plugin boundary, not an unconditional custom-op dependency.
- H5c: SVD-CUT value can be validated on affine models, rank-deficient
  covariance placement, and nonlinear low-dimensional fixtures before any
  derivative claims.

### Gap 6: SVD-CUT Gradient And Hessian

Analytic SVD-CUT derivatives are not yet implemented.

Hypotheses:

- H6a: fixed quadrature-rule moment derivatives are straightforward once the
  factor derivatives are supplied.
- H6b: SVD/eigen first and second derivatives are valid only on smooth
  spectral branches with recorded gaps and inactive floors.
- H6c: full SVD-CUT Hessians should be certified by finite differences,
  TensorFlow autodiff in smooth regions, branch diagnostics, and Hessian
  symmetry checks.

### Gap 7: Compiled, GPU, And HMC Readiness

The current evidence is mostly CPU/static-shape unit testing.  GPU/XLA and HMC
promotion remain downstream.

Hypotheses:

- H7a: QR and SVD linear value/derivative backends can be compiled for fixed
  observation length and fixed state/observation dimensions.
- H7b: SVD sigma-point and CUT filters need explicit compile-time, first-call,
  and steady-state benchmarks before performance claims.
- H7c: HMC should start only after the same model/backend pair has value,
  derivative, and compiled parity evidence.

### Gap 8: Client Switch-Over

MacroFinance and the DSGE client still own local filtering code.

Hypotheses:

- H8a: MacroFinance can switch first for linear value and QR analytic
  derivatives, because BayesFilter now has the closest production spine.
- H8b: DSGE generic nonlinear switch-over should wait for TF structural
  protocols, quadrature, SVD sigma-point value, and deterministic-completion
  metadata.
- H8c: SGU production filtering should remain blocked until the DSGE client
  produces a causal, local, residual-closing value target.

### Gap 9: Legacy NumPy Surface

BayesFilter still contains NumPy implementation modules for historical and
reference paths.

Hypotheses:

- H9a: legacy NumPy modules should remain during migration but must not be the
  production front door.
- H9b: after TF parity and switch-over, legacy modules can move to an archival
  or testing namespace and production source checks can enforce the TF/TFP
  policy.

## Closure Plan

### Phase 0: Evidence Freeze And Scope Guard

Objective:
- freeze the current repo state and protect unrelated dirty files.

Actions:
- record BayesFilter `git status`, latest commit, and branch relation to
  `origin/main`;
- record current untracked sidecars separately;
- run CPU-only full test suite;
- run source hygiene for TF production modules.

Exit gate:
- continue only if baseline tests pass and no unrelated dirty files need to be
  touched.

### Phase 1: Linear Spine Completion

Objective:
- close the remaining linear production gaps before nonlinear work resumes.

Subphases:

1. Masked QR analytic derivatives.
   - Implement score/Hessian for the static dummy-row masked QR backend.
   - Tests: all-true mask equals dense QR; all-missing period gives zero
     likelihood, score, and Hessian contribution; sparse masks match
     solve/covariance references where feasible.
2. Time-varying LGSSM derivative support, only if client switch-over requires
   it.
   - Tests: per-time matrices/covariances match dense references and preserve
     static-shape graph behavior.
3. MacroFinance linear compatibility smoke.
   - Tests: BayesFilter value/score/Hessian match MacroFinance provider
     fixtures without NumPy conversion in production paths.

Stop rule:
- do not start SVD derivatives or nonlinear filtering if masked QR derivatives
  fail and the failure cannot be localized.

### Phase 2: TF SVD Linear Value Backend

Objective:
- add value robustness for singular and near-singular linear Gaussian systems.

Actions:
- port MacroFinance TF SVD dense and masked value backends;
- define the TF eig/SVD factor interface;
- record raw spectrum, floored spectrum, floor count, PSD residual, and
  implemented covariance;
- add derivative-status labels without claiming SVD derivative readiness.

Tests:
- well-conditioned parity with Cholesky and QR;
- rank-deficient process covariance;
- masked missing rows;
- near-singular innovation covariance with explicit regularization metadata;
- no NumPy import or `.numpy()` in production modules.

Stop rule:
- if the implemented covariance cannot be reported precisely, do not expose
  the backend as production.

### Phase 3: TF Structural Protocol And Quadrature

Objective:
- replace legacy NumPy structural references with TF-native generic contracts.

Actions:
- implement `bayesfilter/nonlinear/protocols_tf.py`;
- implement TF cubature and UKF quadrature;
- validate deterministic completion, partition metadata, innovation dimension,
  and observation dimension;
- keep full-state mixed structural filtering blocked unless labeled as an
  approximation.

Tests:
- deterministic identities hold pointwise;
- scalar nonlinear fixtures match analytic references;
- affine LGSSM sigma-point cases match exact Kalman where the rule should be
  exact;
- source hygiene remains TF/TFP-only.

Stop rule:
- do not port DSGE nonlinear filters until the generic protocol passes toy and
  affine controls.

### Phase 4: Generic TF SVD Sigma-Point Filter

Objective:
- consolidate the generic SVD sigma-point infrastructure from the DSGE client.

Actions:
- port SVD factorization and sigma-point prediction/update primitives;
- default to `tf.linalg.eigh`;
- define optional robust-eigh plugin boundary;
- preserve diagnostics for failure code, first failing batch, and spectrum.

Tests:
- scalar LGSSM exact-Kalman parity;
- rank-deficient covariance placement;
- nonlinear observation fixture against TF quadrature reference;
- finite gradients on smooth fixtures;
- static-shape graph reuse.

Stop rule:
- if robust-eigh is required for basic tests, keep the backend experimental
  until the plugin boundary is explicit.

### Phase 5: CUT4-G And SVD-CUT Value

Objective:
- add higher-order deterministic integration without overclaiming scalability.

Actions:
- implement `TFCUT4GRule`;
- add moment tests through the intended CUT order;
- combine CUT4-G with SVD placement in the generic sigma-point filter;
- record point count, state dimension, memory shape, and approximation label.

Tests:
- weights positive and sum to one;
- second and fourth moment identities;
- affine exactness under SVD/eigen placement;
- rank-deficient covariance placement diagnostics;
- CPU static-shape gate before any GPU claim.

Stop rule:
- do not claim GPU practicality until Phase 7 benchmarks pass.

### Phase 6: SVD-CUT Score And Hessian

Objective:
- derive and certify analytic gradients and Hessians for SVD-CUT on smooth
  branches only.

Subphases:

1. Fixed quadrature-rule derivatives with supplied factor derivatives.
2. SVD/eigen first derivatives on simple-spectrum inactive-floor branches.
3. SVD/eigen second derivatives on certified smooth branches.
4. Full SVD-CUT likelihood score/Hessian.

Tests:
- TF finite-difference score/Hessian on small fixtures;
- TensorFlow autodiff parity in smooth regions;
- branch/floor blocked labels near repeated spectra and active floors;
- Hessian symmetry;
- implemented-law metadata.

Stop rule:
- any branch ambiguity or active floor must block derivative certification or
  label the derivative target as the implemented/floored law.

### Phase 7: Compiled And GPU/XLA Gates

Objective:
- prove static-shape compiled behavior and bounded performance claims.

Actions:
- add benchmark JSON schema;
- run CPU-only compile and graph-reuse tests by default;
- run GPU/XLA probes only with escalated sandbox permissions under
  `AGENTS.md`;
- report compile time, first-call time, steady-state time, device, dtype,
  state dimension, observation dimension, point count, and batch/chain count.

Tests:
- eager vs compiled value parity;
- eager vs compiled gradient/Hessian parity where supported;
- no retracing for same static shapes;
- bounded CUT point-count benchmarks.

Stop rule:
- do not start HMC or client GPU claims from finite smoke tests alone.

### Phase 8: Client Switch-Over

Objective:
- replace duplicated filtering code one backend at a time.

MacroFinance order:

1. dense linear TF value;
2. QR analytic score/Hessian;
3. masked linear value and derivatives if needed;
4. SVD value diagnostics;
5. SVD derivatives only after branch gates pass.

DSGE order:

1. generic TF quadrature and SVD sigma-point value;
2. structural partition and deterministic completion metadata;
3. model-specific adapters;
4. SVD-CUT value;
5. derivative/JIT/HMC only after value and compiled parity.

SGU rule:
- no production SGU filtering switch-over until a causal, local,
  residual-closing value target earns `sgu_causal_filtering_target_passed`.

Stop rule:
- switch one backend at a time and require parity before replacing client
  imports.

### Phase 9: Legacy Surface Cleanup

Objective:
- make the TF/TFP implementation the public production surface.

Actions:
- mark old NumPy implementation modules as legacy/reference;
- remove or relocate public NumPy filtering exports after TF replacements pass;
- keep test-side NumPy for simple assertions only;
- add source checks that production filtering imports do not depend on NumPy.

Exit gate:
- production BayesFilter filtering imports resolve to TF/TFP modules or clearly
  labeled testing/reference modules.

## Immediate Next Actions

1. Run a scoped Phase 1A plan/audit/execute cycle for masked QR analytic
   derivatives.
2. In parallel planning notes, decide whether MacroFinance switch-over needs
   time-varying LGSSM derivatives before SVD value work.
3. Keep SGU in the diagnostic lane until the DSGE client derives a causal
   predictive future-marginal-utility law or an augmented-state convention.
4. Do not start SVD-CUT derivative work until SVD linear value diagnostics and
   generic TF sigma-point value gates are complete.
