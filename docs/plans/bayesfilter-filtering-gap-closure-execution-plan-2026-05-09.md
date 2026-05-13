# Plan: BayesFilter filtering gap-closure execution

## Date

2026-05-09

## Purpose

This plan converts the current goals, remaining gaps, and hypotheses into an
execution sequence.  It is the operational companion to:

```text
docs/plans/bayesfilter-filtering-goals-gaps-hypotheses-closure-plan-2026-05-09.md
docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-2026-05-06.md
docs/plans/bayesfilter-sgu-production-target-gap-closure-plan-2026-05-09.md
```

The central split is:

- BayesFilter can continue generic TF/TFP filtering backend work.
- SGU production filtering remains blocked until the DSGE client derives a
  causal, local, residual-closing value target.

## Goals To Achieve

### Goal 1: Complete the generic TF/TFP linear filtering spine

BayesFilter should own production TensorFlow implementations for linear
Gaussian filtering, likelihood, score, and Hessian paths without relying on
client-local filtering code.

Done:
- dense Cholesky value;
- masked Cholesky value;
- dense QR value;
- masked QR value;
- dense QR score/Hessian;
- non-production covariance/solve derivative references in
  `bayesfilter.testing`.

Still needed:
- masked QR score/Hessian;
- optional time-varying LGSSM derivatives if client switch-over requires them;
- MacroFinance compatibility smoke tests.

### Goal 2: Add robust linear SVD/eigen value filtering

BayesFilter needs a TF SVD/eigen value backend for singular and near-singular
linear systems with explicit implemented-law diagnostics.

Done:
- Cholesky/QR production paths can cover regular linear cases.

Still needed:
- TF SVD/eigen dense and masked value backend;
- spectrum, floor, PSD residual, and implemented covariance diagnostics;
- derivative-status blockers near repeated spectra or active floors.

### Goal 3: Build generic TF structural filtering protocols

BayesFilter should represent structural nonlinear state-space models in a
generic way, including deterministic completion and declared innovation-space
integration, without encoding DSGE economics.

Done:
- BayesFilter metadata can represent current DSGE adapter status.
- SGU state-identity support evidence exists in the DSGE client.

Still needed:
- TF structural protocol interfaces;
- TF cubature and UKF rules;
- low-dimensional nonlinear fixtures;
- affine LGSSM exactness controls.

### Goal 4: Port generic TF SVD sigma-point and CUT value filters

BayesFilter should own generic SVD sigma-point and SVD-CUT value filters before
claiming nonlinear HMC readiness.

Done:
- DSGE client has useful generic SVD sigma-point source material.

Still needed:
- TF SVD factor placement with diagnostics;
- generic sigma-point prediction/update primitives;
- CUT4-G rule and moment tests;
- rank-deficient covariance placement tests;
- static-shape value gates.

### Goal 5: Certify SVD-CUT derivatives only after value gates

Analytic gradients and Hessians for SVD-CUT should be certified only on smooth
spectral branches with recorded validity regions.

Still needed:
- fixed quadrature-rule derivatives;
- SVD/eigen derivative branch diagnostics;
- finite-difference and TensorFlow autodiff parity;
- Hessian symmetry and implemented-law checks.

### Goal 6: Open compiled/GPU/HMC gates only in order

Compiled and GPU evidence should certify fixed model/backend pairs.  HMC must
wait for value, derivative, and compiled parity evidence.

Still needed:
- CPU compile and graph-reuse tests;
- escalated GPU/XLA benchmarks only after static-shape gates;
- HMC plans only after value/derivative/compiled gates pass.

### Goal 7: Switch clients over one backend at a time

MacroFinance should switch first for linear value and QR derivatives.  DSGE
nonlinear switch-over should wait for generic TF structural/SVD value gates.

Still needed:
- MacroFinance parity smoke;
- DSGE generic nonlinear parity only after protocols and sigma-point value
  filters pass;
- no SGU production switch-over until `sgu_causal_filtering_target_passed`.

## Current Blocking Facts

### SGU production filtering

Allowed evidence:

```text
sgu_state_identity_completion_passed
sgu_two_slice_projection_diagnostic_passed
sgu_causal_control_anchor_residual_closure_diagnostic_passed
```

Active blockers:

```text
blocked_sgu_causal_projection_residual_not_closed
blocked_sgu_causal_anchor_projection_nonlocal
blocked_sgu_two_slice_projection_not_filter_transition
blocked_sgu_projection_is_new_target_not_gate_b
blocked_sgu_expectation_averaging_does_not_rescue_quadratic_gate
blocked_sgu_quadratic_policy_not_better_than_linear_residual
blocked_nonlinear_equilibrium_manifold_residual
```

Key mechanism:
- the `static_foc` anchor closes residuals, but the closure is dominated by
  future marginal-utility movement above the locality gate.

Conclusion:
- SGU remains in the model-owned derivation lane, not the BayesFilter
  production filtering lane.

## Remaining Gaps And Hypotheses

### Gap A: masked QR analytic derivatives

Hypotheses:
- H-A1: static dummy-row masking can reuse dense QR derivative recursions.
- H-A2: all-missing rows contribute zero likelihood, score, and Hessian
  increments while prediction derivatives still advance.
- H-A3: all-true masks reduce exactly to dense QR.

### Gap B: SVD/eigen linear value diagnostics

Hypotheses:
- H-B1: TF eig/SVD value filtering matches Cholesky/QR on well-conditioned
  cases.
- H-B2: rank-deficient cases must report the implemented/floored covariance.
- H-B3: derivative claims must be blocked near repeated spectra and active
  floors.

### Gap C: generic structural protocols

Hypotheses:
- H-C1: deterministic completion can be represented pointwise as a generic
  model contract.
- H-C2: integration should occur in declared innovation space by default.
- H-C3: toy nonlinear fixtures can certify the protocol before DSGE switch-over.

### Gap D: sigma-point/CUT value filters

Hypotheses:
- H-D1: DSGE generic SVD sigma-point source can be ported without DSGE
  economics.
- H-D2: CUT4-G should be dimension-bounded and labeled by point count.
- H-D3: affine exactness and rank-deficient placement are sufficient first
  value gates.

### Gap E: SVD-CUT derivatives

Hypotheses:
- H-E1: fixed quadrature derivatives are simpler than factor derivatives and
  should be certified first.
- H-E2: SVD/eigen derivatives are valid only on smooth branches.
- H-E3: active floors certify the implemented law only, not the raw law.

### Gap F: SGU production value target

Hypotheses:
- H-F1: future marginal-utility timing is the active SGU causal blocker.
- H-F2: an augmented predictive state or source-backed expectation term may
  repair locality.
- H-F3: if no derivation repairs locality, SGU remains diagnostic-only beyond
  state-identity completion.

### Gap G: compiled/GPU/HMC/client switch-over

Hypotheses:
- H-G1: compile parity is meaningful only after eager value/derivative gates.
- H-G2: GPU performance claims need explicit escalated benchmark artifacts.
- H-G3: client switch-over should replace one backend at a time.

## Execution Plan

### Phase 0: evidence freeze and audit

Actions:
- record current `git status` and latest commits;
- run CPU-only BayesFilter suite or scoped backend suite;
- run source hygiene for production TF modules;
- audit this plan for premature HMC, SGU, or derivative promotion.

Exit gate:
- continue only if baseline tests pass and dirty files are scoped.

### Phase 1A: masked QR analytic derivatives

Actions:
- implement or complete masked QR score/Hessian support;
- test all-true mask equals dense QR;
- test all-missing row contributes zero likelihood/score/Hessian increment;
- compare sparse masks against existing references where feasible.

Exit gate:
- masked QR derivatives pass before SVD or nonlinear work resumes.

### Phase 1B: MacroFinance linear compatibility

Actions:
- run value/score/Hessian parity against MacroFinance provider fixtures;
- decide whether time-varying LGSSM derivatives are required before
  switch-over.

Exit gate:
- switch-over plan names exact supported backends and unsupported cases.

### Phase 2: TF SVD/eigen linear value backend

Actions:
- add SVD/eigen dense and masked value backends;
- report spectrum/floor/implemented covariance diagnostics;
- add derivative-status blocked labels.

Exit gate:
- Cholesky/QR parity passes on regular cases and singular cases report the
  implemented law.

### Phase 3: TF structural protocol and quadrature

Actions:
- add TF structural protocol definitions;
- add cubature and UKF rules;
- validate deterministic completion and innovation-space integration on toy
  fixtures.

Exit gate:
- generic protocol passes before DSGE nonlinear switch-over begins.

### Phase 4: generic TF SVD sigma-point value

Actions:
- port generic SVD sigma-point prediction/update primitives;
- define optional robust-eigh plugin boundary;
- test affine exactness, rank-deficient placement, diagnostics, and static
  graph reuse.

Exit gate:
- value gate passes before CUT or derivative claims.

### Phase 5: CUT4-G and SVD-CUT value

Actions:
- implement CUT4-G rule;
- test weight and moment identities;
- combine CUT4-G with SVD placement;
- record dimension, point count, memory shape, and approximation label.

Exit gate:
- CPU static-shape value tests pass before GPU or derivative claims.

### Phase 6: SVD-CUT score/Hessian

Actions:
- certify quadrature derivatives;
- certify SVD/eigen derivatives only on smooth branches;
- add finite-difference, autodiff, and Hessian symmetry gates.

Exit gate:
- derivatives earn explicit validity-region labels or remain blocked.

### Phase 7: compiled/GPU gates

Actions:
- add eager/compiled parity tests;
- run GPU/XLA only with escalated permissions when needed;
- record compile time, first-call time, steady-state time, dimensions, point
  count, dtype, and device.

Exit gate:
- no HMC until compiled value/derivative parity exists for the target.

### Phase 8: client switch-over

Actions:
- switch MacroFinance linear backends first;
- switch DSGE generic nonlinear paths only after Phase 3--5 value gates;
- keep SGU production filtering blocked until the DSGE client earns:

```text
sgu_causal_filtering_target_passed
```

Exit gate:
- each client switch-over has parity tests and rollback boundaries.

### Phase 9: legacy NumPy cleanup

Actions:
- mark old NumPy modules as reference/legacy;
- move non-production code to testing/reference namespaces;
- enforce TF/TFP production import hygiene.

Exit gate:
- public production imports resolve to TF/TFP modules or explicitly labeled
  reference modules.

## Immediate Next Step

Start with Phase 1A: masked QR analytic derivatives.  In parallel, the DSGE
client may pursue SGU marginal-utility timing derivation, but that work should
not block generic BayesFilter backend completion.
