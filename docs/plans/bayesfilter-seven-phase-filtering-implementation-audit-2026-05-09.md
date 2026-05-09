# Audit: Seven-phase BayesFilter filtering implementation plan

## Date

2026-05-09

## Plan Under Audit

```text
docs/plans/bayesfilter-seven-phase-filtering-implementation-plan-2026-05-09.md
```

## Auditor Stance

Pretend the seven-phase plan was written by another developer.  Check whether
it can be executed safely without letting linear compatibility, SVD/eigen
regularization, structural nonlinear protocols, CUT derivatives, GPU/XLA
claims, HMC readiness, or client switch-over outrun their evidence gates.

## Verdict

Approved with a gated automatic execution boundary.

The phase ordering is correct:

1. MacroFinance linear compatibility;
2. SVD/eigen linear value;
3. TF structural nonlinear protocol;
4. SVD sigma-point value;
5. CUT/SVD-CUT value;
6. SVD-CUT derivatives;
7. compiled/GPU/HMC/client switch-over.

The plan correctly makes MacroFinance compatibility the next step before new
backend families.  However, all seven phases should not be executed as one
uninterrupted implementation pass.  The later phases depend on evidence that
Phase 1B has not yet produced, especially whether MacroFinance needs
time-varying LGSSM derivatives before SVD/eigen work begins.

## Required Corrections And Controls

1. Treat Phase 1B as the first executable phase.  It is evidence gathering plus
   compatibility tests, not a client switch-over.
2. Do not edit MacroFinance during Phase 1B.  BayesFilter may inspect
   MacroFinance and borrow fixtures, but compatibility artifacts should live in
   BayesFilter.
3. Do not add MacroFinance as a production dependency.  Any donor fixtures must
   be static fixtures or test-only helpers.
4. Keep NumPy out of production BayesFilter TF modules.  NumPy remains allowed
   in tests and `bayesfilter.testing`.
5. If MacroFinance requires time-varying LGSSM derivatives for first
   switch-over, insert Phase 1C and stop before Phase 2 unless Phase 1C is
   implemented and tested.
6. Do not start Phase 2 until Phase 1B either passes or explicitly states that
   MacroFinance compatibility is not a blocker for SVD/eigen value work.
7. Do not start Phase 3 until Phase 2 value diagnostics are implemented and
   singular/floored implemented-law semantics are tested.
8. Do not start Phase 5 until Phase 4 generic sigma-point value filters pass
   affine and rank-deficient placement tests.
9. Do not start Phase 6 until SVD-CUT value gates pass and smooth-branch
   derivative assumptions are specified.
10. Do not run GPU/CUDA/XLA benchmark claims without escalated sandbox
    permissions and recorded device/shape artifacts.
11. Do not promote HMC readiness until value, derivative, and compiled parity
    pass for the exact target model/backend pair.
12. Keep SGU production filtering blocked until the DSGE client provides
    `sgu_causal_filtering_target_passed`.

## Primary Criteria And Veto Diagnostics

### Phase 1B Primary Criterion

BayesFilter reproduces MacroFinance static linear value and derivative
fixtures for dense and masked observations, or the remaining mismatch is
localized to a documented convention with a clear implementation follow-up.

### Phase 1B Veto Diagnostics

Stop after Phase 1B if:

- MacroFinance fixtures cannot be identified without editing MacroFinance;
- MacroFinance uses time-varying derivative tensors for the first target path;
- dense QR value parity fails and cannot be localized;
- dense or masked QR score/Hessian parity fails and cannot be localized;
- compatibility requires production NumPy or a production MacroFinance
  dependency.

### Phase 2 Primary Criterion

SVD/eigen linear value filtering matches QR/Cholesky on regular cases and
reports implemented-law diagnostics on singular or floored cases.

### Phase 2 Veto Diagnostics

Stop after Phase 2 if:

- the code cannot distinguish raw covariance from implemented/floored
  covariance;
- derivative metadata accidentally implies SVD derivative readiness;
- regular cases fail Cholesky/QR parity.

### Phase 3 Primary Criterion

Generic TF structural protocols reproduce affine LGSSM controls and represent
deterministic completion pointwise.

### Phase 3 Veto Diagnostics

Stop after Phase 3 if:

- DSGE-specific economics leak into BayesFilter production protocols;
- deterministic completion can only be represented as a hidden singular
  covariance artifact;
- affine controls fail against the linear backend.

### Phase 4 Primary Criterion

Generic UKF/cubature value filters with SVD/eigen placement pass moment,
affine, rank-deficient placement, and static-shape tests.

### Phase 4 Veto Diagnostics

Stop after Phase 4 if:

- point placement leaves the implemented support;
- affine exactness fails;
- same-shape graph reuse fails.

### Phase 5 Primary Criterion

CUT/SVD-CUT value filters pass documented moment identities, affine controls,
rank-deficient support tests, and point-count diagnostics.

### Phase 5 Veto Diagnostics

Stop after Phase 5 if:

- CUT4-G point construction fails moment identities;
- point count is too large for target dimensions without a benchmark plan;
- SVD placement conflicts with CUT rule geometry.

### Phase 6 Primary Criterion

SVD-CUT score/Hessian passes finite-difference/autodiff parity on smooth
branches and fails closed under invalid spectral/regularization branches.

### Phase 6 Veto Diagnostics

Stop after Phase 6 if:

- repeated spectra or active floors cannot be detected;
- Hessian symmetry fails;
- derivative target metadata claims the raw law under regularization.

### Phase 7 Primary Criterion

Compiled/GPU/HMC/client switch-over claims are backed by exact backend/model
parity artifacts and rollback boundaries.

### Phase 7 Veto Diagnostics

Stop if:

- eager and compiled results disagree;
- GPU benchmarks were run without escalated permissions;
- HMC is requested before value, derivative, and compiled parity all exist.

## Executable Boundary For This Pass

Proceed automatically through Phase 1B.

Continue to Phase 2 only if Phase 1B passes and explicitly records that no
Phase 1C time-varying derivative implementation is required before SVD/eigen
value work.

If Phase 1B discovers that time-varying derivatives are required, stop and ask
for direction before Phase 2.

## Audit Outcome

The seven-phase program is sound as a master program.  The first executable
gate is Phase 1B MacroFinance linear compatibility.  Later phases remain
valid, but each must be justified by the previous phase's primary criterion
and veto diagnostics.
