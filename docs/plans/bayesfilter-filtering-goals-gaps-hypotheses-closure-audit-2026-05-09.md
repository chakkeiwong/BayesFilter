# Audit: BayesFilter filtering goals/gaps closure plan

## Date

2026-05-09

## Plan Under Audit

`docs/plans/bayesfilter-filtering-goals-gaps-hypotheses-closure-plan-2026-05-09.md`

## Auditor Stance

Pretend the closure plan was written by another developer.  Check whether the
plan can be executed safely without mixing generic BayesFilter backend work,
client-owned DSGE economics, SVD derivative claims, GPU performance claims, or
HMC promotion.

## Verdict

Approved with a narrower automatic execution boundary.

The plan correctly separates two lanes:

- generic TF/TFP BayesFilter backend completion;
- model-specific structural evidence owned by the DSGE client and
  MacroFinance.

The plan also correctly states that generic backend work may continue while
SGU remains blocked from production filtering.  However, the plan is too broad
to execute end-to-end in one automatic pass.  SVD value, structural protocols,
generic SVD sigma-point filtering, CUT4-G, SVD-CUT derivatives, GPU/XLA gates,
and client switch-over all have additional donor and evidence dependencies.

## Required Corrections And Controls

1. Treat Phase 0 as mandatory before coding.  Record the exact branch state,
   dirty files, baseline tests, and source-hygiene status.
2. Split Phase 1 into subphases.  Phase 1A is masked QR analytic derivatives.
   Time-varying derivatives and MacroFinance switch-over smoke tests should be
   separate later gates.
3. Do not begin SVD value work unless Phase 1A passes.  SVD value is a separate
   production backend with regularization metadata, not a derivative shortcut.
4. Do not begin nonlinear structural protocols, UKF/CUT, or SVD sigma-point
   work before the linear spine is stable.
5. Do not begin SVD-CUT score/Hessian until SVD value and generic sigma-point
   value pass.
6. Do not run GPU/XLA benchmarks without escalated sandbox permissions.
7. Do not edit MacroFinance or `/home/chakwong/python` in this BayesFilter pass.
8. Keep the Windows `:Zone.Identifier` sidecars untracked and out of commits.
9. Keep production TF modules free of NumPy imports and `.numpy()` calls.

## Executable Boundary For This Pass

Proceed automatically through:

1. Phase 0: evidence freeze and scope guard.
2. Phase 1A: masked QR analytic derivatives.

Stop after Phase 1A unless the user explicitly asks to continue to Phase 1B or
Phase 2.  The reason is that Phase 1B depends on client requirements for
time-varying matrices, and Phase 2 starts a new SVD backend family with
separate regularization semantics.

## Phase 1A Acceptance Criteria

Masked QR analytic derivatives must satisfy:

- all-true mask equals dense QR likelihood, score, and Hessian;
- sparse masks match an independent masked autodiff reference on a smooth
  small fixture;
- an all-missing period contributes exactly zero likelihood, zero score, and
  zero Hessian contribution while still advancing state prediction
  derivatives;
- Hessian is symmetric;
- static-shape graph reuse passes for same-shaped masks;
- production module source check finds no NumPy import and no `.numpy()` call.

## Stop Rules

Stop and ask for direction if:

- baseline tests fail before coding;
- masked QR value semantics cannot be matched;
- masked QR derivative formulas disagree with autodiff and the error cannot be
  localized;
- the implementation would require NumPy in production modules;
- TensorFlow graph behavior requires dropping static-shape guarantees.

## Audit Outcome

The closure plan is sound as a master plan.  The safe automatic implementation
scope for this run is Phase 0 plus Phase 1A only.
