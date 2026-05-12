# Audit: BayesFilter V1 Nonlinear Derivative-Provider Gap Closure Plan

## Date

2026-05-13

## Audit Role

This is a second-developer audit of
`docs/plans/bayesfilter-v1-nonlinear-derivative-provider-gap-closure-plan-2026-05-13.md`.

## Verdict

Approved for scoped execution with the tightened gates in the plan.

## Missing-Point Check

The plan covers the important gaps left by the previous nonlinear passes:

- explicit derivative providers for Models B--C;
- score finite-difference tests against the implemented value filters;
- branch summaries over small parameter boxes;
- default Model C deterministic-degeneracy behavior;
- documentation and reset-memo provenance;
- no premature Hessian, GPU/XLA, HMC, MacroFinance, DSGE, or Chapter 18b work.

## Ambiguity Tightening

The main ambiguity is Model C.  The default fixture has a deterministic phase
coordinate with zero initial variance.  The current analytic SVD score branch
is explicitly a smooth simple-spectrum/no-active-floor branch, so it should not
be silently promoted to the default degenerate Model C law.  The plan resolves
this by:

- adding the derivative provider;
- testing score correctness on a nondegenerate phase-state variant;
- requiring the default zero-phase-variance fixture to remain blocked until a
  future fixed-null-direction derivative policy is derived and tested.

This is not drift.  The smooth variant is a derivative-provider test fixture,
while the default value benchmark remains unchanged.

## Risks

R1. Finite-difference tolerances may need to be looser than the affine test
because nonlinear filters propagate approximation and factorization
sensitivities over multiple time steps.

Mitigation:
- use centered finite differences and compare to the same implemented value
  backend, not to a dense oracle or exact likelihood.

R2. Model C may expose the current branch limitation more strongly than Model B.

Mitigation:
- count that as a successful diagnostic if the default degenerate fixture is
  blocked by active floors and the smooth variant passes.

R3. Adding derivative helpers to `bayesfilter/testing` could be confused with
production API expansion.

Mitigation:
- keep the helpers in testing fixtures and document them as validation tools.

## Stop Rules

Stop and ask for direction if:

- Model B score finite-difference parity fails after formula audit;
- smooth Model C cannot pass without active floors or weak spectral gaps;
- default Model C unexpectedly passes while still labeled as
  smooth/no-active-floor;
- any required action crosses into MacroFinance, DSGE, Chapter 18b, structural
  SVD/SGU plans, or the shared monograph reset memo.
