# Audit: structural SVD six-blocker closure plan

## Date

2026-05-06

## Plan reviewed

- `docs/plans/bayesfilter-structural-svd-six-blocker-closure-plan-2026-05-06.md`

## Audit stance

Pretending to be another developer, I audited the six-blocker plan for
dependency order, missing gates, ownership boundaries, and claim discipline.

## Findings

1. The plan correctly starts with model residuals rather than derivative, JIT,
   or HMC work.  This is the right scientific dependency: an HMC target cannot
   be made correct by sampler diagnostics if the structural completion map is
   wrong or missing.
2. The plan correctly separates the three DSGE blockers:
   - Rotemberg second-order/pruned `dy` identity;
   - SGU nonlinear equilibrium residuals;
   - EZ timing/partition provenance.
3. The plan correctly keeps BayesFilter generic.  DSGE model equations and
   timing remain client-owned in `/home/chakwong/python`.
4. The plan correctly treats derivative/Hessian certification as a separate
   blocker from value correctness.  Existing BayesFilter guardrails are not the
   same as certified SVD/eigen gradients.
5. The plan correctly treats compiled static-shape parity as separate from
   derivative correctness.
6. The plan correctly blocks HMC until model residuals, derivative safety, and
   compiled parity are all closed for the same model/backend pair.

## Missing points found

No blocking omission was found.  Two execution clarifications are required:

- If client-side strong residual tests pass by asserting blocker labels, that is
  a successful blocker evaluation, not a model promotion.
- If Blockers 1--3 remain blocked, then Blockers 4--6 should be evaluated only
  as BayesFilter guardrail/provenance gates.  Do not implement derivative,
  compiled, or HMC backends for models whose residual gates are blocked.

## Decision

Approve the plan for execution with conservative stop rules.  Continue through
all blockers as evaluations.  Promote only evidence that actually passes; record
blocked gates explicitly.
