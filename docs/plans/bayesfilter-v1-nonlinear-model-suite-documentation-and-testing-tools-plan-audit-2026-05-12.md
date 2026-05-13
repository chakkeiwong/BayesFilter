# Audit: BayesFilter V1 Nonlinear Model Suite Documentation And Testing Tools Plan

## Date

2026-05-12

## Plan Audited

```text
docs/plans/bayesfilter-v1-nonlinear-model-suite-documentation-and-testing-tools-plan-2026-05-12.md
```

## Audit Stance

Pretending to be a separate developer, I approve the plan for execution after
tightening.  The tightened version correctly keeps this phase on reusable
benchmark models and test oracles rather than drifting into analytic gradients,
HMC, GPU/XLA, or client switch-over.

## Required Tightenings Applied

1. The plan no longer describes angle residual handling as a first-rung goal.
   Bearings-only and radar tracking remain deferred.
2. The write lane now includes `docs/references.bib` only for accepted
   benchmark citations and removes Chapter 18 from this subplan's active
   documentation lane.
3. Model C now explicitly acknowledges the production interface issue: the
   canonical nonlinear growth law is time-inhomogeneous, while the current
   structural TF transition has no explicit time argument.  The accepted
   testing fixture therefore uses an autonomous deterministic phase coordinate.
4. Dense quadrature is labeled as a moment-matched Gaussian projection oracle
   for sigma-point tests, not as the exact nonlinear likelihood.
5. SMC reference implementation is out of scope for this first pass.

## Remaining Risks

R1. The univariate nonlinear growth equation is standard but the local
research-assistant index has no paper summary.  The execution pass must record
external DOI/source verification and avoid overclaiming exact parameter
settings if the full primary text is unavailable.

R2. The dense tensor-product quadrature oracle is suitable for tiny dimensions
only.  The execution pass must keep it under `bayesfilter/testing` and avoid
presenting it as a production reference filter.

R3. Model C's autonomous phase-state embedding tests BayesFilter mechanics but
is not identical to adding a general time-inhomogeneous transition API.  The
documentation must say this plainly.

R4. Adding bibliography entries touches a shared documentation file.  The pass
must keep changes minimal and register them in the source map.

## Execution Approval

Approved with the following stop rules:
- stop if Models A-C cannot be built without production filter API changes;
- stop if source evidence for Model C is too weak to document the canonical
  law;
- stop if tests require MacroFinance, DSGE, GPU, HMC, or broad SMC code;
- stop if out-of-lane files must be staged to commit.
