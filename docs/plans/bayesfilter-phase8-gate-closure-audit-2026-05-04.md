# Audit: Phase 8 executable gate closure

## Scope

This audit reviews
`docs/plans/bayesfilter-phase8-gate-closure-plan-2026-05-04.md` before
execution.

## Findings

### Completeness

The plan covers all four hypotheses recorded by the prior reset memo:

- large-scale dense/masked derivative readiness;
- cross-currency derivative coverage and oracle discrepancy readiness;
- production exposure fail-closed readiness;
- HMC chain diagnostics beyond target readiness.

### Risk Review

The largest risk is accidental promotion.  The gate helpers must not turn
metadata presence into readiness.  They should be explicitly conservative:
missing support, missing rows, failed oracle checks, blockers, weak
identification, divergences, low ESS, and high R-hat all keep readiness false.

The second risk is hidden implementation migration.  The plan correctly keeps
BayesFilter responsible for metadata normalization and gate decisions only.
MacroFinance remains responsible for numerical likelihood, derivative, oracle,
production, TensorFlow, and sampler implementations.

The third risk is overclaiming HMC convergence.  The diagnostic gate may report
that thresholds pass, but it must preserve the distinction between target
readiness and sampler evidence.

### Required Amendments

Execution should ensure:

1. Gate result objects are immutable dataclasses.
2. Gate helpers return explicit blocker/reason tuples.
3. Sparse large-scale panels fail closed unless masked derivative support is
   declared for the requested order.
4. Cross-currency coverage checks compare coverage rows against provider
   parameter names.
5. Oracle checks are bounded and caller-supplied so BayesFilter does not import
   MacroFinance oracle logic.
6. Production exposure requires final identification evidence, not merely weak
   or fixture evidence.
7. HMC diagnostics require target readiness and do not infer convergence from
   finite target checks alone.
8. The pre-existing dirty chapter file remains unstaged.

## Audit Decision

No blocking issue. Execute phase by phase with reset-memo updates after each
phase.
