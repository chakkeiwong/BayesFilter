# Audit: SGU residual-source, pruned-state, and projection diagnostics plan

## Audit Role

This is a second-developer audit of
`docs/plans/bayesfilter-sgu-residual-source-pruned-projection-plan-2026-05-07.md`.
The audit checks whether the plan can test the stated hypotheses without
turning diagnostics into unsupported promotion claims.

## Verdict

Approved for bounded execution.  The plan is careful enough to run because it
separates three targets that are easy to confuse:

1. perturbation-policy residual comparison;
2. pruned-state residual comparison;
3. joint nonlinear state-control projection.

The most important stop rule is that a successful joint projection does not
resurrect the failed Gate B label.  It creates a new candidate target only.

## Required Clarifications

1. H1 should compare policies on the same state-identity-completed total-state
   grid.  Otherwise per-equation attribution is confounded by state support.
2. H2 should preserve the stochastic first-order path and complete only
   deterministic coordinates in either the total side-state or a documented
   first-plus-side decomposition.
3. H3 must report adjustment sizes.  A residual-zero projection is not useful
   if it moves controls or deterministic states nonlocally.
4. The exact nonlinear blocker should remain in tests and docs:

```text
blocked_nonlinear_equilibrium_manifold_residual
```

5. The result note should explicitly state whether BayesFilter changes are
   justified.  The expected answer is no unless a generic metadata field is
   missing.
6. All TensorFlow-backed diagnostics should run CPU-only unless deliberately
   probing GPU.  This avoids conflating CUDA sandbox behavior with residual
   evidence.

## Missing Points Closed By Audit

- Add explicit blocker labels:

```text
blocked_sgu_pruned_state_comparison_not_residual_improving
blocked_sgu_projection_is_new_target_not_gate_b
```

- Add a result note in `/home/chakwong/python/docs/plans` even if all
  hypotheses fail.  Negative evidence matters here.
- Update both reset memos with phase-by-phase execution outcomes and next-phase
  justification.

## Execution Stop Rules

Stop before BayesFilter adapter/backend changes, derivative certification,
compiled parity, or HMC work if any of the following holds:

```text
pruned_quadratic_rms >= linear_rms
pruned_quadratic_max > linear_max
```

or:

```text
joint projection closes residuals but is a new target
```

The second condition is not a failure; it is a scope boundary.

## Audit Outcome

The plan has no blocking omissions after the clarifications above.  Execute it
as a diagnostic evidence pass, not a production implementation pass.
