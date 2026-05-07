# Audit: SGU joint state-control projection design plan

## Audit Role

This is an independent second-developer audit of
`docs/plans/bayesfilter-sgu-joint-state-control-projection-design-plan-2026-05-07.md`.
The audit checks whether the plan can be executed without converting a local
least-squares residual closure into an unsupported filtering, derivative,
compiled, or HMC claim.

## Verdict

Approved for bounded execution.

The plan is correctly scoped because it separates three targets that are easy
to confuse:

1. state-identity completion over deterministic SGU coordinates;
2. causal predictive projection with current controls fixed at the policy
   value for the current state;
3. two-slice/offline projection that may adjust current controls and therefore
   cannot automatically be a filtering transition.

The most important safety requirement is that a successful two-slice solve
must preserve the blocker:

```text
blocked_sgu_two_slice_projection_not_filter_transition
```

unless a separate timing derivation justifies current-control adjustment inside
the predictive filtering recursion.

## Required Clarifications

1. The causal projection must keep both:

```text
y_t = policy(x_t)
x'_S = (a', zeta', mu')
```

2. The two-slice projection must include an explicit selection rule, even if
   the regularization weight is small.  Otherwise the 20-unknown,
   15-residual system is underidentified.
3. Conditioning diagnostics must refer to the pure canonical-residual
   Jacobian, not only the augmented regularized objective.
4. Stress-grid failure should narrow the validity region or block production
   promotion.  It should not invalidate a local diagnostic result if the result
   is labeled narrowly.
5. BayesFilter code changes are justified only if a causal projection passes
   and existing metadata cannot represent the target honestly.
6. The existing failed labels must remain blocked:

```text
sgu_quadratic_policy_residual_improvement_passed
sgu_combined_structural_approximation_target_passed
blocked_nonlinear_equilibrium_manifold_residual
```

## Missing Points Closed By Audit

- Add an explicit causal-failure label if causal mode does not close residuals:

```text
blocked_sgu_causal_projection_residual_not_closed
```

- Add an explicit two-slice diagnostic label only if locality and residual
  thresholds pass:

```text
sgu_two_slice_projection_diagnostic_passed
```

- Keep the two-slice filtering blocker paired with any two-slice pass:

```text
blocked_sgu_two_slice_projection_not_filter_transition
```

- The result note must report whether BayesFilter was changed.  The expected
  answer is no unless the causal target passes.

## Execution Stop Rules

Stop before BayesFilter backend/filter changes, derivative certification,
compiled parity, or HMC validation if either condition holds:

```text
causal_projected_max_residual > 1e-7
```

or:

```text
causal_current_control_anchor_error > 1e-10
```

If two-slice projection passes while causal projection fails, continue only
through diagnostic documentation, conditioning/stress characterization, reset
memo updates, and provenance.  Do not start production integration.

## Audit Outcome

The plan has no blocking omissions after the clarifications above.  Execute it
as a target-classification and diagnostic-hardening pass, not as a production
BayesFilter or HMC implementation pass.
