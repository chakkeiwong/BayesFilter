# Audit: SGU causal current-control anchor closure plan

## Audit Role

This is a second-developer audit of
`docs/plans/bayesfilter-sgu-causal-control-anchor-closure-plan-2026-05-08.md`.
The audit checks whether the plan closes the remaining causal SGU hypothesis
without reopening already blocked BayesFilter, derivative/JIT, or HMC work.

## Verdict

Approved for bounded execution.

The plan targets the correct remaining gap: after the H1-H6 DSGE pass,
expectation averaging, two-slice projection, and downstream promotion gates are
already decided.  The untested route is whether a timing-consistent
current-control anchor can make causal projection both residual-closing and
local.

## Required Clarifications

1. Residual closure is not enough.  The plan must require locality:

```text
next_control_adjustment_norm <= 2e-2
deterministic_state_adjustment_norm <= 1e-2
```

2. The static-FOC anchor must be treated as a diagnostic unless a derivation
shows that its use of the completed next state is valid in the predictive
transition.

3. Tests must keep the existing blocked labels active when an anchor is
nonlocal:

```text
blocked_sgu_causal_anchor_projection_nonlocal
blocked_sgu_causal_projection_residual_not_closed
```

4. BayesFilter code must not change unless an anchor passes both residual and
locality gates.  The expected outcome is still no BayesFilter code change.

5. HMC, derivative, and compiled work must remain blocked if the best anchor is
nonlocal.

## Missing Points Closed By Audit

- Add a dedicated result note in the DSGE client even if all anchors fail
  production gates.
- Record whether residual-closing anchors fail due to locality rather than
  residuals.
- Update both reset memos with phase-by-phase interpretation.

## Execution Stop Rules

Stop before BayesFilter backend/filter work, derivative/JIT, or HMC if:

```text
best_anchor_projected_max > 1e-7
```

or:

```text
best_anchor_next_control_adjustment_norm > 2e-2
```

## Audit Outcome

The plan has no blocking omissions after these clarifications.  Execute it as
a diagnostic closure pass, not a production integration pass.
