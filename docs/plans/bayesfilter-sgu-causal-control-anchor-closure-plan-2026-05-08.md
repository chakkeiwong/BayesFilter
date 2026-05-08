# Plan: SGU causal current-control anchor closure

## Question

Can a timing-consistent current-control anchor make SGU causal projection a
production filtering target, or does causal SGU remain blocked even after
alternative anchors are tested?

## Motivation

The prior SGU passes established:

- state identities close;
- full quadratic residual failure is driven by volatility correction effects
  in `H[3]` through `H[6]`;
- symmetric-shock expectation averaging does not rescue the quadratic gate;
- two-slice projection closes residuals but is not a predictive filtering
  transition;
- causal projection with the full second-order current-control anchor fails
  residual and locality gates.

The remaining plausible causal route is to change the current-control anchor
without violating timing.  This plan tests whether a first-order,
quadratic-without-constant, or current-period static-FOC anchor can support a
causal projection that is both residual-closing and local.

## Scope

- Owner: DSGE client `/home/chakwong/python`.
- BayesFilter role: planning, provenance, adapter-guard verification only.
- Runtime: targeted SGU contract tests under one minute.
- Execution mode: CPU-only diagnostics with `DSGE_FORCE_CPU=1` and
  `CUDA_VISIBLE_DEVICES=-1`.
- Non-goals: no BayesFilter backend/filter code changes, no derivative/JIT,
  no HMC, no EZ determinacy branch.

## Hypotheses

H1. The causal residual obstruction is the full second-order current-control
anchor, not state identity completion.

H2. A timing-consistent current-control anchor can reduce residuals, but may
still fail locality if next-control moves are too large.

H3. A current-period static-FOC anchor can satisfy same-period static
equations, but it still may not produce a local predictive transition.

H4. If every timing-consistent anchor fails residual or locality gates, SGU
production remains state-identity-only and two-slice diagnostic-only.

## Anchors To Test

1. `second_order`: existing full second-order policy with `g_ss`.
2. `quadratic_without_constant`: quadratic policy with `g_ss` removed.
3. `first_order`: first-order policy.
4. `static_foc`: solve current controls against current/static equations while
   holding the state-identity next state and second-order next controls fixed.

All anchors must be timing-consistent: they may use `x_t`, the model solution,
and the state-identity-completed next state where the SGU current static
equations already depend on `kfu_t`, but they must not use a two-slice
least-squares adjustment to move current controls after seeing future residual
closure.

## Success Criteria

A causal anchor earns:

```text
sgu_causal_control_anchor_projection_passed
```

only if, on the declared default grid:

```text
projected_max <= 1e-7
projected_rms <= 1e-8
next_control_adjustment_norm <= 2e-2
deterministic_state_adjustment_norm <= 1e-2
current_control_anchor_error <= 1e-10
max_nfev <= 500
```

If residuals close but locality fails, the label remains blocked:

```text
blocked_sgu_causal_anchor_projection_nonlocal
```

If residuals do not close, preserve:

```text
blocked_sgu_causal_projection_residual_not_closed
```

## Phases

### Phase 0: plan, audit, and reset memo

- Record the existing H1-H6 commit evidence.
- Audit this plan as a second developer.
- Update the BayesFilter and DSGE reset memos before code edits.

### Phase 1: implement anchor diagnostics

- Add a focused client helper/test module for alternative current-control
  anchors.
- Do not change BayesFilter code.
- Reuse existing `SGUEstimable.project_sgu_state_control(...)` semantics where
  possible, but allow tests to call local anchor-specific causal projections
  if that is less invasive.

### Phase 2: run anchor tests

- Test each anchor on the default grid.
- Report residual RMS/max, next-control adjustment norm, deterministic-state
  adjustment norm, and max function evaluations.

### Phase 3: decide SGU target status

- If one anchor passes residual and locality gates, plan a separate production
  metadata pass.
- If anchors close residuals but are nonlocal, record that causal production
  remains blocked for locality.
- If anchors fail residual closure, record causal production blocked for
  residuals.

### Phase 4: BayesFilter and downstream gate

- Run BayesFilter DSGE adapter guard.
- If no causal anchor passes, keep BayesFilter backend/filter, derivative/JIT,
  and HMC blocked.

### Phase 5: final verification and commit

- Run focused client tests and targeted SGU structural tests.
- Run BayesFilter YAML and diff checks.
- Commit scoped DSGE and BayesFilter files only.

## Interpretation Rule

- Passing residuals alone is not enough; locality is required for production.
- Static-FOC current controls may become a future derivation target only if
  they reduce residuals with local next-control moves.
- If the best anchor is residual-closing but nonlocal, the next mathematical
  question is why next controls must move so far, not whether BayesFilter can
  estimate the model today.
