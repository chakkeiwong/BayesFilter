# Plan: BayesFilter v1 DSGE Read-only Target Inventory

## Date

2026-05-10

## Purpose

This plan defines a future read-only inventory of DSGE compatibility targets.
It does not edit `/home/chakwong/python`.  It keeps DSGE economics in the DSGE
project and treats BayesFilter as a generic filtering library.

## Scope

Read-only inventory candidates:

- toy affine controls;
- existing UKF/SVD examples;
- CUTSRUKF experimental examples;
- non-SGU nonlinear fixtures with observation data;
- SGU candidates only as diagnostic or blocked targets until locality passes.

Out of scope:

- DSGE source edits;
- SGU solver changes;
- DSGE production default switch-over;
- BayesFilter production imports from DSGE.

## Classification Labels

```text
ready_external_compatibility_fixture
needs_test_only_bridge
blocked_by_model_law
diagnostic_only
blocked_sgu_causal_locality
```

## Inventory Questions

For each candidate:

1. What are the state, shock, observation, and deterministic-completion blocks?
2. Is the one-step transition causal and local?
3. Are observations and reference filter outputs available?
4. Is the stochastic integration rank small enough for sigma-point/CUT rules?
5. Are deterministic residual diagnostics available?
6. Does the target require hidden regularization?
7. Can the target be represented without BayesFilter importing DSGE economics?

## SGU Stop Rule

SGU production filtering remains blocked unless a DSGE-owned test proves:

```text
sgu_causal_filtering_target_passed
```

Residual-closing two-slice or future-marginal-utility repairs remain diagnostic
or smoother candidates, not one-step predictive filtering targets.

## Output Artifact

Future read-only inventory should produce:

```text
docs/plans/bayesfilter-v1-dsge-readonly-target-inventory-result-YYYY-MM-DD.md
```

Do not run this inventory in the current pass unless explicitly requested.
