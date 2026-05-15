# BayesFilter V1 P8 External Integration Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase P8 / R9 in:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

## Entry Gate

P8 starts only after local V1 gates stabilize:
- focused and full CPU tests pass;
- P1 derivative matrix is current;
- P2/P3 nonlinear branch and benchmark claims are current;
- optional GPU/HMC/reference claims are either documented or explicitly
  deferred.

## Motivation

MacroFinance and DSGE are important clients, but V1 should stay independently
testable and should not create production coupling too early.  This phase
prepares an integration plan; it does not switch clients by default.

## Required Decisions

The integration plan must decide:
- which client target goes first;
- whether the bridge is test-only or production;
- what API is frozen enough to call from the client;
- what parity tests are required;
- what rollback path exists;
- who owns client-side changes.

## Primary Gate

P8 passes if the integration plan can be reviewed without requiring changes to
external source trees.

## Veto Diagnostics

Stop and ask for direction if:
- BayesFilter imports MacroFinance or DSGE as a production dependency;
- external source is modified from the V1 lane;
- SGU economics are promoted without reopening the structural/DSGE lane;
- optional live checks are described as default CI.

## Expected Artifacts

```text
docs/plans/bayesfilter-v1-p8-external-integration-result-2026-05-14.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```
