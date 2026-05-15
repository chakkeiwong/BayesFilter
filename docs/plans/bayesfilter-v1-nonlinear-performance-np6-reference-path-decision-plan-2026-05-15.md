# BayesFilter V1 Nonlinear Performance NP6 Reference Path Decision Plan

## Date

2026-05-15

## Governing Master Program

This plan executes Phase NP6 in:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
```

## Purpose

Decide whether the NumPy reference nonlinear filters should remain
reference-only or receive separate TensorFlow performance backends.

## Entry Gate

NP6 may start after NP0 classifies reference-only paths and after NP1-NP5
identify whether a concrete compiled/GPU use case remains unmet by production
TensorFlow nonlinear paths.

## Evidence Contract

Question:

- Is a TensorFlow rewrite justified for `StructuralSVDSigmaPointFilter` or
  `particle_filter_log_likelihood`?

Baseline:

- NP0 inventory and NP1-NP5 evidence.

Primary criterion:

- A rewrite is allowed only if a concrete downstream use needs compiled/GPU
  behavior not already covered by production TensorFlow paths.

Veto diagnostics:

- reference semantics are replaced by an unvalidated fast path;
- a rewrite is proposed only for generic performance curiosity;
- particle-filter randomness/resampling policy lacks a separate evidence
  contract;
- NumPy reference status is hidden from users.

Explanatory diagnostics only:

- current eager NumPy runtime;
- user-facing convenience arguments;
- possible future GPU usefulness.

What will not be concluded:

- particle-filter correctness under TensorFlow;
- differentiable resampling readiness;
- HMC readiness;
- production default changes.

Artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-YYYY-MM-DD.md
```

## Decision Branches

Allowed decisions:

- defer and document reference-only status;
- plan a separate TensorFlow bootstrap particle backend;
- plan a separate TensorFlow reference sigma-point backend only if it differs
  from existing production paths in a necessary way.

Any implementation branch must create a new plan before code changes.

## Continuation Rule

Continue to NP7 after recording the reference-path decision and any required
future subplans.  Do not start reference-path implementation inside NP6.
