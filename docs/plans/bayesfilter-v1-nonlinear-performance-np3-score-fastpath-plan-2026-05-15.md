# BayesFilter V1 Nonlinear Performance NP3 Score Fast-Path Plan

## Date

2026-05-15

## Governing Master Program

This plan executes Phase NP3 in:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
```

## Purpose

Optimize certified analytic score paths without weakening branch-certification
contracts or changing derivative semantics.

## Entry Gate

NP3 may start only after NP1 records score-path benchmark baselines for cells
with certified branch behavior.  Model C rows must preserve the structural
fixed-support branch where applicable.

## Evidence Contract

Question:

- Can analytic score runtime be reduced while preserving score values,
  branch diagnostics, and derivative derivation scope?

Baseline:

- NP1 score benchmark rows and existing finite-difference/reference evidence.

Primary criterion:

- An optimized score path passes branch diagnostics and score parity, cites a
  proof obligation or derivation artifact, and improves a predeclared score
  shape class.

Material worsening rule:

- Unless a phase-specific result note predeclares a stricter rule, a required
  row is materially worse if median steady-state wall time regresses by more
  than 10 percent or process RSS increases by more than 10 percent relative to
  the NP1 baseline for the same shape/mode/device.
- If timing noise or memory measurement uncertainty makes the 10 percent rule
  ambiguous, the row is explanatory only and cannot support promotion.

Veto diagnostics:

- branch assertions are removed rather than moved behind an explicit precheck;
- Model C structural fixed-support behavior is weakened;
- parameter-axis vectorization lacks a proof obligation;
- parity is used as the only derivative correctness evidence;
- Hessian readiness is implied.

Explanatory diagnostics only:

- speedup on a single tiny parameter dimension;
- branch-grid timing without precheck artifact;
- score rows without reference/finite-difference context.

What will not be concluded:

- nonlinear Hessian readiness;
- HMC readiness or convergence;
- default score backend policy;
- GPU score speedup.

Artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-proof-obligation-YYYY-MM-DD.md
docs/benchmarks/bayesfilter-v1-nonlinear-performance-score-fastpath-*.json
```

## Candidate Changes

Candidates:

- vectorize the parameter-axis loop for Kalman-gain derivatives;
- reuse eigensystem solves where algebraically identical;
- add a score-only fast path for callers that do not need full diagnostics;
- separate branch precheck from steady-state timing while recording the
  branch artifact used by each timing row.

## Mathematical Pre-Gate

Before production implementation:

1. Name the derivative expression preserved by the transformation.
2. Cite the existing derivation or write a proof-obligation artifact.
3. State tensor shapes and parameter-axis conventions.
4. Prove equality between the old scalar-parameter loop and the proposed
   batched/vectorized expression.
5. Record the proof artifact path in the NP3 result run manifest.

## Required Tests

Minimum focused tests for accepted code changes:

- score parity against existing implementation;
- value parity where score path returns value;
- finite-difference/reference score check for the affected model/backend cells;
- branch diagnostics before timing;
- graph parity;
- XLA parity only for supported cells from NP4.

## Continuation Rule

Continue to NP4 only if accepted score-path changes carry proof artifacts,
branch diagnostics, and parity evidence.

If no score change is accepted, record a deferral result and continue to NP4
only with current score paths.
