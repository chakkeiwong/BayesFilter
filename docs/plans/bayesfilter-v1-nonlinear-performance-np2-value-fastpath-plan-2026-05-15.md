# BayesFilter V1 Nonlinear Performance NP2 Value Fast-Path Plan

## Date

2026-05-15

## Governing Master Program

This plan executes Phase NP2 in:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
```

## Purpose

Optimize value-only TensorFlow nonlinear filters while preserving high-level
wrapper behavior, diagnostics, and numerical semantics.

## Entry Gate

NP2 may start only after NP1 provides value-path benchmark baselines with
parity, branch, shape, and timing metadata.

## Evidence Contract

Question:

- Which value-path changes improve runtime or memory without changing
  log-likelihood, filtered-state outputs, or diagnostic meaning?

Baseline:

- NP1 value benchmark rows before optimization.

Primary criterion:

- A candidate value fast path preserves eager/graph/XLA parity and improves a
  predeclared shape class without materially worsening required rows.

Material worsening rule:

- Unless a phase-specific result note predeclares a stricter rule, a required
  row is materially worse if median steady-state wall time regresses by more
  than 10 percent or process RSS increases by more than 10 percent relative to
  the NP1 baseline for the same shape/mode/device.
- If timing noise or memory measurement uncertainty makes the 10 percent rule
  ambiguous, the row is explanatory only and cannot support promotion.

Veto diagnostics:

- high-level wrapper diagnostics disappear or change meaning;
- algebraic rewrites lack a derivation artifact;
- branch or floor behavior changes without an explicit accepted proof gate;
- XLA support regresses for a cell claimed as supported;
- benchmark-only code becomes a hidden production dependency.

Explanatory diagnostics only:

- first-call timing;
- RSS deltas;
- tiny-shape improvement without small-shape confirmation.

What will not be concluded:

- default backend change;
- score-path speedup;
- GPU speedup;
- exact nonlinear likelihood quality.

Artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-derivation-YYYY-MM-DD.md
docs/benchmarks/bayesfilter-v1-nonlinear-performance-value-fastpath-*.json
```

## Allowed Candidate Changes

Low-risk candidates:

- tensor-only log-likelihood helper beneath existing wrappers;
- diagnostics-level option that keeps full wrapper diagnostics available;
- sigma-point rule precomputation at call sites;
- compiled-friendly filtered-state storage;
- benchmark-only variants behind explicit experimental labels.

Higher-risk candidates requiring derivation:

- block-factor alternatives to full augmented eigendecomposition;
- replacing full innovation precision with narrower solves;
- covariance update algebra changes;
- `tf.while_loop` or `tf.scan` rewrites that alter trace structure.

## Mathematical Pre-Gate

Before a higher-risk candidate can be implemented for production:

1. Write a derivation artifact in project notation.
2. State the exact original expression and proposed expression.
3. State assumptions on symmetry, flooring, positive definiteness, and solve
   exactness.
4. State executable assertions or branch preconditions.
5. Record the derivation artifact path in the NP2 result run manifest.

If the derivation is absent or inconclusive, the candidate may only appear in
an experimental benchmark branch with no promotion claim.

## Required Tests

Minimum focused tests for accepted code changes:

- eager versus existing implementation parity;
- graph parity;
- XLA parity for NP4-supported cells, if the path claims XLA support;
- `return_filtered=False` and `return_filtered=True` behavior where touched;
- diagnostics-level behavior for high-level wrappers.

## Continuation Rule

Continue to NP3 or NP4 only if every accepted value-path change has a result
artifact, parity tests, benchmark evidence, and non-implication text.

If no value fast path is accepted, record a deferral result and continue to
NP3/NP4 only if benchmark baselines remain usable.
