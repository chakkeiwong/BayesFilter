# BayesFilter V1 P3 Benchmark Refresh Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase P3 / R4 in:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

## Entry Gate

P3 may start only after P1 and P2 have current derivative-status and branch-box
evidence.

## Motivation

Benchmarks should summarize what the nonlinear filters do, but they must not
overclaim exact nonlinear likelihood certification.  This phase refreshes the
nonlinear benchmark artifacts with derivative and branch metadata so later HMC,
GPU/XLA, and integration plans can cite one coherent evidence source.

## Scope

Models:
- Model A as an exact affine Gaussian control;
- Model B on the P2 stable branch box;
- Model C only at the scope justified by P2.

Backends:
- SVD cubature;
- SVD-UKF;
- SVD-CUT4.

## Required Metadata

Each benchmark row must include:
- model and backend;
- point count and polynomial degree;
- value status and score status;
- score branch label;
- finite-score status;
- active-floor and weak-gap counts;
- deterministic residual;
- support residual;
- structural-null diagnostics, if applicable;
- reference type: exact affine, dense projection, finite difference, or
  diagnostic Monte Carlo;
- runtime tier: default CPU, extended CPU, optional GPU, or optional external.

## Primary Gate

P3 passes if:
- benchmark artifacts reproduce from documented commands;
- every score row carries branch metadata;
- exactness claims are limited to the reference type actually used.

## Veto Diagnostics

Stop and ask for direction if:
- dense one-step projection errors are described as exact full nonlinear
  likelihood errors;
- a benchmark requires GPU or external projects by default;
- score metadata is missing for any nonlinear score row;
- results would imply HMC, Hessian, or client readiness without the later
  gates.

## Expected Artifacts

```text
docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-*.json
docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-*.md
docs/plans/bayesfilter-v1-p3-benchmark-refresh-result-2026-05-14.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Chapter 28 may be updated only if the benchmark changes documented claims.

## Continuation Rule

Continue to P4 only if P3 identifies a target whose value, score, and branch
metadata are stable enough for a tiny HMC readiness plan.
