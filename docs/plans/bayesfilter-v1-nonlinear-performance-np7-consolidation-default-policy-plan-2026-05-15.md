# BayesFilter V1 Nonlinear Performance NP7 Consolidation Default Policy Plan

## Date

2026-05-15

## Governing Master Program

This plan executes Phase NP7 in:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
```

## Purpose

Consolidate NP0-NP6 evidence into accepted optimizations, optional paths,
deferred work, and default-policy decisions.

## Entry Gate

NP7 may start only after each prior phase has either:

- a result artifact; or
- a structured blocker/deferral note.

## Evidence Contract

Question:

- What nonlinear performance policy is justified by NP0-NP6?

Baseline:

- Original implementation and benchmark artifacts before the performance
  program.

Primary criterion:

- Every accepted change has engineering correctness, numerical/branch, compile,
  and shape-specific timing evidence.

Veto diagnostics:

- optional paths are promoted to defaults without medium-shape evidence;
- timing evidence promotes correctness or numerical validity by itself;
- GPU results are generalized beyond tested shapes;
- unsupported XLA cells are hidden;
- Hessian, HMC, or exact likelihood claims are implied.

Explanatory diagnostics only:

- tiny-shape speedups;
- first-call timing;
- memory notes without allocator-level attribution;
- dense projection errors for Models B-C.

What will not be concluded:

- production deployment readiness outside BayesFilter;
- HMC convergence;
- nonlinear Hessian readiness;
- exact nonlinear likelihood quality for Models B-C.

Artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-final-summary-YYYY-MM-DD.md
```

## Required Ledgers

Engineering correctness ledger:

- API compatibility;
- result-container behavior;
- eager/graph/XLA parity;
- test status;
- source-map/reset-memo status.

Numerical validity ledger:

- value and score branch status;
- finite-difference/reference evidence;
- floor/eigen diagnostics;
- deterministic/support residuals;
- Monte Carlo uncertainty where applicable.

Performance ledger:

- compile time;
- steady-state time;
- memory notes;
- CPU/GPU trust labels;
- shape-specific comparisons;
- non-claims.

Evidence from one ledger cannot promote another ledger without the checks for
that ledger.

## Required Decision Table

Columns:

- decision;
- affected function/backend;
- accepted, optional, deferred, or blocked;
- correctness status;
- numerical/branch status;
- XLA status;
- CPU/GPU status;
- primary timing result;
- memory status;
- default-policy result;
- uncertainty;
- next justified action;
- what is not concluded.

## Post-Run Red-Team Note

The final summary must include:

- strongest alternative explanation for any accepted speedup;
- what result would overturn the default-policy decision;
- weakest part of the evidence;
- whether the uncertainty belongs to engineering correctness, numerical
  validity, or performance evidence.

## Continuation Rule

NP7 closes the program if the final summary has no unresolved mandatory cells.
If a mandatory cell is blocked, NP7 must produce a repair plan rather than a
release or default-policy claim.
