# Phase P9 plan: debugging crosswalk into verification contract

## Date

2026-05-14

## Target chapter

- `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`

## Governing prerequisites and lane guard

- Required prior results: P0, P1, P2, and passed P3-P8 results.
- P2 established bibliography-spine support only; diagnostics may cite source
  families for provenance but must not use citations as validation evidence.
- Allowed write set: this target chapter and the P9 result artifact.  Touch
  shared files only if the result records a necessary citation/label reason.
- Before editing, record branch, `git status --short`, out-of-lane dirty files,
  and this write set.

## Purpose

Turn the short debugging crosswalk into an equation-indexed verification
contract.  A coding agent or reviewer should be able to use the chapter to
design tests and interpret failures without guessing which mathematical layer is
responsible.

## Required implementation instructions

1. Add an equation-to-test matrix for:
   - filtering recursion;
   - bootstrap likelihood estimator;
   - homotopy derivative;
   - EDH/LEDH endpoint;
   - PF-PF weight;
   - log-determinant ODE;
   - categorical discontinuity;
   - soft-resampling bias;
   - EOT/Sinkhorn marginal constraints;
   - barycentric projection;
   - learned-map residual;
   - HMC value-gradient contract.
2. Define minimal controlled examples:
   - linear-Gaussian recovery;
   - synthetic affine flow;
   - scalar nonlinear observation;
   - two-particle soft-resampling bias;
   - small Sinkhorn problem with known marginals;
   - permutation-equivariance learned-map test;
   - finite-difference HMC gradient test.
3. For each diagnostic, state:
   - required inputs;
   - expected output;
   - tolerated numerical error;
   - failure interpretation;
   - what the failure does not prove.
4. Add promotion thresholds:
   - exploratory implementation;
   - credible research implementation;
   - bank-facing research claim;
   - production/governance claim.
5. Link each diagnostic to:
   - chapter section;
   - equation label;
   - source family;
   - implementation quantity.

## Required source use

- All revised DPF chapters.
- Source-role notes from P2.
- Implementation implications from P4-P8.
- Because P2 found no local ResearchAssistant summaries, source use is
  bibliography-spine provenance unless reviewed source evidence is added later.
  Diagnostics must be equation- and claim-local.

## Audit rules

- Every diagnostic must be tied to an equation or claim.
- Do not recommend tuning before target/value-gradient checks.
- Do not let learned-map diagnostics stand in for teacher-map diagnostics.
- Do not claim a test validates more than it actually checks.

## Required local tests/checks

- Confirm all referenced equation labels exist.
- Confirm no diagnostic refers to a method not defined in earlier chapters.
- Search for `validate` and ensure the scope is narrow.
- Check table widths and PDF readability after insertion.
- Create a P9 result artifact with an actual completion date in the filename.
- Record a derivation/diagnostic-obligation table and mark each obligation as
  `MathDevMCP`, `manual`, or `blocked`, with reasons for any manual fallback.
- Run the established LaTeX build and record DPF-local warnings.

## Veto diagnostics

The phase fails if:

- diagnostics remain qualitative;
- equations are not linked to tests;
- failure interpretations are overbroad;
- target-status distinctions disappear from implementation advice;
- the crosswalk remains too short to guide a coding agent.
- the phase edits or stages student-baseline files;
- bibliography-spine source support is described as ResearchAssistant-reviewed;
- MathDevMCP/manual derivation-obligation evidence is omitted;
- build or table-readability status is omitted.

## Exit gate

The chapter is ready only if it can serve as a concrete verification checklist
for DPF implementation and reviewer audit.
