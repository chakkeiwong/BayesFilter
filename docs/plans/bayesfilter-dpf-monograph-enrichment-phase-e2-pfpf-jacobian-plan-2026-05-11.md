# Phase E2 plan: PF-PF and Jacobian/log-det enrichment

## Date

2026-05-11

## Purpose

Deepen the PF-PF / proposal-correction chapter so that the change-of-variables
logic, corrected-weight construction, and Jacobian/log-determinant machinery are
monograph-grade and implementation-useful.

## Target chapter

- `docs/chapters/ch19c_dpf_implementation_literature.tex`

## Main questions

1. Exactly what density does the transported particle have?
2. What does the PF-PF correction restore mathematically?
3. What remains approximate even after proposal correction?
4. What should an implementation compute to remain faithful to the intended
   target?

## Required enrichment topics

- fuller change-of-variables derivation;
- clearer distinction between pre-flow proposal, post-flow proposal, and target;
- Jacobian matrix evolution and log-determinant reduction;
- comparison of EDH/PF and LEDH/PF beyond verbal contrast;
- more explicit discussion of finite-particle, discretization, and numerical
  Jacobian issues.

## Required source lanes

- PF-PF / invertible particle-flow literature;
- CIP LEDH/PF-PF material;
- student materials only as support for implementation issue identification.

## Required outputs

1. derivation ledger for the PF-PF correction formulas;
2. source-backed explanation of what proposal correction restores;
3. implementation issue map for Jacobian/log-det evaluation;
4. stronger exact/unbiased/approximate interpretation note.
5. explicit note on how Jacobian/log-det mistakes would manifest in
   implementation outputs or diagnostics.

## Required tables

### Table E2-1: PF-PF quantity map

| Quantity | Mathematical role | Source support | Implementation burden |
| --- | --- | --- | --- |

### Table E2-2: restored vs. not restored

| Aspect | Restored by PF-PF correction? | Why / why not |
| --- | --- | --- |

### Table E2-3: debugging relevance map

| Failure symptom | Likely mathematical source | Relevant literature | Suggested diagnostic |
| --- | --- | --- | --- |

### Table E2-4: exact identity versus practical approximation

| Component | Exact identity? | Approximation source | Observable failure symptom |
| --- | --- | --- | --- |

## Primary criterion

The chapter must make the proposal-correction story explicit enough that a
reader can no longer confuse raw flow transport with a corrected particle method.

## Veto diagnostics

Do not proceed if:
- the change-of-variables argument is still only sketched;
- the chapter still sounds as if PF-PF makes the nonlinear filter exact in
  general;
- the practical Jacobian/log-det burden is still unclear.

## Exit gate

Proceed to E3 only after the PF-PF chapter can serve as a mathematically clear
reference for implementation and later HMC interpretation.
