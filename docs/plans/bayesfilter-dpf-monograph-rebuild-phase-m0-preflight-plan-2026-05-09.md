# Phase M0 plan: DPF monograph rebuild preflight and supersession audit

## Date

2026-05-09

## Purpose

This phase establishes the starting state for the DPF monograph rebuild.  It
records why the previous pass is insufficient, which existing documents remain
governing inputs, which ones are provisional or superseded for reader-facing
purposes, and what source lanes must be prepared before the literature survey
begins.

## Questions answered by this phase

1. Which existing BayesFilter DPF planning and chapter artifacts remain useful?
2. Which artifacts should no longer govern reader-facing monograph prose?
3. What exact source families and tool workflows are required for the rebuild?
4. What is the clean dependency path for the remaining phases?

## Inputs

- `docs/differentiable-particle-filter-program.md`
- `docs/plans/bayesfilter-differentiable-particle-filter-phase1-audit-plan-2026-05-08.md`
- `docs/plans/bayesfilter-differentiable-particle-filter-governing-and-phase1-review-2026-05-08.md`
- `docs/plans/bayesfilter-student-dpf-baseline-consolidation-plan-2026-05-08.md`
- the current DPF draft chapters under `docs/chapters/`
- relevant CIP monograph chapters

## Required output

Produce a short audit note that classifies each existing DPF artifact as one of:

- governing input;
- source/provenance input;
- experimental comparison input;
- reader-facing draft to rewrite heavily;
- superseded for exposition but retained for provenance.

## Audit points

- Confirm that the earlier DPF chapter pass failed primarily on exposition
  quality rather than on source provenance.
- Confirm which mathematical scope decisions remain valid:
  EDH -> EDH/PF -> soft/OT -> LEDH/neural OT.
- Confirm that the student lanes remain critique/coverage inputs rather than
  monograph authorities.

## Exit gate

Proceed only once the project has an explicit supersession map so later phases do
not accidentally inherit the wrong prose standard.
