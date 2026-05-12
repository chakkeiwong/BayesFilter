# Phase E6 result: literature-to-debugging crosswalk

## Date

2026-05-12

## Purpose

This note records the E6 phase of the DPF monograph enrichment round: converting
the enriched DPF chapters into a practical debugging reference.

## Plan audit

Pretending to be another developer, the E6 plan was audited before execution.
The audit concluded that the plan stayed in the correct monograph lane but
needed one tightening: the crosswalk should be embedded in the monograph itself,
not only recorded as a planning artifact.

Plan audit note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e6-plan-audit-2026-05-12.md`

## Execution

Reader-facing chapter added:
- `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`

Main book integration:
- added `\input{chapters/ch19f_dpf_debugging_crosswalk}` to `docs/main.tex`
  after the DPF HMC target chapter.

The chapter adds:
- a diagnostic order for localizing DPF failures in the computation graph;
- the required literature-to-debugging crosswalk table;
- an implementation-facing issue taxonomy for future coding agents;
- an explicit boundary statement that the chapter routes debugging hypotheses
  but does not validate any implementation.

## Coverage

The crosswalk covers:
- particle degeneracy / ESS collapse;
- flow stiffness;
- EDH/LEDH approximation mismatch;
- PF-PF weight and Jacobian/log-det mismatch;
- soft-resampling bias and zero-gradient problems;
- OT regularization and finite-iteration Sinkhorn issues;
- learned-OT extrapolation and residual issues;
- DPF-HMC target mismatch;
- value-object versus gradient-object mismatch.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed
  from `docs/`.
- After converting the crosswalk from a page-sized float to a `longtable`,
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed
  again.
- `git diff --check` passed.
- Focused log scan found no undefined citations, no undefined references, no
  duplicate Hyperref destination warnings, no rerun requests, no `Float too
  large` diagnostics, and no LaTeX errors.  The only match to the scan pattern
  was the package name `rerunfilecheck`.
- Text audit confirmed the required issue categories and the coding-agent issue
  taxonomy are present.

## Audit

Primary criterion: satisfied.

The chapter gives a future implementation agent a concrete route from observed
failure to mathematical layer, chapter section, source cluster, and next
diagnostic.

Veto diagnostics:
- crosswalk too abstract: cleared by row-level diagnostics;
- issue-to-literature mapping vague: cleared by chapter labels and citations;
- artifact not useful for implementation diagnosis: cleared by the coding-agent
  issue taxonomy and diagnostic order.

## Next phase justified?

Yes.

Phase E7 is justified because the DPF block now contains the enriched chapters
and the reader-facing debugging crosswalk needed for a final round-level audit.
