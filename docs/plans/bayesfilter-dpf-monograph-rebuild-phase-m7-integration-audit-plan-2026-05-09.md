# Phase M7 plan: DPF monograph integration, bibliography, and final audit

## Date

2026-05-09

## Purpose

Plan the final integration pass for the rebuilt DPF monograph block.

## Scope

- insert the final chapter set into `docs/main.tex`;
- update `docs/references.bib` conservatively with source-audited entries;
- update `docs/source_map.yml` with the new DPF chapter lineage;
- run compilation, citation, and cross-reference checks;
- write final audit/result notes for the monograph rebuild.

## Required output

Produce an integration checklist and final audit checklist.

## Integration checklist

- final chapter order approved;
- all renamed/split/replaced chapter files reflected in `docs/main.tex`;
- old draft artifacts either removed from the chapter graph or explicitly
  retired;
- bibliography keys verified and conflicts resolved;
- source map updated with imported source lineage and status;
- adjacent chapters checked for cross-reference coherence.

## Final audit checklist

- no undefined citations or references after final compile;
- no unsupported production/HMC claims introduced by prose tightening;
- mathematical comparisons remain source-backed;
- student-derived observations appear only where they sharpen mathematical or
  implementation comparison;
- the final text reads as a monograph treatment, not an internal memo.

## Exit gate

The rebuild is complete only when the final chapter block compiles cleanly and a
final audit note can defend the mathematical rigor, self-containment, and HMC
claim discipline of the result.
