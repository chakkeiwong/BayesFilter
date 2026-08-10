# Curated promotion note for migrating the Fable rewrite into canonical `docs/`

- **Date:** 2026-08-06
- **Source branch:** `docs/fable-rewrite/monograph/`
- **Target branch:** canonical `docs/`
- **Purpose:** record what should be promoted from the standalone rewrite after the replacement-verdict-driven repair pass, and what should not be copied wholesale.

## Promotion decision

The standalone rewrite branch is the correct migration base. The canonical `docs/` tree should not remain the primary development version because it is unbuildable and materially worse on several confirmed mathematical, target-definition, citation, label, and policy defects.

However, promotion should be **curated**, not a directory replacement. The rewrite tree still contains standalone-snapshot packaging and generated state that should not be copied wholesale.

## What to promote

### Core monograph source files
Promote the reviewed diffs from the active rewrite root into canonical `docs/` for:

- changed active chapters in `docs/fable-rewrite/monograph/chapters/`
- changed active appendices in `docs/fable-rewrite/monograph/appendices/`
- `docs/fable-rewrite/monograph/references.bib`
- any required include-path changes in `docs/fable-rewrite/monograph/main.tex` only if they are source-semantic rather than standalone-path packaging

### Required figure assets
Promote only the figure assets needed for canonical build closure, specifically the five PNGs that replace the missing canonical PDF figure inputs for the SSL-LSTM visual-validation figures.

Those should be placed at a stable canonical asset location under `docs/` rather than by copying the full standalone `plans/artifacts` subtree unchanged.

### Specific rewrite gains to preserve
The following content improvements should be retained during promotion:
- corrected Reader Map;
- corrected finite-fallback target wording;
- corrected HMC exact-endpoint-MH vs surrogate/wrong-value distinction;
- repaired PF no-resampling algorithm and proof-object wording;
- corrected LEDH centered-information offset;
- narrowed GenUT attribution to a local variant where appropriate;
- corrected structural-UKF comparison row;
- active-root label/citation cleanup;
- quadrature worked-example additions;
- stop-gradient partial-derivative wording;
- canonical Contract E / chunk-policy / tuning-identity binding;
- squared-TT scaling/orientation/interface repairs made in the replacement-directed pass;
- explicit nonclaims where theorem-level source support remains open.

## What not to promote wholesale
Do **not** copy these directly into canonical `docs/` as part of promotion:

- generated LaTeX build products:
  - `main.aux`, `main.bbl`, `main.blg`, `main.fdb_latexmk`, `main.fls`, `main.log`, `main.out`, `main.toc`
- the standalone built PDF itself as a source artifact
- `.mathdevmcp/latex_index.json`
- copied experiment-artifact trees not actually required by the canonical LaTeX build
- inactive predecessor chapter snapshot files that are present only because the standalone tree was created as a bounded repair snapshot

## Promotion gate
Before applying the curated diff into canonical `docs/`, require:

1. canonical build closure after asset migration,
2. no undefined citations,
3. no undefined references,
4. no duplicate active labels,
5. no live provisional bibliography records in the promoted canonical source,
6. one final bounded independent review of the repaired blocker set,
7. a short migration result note recording:
   - promoted files,
   - preserved historical artifacts,
   - remaining explicit nonclaims,
   - exact canonical build command.

## Historical artifacts to preserve, not treat as live source
Keep as historical evidence:
- old canonical review/audit notes,
- standalone rewrite audit notes,
- old canonical PDF,
- standalone rewrite PDF,
- recovery, freeze, and verdict notes.

These are evidence records, not canonical monograph source.

## Final instruction
Promote the rewrite by **file diff selection**, not by copying `docs/fable-rewrite/monograph/` into place. The target outcome is a canonical `docs/` tree that inherits the rewrite's mathematical and mechanical gains without importing standalone-snapshot debris.
