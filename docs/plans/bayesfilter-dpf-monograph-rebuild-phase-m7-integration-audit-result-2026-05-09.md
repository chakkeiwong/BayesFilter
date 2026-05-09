# Phase M7 result: integration and final audit protocol for the DPF monograph rebuild

## Date

2026-05-09

## Purpose

This note defines the final integration checklist and final audit checklist for
the rebuilt DPF monograph block.

## Integration checklist

The final DPF rebuild should not be integrated into `docs/main.tex` until all of
the following are settled.

1. **Final chapter order approved**
   - the actual rebuilt chapter list is fixed;
   - obsolete draft chapter files are either retired or clearly superseded.

2. **Main document integration**
   - `docs/main.tex` reflects the final rebuilt chapter order;
   - the DPF block sits in a position consistent with nonlinear filtering and the
     later HMC chapters.

3. **Bibliography integration**
   - `docs/references.bib` contains the source-audited keys required for the
     rebuilt DPF block;
   - duplicate or conflicting keys are resolved;
   - bibliography entries reflect preferred primary sources rather than only
     secondary or student references.

4. **Source map integration**
   - `docs/source_map.yml` records the lineage of the final DPF chapters;
   - retired draft artifacts remain only as provenance history, not active
     exposition dependencies.

5. **Cross-reference coherence**
   - the final DPF block cross-references nonlinear-filtering, filter-choice,
     HMC, transport/surrogate, and structural-model chapters coherently.

## Final audit checklist

1. no undefined citations after final compile;
2. no undefined references after final compile;
3. no duplicate labels or silently conflicting chapter insertions;
4. no unsupported high-stakes claims such as exactness, unbiasedness,
   convergence, or HMC correctness without explicit source or derivation basis;
5. no reader-facing drift back toward repo commentary or governance language;
6. student-material discussion appears only where mathematically or
   implementation-relevantly justified;
7. the final text reads as a monograph treatment rather than a plan document.

## Completion requirement

The DPF monograph rebuild is not complete merely because plans exist.  It is
complete only after the reader-facing chapters are rewritten, integrated, and
successfully audited against the standards above.

## Audit

This final integration protocol is consistent with the earlier phases and closes
the loop on the planning program.  No missing planning bucket remains obvious at
this level.

## Program-level next phase justified?

Yes.

The planning program is complete enough to support the actual DPF monograph
rewrite and integration work.  The next substantive step after this planning set
is a drafting execution pass governed by the M6 protocol.
