# Phase M6 result: drafting-execution and equation-audit protocol

## Date

2026-05-09

## Purpose

This note defines the execution protocol for the eventual reader-facing rewrite
of the DPF monograph block.

## Core conclusion

The drafting phase must be governed by a chapter-by-chapter derivation and claim
ledger, not by freeform prose drafting.  The previous failed DPF pass shows that
writing too early, without a derivation-first discipline, leads to commentary
instead of monograph mathematics.

## Required execution cycle per chapter

Every rebuilt DPF chapter should follow:

```text
source map -> derivation map -> draft -> tool audit -> revise -> compile -> claim audit
```

This should be executed chapter by chapter, not only at the end of the entire
block.

## Per-chapter checklist

For every final DPF chapter:

1. section outline completed;
2. source list completed;
3. theorem/equation obligations identified;
4. derivation ownership marked as:
   - derive fully,
   - source and adapt,
   - compare variants,
   - human review required;
5. first full draft written;
6. MathDevMCP run on bounded formulas;
7. ResearchAssistant support recorded for literature-heavy claims;
8. implementation implications stated mathematically, not as repo commentary;
9. compile pass completed;
10. claim audit completed.

## Required audit artifacts during drafting

### A. equation-audit note

A short per-drafting-pass note listing:
- equations audited with MathDevMCP;
- equations not yet certified by tool support;
- equations requiring manual derivation checks.

### B. claim-audit note

A short pass note listing:
- source-backed claims;
- approximation claims;
- unresolved or human-review-required claims;
- any student-report disagreement still unresolved.

### C. replacement map

A running note stating which old DPF draft chapter text is retired or replaced,
so the rewrite does not accidentally preserve weak prose out of inertia.

## Tidy-up requirements

At the end of each drafting subphase:
- remove or rewrite repo-facing file commentary from reader-facing chapter prose;
- keep only provenance-relevant material in plans and source-map notes;
- verify that implementation implications remain mathematical rather than casual;
- ensure that student-material references appear only where they sharpen coverage
  or mathematical comparison.

## Exit gate for drafting phase

The drafting phase may be considered complete only when:
- every load-bearing claim has an explicit evidence status;
- the rebuilt chapter block compiles;
- the exposition reads as a mathematical monograph treatment rather than an
  internal memo;
- the HMC assessment remains tied to the rung-by-rung target analysis.

## Audit

The execution protocol is now explicit enough to prevent repeating the main
failure mode of the prior DPF pass.  No blocker prevents the final integration
planning phase.

## Next phase justified?

Yes.

Phase M7 is justified because the drafting protocol is now explicit enough to
support a final integration and audit checklist.
