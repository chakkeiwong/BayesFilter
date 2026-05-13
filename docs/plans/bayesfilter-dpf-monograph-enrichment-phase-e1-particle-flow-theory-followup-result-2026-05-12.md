# Phase E1 follow-up result: particle-flow chapter deepening pass

## Date

2026-05-12

## Purpose

This note records the direct chapter-deepening pass required after the initial
E1 execution found that the particle-flow chapter was still too compressed to
meet the enrichment standard.

## Execution

Deepened `docs/chapters/ch19b_dpf_literature_survey.tex` substantially in the
following ways:

1. **Homotopy section expanded**
   - strengthened the explanation of endpoint exactness;
   - emphasized the role of the normalizing constant;
   - made the debugging implication of an incorrectly normalized homotopy
     explicit.

2. **EDH section expanded**
   - clarified where the exact conservation law ends and where Gaussian closure
     begins;
   - made the closure approximation itself more explicit;
   - added a clearer implementation/debugging reading of the EDH construction.

3. **LEDH section expanded**
   - added the local linearization equation explicitly;
   - added the local information-vector construction explicitly;
   - clarified the added approximation layer and the new numerical fragilities.

4. **Stiffness section expanded**
   - moved stiffness from a short warning to a richer debugging-oriented
     discussion;
   - connected small observation noise, local precision, and unstable explicit
     integration more directly to implementation symptoms.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- keyword audit confirmed that EDH, LEDH, Gaussian closure, exact special case,
  and stiffness are now present in a more substantial form.

## Audit

### Re-check against the original E1 veto diagnostics

- **EDH too compressed**: cleared.
- **LEDH too compressed**: cleared.
- **stiffness too short**: cleared.

### Primary criterion

Satisfied after the deepening pass.

The chapter is still part of a larger enrichment round, but it is now materially
closer to the intended monograph standard and no longer fails the three specific
E1 veto diagnostics that blocked progress.

## Interpretation

This follow-up confirms the main value of the enrichment-round discipline.  The
first E1 pass correctly stopped the workflow before a too-brief chapter could be
treated as good enough.  The direct deepening pass now gives the chapter the
kind of derivation depth and debugging relevance it lacked.

## Next phase justified?

Yes.

Phase E2 is now justified because the particle-flow chapter has been deepened
enough that the PF-PF chapter can build on a stronger mathematical foundation
rather than on a compressed summary.
