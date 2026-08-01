# Contract E Canonical Migration Phase 1 Scientific Review, Iteration 2

Date: 2026-07-13

Reviewer: fresh bounded Codex substitute after Claude repository disclosure was
platform-blocked. Read-only scope was the repaired normative specification and
numerical/statistical design freeze.

## Material Findings

1. Bonferroni coverage was described as exactly 95% under the marginal Student-t
   assumptions. Bonferroni guarantees at least 95%; equality generally does not
   hold for dependent statistics.
2. The FD selection rule left the coarsest estimate without a truncation
   estimate, assumed an exact 2:1 ratio after representability adjustment, and
   did not define the middle of an even-cardinality run.

## Verdict

`VERDICT: REVISE`

## Repair Disposition

The design now says at least 95% under the stated assumptions. It admits only
symmetric representable endpoints, uses actual adjacent-step ratios in the
Richardson estimate, makes the coarsest estimate ineligible, and defines exact
run, tie, and even-cardinality selection rules. Iteration 3 re-reviewed these
repairs.
