# Phase P12 plan: build, PDF review, and hostile-reader audit

## Date

2026-05-14

## Target scope

The compiled DPF chapter block in document order.

## Governing prerequisites and lane guard

- Required prior results: P0, P1, P2, and passed P3-P11 results.
- P2 remains the source-grounding register unless superseded by a later
  reviewed source-intake artifact.
- Allowed write set: P12 audit artifact and narrowly scoped DPF reviewer-grade
  chapter fixes if the audit finds immediate repairable issues.  Do not touch
  student-baseline files.
- Before auditing, record branch, `git status --short`, out-of-lane dirty files,
  and this write set.

## Purpose

Audit the revised material as a skeptical academic panel might read it.  This
phase checks whether the document itself is persuasive, self-contained, and
honest after the source-level mathematical rewrites.

## Implementation instructions

1. Compile the document using the established build route.
2. Record:
   - command used;
   - success/failure;
   - warnings relevant to DPF chapters;
   - undefined references/citations;
   - overfull tables or unreadable layouts.
3. Review the PDF in DPF chapter order.
4. For each chapter, check:
   - local problem statement;
   - notation/object inventory;
   - assumptions;
   - derivations;
   - literature synthesis;
   - claim-status boundaries;
   - implementation diagnostics;
   - limitations.
5. Run overclaim searches:
   - exact;
   - unbiased;
   - consistent;
   - validated;
   - robust;
   - HMC-ready;
   - production;
   - optimal;
   - guarantee;
   - solves;
   - proves;
   - suitable;
   - credible;
   - first serious.
6. For each hit, verify that assumptions and limitations are local.
7. Check whether tables have become substitutes for derivations.
8. Check whether a non-specialist mathematical reviewer would need to search
   papers for a load-bearing step.

## Required result artifact

Create:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-hostile-reader-audit-{YYYY-MM-DD}.md`

## Audit rules

- Read the PDF, not only source files.
- Do not treat polished prose as proof.
- Do not let claim-boundary caveats appear only after the claim has already been
  made strongly.
- Do not ignore layout failures; unreadable tables are failed exposition.

## Veto diagnostics

The phase fails if:

- the PDF cannot be built and no bounded fallback audit is recorded;
- DPF-local references or citations remain unresolved;
- overclaim terms lack local assumptions;
- derivation gaps survive because citations exist;
- banking suitability reads stronger than the evidence supports.
- source-support language conflicts with P2 or later reviewed-source artifacts;
- student-baseline files are edited, staged, or treated as monograph evidence.

## Exit gate

Proceed to final readiness only when the revised DPF block can survive a
hostile but fair reading without relying on reader goodwill or outside paper
searches.
