# Phase P13 plan: final readiness report and reset memo update

## Date

2026-05-14

## Target scope

The complete reviewer-grade DPF revision round.

## Governing prerequisites and lane guard

- Required prior results: passed P0-P12 results.
- P13 must distinguish bibliography-spine support from reviewed
  ResearchAssistant support using P2 and any later source artifacts.
- Allowed write set: P13 final readiness artifact, reviewer-grade DPF reset
  memo, and narrowly scoped reviewer-grade planning metadata.  Do not touch
  student-baseline files.
- Before closeout, record branch, `git status --short`, out-of-lane dirty
  files, and this write set.

## Purpose

Close the round with an honest readiness report.  The final artifact must not
declare success by tone.  It must state what was revised, what was audited, what
remains weak, and which bank-facing claims still require experiments.

## Implementation instructions

1. Summarize every phase P0-P12:
   - completed;
   - partially completed;
   - blocked;
   - superseded.
2. Summarize chapter changes:
   - SMC baseline;
   - EDH/LEDH;
   - PF-PF;
   - resampling and OT/Sinkhorn;
   - learned OT;
   - HMC/banking suitability;
   - debugging verification contract.
3. Summarize source support:
   - primary sources used;
   - bibliography-only sources;
   - missing sources;
   - claims weakened due to source gaps.
4. Summarize derivation audit:
   - obligations passed;
   - obligations repaired;
   - obligations manually checked;
   - obligations unresolved.
5. Summarize build/PDF status.
6. Summarize hostile-reader risks:
   - mathematical risks;
   - source risks;
   - implementation risks;
   - banking governance risks.
7. Separate:
   - writing-complete items;
   - experiment-required items;
   - implementation-required items;
   - reviewer-risk items.
8. Update the DPF reset memo or create a reviewer-grade reset memo.

## Required result artifact

Create:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-final-readiness-report-{YYYY-MM-DD}.md`

Also update or create:

- reviewer-grade DPF reset memo, if the round has produced substantial changes.
  This must be a DPF monograph memo, not the student-baseline reset memo.

## Audit rules

- Do not call the DPF block ready unless P11 and P12 passed.
- Do not hide unresolved derivation or source gaps.
- Do not describe experiment-required claims as writing-complete.
- Do not merge student-baseline workstream results into this closeout unless
  explicitly relevant and source-grounded.

## Veto diagnostics

The phase fails if:

- the final report is optimistic but not evidence-backed;
- remaining banking validation gaps are hidden;
- unresolved mathematical weaknesses are labeled as polish;
- build or PDF status is omitted;
- reset memo is not updated.
- source-support status conflates bibliography-spine support with
  ResearchAssistant-reviewed support;
- student-baseline files are edited, staged, or merged into this closeout.

## Exit gate

The round is complete only when the final report lets a future agent or human
reviewer know whether the DPF block is ready, partially ready, or still blocked
against the master-program standard.
