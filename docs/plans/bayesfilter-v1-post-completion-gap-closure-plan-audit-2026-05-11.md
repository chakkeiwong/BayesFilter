# Audit: BayesFilter v1 Post-completion Gap-closure Plan

## Date

2026-05-11

## Scope

This audit reviews:

```text
docs/plans/bayesfilter-v1-post-completion-gap-closure-plan-2026-05-11.md
```

It is written as an independent developer review before execution in the
BayesFilter v1 external-compatibility lane.

## Verdict

Approved for scoped execution with conditional gates.

The plan correctly treats the committed v1 external-compatibility phase as the
starting point and does not reopen MacroFinance, DSGE, structural SVD/SGU, or
Chapter 18/18b ownership.  Its strongest immediate move is the CPU-local QR
score/Hessian diagnostic ladder, because that is the only gap with concrete
BayesFilter-local evidence of a performance and memory issue.

## Required Clarifications

1. GPU/XLA-GPU must remain conditional.
   The plan should run escalated GPU probes before any GPU benchmark.  If the
   escalated probe fails or TensorFlow cannot see a GPU, the phase can still
   close by recording GPU as blocked by real escalated evidence.  A non-GPU
   result must not block CPU-local closure.

2. Optional live MacroFinance must remain optional.
   The MacroFinance compatibility test can be run read-only when local
   preconditions are available.  A skip or read-only failure must be recorded
   as external evidence, not as a BayesFilter-local CI failure.

3. DSGE work is design-only in this lane.
   Rotemberg and EZ bridge requirements may be specified, but no DSGE source or
   production BayesFilter import should be added.  SGU remains blocked.

4. SVD-CUT derivative work must stay diagnostic.
   Existing SVD-CUT derivative tests are useful branch evidence, but derivative
   and HMC promotion require active-floor and spectral-gap diagnostics to be
   explicit.

## Missing Points Checked

- The plan names the lane-specific reset memo.
- It forbids staging shared monograph reset memo changes.
- It has falsifiable hypotheses for QR derivative cost, CPU scaling, GPU
  evidence, optional external compatibility, DSGE fixtures, HMC target
  selection, SVD-CUT branch safety, and CI tiering.
- It contains a final commit boundary with staged-file checks.

## Veto Diagnostics

Stop or downgrade the relevant phase if any of these occur:

- a required edit falls outside `docs/plans/bayesfilter-v1-*.md`,
  `docs/benchmarks`, `tests/test_v1_public_api.py`, or v1 source-map entries;
- a GPU check is run without escalated permissions;
- a CPU-hidden run is described as GPU evidence;
- optional MacroFinance or DSGE checks require source edits;
- SVD-CUT derivative or HMC labels exceed branch diagnostics;
- out-of-lane dirty files are staged.

## Execution Approval

Proceed through the CPU-local phases first.  Continue automatically when the
primary criterion passes or a blocked status is explicitly supported by the
phase evidence.  Human direction is only needed if a lane-boundary veto appears
or if the next phase would require changing an external project.
