# Audit: MP4 flow and DPF readiness review plan

## Date

2026-05-11

## Scope

This audit reviews:

- `docs/plans/bayesfilter-student-dpf-baseline-mp4-flow-dpf-readiness-review-plan-2026-05-11.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`.

The audit is written as if by a second developer before MP4 execution.

## Disposition

Approve with controls.

The plan addresses the correct next gap in the quarantined student-baseline
stream.  It properly treats flow and DPF code as comparison-only evidence and
does not authorize production changes or vendored-code edits.

## Required Controls

1. Keep MP4 as a readiness review, not a filter-comparison run.  Static
   inventory, imports, and signature inspection are allowed; filter
   instantiation and filter/update execution are not part of this phase.
2. Preserve lane separation.  Do not edit or stage DPF monograph writing files,
   DPF monograph reset memos, `docs/chapters/ch19*.tex`,
   `docs/references.bib`, or production `bayesfilter/` code.
3. Preserve student snapshots.  Do not edit vendored files under
   `experiments/student_dpf_baselines/vendor/`.
4. Keep kernel PFF out of first-candidate selection.  MP3 classified kernel PFF
   as `excluded_pending_debug` because bounded runs consistently hit
   `max_iterations`.
5. Separate deterministic EDH/LEDH flow readiness from DPF, neural OT,
   differentiable resampling, HMC, MLE, and experiment-script readiness.
6. Treat import failures and unclear APIs as structured blockers.  Do not
   install dependencies or broaden the environment.
7. Require a concrete later candidate to specify both implementation paths,
   fixture, metric, runtime cap, and blocker plan.  If that cannot be stated,
   MP4 must end with a blocker-only decision.

## Ambiguities Tightened Before Execution

The plan was tightened to make explicit that MP4 may inspect constructor and
method signatures but must not instantiate filters, run notebooks, call
experiment scripts, or execute filtering/training/HMC entry points.

## Decision

Proceed automatically through MP4.0-MP4.4 if each phase satisfies its exit
gate.  Stop for direction only if:

- inventory requires edits outside the student-baseline lane;
- import/signature probes require vendored-code edits or dependency changes;
- no structured readiness classification can be produced;
- monograph and student-baseline files cannot be staged separately.
