# Audit: student DPF baseline gap-closure plan

## Date

2026-05-10

## Scope

This audit reviews:

- `docs/plans/bayesfilter-student-dpf-baseline-gap-closure-plan-2026-05-09.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
- the current `experiments/student_dpf_baselines/` scaffold.

The audit is written as if by a second developer before executing the plan.

## Disposition

Approve with execution controls.

The plan correctly keeps the student experimental-baseline stream separate from
the DPF monograph writing stream.  It also correctly blocks adapter work until
`advanced_particle_filter` has a targeted reproduction result.  The plan is
safe to execute as long as the controls below are enforced.

## Required controls before and during execution

### 1. Use current execution date for new outputs

The plan was written on 2026-05-09, but execution is happening on 2026-05-10.
New reports and JSON outputs should use `2026-05-10` in filenames.  The plan
document itself can remain dated 2026-05-09 because it is the controlling plan.

### 2. Keep monograph changes out of this stream

The current working tree contains unrelated DPF monograph writing changes.
This execution must not edit, stage, or commit:

- `docs/chapters/ch19_particle_filters.tex`;
- `docs/references.bib`;
- DPF monograph rebuild plan/result files;
- the DPF monograph rebuild reset memo.

Final staging must be path-scoped to student-baseline files only.

### 3. Preserve student snapshots

Do not patch vendored student code in place.  If a compatibility workaround is
needed, implement it in an adapter or runner and record it as such.

### 4. Treat partial reproduction as evidence, not correctness

If `advanced_particle_filter` passes one targeted test and fails another, the
adapter gate may still be justified, but the failure must remain visible in the
report.  Passing student tests is not a correctness certificate for BayesFilter.

### 5. Make failures machine-readable

Every runner should emit structured JSON records for both success and failure.
Expected failures should become `BaselineResult` records, not uncaught
exceptions that abort the full panel.

### 6. Avoid broad dependency changes

Do not install packages as part of this cycle.  Missing dependencies should be
recorded as blockers unless the current environment already has them.

### 7. Guard import boundaries

The adapters may import vendored student code from `experiments/`.  Production
`bayesfilter/` modules and normal tests must not import from student baselines.
Run an import-boundary search before committing.

## Missing points added by this audit

The execution should add:

- a reproduction report for `advanced_particle_filter` using 2026-05-10;
- a JSON reproduction record under `reports/outputs/`;
- a minimal common result contract before per-student adapters;
- a small linear-Gaussian fixture catalog before comparison;
- a final result report with explicit next hypotheses;
- a reset-memo execution log that records every phase decision.

## Decision

Proceed through phases G0-G7 automatically if each phase satisfies its exit
gate.  Stop for direction only if:

- `advanced_particle_filter` cannot produce any interpretable targeted
  reproduction result;
- adapter work requires modifying vendored student code in place;
- comparison requires production `bayesfilter/` edits;
- final staging cannot be separated from unrelated monograph work.
