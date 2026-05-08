# Audit: student DPF baseline consolidation plan

## Date

2026-05-09

## Scope

This note audits:

- `docs/plans/bayesfilter-student-dpf-baseline-consolidation-plan-2026-05-08.md`;
- `experiments/student_dpf_baselines/`;
- the current student repository snapshots under
  `experiments/student_dpf_baselines/vendor/`.

The audit is written from the perspective of another developer checking
whether the plan is safe, complete, and executable before continuing.

## Disposition

Approve with execution controls.

The plan is directionally correct and contains the major controls needed for a
quarantined experimental comparison track.  It correctly separates student
code from production BayesFilter code, requires independent Kalman/UKF-style
references, and prevents student-vs-student agreement from being treated as
correctness evidence.

## Strengths

1. The quarantine boundary is explicit: no production `bayesfilter/` module may
   import from `experiments/`.
2. The source snapshots are pinned to exact commits.
3. The plan requires provenance, permission, dependency, reproduction,
   adapter, fixture, reference, comparison, analysis, and report phases.
4. The result contract explicitly allows unavailable fields to be null, which
   avoids fabricating metrics that student code does not expose.
5. The plan compares each student implementation to independent references,
   not only to the other student implementation.
6. The plan records a storage issue that must be resolved before commit: the
   current vendored directories are nested Git repositories.

## Issues and Required Controls

### 1. Nested Git repositories must be normalized before commit

The current snapshots contain internal `.git` directories.  The parent repo
should not accidentally commit embedded repository pointers or rely on
unplanned submodule semantics.

Required control: execute Phase S1A before dependency or adapter work.

### 2. Vendored code may contain generated or large artifacts

The shallow inventory shows PDFs, figures, NPZ/CSV results, `.DS_Store`, and
possibly pretrained weights in the vendored snapshots.  These may be useful for
reproduction, but they increase repository size and noise.

Required control: before final commit, record whether these artifacts are kept
as part of the reproducibility snapshot or excluded in a later cleanup plan.
Do not delete them during this pass unless deletion is explicitly scoped and
the source snapshot remains reproducible.

### 3. Environment isolation is mandatory

The two student repositories may have different dependency assumptions.
`2026MLCOE` appears to lack a requirements file in the shallow audit, while
`advanced_particle_filter` has `requirements.txt`.

Required control: Phase S2 must classify each repo as runnable, runnable with
environment work, or blocked.  Do not install dependencies globally as an
unrecorded side effect.

### 4. Reproduction should precede adaptation

The plan correctly says not to rewrite the implementations before reproducing
original behavior.  This must be enforced during execution.

Required control: Phase S3 should run or classify original examples before
Phase S4 adapters claim any comparison status.

### 5. Adapter results must preserve failure information

The adapter schema should make blocked and failed states first-class.  This is
necessary because one or both student implementations may not map cleanly onto
BayesFilter fixtures.

Required control: `BaselineResult` must include `status` and
`failure_reason`, and runners must serialize failures rather than raising
uncaught exceptions for the full panel.

### 6. Experimental success must not promote HMC

Some student code includes HMC or MLE-related experiments.  Those may be
recorded as reproduction artifacts, but they do not justify BayesFilter HMC
promotion.

Required control: reports must use `comparison_only`,
`filter_consistency_evidence`, or `sampler_not_justified`, never
`strict_convergence_evidence`.

## Missing Points Added by This Audit

The execution pass should add or enforce:

- a `vendor/SNAPSHOT.md` provenance record, already present;
- a storage-normalization phase before commit;
- a dependency-audit report under `experiments/student_dpf_baselines/reports/`;
- a first report even if both repositories are partially blocked;
- a final import-boundary check with `rg -n "experiments" bayesfilter tests`;
- a scoped commit that excludes unrelated dirty documentation files.

## Decision

Proceed through the plan without further human intervention as long as:

- storage normalization succeeds;
- dependency and reproduction failures are recorded rather than silently
  bypassed;
- no production code imports from `experiments/`;
- final staging remains scoped to the DPF/student-baseline work.

Ask for direction only if:

- a phase requires modifying production BayesFilter code;
- a dependency install would need broad environment changes;
- student code redistribution status blocks committing the vendored snapshot;
- the snapshots cannot be made reproducible.
