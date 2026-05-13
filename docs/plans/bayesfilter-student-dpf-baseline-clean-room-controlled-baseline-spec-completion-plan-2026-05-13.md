# Plan: complete student DPF clean-room controlled-baseline specification

## Date

2026-05-13

## Status

Execution authorized on 2026-05-13 by the current user instruction.  Execute the
phase in the student DPF experimental-baseline lane only, then make one
path-scoped commit.  Do not push.

This plan belongs only to the quarantined student DPF experimental-baseline
lane.  It does not authorize monograph rebuild work, production BayesFilter
edits, experiment execution, student-code copying, or pushes.

## Lane boundary

Allowed write set for the later execution of this phase:

- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-plan-2026-05-12.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-audit-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-result-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
- optionally `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md` only if the active next-phase pointer must be updated.

Out of scope:

- DPF monograph rebuild/enrichment files;
- `docs/chapters/ch19*.tex`;
- `docs/references.bib`;
- production `bayesfilter/` code;
- vendored student code edits;
- `experiments/student_dpf_baselines/vendor/`;
- execution of new experiments;
- pushes unless separately requested.

## Phase goal

Convert the confirmed student-baseline evidence into a BayesFilter-owned
clean-room controlled-baseline specification.  The output should be a
specification and audit package that a later implementation phase can follow
without importing or copying student code.

This phase should answer:

1. What fixture contract should the clean-room baseline implement?
2. What target settings are justified by the current student-baseline evidence?
3. Which metrics are required, optional, or implementation-specific?
4. Which caveats must be preserved so proxy evidence is not overclaimed?
5. What gates must a later clean-room implementation pass before code is
   written, compared, committed, or promoted?

## Current evidence to preserve

The full-horizon confirmation panel supports a specification phase with caveats:

- fixtures: `range_bearing_gaussian_low_noise` and
  `range_bearing_gaussian_moderate`;
- horizon: 20 observations;
- particles: 128;
- seeds: 31, 43, 59, 71, 83;
- low-noise target setting: 20 flow steps;
- moderate-noise setting: keep both 10 and 20 flow steps as diagnostic variants;
- decision: `confirmation_ready_with_caveats`;
- interpretation: proxy-only comparison evidence, not production correctness.

## Remaining gaps

### Gap 1: fixture contract is not clean-room formalized

The student-baseline reports identify the fixture names and broad model class,
but a future clean-room implementation needs a precise BayesFilter-owned
contract for state layout, observation layout, transition law, observation law,
covariance regimes, initial distribution, horizon, seed policy, and output
schema.

### Gap 2: metric contract is not independent of student implementations

Existing reports include useful metrics, but some diagnostics are
implementation-specific.  A future clean-room baseline needs required metrics,
optional metrics, null-handling rules, and interpretation limits.

### Gap 3: moderate-noise policy is intentionally unresolved

The confirmation panel did not justify a single moderate-noise flow-step winner.
The specification must carry forward both 10-step and 20-step variants as
diagnostics.

### Gap 4: clean-room boundary needs enforceable gates

The project needs explicit checks that a later implementation does not import
student vendor snapshots, copy student control flow, edit production code, or
promote student results into production claims.

### Gap 5: later execution sequencing is not yet locked

The current plan states what should be specified, but the next execution needs a
phase-by-phase cycle: preflight, write spec, audit, result, memo update, and
stop-before-commit unless a commit is explicitly requested.

## Hypotheses

### H1: a clean-room fixture contract can be specified without student code

Test in this phase:

- write the fixture contract using model statements and result reports only;
- include all required fixture fields;
- verify the document does not instruct future developers to import or copy
  student code.

Pass criterion:

- a future developer can implement a fixture generator from the specification
  without reading student source files.

Veto:

- the specification depends on student classes, functions, control flow, or
  numerical tricks.

### H2: target settings are fixed enough for a first clean-room baseline

Test in this phase:

- specify low-noise N128/steps20;
- specify moderate-noise N128/steps10 and N128/steps20 as diagnostic variants;
- explain that moderate-noise 20 steps is not a universal winner.

Pass criterion:

- target settings match the confirmation result and preserve the caveat.

Veto:

- the specification presents moderate-noise 20 steps as uniformly superior or
  expands the grid before the first clean-room implementation.

### H3: metrics can preserve proxy-only interpretation

Test in this phase:

- define required metrics: state RMSE, position RMSE, final-position error,
  observation proxy RMSE, runtime seconds, finite-output checks;
- define optional diagnostics: average ESS, minimum ESS, resampling count, log
  likelihood if available;
- require null or structured-blocker handling for unavailable metrics.

Pass criterion:

- the specification separates comparison metrics from correctness claims.

Veto:

- student agreement, ESS, resampling semantics, or likelihood values are used as
  correctness certificates.

### H4: implementation boundary can be audited mechanically and by review

Test in this phase:

- specify import-boundary searches;
- specify code-review checks for copied student code;
- specify path boundaries for future implementation;
- prohibit production `bayesfilter/` edits unless a separate production plan is
  approved.

Pass criterion:

- a future implementation phase has explicit clean-room gates before code is
  accepted.

Veto:

- future implementation is allowed to import from
  `experiments/student_dpf_baselines/vendor/` or edit vendored student files.

### H5: the next decision can be made from the spec result

Test in this phase:

- write a result note with one of these decisions:
  - `clean_room_spec_ready_for_implementation_plan`;
  - `clean_room_spec_ready_with_caveats`;
  - `needs_spec_revision`;
  - `blocked_or_excluded`.

Pass criterion:

- the result clearly states whether a later implementation plan is justified.

Veto:

- the result cannot classify the next move without additional evidence.

## Execution plan for the later phase

### SP0: preflight and lane guard

Actions:

1. Record current `git status --short`.
2. Confirm the active lane is student DPF experimental baseline.
3. Confirm this plan and the 2026-05-12 source plan exist.
4. Confirm no production import contamination:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

5. Confirm vendored student files are not dirty.
6. Record SP0 in the student reset memo only.

Exit gate:

- proceed only if all required edits remain in the allowed write set.

### SP1: write the clean-room specification

Actions:

1. Create:
   `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`.
2. Include:
   - scope and provenance;
   - fixture contract;
   - target settings;
   - metrics and diagnostics;
   - acceptance and veto gates;
   - caveats;
   - later implementation outline.
3. State all clean-room prohibitions explicitly.
4. Carry forward the moderate-noise diagnostic-variant policy.

Exit gate:

- the specification is complete enough for independent audit.

### SP2: audit the specification

Actions:

1. Create:
   `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-audit-2026-05-13.md`.
2. Audit for:
   - lane drift;
   - missing fixture fields;
   - missing metrics;
   - overclaiming;
   - student-code copying risk;
   - production-code drift;
   - unclear future gates.
3. Tighten the specification if the audit finds non-blocking ambiguity.

Exit gate:

- no blocking audit findings remain.

### SP3: write result and reset memo update

Actions:

1. Create:
   `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-result-2026-05-13.md`.
2. Record:
   - goals;
   - gaps closed;
   - hypothesis results;
   - caveats;
   - next decision label.
3. Update only:
   `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`.
4. Update the student master program only if the active next-phase pointer must
   change.

Exit gate:

- result decision is one of the H5 decision labels.

### SP4: verification and scoped commit

Actions:

1. Run `git diff --check`.
2. Review `git status --short` and verify no monograph, production, references,
   vendored student, or generated-output files are part of this phase.
3. Stage only the student-baseline files created or updated by this phase.
4. Commit the scoped student-baseline documentation changes.
5. Do not push.

Exit gate:

- scoped commit exists and excludes monograph-lane files, production code,
  references, vendored student code, and generated experiment outputs.

## Stop rules

Stop and ask for direction if:

- any required edit touches monograph rebuild/enrichment files;
- any required edit touches production `bayesfilter/` code;
- the clean-room specification needs student implementation details;
- vendored student files are dirty or must be edited;
- the fixture parameters cannot be stated without reading student source code;
- the result cannot classify the next move using the H5 labels.

## Expected deliverables

After execution, the phase should have produced:

1. a clean-room controlled-baseline specification;
2. an independent audit note;
3. a result note with hypothesis outcomes and next decision;
4. a student reset-memo update;
5. no experiment outputs;
6. no production code changes;
7. no monograph-lane changes;
8. one path-scoped commit;
9. no push unless separately requested.
