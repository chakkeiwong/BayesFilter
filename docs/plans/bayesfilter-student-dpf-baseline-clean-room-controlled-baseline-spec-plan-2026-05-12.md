# Plan: student DPF clean-room controlled-baseline specification

## Date

2026-05-12

## Status

Execution authorized on 2026-05-13 for the quarantined student DPF
experimental-baseline stream.  This remains a specification phase only.  It
does not execute experiments, edit production code, or implement a clean-room
baseline.

This plan follows:

- full-horizon confirmation plan:
  `docs/plans/bayesfilter-student-dpf-baseline-full-horizon-edh-pfpf-confirmation-plan-2026-05-12.md`;
- full-horizon confirmation result:
  `experiments/student_dpf_baselines/reports/student-dpf-baseline-full-horizon-edh-pfpf-confirmation-result-2026-05-12.md`;
- student-baseline master program:
  `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`;
- student-baseline reset memo:
  `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`.

## Lane Boundary

Owned paths for this specification phase:

- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-plan-2026-05-12.md`;
- optional follow-on audit/result documents under `docs/plans/` whose filenames
  begin with `bayesfilter-student-dpf-baseline-clean-room-`;
- optional future controlled-baseline experiment area only if a later execution
  plan explicitly creates it:
  `experiments/controlled_dpf_baseline/`;
- student-baseline reset/master documents:
  - `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
  - `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`.

Out of scope:

- DPF monograph writing, rebuild, or enrichment files;
- DPF monograph reset memos;
- `docs/chapters/ch19*.tex`;
- `docs/references.bib`;
- production `bayesfilter/` code;
- editing vendored student code in place;
- copying student implementation code into any BayesFilter-owned code;
- importing student code from a clean-room baseline;
- kernel PFF, stochastic flow, DPF, dPFPF, neural OT, differentiable
  resampling, HMC, MLE, notebook, or plotting workflow claims;
- executing any experiment in this plan phase.

Known drift hazard:

- the worktree may contain dirty or untracked monograph-lane files from other
  agents.  They must not be edited, staged, committed, or used as evidence for
  this phase.

Concrete execution artifacts for 2026-05-13:

- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-completion-plan-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-audit-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-result-2026-05-13.md`.

## Goal

Write a caveated clean-room controlled-baseline specification that translates
the confirmed student-baseline evidence into a BayesFilter-owned experimental
contract without copying or importing student implementation code.

The output of this phase should be a specification document, not an
implementation.  It should define:

- the fixture contract;
- the metric contract;
- the seed policy;
- the first clean-room target settings;
- diagnostics and caveats;
- exclusions;
- release and staging gates for any later clean-room implementation phase.

## Current Evidence

The full-horizon EDH/PFPF confirmation panel produced:

- 30/30 successful records;
- no runtime warnings;
- no nonfinite outputs;
- particles fixed at `128`;
- additional seeds: `31`, `43`, `59`, `71`, `83`;
- full 20-observation nonlinear range-bearing fixtures;
- decision: `confirmation_ready_with_caveats`.

Confirmed low-noise setting:

- fixture: `range_bearing_gaussian_low_noise`;
- horizon: 20 observations;
- particles: `128`;
- flow steps: `20`;
- advanced median average ESS about `27.6`, median position RMSE about
  `0.0475`, median observation proxy RMSE about `0.0162`;
- MLCOE median average ESS about `42.7`, median position RMSE about `0.0468`,
  median observation proxy RMSE about `0.0156`.

Moderate-noise caveat:

- fixture: `range_bearing_gaussian_moderate`;
- particles: `128`;
- advanced did not benefit from 20 flow steps on median position RMSE or
  observation proxy RMSE;
- MLCOE improved median observation proxy RMSE with 20 flow steps but did not
  improve median position RMSE;
- policy recommendation: `moderate_keep_both_as_diagnostic`.

Interpretation constraints:

- the evidence is proxy-only;
- student agreement is not correctness evidence;
- ESS and resampling semantics are implementation-specific diagnostics;
- student code is not production code and must not be copied.

## Goals for This Phase

1. Specify a clean-room fixture contract using BayesFilter-owned code only.
2. Specify the first controlled baseline target settings.
3. Specify metrics and acceptance diagnostics that preserve proxy-only
   interpretation.
4. Specify how to carry forward the moderate-noise flow-step caveat.
5. Specify what must be excluded from the clean-room baseline.
6. Define execution gates for a later implementation plan.

## Remaining Gaps

### Gap 1: clean-room fixture contract is not formalized

The student-baseline stream uses shared nonlinear range-bearing fixtures, but a
clean-room baseline needs an explicit BayesFilter-owned fixture contract.

Closure target:

- define state dimension, observation dimension, transition model, process
  covariance, observation model, observation covariance regimes, initial
  distribution, horizon, and seed policy in a specification document.

### Gap 2: clean-room metric contract is not formalized

The reports already use state RMSE, position RMSE, final-position error,
observation proxy RMSE, ESS, resampling count, runtime, and finite-output
checks.  A clean-room baseline needs a metric contract that does not depend on
student-specific fields.

Closure target:

- define required metrics and required handling of unavailable diagnostics;
- separate algorithm-state metrics from implementation-specific diagnostics.

### Gap 3: moderate-noise flow-step policy is caveated

The confirmation panel did not justify a single moderate-noise flow-step policy
across both student implementations.

Closure target:

- specify moderate-noise 10-step and 20-step variants as diagnostic variants,
  not as a single winner.

### Gap 4: clean-room implementation boundary is not yet explicit enough

The next phase must not import student code or copy implementation details.

Closure target:

- specify that future clean-room code may use only fixture definitions,
  mathematical model statements, metric definitions, and result schemas derived
  from the reports;
- explicitly prohibit copying student classes, functions, control flow, or
  numerical tricks.

### Gap 5: later execution gates are not defined

A future implementation plan needs clear gates before code is written.

Closure target:

- define preflight, implementation, comparison, audit, and commit gates for a
  later clean-room controlled-baseline phase.

## Hypotheses for the Specification Phase

### S1: clean-room fixture contract can be specified without student code

The range-bearing fixtures can be specified using BayesFilter-owned model
statements and data-generation rules, without copying implementation code from
either student snapshot.

Primary criterion:

- specification names all fixture parameters and generation rules needed for a
  clean-room implementation;
- specification identifies any values to reproduce exactly versus values that
  may be regenerated by a BayesFilter-owned fixture generator.

Veto diagnostics:

- specification relies on importing student code;
- specification instructs a future developer to copy student implementation
  functions.

### S2: first target settings are fixed and caveated

The first clean-room target setting can be fixed from the confirmation evidence:

- low-noise: 128 particles, 20 flow steps;
- moderate-noise: 128 particles, 10 and 20 flow steps as diagnostic variants.

Primary criterion:

- specification states these settings and explains the evidence behind them.

Veto diagnostics:

- specification presents moderate-noise 20 steps as uniformly superior;
- specification ignores the moderate-noise caveat.

### S3: metric contract preserves proxy-only interpretation

The clean-room baseline can use simulated-state and observation-proxy metrics
without turning them into production correctness claims.

Primary criterion:

- specification defines required metrics and interpretation warnings;
- specification forbids using student agreement as correctness evidence.

Veto diagnostics:

- specification claims production correctness from student-baseline results;
- specification uses implementation-specific ESS/resampling semantics as
  cross-implementation correctness metrics.

### S4: implementation boundary is enforceable

A later implementation can be audited for no student-code copying or imports.

Primary criterion:

- specification includes import-boundary checks and code-review checks for any
  future controlled-baseline implementation.

Veto diagnostics:

- future plan would be allowed to import from
  `experiments/student_dpf_baselines/vendor/`;
- future plan would be allowed to edit production `bayesfilter/` without a
  separate production plan.

### S5: next execution decision is clear

The specification phase should decide whether a later clean-room implementation
plan is justified.

Primary criterion:

- result decision is one of:
  - `clean_room_spec_ready_for_implementation_plan`;
  - `clean_room_spec_ready_with_caveats`;
  - `needs_spec_revision`;
  - `blocked_or_excluded`.

Veto diagnostics:

- evidence remains too ambiguous to write an implementation plan.

## Required Specification Contents

The planned specification document should include the following sections.

### 1. Scope and provenance

Required content:

- this is a clean-room controlled-baseline specification;
- student evidence is comparison-only;
- no student code may be copied or imported;
- source evidence files and commits are cited for provenance only.

### 2. Fixture contract

Required content:

- fixture names:
  - `range_bearing_gaussian_moderate`;
  - `range_bearing_gaussian_low_noise`;
- state vector convention:
  - `[px, py, vx, vy]`;
- observation convention:
  - range and bearing;
- horizon:
  - full 20-observation fixtures;
- initial distribution:
  - mean and covariance must be specified from BayesFilter-owned fixture
    definitions;
- transition:
  - constant-velocity linear Gaussian transition;
- observation:
  - nonlinear range-bearing Gaussian observation;
- covariance regimes:
  - moderate and low-noise observation covariance values;
- seed policy:
  - fixed declared seed lists for reproducibility;
  - future runs must record seed lists.

### 3. Target settings

Required content:

- particles:
  - first target: `128`;
- flow steps:
  - low-noise: `20`;
  - moderate-noise: `10` and `20` as diagnostic variants;
- horizon:
  - full 20 observations;
- initial clean-room run should avoid expanding the grid.

### 4. Metrics and diagnostics

Required metrics:

- state RMSE;
- position RMSE;
- final-position error;
- observation proxy RMSE;
- runtime seconds;
- finite-output checks.

Optional or implementation-specific diagnostics:

- average ESS;
- minimum ESS;
- resampling count;
- log likelihood if available.

Rules:

- missing optional metrics must be recorded as null or structured blockers;
- likelihoods must not be used as primary cross-implementation evidence unless
  target semantics are independently verified;
- ESS and resampling semantics must be labeled.

### 5. Acceptance and veto gates

Required gates for a later implementation phase:

- generated outputs are finite;
- no unbounded runtime;
- artifacts remain small enough for repository history or are explicitly
  excluded from Git;
- import-boundary search confirms no imports from student vendor snapshots;
- code review confirms no copied student implementation code;
- clean-room code lives outside production `bayesfilter/` unless a separate
  production plan is approved.

### 6. Caveats to carry forward

Required caveats:

- comparison-only evidence;
- proxy metrics, not correctness certificates;
- moderate-noise flow-step policy remains diagnostic;
- ESS/resampling semantics are implementation-specific;
- no HMC, production, or monograph claims.

### 7. Later implementation phases

Required future phase outline:

1. Plan and audit clean-room implementation.
2. Implement BayesFilter-owned fixture generator and result schema.
3. Implement or wrap a BayesFilter-owned experimental baseline algorithm without
   copying student code.
4. Run low-noise N128/steps20 and moderate N128/steps10/20 diagnostic variants.
5. Compare against student-baseline reports as external benchmark evidence only.
6. Update student reset memo and commit path-scoped changes.

## Phase SP0: Specification Preflight

Actions:

1. Record current Git status.
2. Confirm this plan exists and remains in the student experimental lane.
3. Confirm no production import boundary:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

4. Confirm no vendored student files are dirty.
5. Update only the student reset memo with SP0 result.

Exit gate:

- proceed only if edits can remain in the student-baseline lane.

## Phase SP1: Write Specification

Actions:

1. Write a specification document under `docs/plans/`, for example:
   `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`.
2. Include every required content section above.
3. Include explicit clean-room import/copying prohibitions.
4. Include explicit moderate-noise caveat.
5. Include a future implementation-phase outline.

Exit gate:

- specification is complete enough for independent audit.

## Phase SP2: Independent Audit

Actions:

1. Audit the specification as another developer.
2. Check for:
   - lane drift;
   - student-code copying risk;
   - production-code drift;
   - missing fixture fields;
   - missing metric definitions;
   - missing caveats;
   - unclear future gates.
3. Write audit results under `docs/plans/`.

Exit gate:

- no unresolved audit finding blocks a later implementation plan.

## Phase SP3: Reset Memo and Commit

Actions:

1. Update only the student reset memo with specification result and next
   justified work.
2. Update the student master program if the current next move changes.
3. Run `git diff --check`.
4. Stage and commit only student-baseline plan/reset/master files.

Exit gate:

- scoped commit excludes monograph-lane files, production code, references,
  experiments outputs not created by this phase, and vendored student code.

## Stop Rules

Stop and ask for direction if:

- specification requires edits outside the student-baseline lane;
- specification requires copying student implementation code;
- specification requires production `bayesfilter/` edits;
- specification requires monograph-lane edits;
- clean-room boundary cannot be stated clearly;
- result cannot be classified with one of the S5 decision labels.
