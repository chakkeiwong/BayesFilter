# Plan: student DPF baseline hypothesis closure

## Date

2026-05-10

## Status

Active follow-on plan for the quarantined student DPF experimental-baseline
stream.

This plan continues from commit `7f57486`, which completed the first
linear-Gaussian adapter and comparison cycle.  It tests the three explicit
hypotheses recorded in
`experiments/student_dpf_baselines/reports/student-dpf-baseline-gap-closure-result-2026-05-10.md`.

## Scope boundary

This is student-baseline experimental work only.

Owned paths:

- `experiments/student_dpf_baselines/`;
- student-baseline plans and audits under `docs/plans/`;
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`.

Out of scope:

- DPF monograph writing files;
- DPF monograph reset memo;
- `docs/chapters/ch19*.tex`;
- `docs/references.bib`;
- production `bayesfilter/` code;
- treating student code as production or correctness authority.

## Motivation

The previous cycle established that both student snapshots can be called through
quarantined adapters and agree with an independent Kalman reference on small
linear-Gaussian fixtures.  That result is useful but narrow.  The remaining
gaps are:

1. whether particle-filter diagnostics diverge under larger or lower-noise
   linear-Gaussian panels;
2. whether `advanced_particle_filter` kernel PFF is reproducible in isolation
   after the observed `.F.` partial failure and non-completion;
3. whether nonlinear fixtures expose adapter or implementation assumptions
   before any broader flow/DPF comparison.

## Hypotheses

### H1: linear stress

The two student implementations will remain reference-consistent on larger
linear-Gaussian panels, while particle-filter ESS/runtime diagnostics will
diverge as observation noise decreases.

### H2: kernel PFF isolation

The observed `advanced_particle_filter/tests/test_kernel_pff.py` partial failure
is reproducible in isolation and can be classified as either an environment
issue, an algorithm/test sensitivity, or a long-running test design issue.

### H3: nonlinear smoke

Small nonlinear fixtures will expose implementation assumptions around
Jacobian shape, covariance regularization, and observation mapping before any
production-quality conclusion can be drawn.

## Success criteria

This cycle succeeds when:

1. a larger/low-noise linear stress panel is run and summarized;
2. kernel PFF reproduction is classified with command-level evidence;
3. at least one nonlinear smoke fixture is attempted through quarantined
   adapters or explicitly blocked with structured reasons;
4. the reset memo records phase results, interpretations, and next justified
   actions;
5. final staging and commit include only student-baseline files.

## Veto diagnostics

Stop before the next phase if:

- a phase requires editing production `bayesfilter/` code;
- a phase requires editing DPF monograph writing files or reset memo;
- a phase requires patching vendored student code in place;
- kernel PFF or nonlinear work produces unclear failures that cannot be
  classified without broad dependency or environment changes;
- generated artifacts are too large for a normal repository commit.

## Phase H0: preflight and lane guard

### Primary criterion

Confirm the student-baseline lane can proceed without staging unrelated
monograph files.

### Actions

1. Record Git status.
2. Confirm active student memo and previous result report are present.
3. Confirm no production import boundary was introduced by the prior cycle.
4. Update the student reset memo with this phase result.

### Exit gate

Proceed only if all new edits can remain under the student-baseline scope.

## Phase H1: larger and low-noise linear stress panel

### Primary criterion

Run a controlled linear-Gaussian stress panel that exercises particle-filter
diagnostics beyond the first smoke panel.

### Actions

1. Add larger linear fixtures:
   - longer 1D random-walk or AR-style fixture;
   - longer constant-velocity fixture;
   - lower-observation-noise constant-velocity fixture.
2. Add a stress-panel runner using existing adapters.
3. Record:
   - Kalman reference agreement;
   - advanced bootstrap-PF log-likelihood diagnostics;
   - average/min ESS;
   - resampling count;
   - runtime.
4. Summarize whether H1 is supported, partially supported, or not supported.

### Exit gate

Proceed if the panel completes or produces structured adapter failures.

## Phase H2: focused kernel PFF reproduction

### Primary criterion

Classify the `advanced_particle_filter` kernel PFF issue with bounded,
command-level evidence.

### Actions

1. Run each kernel PFF test individually with a bounded timeout:
   - `test_kernel_pff_lgssm`;
   - `test_kernel_pff_convergence`;
   - `test_scalar_vs_matrix_kernel`.
2. If pytest collection or naming differs, record the actual discovered names.
3. Capture pass/fail/timeout status and short failure summaries.
4. Do not patch vendored code.

### Exit gate

Proceed if the issue is classified as pass, fail, or timeout with a named test.
Stop if the failure is ambiguous and cannot be reproduced with bounded commands.

## Phase H3: nonlinear smoke fixtures

### Primary criterion

Attempt at least one small nonlinear fixture through the comparison harness,
while keeping failure states structured and non-promotional.

### Actions

1. Add a minimal nonlinear range-bearing fixture if feasible from the
   `advanced_particle_filter` local model.
2. Add an advanced-only nonlinear smoke runner first.
3. Attempt MLCOE nonlinear adapter only if it can be wrapped without modifying
   vendored code.
4. Record whether nonlinear comparison is runnable, adapter-blocked, or
   assumption-blocked.

### Exit gate

Proceed to synthesis if nonlinear smoke is either runnable or explicitly
blocked with a precise reason.

## Phase H4: synthesis, audit, and handoff

### Primary criterion

Produce a result report and reset-memo update that decide the next justified
student-baseline phase.

### Actions

1. Write a result report under `experiments/student_dpf_baselines/reports/`.
2. Write JSON summaries under `reports/outputs/`.
3. Run import-boundary and syntax checks.
4. Update the reset memo with all phase decisions.
5. Commit only student-baseline files.

### Exit gate

The commit is scoped, auditable, and does not include monograph work.
