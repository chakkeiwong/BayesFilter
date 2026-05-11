# Plan: student DPF full-horizon EDH/PFPF confirmation

## Date

2026-05-12

## Status

Active next-phase plan for the quarantined student DPF experimental-baseline
stream.

Tightened on 2026-05-12 before execution to remove ambiguity about low-noise
material-degradation thresholds, moderate-noise flow-step policy, audit
artifact, and reset-memo ownership.

This plan follows:

- replicated EDH/PFPF panel:
  `docs/plans/bayesfilter-student-dpf-baseline-replicated-edh-pfpf-panel-plan-2026-05-11.md`;
- full-horizon EDH/PFPF sensitivity panel:
  `docs/plans/bayesfilter-student-dpf-baseline-full-horizon-edh-pfpf-sensitivity-plan-2026-05-12.md`;
- full-horizon sensitivity result:
  `experiments/student_dpf_baselines/reports/student-dpf-baseline-full-horizon-edh-pfpf-sensitivity-result-2026-05-12.md`.

## Lane Boundary

Owned paths:

- `experiments/student_dpf_baselines/`;
- student-baseline plans, audits, reports, and reset memo under `docs/plans/`;
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`.

Out of scope:

- DPF monograph writing, rebuild, or enrichment files;
- DPF monograph reset memos;
- `docs/chapters/ch19*.tex`;
- `docs/references.bib`;
- production `bayesfilter/` code;
- editing vendored student code in place;
- copying student implementation code into production;
- kernel PFF, stochastic flow, DPF, dPFPF, neural OT, differentiable
  resampling, HMC, MLE, notebook, or plotting workflow claims;
- broad grid expansion beyond the fixed confirmation settings below.

Known drift hazard:

- the worktree may contain dirty or untracked monograph-lane files from other
  agents.  They must not be edited, staged, committed, or used as evidence for
  this phase.

## Goal

Confirm that the full-horizon EDH/PFPF experimental baseline is stable over
additional seeds using a small fixed setting selected from the sensitivity
evidence.

The purpose is not to keep expanding the student-code experiment.  The purpose
is to decide whether the student-baseline evidence is stable enough to inform a
later clean-room controlled baseline specification without copying student code.

This phase remains comparison-only.  It must not certify correctness,
production quality, HMC suitability, or cross-student superiority.

## Current Evidence

The full-horizon sensitivity panel ran 32/32 planned records successfully:

- fixtures: `range_bearing_gaussian_moderate`,
  `range_bearing_gaussian_low_noise`;
- horizon: full 20-observation fixtures;
- seeds: `17`, `23`;
- particles: `64`, `128`;
- flow steps: `10`, `20`;
- implementations: advanced `EDHParticleFilter`, MLCOE `PFPF_EDH`;
- no runtime warnings;
- decision: `full_horizon_sensitivity_ready`.

Observed patterns:

- increasing particles to 128 clearly improved low-noise ESS pressure for both
  implementations;
- 20 flow steps often improved observation proxy RMSE at bounded runtime cost;
- 20 flow steps were not uniformly better for position RMSE;
- advanced moderate-noise at 128 particles did not benefit from 20 flow steps;
- ESS and resampling-count semantics remain implementation-specific.

## Goals for This Phase

1. Test additional seeds at full horizon with fixed pragmatic settings.
2. Confirm that 128 particles continue to reduce low-noise pressure.
3. Confirm whether 20 flow steps should be used generally or only on low-noise
   cases.
4. Preserve implementation-specific ESS and resampling semantics.
5. Produce a clear decision about whether to proceed to a clean-room controlled
   baseline specification.

## Fixed Confirmation Settings

The confirmation panel intentionally avoids a broad grid.  It fixes settings as
follows:

- fixtures:
  - `range_bearing_gaussian_moderate`;
  - `range_bearing_gaussian_low_noise`;
- horizon: full fixture horizon, 20 observations;
- seeds: `31`, `43`, `59`, `71`, `83`;
- implementations:
  - `advanced_particle_filter` `EDHParticleFilter`;
  - `2026MLCOE` `PFPF_EDH`;
- particles: `128` for all confirmation runs;
- flow steps:
  - low-noise fixture: `20`;
  - moderate-noise fixture: `10` and `20` only as a narrow confirmation
    comparison.

Planned records:

- low-noise: 5 seeds x 2 implementations x 1 flow setting = 10 records;
- moderate-noise: 5 seeds x 2 implementations x 2 flow settings = 20 records;
- total planned records: 30.

Runtime-warning threshold:

- 45 seconds per implementation/fixture/seed/flow-step run.

Artifact-size threshold:

- any generated artifact above 1 MB requires explicit justification before
  staging.

## Remaining Gaps

### Gap 1: seed stability is not yet established

The full-horizon sensitivity panel used only seeds `17` and `23`.  That is
enough for bounded sensitivity, but not enough to rely on the selected
experimental setting as a stable baseline artifact.

Closure target:

- run five additional seeds at the selected full-horizon settings.

### Gap 2: selected low-noise setting needs confirmation

The sensitivity panel suggests full horizon, 128 particles, and 20 flow steps is
a pragmatic low-noise setting because it improves ESS pressure and often
improves observation proxy RMSE.

Closure target:

- confirm low-noise behavior at 128 particles and 20 flow steps over additional
  seeds.

### Gap 3: moderate-noise flow-step policy is ambiguous

The sensitivity panel did not show uniform benefit from 20 flow steps on
moderate noise, especially for advanced at 128 particles.

Closure target:

- compare 10 versus 20 flow steps on the moderate-noise fixture only, at 128
  particles, over additional seeds.

### Gap 4: clean-room baseline inputs are not yet frozen

The student experimental stream should inform a clean-room controlled baseline
only through fixtures, metrics, settings, and observed failure modes, not by
copying implementation code.

Closure target:

- produce a confirmation report that identifies the recommended clean-room
  fixture, settings, metrics, and caveats.

### Gap 5: comparison remains proxy-only

Latent-state RMSE and observation proxy RMSE are useful diagnostics because the
fixtures are simulated.  They are not correctness certificates.

Closure target:

- report latent-state RMSE, final-position error, observation proxy RMSE, ESS,
  resampling, runtime, finite-output checks, and failure blockers;
- preserve comparison-only interpretation.

## Hypotheses

### C1: selected full-horizon setting is seed-stable

At full horizon with 128 particles, the selected settings remain runnable and
finite across additional seeds.

Primary criterion:

- every planned run either returns finite structured metrics or a structured
  blocker;
- at least 90 percent of planned records are `ok`;
- no implementation fails all records on any fixture.

Veto diagnostics:

- running the panel requires vendored-code edits, production imports, or target
  changes;
- nonfinite outputs dominate any implementation/fixture group;
- one implementation fails all records on a fixture with an unclassified error.

### C2: low-noise 128-particle pressure reduction persists

The low-noise fixture at 128 particles and 20 flow steps should keep ESS and
resampling pressure within the previously observed full-horizon range.

Primary criterion:

- low-noise median average ESS is not materially worse than the prior
  full-horizon 128-particle, 20-step sensitivity medians for each implementation;
- low-noise median resampling count does not exceed the full 20-step horizon for
  either implementation;
- finite outputs remain dominant.

Material-worse threshold:

- median average ESS below 75 percent of the sensitivity-panel reference median
  is materially worse;
- median position RMSE or observation proxy RMSE above 150 percent of the
  sensitivity-panel reference median is materially worse;
- median resampling count above 20 is materially worse.

Reference medians from the sensitivity panel:

- advanced low-noise, N128, steps20:
  - median average ESS about `29.6`;
  - median position RMSE about `0.0466`;
  - median observation proxy RMSE about `0.0165`;
  - median resampling count `18`;
- MLCOE low-noise, N128, steps20:
  - median average ESS about `43.8`;
  - median position RMSE about `0.0480`;
  - median observation proxy RMSE about `0.0166`;
  - median resampling count `17`.

Veto diagnostics:

- ESS or resampling semantics are compared without labels;
- the report claims low-noise correctness or superiority without metric
  support.

### C3: moderate-noise flow-step policy is resolved

The moderate-noise fixture should produce enough evidence to choose either:

- 10 flow steps as the default moderate-noise setting; or
- 20 flow steps as justified by observation proxy RMSE or position RMSE at
  bounded runtime cost.

Primary criterion:

- report compares 10 versus 20 flow steps at fixed fixture, implementation,
  seed set, and 128 particles;
- runtime cost is considered bounded when median 20-step runtime is at most
  2.5 times median 10-step runtime;
- report recommends one of:
  - `moderate_use_10_steps`;
  - `moderate_use_20_steps`;
  - `moderate_keep_both_as_diagnostic`.

Policy rule:

- choose `moderate_use_20_steps` only if both implementations show bounded
  benefit from 20 steps, where benefit means lower median observation proxy RMSE
  or lower median position RMSE at bounded runtime cost;
- choose `moderate_use_10_steps` only if neither implementation shows bounded
  benefit from 20 steps;
- choose `moderate_keep_both_as_diagnostic` when implementations disagree or
  metrics split between position RMSE and observation proxy RMSE.

Veto diagnostics:

- runtime warnings dominate 20-step runs;
- report recommends more flow steps without RMSE or runtime evidence.

### C4: clean-room baseline specification is ready

The confirmation phase should produce enough information to specify a later
clean-room controlled baseline without copying student code.

Primary criterion:

- report identifies:
  - fixture names;
  - horizon;
  - seed policy;
  - particle count;
  - flow-step policy;
  - metrics;
  - ESS/resampling semantics caveat;
  - comparison-only caveat;
  - exclusions from the controlled baseline.

Veto diagnostics:

- the report recommends copying student implementation code;
- the report crosses into production `bayesfilter/` changes;
- the report uses student agreement as correctness evidence.

### C5: next baseline decision is clear

The phase should decide whether to proceed to clean-room controlled baseline
specification.

Primary criterion:

- result decision is one of:
  - `confirmation_ready_for_clean_room_spec`;
  - `confirmation_ready_with_caveats`;
  - `needs_targeted_debug`;
  - `blocked_or_excluded`.

Veto diagnostics:

- evidence remains too ambiguous to justify the next action.

## Phase C0: Preflight and Lane Guard

Actions:

1. Record current Git status.
2. Confirm this plan exists and remains in the student experimental lane.
3. Confirm no production import boundary:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

4. Confirm no vendored student files are dirty.
5. Confirm the independent audit artifact exists:
   `docs/plans/bayesfilter-student-dpf-baseline-full-horizon-edh-pfpf-confirmation-plan-audit-2026-05-12.md`.
6. Update only the student reset memo with the C0 result.

Exit gate:

- proceed only if edits can remain in the student-baseline lane.

## Phase C1: Confirmation Runner

Actions:

1. Add an experiment-owned runner under
   `experiments/student_dpf_baselines/runners/`.
2. Reuse adapter-owned model bridges from the EDH/PFPF runners.
3. Run the fixed confirmation settings:
   - full horizon;
   - seeds `31`, `43`, `59`, `71`, `83`;
   - particles `128`;
   - low-noise fixture with 20 flow steps;
   - moderate-noise fixture with 10 and 20 flow steps.
4. Record structured per-run records for both implementations.
5. Use a 45 second per-run runtime-warning threshold.

Exit gate:

- each planned run returns either finite metrics or a structured blocker.

## Phase C2: Classification and Report

Actions:

1. Summarize by implementation, fixture, and flow steps.
2. Summarize low-noise seed stability against sensitivity-panel reference
   medians.
3. Summarize moderate-noise 10-step versus 20-step tradeoff.
4. Classify C1-C5.
5. Expected artifacts:
   - `experiments/student_dpf_baselines/reports/outputs/full_horizon_edh_pfpf_confirmation_2026-05-12.json`;
   - `experiments/student_dpf_baselines/reports/outputs/full_horizon_edh_pfpf_confirmation_summary_2026-05-12.json`;
   - `experiments/student_dpf_baselines/reports/student-dpf-baseline-full-horizon-edh-pfpf-confirmation-result-2026-05-12.md`.
6. Update only the student reset memo with results and next-phase
   justification.

Exit gate:

- report gives a clear decision and preserves comparison-only interpretation.

## Phase C3: Audit, Tidy, Reset Memo, Commit

Actions:

1. Run syntax checks for edited modules.
2. Run import-boundary search over production code and tests.
3. Run `git diff --check`.
4. Confirm no vendored student code changed.
5. Check generated artifact sizes.
6. Confirm each new generated artifact is below 1 MB or explicitly justified.
7. Update only the student reset memo with completion state and next
   hypotheses.
8. Stage and commit only student-baseline files.

Exit gate:

- scoped commit excludes monograph-lane files, production code, references, and
  vendored student code.

## Stop Rules

Stop and ask for direction if:

- the panel needs edits outside the student-baseline lane;
- either selected path requires vendored-code edits;
- either selected path requires production `bayesfilter/` imports;
- the fixture target must change to make the result run;
- runtime becomes unbounded;
- nonfinite outputs dominate the panel;
- generated artifacts are too large for normal repository history;
- the result cannot be classified with one of the C5 decision labels.
