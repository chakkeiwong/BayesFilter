# Plan: student DPF replicated EDH/PFPF panel

## Date

2026-05-11

## Status

Active next-phase plan for the quarantined student DPF experimental-baseline
stream.

Tightened on 2026-05-12 before execution to remove ambiguity about the seed
set, runtime warning threshold, generated artifacts, audit artifact, and lane
boundary.

This plan follows the bounded EDH/PFPF adapter spike:
`docs/plans/bayesfilter-student-dpf-baseline-edh-pfpf-adapter-spike-plan-2026-05-11.md`.

## Lane Boundary

Owned paths:

- `experiments/student_dpf_baselines/`;
- student-baseline plans, audits, reports, and reset memo under `docs/plans/`.
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`.

Out of scope:

- DPF monograph writing or enrichment files;
- DPF monograph reset memos;
- `docs/chapters/ch19*.tex`;
- `docs/references.bib`;
- production `bayesfilter/` code;
- editing vendored student code in place;
- copying student implementation code into production;
- kernel PFF, stochastic flow, DPF, dPFPF, neural OT, differentiable
  resampling, HMC, MLE, notebook, or plotting workflow claims.

Known drift hazard:

- the repository may contain dirty or untracked DPF monograph files from another
  lane.  They must not be edited, staged, committed, or used as evidence for
  this phase.

## Goal

Complete the next student-baseline experimental phase: a replicated EDH/PFPF
panel comparing the two MP4-selected paths:

- `advanced_particle_filter.filters.edh.EDHParticleFilter`;
- `2026MLCOE` `src.filters.flow_filters.PFPF_EDH`.

The panel should test whether the adapter-spike success survives:

- both existing nonlinear Gaussian range-bearing fixtures;
- fixed seeds `[17, 23, 31]`;
- the same bounded particle and flow-step settings: 64 particles and 10 flow
  steps;
- a 30 second per-run runtime-warning threshold;
- explicit proxy metrics and implementation-specific diagnostics.

This phase remains comparison-only.  It must not certify correctness,
production quality, or HMC suitability.

## Current Evidence

MP4 selected the EDH/PFPF pair as the only immediate flow comparison candidate.

The adapter spike then ran both paths on the reduced moderate-noise nonlinear
fixture:

- advanced `EDHParticleFilter`: status `ok`, position RMSE about `0.0778`,
  average ESS about `8.45`, runtime about `0.559` seconds;
- MLCOE `PFPF_EDH`: status `ok`, position RMSE about `0.0710`, average ESS
  about `24.14`, runtime about `3.91` seconds;
- decision: `adapter_spike_success_needs_replication`.

## Goals for This Phase

1. Test the EDH/PFPF pair on both nonlinear Gaussian range-bearing fixtures.
2. Add fixed repeated seeds `[17, 23, 31]` to separate one-off stochastic
   behavior from stable patterns.
3. Compare moderate-noise versus low-noise behavior using proxy metrics.
4. Preserve implementation-specific ESS and resampling semantics.
5. Decide whether this EDH/PFPF panel can become a stable experimental baseline
   artifact, remains spike-only, or should be blocked/excluded.

The planned run matrix is exactly:

- fixtures:
  - `range_bearing_gaussian_moderate`;
  - `range_bearing_gaussian_low_noise`;
- seeds: `17`, `23`, `31`;
- implementations:
  - `advanced_particle_filter` `EDHParticleFilter`;
  - `2026MLCOE` `PFPF_EDH`;
- total planned records: 12.

## Remaining Gaps

### Gap 1: single-seed evidence is insufficient

The adapter spike used one seed on one reduced fixture.  It did not test
stochastic variability.

Closure target:

- run multiple seeds for both selected implementations.

### Gap 2: low-noise behavior is untested for EDH/PFPF

Particle-flow methods are intended to help with sharp likelihoods, but low
observation noise can also expose stiffness, ESS collapse, or resampling
pressure.

Closure target:

- run the existing low-noise nonlinear Gaussian range-bearing fixture under the
  same bounded settings.

### Gap 3: ESS/resampling semantics differ by implementation

The spike exposed different ESS levels and resampling-count semantics.

Closure target:

- report ESS and resampling pressure per implementation with explicit
  semantics labels;
- do not merge ESS fields into a false common metric unless semantics match.

### Gap 4: runtime stability is unknown

The spike was bounded and fast enough, but replicated panels can expose runtime
or numerical instability.

Closure target:

- keep the panel bounded;
- record runtime per run and summarize medians/maxima;
- classify runtime or nonfinite-output failures explicitly.

### Gap 5: comparison remains proxy-only

Latent-state RMSE is available because the fixture is simulated, but agreement
between student outputs still is not correctness evidence.

Closure target:

- report latent-state RMSE, final-position error, observation proxy RMSE, ESS,
  resampling, runtime, and finite-output checks;
- interpret differences against the fixture and MP2 EKF/UKF/BPF context, not
  as proof of correctness.

## Hypotheses

### R1: both EDH/PFPF paths remain runnable

Both selected implementations can run across both nonlinear fixtures and all
planned seeds without vendored-code edits.

Primary criterion:

- every planned run either returns finite structured metrics or a structured
  blocker; at least one run per implementation and fixture is `ok`.

Veto diagnostics:

- running the panel requires vendored-code edits, production imports, or target
  changes;
- one implementation fails all runs on a fixture with an unclassified error.

### R2: low observation noise increases pressure

Low observation noise should reduce ESS, increase resampling pressure, increase
runtime, or increase RMSE for at least one implementation relative to the
moderate-noise fixture.

Primary criterion:

- fixture-level summaries show a directional low-noise pressure signal in ESS,
  resampling count, runtime, or RMSE for at least one implementation.

Veto diagnostics:

- ESS/resampling semantics are compared without labels;
- the report claims low-noise superiority or failure without metric support.

### R3: proxy comparison remains interpretable

The panel can be interpreted through declared target labels and proxy metrics.

Primary criterion:

- report includes fixture, seed, particle count, flow steps, target labels,
  finite-output checks, RMSE metrics, ESS availability, resampling semantics,
  and runtime for every record.

Veto diagnostics:

- report treats student agreement as correctness;
- likelihoods are used as primary cross-student evidence;
- missing metrics are hidden rather than recorded as nulls or blockers.

### R4: runtime remains bounded enough for experimental use

The replicated panel should remain small enough for routine experimental
comparison artifacts.

Primary criterion:

- no individual run exceeds the planned runtime-warning threshold, or any
  exceedance is classified and does not prevent structured reporting;
- planned runtime-warning threshold is 30 seconds per implementation/fixture/seed
  run.

Veto diagnostics:

- runtime becomes unbounded;
- generated artifacts become too large for normal repository history;
- nonfinite outputs dominate the panel.

### R5: next baseline decision is clear

The phase should decide whether EDH/PFPF becomes a replicated experimental
baseline, needs more debugging, or should remain spike-only.

Primary criterion:

- result decision is one of:
  - `replicated_panel_ready`;
  - `replicated_panel_ready_with_caveats`;
  - `needs_targeted_debug`;
  - `blocked_or_excluded`.

Veto diagnostics:

- evidence remains too ambiguous to justify the next action.

## Phase RP0: Preflight and Lane Guard

Actions:

1. Record current Git status.
2. Confirm this plan exists and the student reset memo points to this lane.
3. Confirm no production import boundary:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

4. Confirm no vendored student files are dirty.
5. Confirm the independent audit artifact exists:
   `docs/plans/bayesfilter-student-dpf-baseline-replicated-edh-pfpf-panel-plan-audit-2026-05-12.md`.
6. Update only the student reset memo with the RP0 result.

Exit gate:

- proceed only if edits can remain in the student-baseline lane.

## Phase RP1: Replicated Panel Runner

Actions:

1. Add or extend an experiment-owned runner under
   `experiments/student_dpf_baselines/runners/`.
2. Reuse the adapter-owned bridges from the spike.
3. Run:
   - fixtures: `range_bearing_gaussian_moderate`,
     `range_bearing_gaussian_low_noise`;
   - reduced horizon: 8 observations first;
   - seeds: `17`, `23`, `31`;
   - particles: 64;
   - flow steps: 10.
4. Record structured per-run records for both implementations.
5. Use a 30 second per-run runtime-warning threshold.

Exit gate:

- each planned run returns either finite metrics or a structured blocker.

## Phase RP2: Classification and Report

Actions:

1. Summarize by implementation, fixture, and seed.
2. Summarize low-noise pressure signals.
3. Classify R1-R5.
4. Write JSON outputs and a Markdown report under
   `experiments/student_dpf_baselines/reports/`.
5. Expected artifacts:
   - `experiments/student_dpf_baselines/reports/outputs/replicated_edh_pfpf_panel_2026-05-12.json`;
   - `experiments/student_dpf_baselines/reports/outputs/replicated_edh_pfpf_panel_summary_2026-05-12.json`;
   - `experiments/student_dpf_baselines/reports/student-dpf-baseline-replicated-edh-pfpf-panel-result-2026-05-12.md`.
6. Update only the student reset memo with results and next-phase
   justification.

Exit gate:

- report gives a clear decision and preserves comparison-only interpretation.

## Phase RP3: Audit, Tidy, Reset Memo, Commit

Actions:

1. Run syntax checks for edited modules.
2. Run import-boundary search over production code and tests.
3. Run `git diff --check`.
4. Confirm no vendored student code changed.
5. Check generated artifact sizes.
6. Confirm each new generated artifact is small enough for normal repository
   history; any artifact over 1 MB requires an explicit justification before
   staging.
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
- the result cannot be classified with one of the R5 decision labels.
