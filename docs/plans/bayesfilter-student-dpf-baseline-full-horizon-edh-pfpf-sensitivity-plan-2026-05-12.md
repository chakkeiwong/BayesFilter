# Plan: student DPF full-horizon EDH/PFPF sensitivity panel

## Date

2026-05-12

## Status

Active next-phase plan for the quarantined student DPF experimental-baseline
stream.

This plan follows the replicated EDH/PFPF panel:
`docs/plans/bayesfilter-student-dpf-baseline-replicated-edh-pfpf-panel-plan-2026-05-11.md`.

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
  resampling, HMC, MLE, notebook, or plotting workflow claims.

Known drift hazard:

- the repository may contain dirty or untracked monograph-lane files from
  another agent.  They must not be edited, staged, committed, or used as
  evidence for this phase.

## Goal

Test whether the replicated EDH/PFPF baseline remains usable when moving from
the reduced 8-observation panel to the full 20-observation nonlinear
range-bearing fixtures and a bounded particle/flow-step sensitivity grid.

This phase remains comparison-only.  It must not certify correctness,
production quality, HMC suitability, or cross-student superiority.

## Current Evidence

The replicated panel produced 12/12 `ok` records on reduced 8-observation
fixtures using:

- fixtures: `range_bearing_gaussian_moderate`,
  `range_bearing_gaussian_low_noise`;
- seeds: `17`, `23`, `31`;
- particles: `64`;
- flow steps: `10`;
- implementations: advanced `EDHParticleFilter`, MLCOE `PFPF_EDH`;
- decision: `replicated_panel_ready`.

Low-noise pressure was clear for MLCOE across ESS, resampling count, runtime,
and position RMSE.  For advanced, low-noise pressure appeared mainly in lower
minimum ESS and slightly higher runtime.

## Goals for This Phase

1. Test both EDH/PFPF implementations on the full 20-observation fixtures.
2. Measure whether increasing particles from 64 to 128 reduces low-noise ESS
   and resampling pressure.
3. Measure whether increasing flow steps from 10 to 20 improves proxy RMSE
   enough to justify additional runtime.
4. Preserve implementation-specific ESS and resampling semantics.
5. Decide whether the EDH/PFPF baseline can support a controlled full-horizon
   sensitivity artifact or needs targeted debug.

The planned run matrix is exactly:

- fixtures:
  - `range_bearing_gaussian_moderate`;
  - `range_bearing_gaussian_low_noise`;
- horizon: full fixture horizon, 20 observations;
- seeds: `17`, `23`;
- particles: `64`, `128`;
- flow steps: `10`, `20`;
- implementations:
  - `advanced_particle_filter` `EDHParticleFilter`;
  - `2026MLCOE` `PFPF_EDH`;
- total planned records: 32.

Runtime-warning threshold:

- 45 seconds per implementation/fixture/seed/particle-count/flow-step run.

Artifact-size threshold:

- any generated artifact above 1 MB requires explicit justification before
  staging.

## Remaining Gaps

### Gap 1: full-horizon behavior is untested

The replicated panel used only the first 8 observations.  It does not prove the
same adapter paths remain bounded and finite for the full 20-observation
fixtures.

Closure target:

- run both selected implementations on the full fixture horizon.

### Gap 2: particle-count sensitivity is unknown

The replicated panel used 64 particles only.  Low-noise pressure may improve
with more particles, or it may expose implementation-specific runtime and
resampling costs.

Closure target:

- compare 64 and 128 particles using the same fixtures and seeds.

### Gap 3: flow-step sensitivity is unknown

The replicated panel used 10 flow steps only.  More flow steps may improve
proxy RMSE or may only increase runtime.

Closure target:

- compare 10 and 20 flow steps using the same fixtures, seeds, and particle
  counts.

### Gap 4: ESS/resampling semantics remain implementation-specific

The two implementations expose ESS and resampling pressure differently.

Closure target:

- report ESS and resampling pressure with explicit semantics labels;
- do not collapse them into one cross-implementation correctness metric.

### Gap 5: comparison remains proxy-only

Latent-state RMSE and observation proxy RMSE are available because the fixtures
are simulated.  These are diagnostics, not correctness certificates.

Closure target:

- report latent-state RMSE, final-position error, observation proxy RMSE, ESS,
  resampling, runtime, finite-output checks, and failure blockers;
- preserve comparison-only interpretation.

## Hypotheses

### FH1: full horizon remains runnable

Both selected implementations can run the full 20-observation fixtures across
the bounded sensitivity grid without vendored-code edits.

Primary criterion:

- every planned run either returns finite structured metrics or a structured
  blocker;
- at least one run per implementation and fixture is `ok`.

Veto diagnostics:

- running the panel requires vendored-code edits, production imports, or target
  changes;
- one implementation fails all runs on a fixture with an unclassified error.

### FH2: more particles reduce low-noise pressure

Increasing particles from 64 to 128 should reduce low-noise ESS pressure,
resampling pressure, or proxy RMSE for at least one implementation.

Primary criterion:

- low-noise summaries show lower pressure at 128 particles than 64 particles in
  at least one of average ESS, minimum ESS, resampling count, position RMSE, or
  observation proxy RMSE.

Veto diagnostics:

- particle-count comparisons mix fixtures, seeds, flow steps, or implementations
  without labels;
- report claims superiority without metric support.

### FH3: more flow steps have bounded benefit

Increasing flow steps from 10 to 20 should either improve proxy RMSE at bounded
runtime cost or be classified as not useful for this grid.

Primary criterion:

- flow-step summaries compare 10 versus 20 steps at fixed fixture,
  implementation, seed, and particle count;
- runtime and proxy-RMSE tradeoffs are reported explicitly.

Veto diagnostics:

- runtime warnings dominate 20-step runs;
- report recommends more flow steps without RMSE or runtime evidence.

### FH4: proxy comparison remains interpretable

The panel can be interpreted through declared target labels and proxy metrics.

Primary criterion:

- report includes fixture, seed, particle count, flow steps, target labels,
  finite-output checks, RMSE metrics, ESS availability, resampling semantics,
  and runtime for every record.

Veto diagnostics:

- report treats student agreement as correctness;
- likelihoods are used as primary cross-student evidence;
- missing metrics are hidden rather than recorded as nulls or blockers.

### FH5: next baseline decision is clear

The phase should decide whether full-horizon EDH/PFPF sensitivity can become a
stable experimental baseline artifact.

Primary criterion:

- result decision is one of:
  - `full_horizon_sensitivity_ready`;
  - `full_horizon_sensitivity_ready_with_caveats`;
  - `needs_targeted_debug`;
  - `blocked_or_excluded`.

Veto diagnostics:

- evidence remains too ambiguous to justify the next action.

## Phase FH0: Preflight and Lane Guard

Actions:

1. Record current Git status.
2. Confirm this plan exists and the student reset memo points to this lane.
3. Confirm no production import boundary:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

4. Confirm no vendored student files are dirty.
5. Confirm the independent audit artifact exists:
   `docs/plans/bayesfilter-student-dpf-baseline-full-horizon-edh-pfpf-sensitivity-plan-audit-2026-05-12.md`.
6. Update only the student reset memo with the FH0 result.

Exit gate:

- proceed only if edits can remain in the student-baseline lane.

## Phase FH1: Sensitivity Runner

Actions:

1. Add an experiment-owned runner under
   `experiments/student_dpf_baselines/runners/`.
2. Reuse adapter-owned model bridges from the EDH/PFPF runner.
3. Run:
   - fixtures: `range_bearing_gaussian_moderate`,
     `range_bearing_gaussian_low_noise`;
   - full horizon: 20 observations;
   - seeds: `17`, `23`;
   - particles: `64`, `128`;
   - flow steps: `10`, `20`.
4. Record structured per-run records for both implementations.
5. Use a 45 second per-run runtime-warning threshold.

Exit gate:

- each planned run returns either finite metrics or a structured blocker.

## Phase FH2: Classification and Report

Actions:

1. Summarize by implementation, fixture, particle count, and flow steps.
2. Summarize particle-count sensitivity on the low-noise fixture.
3. Summarize flow-step runtime/RMSE tradeoffs.
4. Classify FH1-FH5.
5. Expected artifacts:
   - `experiments/student_dpf_baselines/reports/outputs/full_horizon_edh_pfpf_sensitivity_2026-05-12.json`;
   - `experiments/student_dpf_baselines/reports/outputs/full_horizon_edh_pfpf_sensitivity_summary_2026-05-12.json`;
   - `experiments/student_dpf_baselines/reports/student-dpf-baseline-full-horizon-edh-pfpf-sensitivity-result-2026-05-12.md`.
6. Update only the student reset memo with results and next-phase
   justification.

Exit gate:

- report gives a clear decision and preserves comparison-only interpretation.

## Phase FH3: Audit, Tidy, Reset Memo, Commit

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
- the result cannot be classified with one of the FH5 decision labels.
