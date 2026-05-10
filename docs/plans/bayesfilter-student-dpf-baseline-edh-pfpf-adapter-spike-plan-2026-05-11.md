# Plan: student DPF EDH/PFPF adapter spike

## Date

2026-05-11

## Status

Active next-phase plan for the quarantined student DPF experimental-baseline
stream.

This plan follows the MP4 flow and DPF readiness review:
`docs/plans/bayesfilter-student-dpf-baseline-mp4-flow-dpf-readiness-review-plan-2026-05-11.md`.

## Lane Boundary

Owned paths:

- `experiments/student_dpf_baselines/`;
- student-baseline plans, audits, reports, and reset memo under `docs/plans/`.

Out of scope:

- DPF monograph writing files;
- DPF monograph reset memo;
- `docs/chapters/ch19*.tex`;
- `docs/references.bib`;
- production `bayesfilter/` code;
- editing vendored student code in place;
- copying student implementation code into production;
- kernel PFF, stochastic flow, DPF, dPFPF, neural OT, differentiable
  resampling, HMC, or MLE comparison claims.

## Goal

Run the one bounded candidate selected by MP4:

- `advanced_particle_filter.filters.edh.EDHParticleFilter`;
- `2026MLCOE` `src.filters.flow_filters.PFPF_EDH`.

The purpose is not to certify correctness or performance.  The purpose is to
test whether both selected paths can run through adapter-owned bridges on an
existing nonlinear Gaussian range-bearing fixture and produce finite,
comparison-only proxy diagnostics.

## Current Evidence

MP4 found:

- 28 flow/DPF-related candidate surfaces;
- all 28 importable for signature inspection;
- EDH/PFPF-EDH was the only immediate bounded comparison candidate;
- kernel PFF remained excluded due to MP3 max-iteration diagnostics;
- DPF, dPFPF, neural OT, differentiable resampling, stochastic flow, and HMC
  require separate reproduction gates.

MP2 already provides:

- shared nonlinear Gaussian range-bearing fixtures;
- adapter-local advanced and MLCOE model bridges;
- trajectory RMSE and proxy metrics;
- EKF/UKF/BPF comparison context.

## Gaps to Close

### Gap 1: Signature readiness is not runtime readiness

MP4 showed both selected classes import and expose call surfaces.  It did not
instantiate or run either path.

Closure target:

- instantiate each selected path only through adapter-owned runner code;
- run a reduced fixture with explicit particle, horizon, flow-step, and runtime
  bounds;
- convert runtime failures to structured blocker records.

### Gap 2: Model-interface compatibility is untested

The selected paths expect different model contracts:

- advanced expects an `advanced_particle_filter.models.base.StateSpaceModel`;
- MLCOE expects a TensorFlow model with `F`, `Q_filter`, `R_filter`,
  `R_inv_filter`, `h_func`, and `jacobian_h`.

Closure target:

- reuse MP2 adapter-owned bridge patterns;
- do not modify either vendored snapshot;
- classify missing or incompatible assumptions as `blocked_missing_assumption`.

### Gap 3: Metrics must remain proxy metrics

EDH/PFPF outputs are stochastic and implementation-specific.  Cross-student
agreement alone is not correctness evidence.

Closure target:

- report latent-state RMSE, final-position error, average/min ESS if exposed,
  inferred resampling count where available, finite-output checks, and runtime;
- compare against MP2 EKF/UKF context only as a proxy baseline;
- do not compare likelihoods as primary evidence.

### Gap 4: Runtime and numerical risk are unknown

Particle-flow implementations can become slow or unstable.

Closure target:

- use one reduced fixture first;
- cap the first spike at at most 64 particles, at most 10 flow steps, and a
  short horizon;
- stop if either implementation needs broad runtime or model changes.

## Hypotheses

### E1: selected advanced EDH PFPF runs

`advanced_particle_filter.filters.edh.EDHParticleFilter` can run on the reduced
nonlinear Gaussian range-bearing fixture with adapter-owned model construction.

Primary criterion:

- one bounded run returns finite means, finite RMSE, finite runtime, and
  structured ESS/resampling diagnostics if exposed.

Veto diagnostics:

- requires vendored-code edits;
- requires changing the fixture target;
- fails due to undocumented model assumptions that cannot be represented by
  the adapter-owned bridge.

### E2: selected MLCOE PFPF_EDH runs

MLCOE `src.filters.flow_filters.PFPF_EDH` can run on the same reduced fixture
with an adapter-owned TensorFlow model bridge.

Primary criterion:

- one bounded run returns finite means, finite RMSE, finite runtime, and
  structured ESS/resampling diagnostics if exposed.

Veto diagnostics:

- requires vendored-code edits;
- requires production code imports;
- requires changing the MLCOE model contract beyond adapter-owned bridge
  fields.

### E3: proxy comparison is interpretable

If both paths run, their outputs can be interpreted through declared target
semantics and proxy metrics.

Primary criterion:

- the report states target labels, fixture, particle count, flow steps, seed,
  runtime, latent-state RMSE, final-position error, ESS availability, and
  finite-output checks.

Veto diagnostics:

- report treats student agreement as correctness;
- report hides missing metrics instead of recording nulls or blockers;
- likelihoods are used as the primary cross-student metric.

### E4: next phase can be decided

The spike should produce a clear next decision.

Primary criterion:

- decide one of:
  - `edh_pfpf_panel_ready`;
  - `adapter_spike_success_needs_replication`;
  - `blocked_missing_assumption`;
  - `excluded_due_to_runtime_or_numerics`.

Veto diagnostics:

- evidence remains too ambiguous to justify or reject a later replicated panel.

## Phase S0: Preflight and Lane Guard

Actions:

1. Record current Git status.
2. Confirm this plan and its audit are present.
3. Confirm no production import boundary:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

4. Confirm no vendored student files are dirty.
5. Update the student reset memo with the S0 result.

Exit gate:

- proceed only if edits can remain in the student-baseline lane.

## Phase S1: Runner Implementation

Actions:

1. Add an experiment-owned runner under
   `experiments/student_dpf_baselines/runners/`.
2. Reuse existing nonlinear fixture/model bridge logic where possible.
3. Run the reduced fixture:
   - base fixture: `range_bearing_gaussian_moderate`;
   - horizon: at most 8 observations;
   - particles: at most 64;
   - flow steps: at most 10;
   - fixed seed.
4. Record one structured record for each selected implementation.
5. Preserve missing ESS/resampling metrics as null or structured diagnostics.

Exit gate:

- both implementations either return finite structured records or structured
  blockers.

## Phase S2: Execute, Classify, and Report

Actions:

1. Execute the runner with `python -m`.
2. Write JSON outputs and a Markdown report under
   `experiments/student_dpf_baselines/reports/`.
3. Classify E1-E4.
4. Decide whether a replicated EDH/PFPF panel is justified.
5. Update the student reset memo with results and interpretation.

Exit gate:

- result decision is one of the E4 decision labels and is supported by
  structured evidence.

## Phase S3: Audit, Tidy, Reset Memo, Commit

Actions:

1. Run syntax checks for edited modules.
2. Run import-boundary search over production code and tests.
3. Run `git diff --check`.
4. Confirm no vendored student code changed.
5. Check generated artifact sizes.
6. Update the student reset memo with completion state and next hypotheses.
7. Stage and commit only student-baseline files.

Exit gate:

- scoped commit excludes monograph-lane files, production code, references, and
  vendored student code.

## Stop Rules

Stop and ask for direction if:

- the runner needs edits outside the student-baseline lane;
- either selected path requires vendored-code edits;
- either selected path requires production `bayesfilter/` imports;
- the fixture target must be changed to make the result run;
- runtime becomes unbounded or generated artifacts are too large;
- the result cannot be classified with one of the E4 decision labels.
