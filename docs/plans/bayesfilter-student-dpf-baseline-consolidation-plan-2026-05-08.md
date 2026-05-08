# Plan: student DPF baseline consolidation and comparison

## Date

2026-05-08

## Status Update

Updated on 2026-05-09 after the initial student-source snapshot.

Completed:

- `2026MLCOE` was cloned under
  `experiments/student_dpf_baselines/vendor/2026MLCOE` at commit
  `020cfd7f2f848afa68432e95e6c6e747d3d2402d`.
- `advanced_particle_filter` was cloned under
  `experiments/student_dpf_baselines/vendor/advanced_particle_filter` at
  commit `d2a797c330e11befacbb736b5c86b8d03eb4a389`.
- Snapshot provenance was recorded in:
  - `experiments/student_dpf_baselines/sources.yml`;
  - `experiments/student_dpf_baselines/PERMISSIONS.md`;
  - `experiments/student_dpf_baselines/vendor/SNAPSHOT.md`.

Immediate next blocker:

- The vendored repositories are currently nested Git repositories.  Before the
  parent BayesFilter repo commits the experimental snapshot, decide whether to
  keep them as submodules or convert them to plain source snapshots.  The
  recommended choice for this project is plain snapshots, because the goal is
  reproducible internal comparison, not tracking upstream branches.

## Purpose

This plan defines the experimental work needed to consolidate, run, and compare
the two student differentiable particle-filter implementations.  The purpose
is to produce a reproducible experimental baseline for BayesFilter development.

This is not a production implementation plan.  Student code is
`comparison_only` and remains quarantined under:

```text
experiments/student_dpf_baselines/
```

Production BayesFilter code under `bayesfilter/` must not import from, depend
on, or copy implementation code from this experimental track.

## Inputs

Student sources:

- `https://github.com/ljw9510/2026MLCOE`
- `https://github.com/younghwan-cho-dev/advanced_particle_filter`

Permission status:

- The students have granted permission to use and test their code internally
  for this project.
- Redistribution/public release status must still be recorded before any
  vendored source is published outside the project.

BayesFilter references:

- `docs/differentiable-particle-filter-program.md`
- `docs/plans/bayesfilter-differentiable-particle-filter-phase1-audit-plan-2026-05-08.md`
- `experiments/student_dpf_baselines/`
- `experiments/controlled_dpf_baseline/`
- `bayesfilter/testing/structural_fixtures.py`
- `bayesfilter/filters/kalman.py`
- `bayesfilter/filters/sigma_points.py`
- `bayesfilter/filters/particles.py`

## Non-goals

- Do not promote student code into `bayesfilter/`.
- Do not use student results as correctness certification.
- Do not use student code for HMC target construction.
- Do not rewrite their implementations before first reproducing their original
  behavior.
- Do not make production API decisions based only on student implementation
  convenience.

## Success Criteria

The consolidation track succeeds when it produces:

1. provenance records for both student implementations;
2. runnable, isolated snapshots or wrappers;
3. a common fixture and metric harness;
4. reproduction results for each student's original reported examples where
   feasible;
5. shared-fixture comparison results against Kalman and UKF/EKF references;
6. a written consistency report that classifies each implementation as:
   - `student_reproduced`;
   - `student_cross_consistent`;
   - `student_reference_consistent`;
   - `student_reference_failed`;
   - `student_unusable_without_rewrite`;
   - `blocked_missing_dependency`;
   - `blocked_missing_assumption`.

## Directory Plan

Use the existing scaffold:

```text
experiments/student_dpf_baselines/
  PERMISSIONS.md
  README.md
  sources.yml
  adapters/
  fixtures/
  runners/
  reports/
  vendor/
```

Expected additions:

```text
experiments/student_dpf_baselines/
  adapters/
    common.py
    mlcoe_adapter.py
    advanced_particle_filter_adapter.py
  fixtures/
    common_fixtures.py
    fixture_catalog.yml
  runners/
    run_student_baseline_panel.py
    compare_student_outputs.py
  reports/
    student-dpf-baseline-consistency-result-2026-05-08.md
    outputs/
      *.json
  vendor/
    2026MLCOE/
    advanced_particle_filter/
    SNAPSHOT.md
```

If vendoring is not appropriate, replace `vendor/` with external clone paths
recorded in `sources.yml`.  The comparison harness should work through
adapters either way.

Storage policy:

- Preferred: plain source snapshots under `vendor/`, with upstream commit
  hashes recorded in `sources.yml` and `vendor/SNAPSHOT.md`.
- Avoid: live submodules unless a later maintenance policy explicitly requires
  submodule update workflows.
- Avoid: adapters that depend on upstream `main` or any branch name.
- Required: every baseline report must cite the snapshot commit hash.

## Result Contract

All adapters should normalize their output to a common result object or JSON
schema:

```text
BaselineResult
  implementation_name: string
  source_commit: string | null
  fixture_name: string
  seed: int | null
  num_particles: int | null
  dtype: string | null
  status: ok | failed | skipped
  failure_reason: string | null
  log_likelihood: float | null
  likelihood_surrogate: float | null
  filtered_means: array | null
  filtered_covariances: array | null
  particle_means: array | null
  particle_covariances: array | null
  ess_by_time: array | null
  resampling_count: int | null
  runtime_seconds: float | null
  gradient_available: bool
  gradient_value: array | null
  diagnostics: object
```

The adapter must report unavailable fields as null.  It must not fabricate
metrics that the student implementation does not expose.

## Comparison Metrics

For linear-Gaussian fixtures:

- total log-likelihood error against Kalman;
- per-time-step log-likelihood error if available;
- filtered mean trajectory error;
- filtered covariance trajectory error if available;
- particle mean/covariance error if only particles are available;
- ESS trajectory;
- runtime;
- failure modes.

For nonlinear fixtures:

- comparison against BayesFilter sigma-point/UKF-style backend where
  applicable;
- filtered mean trajectory distance;
- predictive log-likelihood or surrogate difference where available;
- particle degeneracy and ESS;
- sensitivity to seed and particle count;
- runtime and numerical failure modes.

For cross-student comparison:

- same fixture, seed, particle count, and dtype where possible;
- difference in log-likelihood or surrogate;
- trajectory distance between filtered means;
- ESS and resampling behavior;
- qualitative agreement/failure classification.

## Phase S0: provenance and permissions

Status: mostly complete.

### Actions

1. Update `experiments/student_dpf_baselines/PERMISSIONS.md` with:
   - student names or source owners;
   - permission date;
   - allowed use: internal testing and comparison;
   - redistribution status;
   - any restrictions.
2. Update `experiments/student_dpf_baselines/sources.yml` with:
   - source URL;
   - commit hash or archive ID;
   - copy date;
   - local path or vendor path;
   - implementation notes.
3. Record whether each repo has dependency files, reports, notebooks, and
   executable examples.

### Current Records

| Source | Commit | Local path | Dependency record | Status |
| --- | --- | --- | --- | --- |
| `2026MLCOE` | `020cfd7f2f848afa68432e95e6c6e747d3d2402d` | `experiments/student_dpf_baselines/vendor/2026MLCOE` | README only found in shallow audit | snapshotted |
| `advanced_particle_filter` | `d2a797c330e11befacbb736b5c86b8d03eb4a389` | `experiments/student_dpf_baselines/vendor/advanced_particle_filter` | `requirements.txt` found | snapshotted |

### Exit Gate

No code is copied or wrapped until provenance and permission records are
complete enough for internal audit.

Current gate status: passed for internal experimental use; redistribution
status remains unconfirmed.

## Phase S1: snapshot or clone sources

Status: clone step complete; storage normalization pending.

### Actions

1. Obtain both student repositories at fixed commits.
2. Prefer clean snapshots under `vendor/` only if internal vendoring is
   appropriate.
3. Otherwise store them externally and record absolute local paths in
   `sources.yml`.
4. Do not edit vendored code in place except for minimal documented patches
   needed to run it.  Prefer wrappers.

5. Normalize storage before committing to BayesFilter:
   - Option A, preferred: remove nested `.git` directories and keep plain
     source snapshots;
   - Option B: convert the two repositories to explicit Git submodules and
     document the submodule update policy;
   - Option C: move clones outside the repo and record absolute local paths in
     `sources.yml`.

Recommendation: use Option A for the first baseline.  It gives a stable,
reviewable snapshot and avoids accidental movement with upstream student work.

### Required Record

For each source:

| Source | Commit | Local path | Dependency file | Original example command | Notes |
| --- | --- | --- | --- | --- | --- |
| `2026MLCOE` | `020cfd7f2f848afa68432e95e6c6e747d3d2402d` | `experiments/student_dpf_baselines/vendor/2026MLCOE` | none found beyond README | pending | nested Git repo currently |
| `advanced_particle_filter` | `d2a797c330e11befacbb736b5c86b8d03eb4a389` | `experiments/student_dpf_baselines/vendor/advanced_particle_filter` | `requirements.txt` | pending | nested Git repo currently |

### Exit Gate

Both sources are reproducibly locatable, or blockers are recorded.

Current gate status: partially passed.  Sources are locatable and pinned, but
the nested-Git storage decision must be resolved before commit.

## Phase S1A: normalize vendored snapshot storage

### Motivation

The current clones are nested Git repositories.  If committed as-is, the parent
repo may record them as embedded repositories or require submodule semantics.
That is unnecessary for this comparison track and increases maintenance risk.

### Actions

1. Confirm the two nested repositories are clean:

   ```bash
   git -C experiments/student_dpf_baselines/vendor/2026MLCOE status --short --branch
   git -C experiments/student_dpf_baselines/vendor/advanced_particle_filter status --short --branch
   ```

2. Confirm the recorded commits still match `sources.yml`.
3. Convert to plain snapshots by removing only the nested `.git` metadata from:

   ```text
   experiments/student_dpf_baselines/vendor/2026MLCOE/.git
   experiments/student_dpf_baselines/vendor/advanced_particle_filter/.git
   ```

4. Preserve `vendor/SNAPSHOT.md` and `sources.yml` as the provenance record.
5. Re-run:

   ```bash
   git status --short
   git diff --check
   ```

### Exit Gate

The parent repo sees the vendored files as ordinary experimental files, not as
submodules or embedded repositories, and provenance still records upstream
commit hashes.

### Rollback Policy

If submodules are preferred later, recreate them deliberately in a separate
commit and document the update workflow.  Do not mix plain-snapshot and
submodule storage in the same result.

## Phase S2: dependency and environment audit

Status: not started.

### Actions

1. Identify Python version and package requirements for each source.
2. Record framework stack:
   - TensorFlow;
   - TFP;
   - NumPy/SciPy;
   - PyTorch/JAX if present;
   - notebook-only dependencies.
3. Attempt import-only smoke checks in isolated commands.
4. Record dependency conflicts with BayesFilter's current environment.

### Required Record

| Source | Python | Framework | Import status | Missing deps | Conflict risk |
| --- | --- | --- | --- | --- | --- |

### Exit Gate

Each source is classified as runnable, runnable with environment work, or
blocked.

## Phase S3: reproduce original student results

Status: not started.

### Actions

1. Locate the original report examples and scripts.
2. Run the smallest original example for each implementation.
3. Record exact commands, seeds, runtime, output artifacts, and deviations
   from the reported result.
4. If notebooks are the only entry point, extract the minimum executable
   commands into a runner without changing algorithm code.

### Required Record

| Source | Example | Command | Reported result | Reproduced result | Status |
| --- | --- | --- | --- | --- | --- |

### Exit Gate

Each implementation is either reproduced, partially reproduced, or blocked with
a specific reason.

## Phase S4: adapter layer

Status: not started.

### Actions

1. Implement `adapters/common.py` with:
   - `BaselineResult`;
   - serialization helpers;
   - status labels;
   - metric helpers.
2. Implement one adapter per student implementation.
3. Adapters must call student code as-is where possible.
4. Adapters must normalize inputs:
   - observations;
   - seed;
   - number of particles;
   - model fixture;
   - dtype;
   - resampling/config options when available.
5. Adapters must normalize outputs to `BaselineResult`.

### Exit Gate

Both adapters can run a trivial smoke fixture or produce a structured
`blocked_*` result.

## Phase S5: common fixtures

Status: not started.

### Actions

Create a fixture catalog with at least:

1. `linear_gaussian_ar2_short`
   - based on `AR2StructuralModel`;
   - exact Kalman reference;
   - short time series for smoke checks.
2. `linear_gaussian_ar2_panel`
   - multiple seeds and particle counts;
   - checks convergence trend to Kalman.
3. `nonlinear_accumulation_short`
   - based on `NonlinearAccumulationModel`;
   - BayesFilter sigma-point comparison where appropriate.
4. `flow_stiffness_small_R`
   - smaller observation noise;
   - tests numerical stability and ESS collapse.

### Required Record

| Fixture | Model | Reference | Seeds | Particle counts | Metrics |
| --- | --- | --- | --- | --- | --- |

### Exit Gate

At least one linear-Gaussian and one nonlinear fixture can be fed through both
adapters or produces explicit adapter blockers.

## Phase S6: reference computations

Status: not started.

### Actions

1. Compute Kalman references for linear-Gaussian fixtures.
2. Compute BayesFilter sigma-point/UKF references for nonlinear fixtures where
   appropriate.
3. Save reference outputs as JSON or NPZ under `reports/outputs/`.
4. Record reference commands and tolerances.

### Exit Gate

Student comparisons use independent BayesFilter references, not only
student-vs-student agreement.

## Phase S7: comparison panel

Status: not started.

### Actions

Run each implementation on each fixture over a small panel:

```text
seeds: 0, 1, 2
particles: 128, 512, 2048 where supported
dtypes: default implementation dtype, plus float64 if supported
```

For each run, save:

- normalized `BaselineResult`;
- raw implementation output when practical;
- command metadata;
- runtime;
- failure reason if any.

### Exit Gate

The output directory contains machine-readable results sufficient to produce
tables without rerunning the experiments.

## Phase S8: consistency analysis

Status: not started.

### Actions

Generate comparison tables:

1. Student A vs Kalman / UKF reference.
2. Student B vs Kalman / UKF reference.
3. Student A vs Student B.
4. Student A/B vs controlled experimental baseline if available.

Classify each run:

| Status | Meaning |
| --- | --- |
| `student_reference_consistent` | Agrees with declared reference within tolerance. |
| `student_cross_consistent` | Students agree with each other within tolerance. |
| `student_reference_failed` | Runs but misses reference tolerance. |
| `student_unusable_without_rewrite` | Requires substantial rewrite before comparison. |
| `blocked_missing_dependency` | Cannot run due to missing or conflicting dependency. |
| `blocked_missing_assumption` | Cannot map algorithm assumptions to fixture. |

### Exit Gate

The analysis separates reference agreement from cross-student agreement.

## Phase S9: report and handoff

Status: not started.

### Actions

Write:

```text
experiments/student_dpf_baselines/reports/student-dpf-baseline-consistency-result-2026-05-08.md
```

The report must include:

- source provenance;
- environment status;
- reproduction status;
- fixture definitions;
- metric tables;
- reference comparison;
- cross-student comparison;
- failure modes;
- implementation ideas worth reimplementing;
- explicit statement that student code remains `comparison_only`.

Also write a short docs result note under `docs/plans/` if the findings affect
the main BayesFilter DPF plan.

### Exit Gate

The report identifies which experimental results are useful for Phase 2
BayesFilter development and which claims remain blocked.

## Quality Gates

Before accepting the consolidation result:

1. No `bayesfilter/` module imports from `experiments/`.
2. Student source provenance is recorded.
3. Every result is tied to a fixture, seed, particle count, and commit/source
   identifier.
4. Reference comparisons use Kalman or BayesFilter nonlinear references where
   available.
5. Cross-student agreement is not treated as correctness.
6. All unavailable metrics are marked null or unavailable, not silently
   omitted.
7. Any required patch to student code is documented separately from the
   original snapshot.

## Recommended Execution Order

Execute in this order:

1. Complete S1A storage normalization.
2. S2 dependency audit.
3. S3 reproduce original examples.
4. S4 adapters.
5. S5 common fixtures.
6. S6 references.
7. S7 comparison panel.
8. S8 consistency analysis.
9. S9 report and handoff.

If a source blocks during S2 or S3, continue with the other source and record
the blocker.  The harness should still be useful for future reruns.

Completed prerequisites:

- S0 provenance and permissions are recorded for internal use.
- S1 source clones are present at fixed commits.
