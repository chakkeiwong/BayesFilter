# Plan: student DPF baseline gap closure

## Date

2026-05-09

## Status

Active plan for closing the remaining gaps in the quarantined student DPF
experimental-baseline stream.

This plan supersedes the open-ended execution portions of
`docs/plans/bayesfilter-student-dpf-baseline-consolidation-plan-2026-05-08.md`
for the next work cycle.  The older plan remains useful as the broader design
record.  This plan is narrower: it starts from the current snapshot state and
defines the concrete steps needed before adapter and comparison work are
justified.

## Scope boundary

This is a student experimental-baseline plan only.

Owned paths:

- `experiments/student_dpf_baselines/`;
- `experiments/controlled_dpf_baseline/`, if a later phase creates the
  controlled reference harness;
- this plan and the student reset memo under `docs/plans/`.

Out of scope:

- `docs/chapters/ch19*.tex`;
- reader-facing DPF monograph plans or reset memo updates;
- production `bayesfilter/` code;
- HMC target claims for BayesFilter.

The active reset memo for this stream is:
`docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`.

## Motivation

The student snapshots already provide useful experimental material, but they
cannot yet serve as a defensible baseline because the two sources are at
different levels of reproduction:

- `2026MLCOE` has passed unit and integration tests in the current local
  environment.
- `advanced_particle_filter` has passed import smoke checks only.  It has not
  yet produced an original-example or targeted-test baseline in this repo.

Starting adapter work before `advanced_particle_filter` has a minimal
reproduction result would be premature.  Adapter failures would be ambiguous:
they could reflect adapter mistakes, environment drift, or broken original
student behavior.  The next phase must therefore establish a small,
documented `advanced_particle_filter` baseline first.

The goal is not to validate student code as correct.  The goal is to create a
reproducible comparison artifact with enough provenance, commands, outputs, and
failure records that later BayesFilter work can use it as experimental evidence
without depending on student implementation quality.

## Current baseline state

Snapshots:

| Source | Snapshot commit | Current status |
| --- | --- | --- |
| `2026MLCOE` | `020cfd7f2f848afa68432e95e6c6e747d3d2402d` | Unit tests and integration tests passed. |
| `advanced_particle_filter` | `d2a797c330e11befacbb736b5c86b8d03eb4a389` | Import smoke checks passed; reproduction pending. |

Existing records:

- `experiments/student_dpf_baselines/sources.yml`;
- `experiments/student_dpf_baselines/PERMISSIONS.md`;
- `experiments/student_dpf_baselines/vendor/SNAPSHOT.md`;
- `experiments/student_dpf_baselines/reports/dependency-audit-2026-05-09.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`.

Known environment caveats:

- current audit observed Python `3.13.13`, TensorFlow `2.20.0`, and TensorFlow
  Probability `0.25.0`;
- `advanced_particle_filter` README says its TensorFlow side was tested with
  older TF/TFP versions;
- `matplotlib` and `numba` were missing during the dependency audit;
- plotting-heavy scripts must be classified separately from non-plotting core
  tests.

## Success criteria

This gap-closure cycle succeeds when:

1. `advanced_particle_filter` has at least one completed targeted original
   baseline, or a precise blocker report if that is impossible.
2. The student reset memo records the baseline result and whether adapter work
   is justified.
3. Adapter work starts only after both snapshots have a minimal reproduction
   status.
4. Any adapters use structured failure results instead of uncaught exceptions
   for expected student-code incompatibilities.
5. No production `bayesfilter/` module imports from `experiments/`.
6. All outputs cite source commit, command, environment, seed, particle count
   where applicable, and status.

## Result labels

Use these labels consistently in reports and machine-readable outputs:

| Label | Meaning |
| --- | --- |
| `reproduced_original` | Original or README-described behavior ran successfully. |
| `reproduced_targeted` | A targeted test/example ran successfully, but not a full original report. |
| `blocked_missing_dependency` | Missing dependency prevents the run. |
| `blocked_environment_drift` | Version mismatch or runtime incompatibility prevents the run. |
| `blocked_missing_assumption` | The run requires undocumented assumptions or unavailable data. |
| `failed_algorithmic` | Code runs but returns invalid numerical behavior. |
| `adapter_justified` | Enough reproduction exists to start adapter normalization. |
| `adapter_not_justified` | Reproduction is too weak or ambiguous to start adapters. |

## Phase G0: preflight and contamination guard

### Motivation

The repository currently contains unrelated DPF monograph writing changes.
This phase prevents the student stream from staging or editing those files.

### Actions

1. Record current Git status.
2. Confirm this plan only touches:
   - `docs/plans/bayesfilter-student-dpf-baseline-gap-closure-plan-2026-05-09.md`;
   - `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
   - later student-baseline experiment files under `experiments/`.
3. Do not edit:
   - `docs/chapters/ch19_particle_filters.tex`;
   - `docs/references.bib`;
   - DPF monograph rebuild plans or memo.

### Implementation details

Use:

```bash
git status --short --branch
git diff -- docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md
```

Before any future commit, stage only student-baseline files.

### Exit gate

Proceed only if the working tree contamination is understood and this stream's
edits remain scoped.

## Phase G1: advanced_particle_filter targeted reproduction

### Motivation

`advanced_particle_filter` is the main unresolved gap.  Adapter work should not
begin until this snapshot has demonstrated at least minimal behavior using its
own tests or README quick start.

### Actions

Run small, non-plotting entry points first:

1. README quick-start Kalman example, translated into a local runner command
   without modifying vendored code.
2. `tests/test_basic.py`, because the README identifies it as an integration
   test that can be run directly.
3. Selected pytest tests:
   - `tests/test_filters.py`;
   - `tests/test_kernel_pff.py`;
   - `tests/test_tf_migration.py` only after NumPy-side tests are recorded,
     because it may be slower and more sensitive to TF/TFP drift.

Avoid first-pass scripts that require plotting or large generated outputs:

- `scripts/dai2022_example1.py`;
- `scripts/filter_comparison.py`;
- acoustic and notebook workflows.

These may be revisited after the minimal baseline is established.

### Implementation details

Use the vendored package as a package from its parent directory:

```bash
PYTHONPATH=experiments/student_dpf_baselines/vendor \
python experiments/student_dpf_baselines/vendor/advanced_particle_filter/tests/test_basic.py
```

For pytest:

```bash
PYTHONPATH=experiments/student_dpf_baselines/vendor \
pytest experiments/student_dpf_baselines/vendor/advanced_particle_filter/tests/test_filters.py -q
```

If the current shell environment cannot run TensorFlow-side tests reliably,
record that as environment drift before using an alternate environment.  Do not
silently change the environment and rerun.

### Output record

Create:

```text
experiments/student_dpf_baselines/reports/advanced-particle-filter-reproduction-2026-05-09.md
experiments/student_dpf_baselines/reports/outputs/advanced_particle_filter_reproduction_2026-05-09.json
```

The report must include:

- exact command;
- working directory;
- `PYTHONPATH`;
- Python, NumPy, SciPy, TensorFlow, and TFP versions where available;
- runtime;
- pass/fail/blocker status;
- relevant stdout summary;
- whether the run justifies adapter work.

### Exit gate

Set one of:

- `adapter_justified`, if at least one meaningful NumPy-side or TF-side
  targeted test passes;
- `adapter_not_justified`, if all targeted tests fail or are blocked;
- `blocked_environment_drift`, if failures are clearly caused by dependency
  mismatch.

Do not proceed to adapters unless the exit state is `adapter_justified`.

## Phase G2: reset memo update and decision checkpoint

### Motivation

The reset memo is the continuity mechanism between agents.  It must contain
the result interpretation, not just a pointer to raw logs.

### Actions

Update
`docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`
with:

- the `advanced_particle_filter` targeted reproduction status;
- link to the reproduction report;
- interpretation of failures or passes;
- explicit decision on whether adapter work is justified.

### Exit gate

Future agents can read the reset memo and know whether to implement adapters
or first address reproduction blockers.

## Phase G3: minimal adapter contract

### Motivation

Once reproduction is established, adapters are needed to compare the two
student implementations on common fixtures without copying their code into
production.

### Actions

Create the common adapter contract:

```text
experiments/student_dpf_baselines/adapters/common.py
```

Define:

- `BaselineStatus`;
- `BaselineResult`;
- JSON serialization helpers;
- exception-to-result conversion.

Minimum fields:

```text
implementation_name
source_commit
fixture_name
seed
num_particles
dtype
status
failure_reason
log_likelihood
likelihood_surrogate
filtered_means
filtered_covariances
particle_means
particle_covariances
ess_by_time
resampling_count
runtime_seconds
gradient_available
gradient_value
diagnostics
```

Adapters must report unavailable fields as `null`.  They must not fabricate
metrics.

### Exit gate

The common contract can serialize both a successful result and a structured
blocked result.

## Phase G4: per-student smoke adapters

### Motivation

The first adapters should be intentionally small.  Their job is to prove that
each snapshot can be called through a stable wrapper and can return a structured
result, not to expose every algorithm.

### Actions

Create:

```text
experiments/student_dpf_baselines/adapters/mlcoe_adapter.py
experiments/student_dpf_baselines/adapters/advanced_particle_filter_adapter.py
```

Initial adapter targets:

- `2026MLCOE`: one linear-Gaussian or already-tested integration path.
- `advanced_particle_filter`: README quick-start Kalman or bootstrap PF path.

Do not adapt HMC, neural OT, or notebook-only workflows in this phase.

### Implementation details

Each adapter should expose one function:

```python
run_smoke_fixture(fixture, *, seed: int, num_particles: int | None) -> BaselineResult
```

Adapters may temporarily add vendored paths to `sys.path` inside a local context
manager, but no production module may import them.

### Exit gate

Both adapters either:

- return `status="ok"` on a small linear-Gaussian fixture; or
- return a structured blocker with `failure_reason`.

## Phase G5: common fixture and reference spine

### Motivation

Student-vs-student agreement is not correctness evidence.  Common fixtures need
independent references, starting with linear-Gaussian Kalman references.

### Actions

Create:

```text
experiments/student_dpf_baselines/fixtures/common_fixtures.py
experiments/student_dpf_baselines/fixtures/fixture_catalog.yml
experiments/student_dpf_baselines/runners/run_reference_fixtures.py
```

Minimum fixtures:

| Fixture | Purpose | Reference |
| --- | --- | --- |
| `lgssm_1d_short` | fastest smoke and serialization check | Kalman |
| `lgssm_cv_2d_short` | matches `advanced_particle_filter` README style | Kalman |
| `lgssm_cv_2d_low_particles` | exposes particle variance and ESS behavior | Kalman |

Defer nonlinear fixtures until the linear spine works.

### Output record

Write reference outputs under:

```text
experiments/student_dpf_baselines/reports/outputs/references/
```

### Exit gate

At least one fixture has a stable serialized Kalman reference and can be passed
to both smoke adapters or receive explicit adapter blockers.

## Phase G6: first comparison panel

### Motivation

The first comparison should be small enough to rerun often and structured
enough to expose differences in likelihood, filtered means, ESS, runtime, and
failure modes.

### Actions

Create:

```text
experiments/student_dpf_baselines/runners/run_student_baseline_panel.py
experiments/student_dpf_baselines/runners/compare_student_outputs.py
```

Initial panel:

```text
fixtures: lgssm_1d_short, lgssm_cv_2d_short
seeds: 0, 1, 2
particles: 128, 512
dtype: implementation default
```

Metrics:

- log-likelihood error against Kalman, if available;
- filtered mean RMSE against Kalman;
- covariance error, if available;
- average ESS, if available;
- runtime;
- structured failure rate.

### Output record

Write:

```text
experiments/student_dpf_baselines/reports/outputs/student_baseline_panel_2026-05-09.json
experiments/student_dpf_baselines/reports/student-dpf-baseline-gap-closure-result-2026-05-09.md
```

### Exit gate

The result report separates:

- reference agreement;
- cross-student agreement;
- implementation failures;
- environment blockers.

## Phase G7: audit, tidy, and handoff

### Motivation

The final artifact must be safe to commit without accidentally promoting
student code into production or mixing with the monograph writing stream.

### Actions

1. Run import-boundary checks:

   ```bash
   rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
   ```

2. Run syntax/import checks on new adapter and runner files.
3. Check generated output size before commit.
4. Update the student reset memo with:
   - phase results;
   - interpretation;
   - next justified phase;
   - stop conditions.
5. Do not update the DPF monograph rebuild reset memo.

### Exit gate

The student baseline stream has a clear next action and all new files are
scoped to the experimental baseline project.

## Stop rules

Stop and ask for direction if:

- a phase requires editing production `bayesfilter/` code;
- a phase requires changing reader-facing DPF monograph files;
- dependency installation would make broad environment changes;
- vendored student code must be patched in place rather than wrapped;
- all `advanced_particle_filter` targeted reproduction attempts fail for
  unclear reasons.

## Next action

Execute Phase G1: run the smallest `advanced_particle_filter` targeted
reproduction, beginning with `tests/test_basic.py` and then selected pytest
tests if the direct script succeeds.
