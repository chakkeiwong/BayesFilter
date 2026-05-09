# Reset memo: student DPF experimental baseline

## Date

2026-05-09

## Active scope

This is the active reset memo for the quarantined student DPF
experimental-baseline stream.

Owned surfaces:
- `experiments/student_dpf_baselines/`;
- `experiments/controlled_dpf_baseline/`;
- student baseline consolidation plans, audits, reports, and adapter notes;
- provenance and comparison-only documentation for vendored student snapshots.

Explicitly out of scope:
- reader-facing DPF monograph chapter rewriting;
- `docs/chapters/ch19*.tex` writing work;
- the DPF monograph rebuild active memo;
- production `bayesfilter/` code unless a later, separate implementation plan
  explicitly authorizes it.

The active reset memo for reader-facing DPF monograph writing is:
`docs/plans/bayesfilter-dpf-monograph-rebuild-reset-memo-2026-05-09.md`.

## Governing constraints

- Student code is comparison-only.
- Student code must not be imported by production `bayesfilter/` modules.
- Student code must not be treated as a correctness certificate.
- Vendored snapshots are internal experimental artifacts for reproduction,
  comparison, and stress testing.
- Keep student-code reproduction failures as structured evidence; do not hide
  them by silently editing vendored snapshots.

## Snapshot/provenance status

The student repositories were copied as plain vendored snapshots under:
- `experiments/student_dpf_baselines/vendor/2026MLCOE/`;
- `experiments/student_dpf_baselines/vendor/advanced_particle_filter/`.

Nested `.git` metadata was removed so the BayesFilter repository owns the
snapshots as plain experimental files.  Provenance is recorded in:
- `experiments/student_dpf_baselines/sources.yml`;
- `experiments/student_dpf_baselines/PERMISSIONS.md`;
- `experiments/student_dpf_baselines/vendor/SNAPSHOT.md`.

Recorded upstream commits:
- `2026MLCOE`: `020cfd7f2f848afa68432e95e6c6e747d3d2402d`;
- `advanced_particle_filter`: `d2a797c330e11befacbb736b5c86b8d03eb4a389`.

## Dependency and smoke status

Dependency audit is recorded in:
`experiments/student_dpf_baselines/reports/dependency-audit-2026-05-09.md`.

Observed environment during the audit:
- Python `3.13.13`;
- TensorFlow `2.20.0`;
- TensorFlow Probability `0.25.0`;
- available packages included `numpy`, `scipy`, `tensorflow`,
  `tensorflow_probability`, and `pytest`;
- missing packages included `matplotlib` and `numba`.

Import-only smoke checks passed for selected modules from both snapshots.

## Reproduction status

Completed:
- `2026MLCOE` unit tests: 8 passed;
- `2026MLCOE` integration tests: 12 passed.

Known caveats:
- the `2026MLCOE` tests emitted many warnings under the current Python/TensorFlow
  environment, but passed on CPU;
- `advanced_particle_filter` import smokes passed, but an original-example
  reproduction has not yet completed;
- the `advanced_particle_filter` README reports a TF/TFP version expectation
  older than the current environment, so future failures must distinguish
  environment drift from algorithm behavior.

## Next justified action

Execute the active gap-closure plan:
`docs/plans/bayesfilter-student-dpf-baseline-gap-closure-plan-2026-05-09.md`.

The immediate phase is targeted `advanced_particle_filter` reproduction.  Run
small non-plotting tests or direct examples before building adapters.  Do not
start adapter normalization until the advanced snapshot has at least one
completed original-example or targeted-test baseline, or until blockers are
recorded explicitly.

## Execution log: gap-closure cycle started 2026-05-10

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-gap-closure-plan-2026-05-09.md`.

Independent audit:
`docs/plans/bayesfilter-student-dpf-baseline-gap-closure-plan-audit-2026-05-10.md`.

### G0: preflight and contamination guard

Status: passed.

Observed state:
- current branch: `dpf-monograph-rebuild`;
- unrelated monograph-writing files are dirty in the working tree;
- no nested `.git` directories were found under
  `experiments/student_dpf_baselines/vendor/`;
- this stream's planned edits remain scoped to
  `experiments/student_dpf_baselines/` and student-baseline plan/memo files.

Interpretation:
- continuing is justified because the student-baseline files can be staged and
  committed separately from the unrelated monograph-writing files;
- final commit must use path-scoped staging and must not include
  `docs/chapters/ch19_particle_filters.tex`, `docs/references.bib`, or DPF
  monograph rebuild files.

Next phase justified: G1 targeted `advanced_particle_filter` reproduction.

### G1: advanced_particle_filter targeted reproduction

Status: passed with caveat.

Report:
`experiments/student_dpf_baselines/reports/advanced-particle-filter-reproduction-2026-05-10.md`.

Machine-readable record:
`experiments/student_dpf_baselines/reports/outputs/advanced_particle_filter_reproduction_2026-05-10.json`.

Observed environment:
- Python `3.13.13`;
- NumPy `2.1.3`;
- SciPy `1.17.1`;
- TensorFlow `2.20.0`;
- TensorFlow Probability `0.25.0`;
- TensorFlow devices: CPU only.

Results:
- `advanced_particle_filter/tests/test_basic.py`: passed, with 3 passed and 0
  failed;
- `advanced_particle_filter/tests/test_filters.py -q`: passed, with 22 tests
  passed;
- `advanced_particle_filter/tests/test_kernel_pff.py -q`: emitted `.F.` and
  did not complete before termination.

Interpretation:
- adapter work is justified for minimal linear-Gaussian Kalman/bootstrap-PF
  smoke fixtures;
- kernel PFF is not yet usable as comparison evidence and needs a later focused
  reproduction/debug pass;
- passing student tests remains comparison evidence only, not BayesFilter
  correctness evidence.

Next phase justified: G2 reset memo decision checkpoint, then G3 minimal
adapter contract.

### G2: reset memo decision checkpoint

Status: passed.

Interpretation:
- the reset memo records the G1 reproduction result, caveat, and adapter gate;
- the next phase is still justified because `advanced_particle_filter` has
  passed targeted tests sufficient for minimal linear-Gaussian smoke adapters.

Next phase justified: G3 minimal adapter contract.

### G3: minimal adapter contract

Status: passed.

Files added:
- `experiments/__init__.py`;
- `experiments/student_dpf_baselines/__init__.py`;
- `experiments/student_dpf_baselines/adapters/__init__.py`;
- `experiments/student_dpf_baselines/adapters/common.py`.

Implementation:
- defines `BaselineStatus` and `BaselineResult`;
- supports JSON serialization of NumPy/TensorFlow-like arrays without importing
  either framework at module import time;
- provides structured blocked/exception result helpers;
- provides a scoped `sys.path` context manager for vendored student imports.

Interpretation:
- syntax and serialization validation passed;
- per-student smoke adapters are justified.

### G4: per-student smoke adapters

Status: passed.

Files added:
- `experiments/student_dpf_baselines/adapters/mlcoe_adapter.py`;
- `experiments/student_dpf_baselines/adapters/advanced_particle_filter_adapter.py`.

Validation:
- both adapters were run on an in-memory 1D linear-Gaussian smoke fixture;
- both returned `status="ok"`;
- both reported the same Kalman log likelihood on that smoke fixture:
  `-1.7950433511918564`.

Implementation notes:
- `advanced_particle_filter_adapter.py` calls the vendored Kalman path and, when
  requested, the vendored bootstrap PF path;
- `mlcoe_adapter.py` calls the vendored TensorFlow `src.filters.classical.KF`
  path and computes the Gaussian predictive log likelihood in the adapter
  because that student KF class does not expose it;
- no vendored student code was modified.

Interpretation:
- G5 common fixture and reference spine is justified;
- first comparison should remain linear-Gaussian until the fixture/reference
  harness is stable.

### G5: common fixture and reference spine

Status: passed.

Files added:
- `experiments/student_dpf_baselines/fixtures/__init__.py`;
- `experiments/student_dpf_baselines/fixtures/common_fixtures.py`;
- `experiments/student_dpf_baselines/fixtures/fixture_catalog.yml`;
- `experiments/student_dpf_baselines/runners/__init__.py`;
- `experiments/student_dpf_baselines/runners/run_reference_fixtures.py`.

Reference outputs:
- `experiments/student_dpf_baselines/reports/outputs/references/lgssm_1d_short.json`;
- `experiments/student_dpf_baselines/reports/outputs/references/lgssm_cv_2d_short.json`;
- `experiments/student_dpf_baselines/reports/outputs/references/lgssm_cv_2d_low_particles.json`;
- `experiments/student_dpf_baselines/reports/outputs/references/summary.json`.

Independent Kalman log likelihoods:
- `lgssm_1d_short`: `-6.103540265912081`;
- `lgssm_cv_2d_short`: `-36.84433645109839`;
- `lgssm_cv_2d_low_particles`: `-29.94835631713834`.

Validation:
- both student smoke adapters matched the independent Kalman references to
  numerical precision on all three fixtures;
- maximum observed absolute log-likelihood difference was approximately
  `1.07e-14`;
- maximum observed filtered-mean RMSE against the reference was approximately
  `4.90e-16`.

Interpretation:
- the linear-Gaussian comparison spine is stable enough for the first panel;
- nonlinear fixtures and kernel PFF should remain deferred.

Next phase justified: G6 first comparison panel.

### G6: first comparison panel

Status: passed.

Files added:
- `experiments/student_dpf_baselines/runners/run_student_baseline_panel.py`;
- `experiments/student_dpf_baselines/runners/compare_student_outputs.py`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/student_baseline_panel_2026-05-10.json`;
- `experiments/student_dpf_baselines/reports/outputs/student_baseline_panel_summary_2026-05-10.json`;
- `experiments/student_dpf_baselines/reports/student-dpf-baseline-gap-closure-result-2026-05-10.md`.

Panel:
- fixtures: `lgssm_1d_short`, `lgssm_cv_2d_short`;
- seeds: `0`, `1`, `2`;
- particle counts for the advanced bootstrap-PF side diagnostics: `128`,
  `512`.

Results:
- `2026MLCOE`: 12/12 runs ok, max log-likelihood error `0`, max filtered-mean
  RMSE versus reference approximately `4.90e-16`;
- `advanced_particle_filter`: 12/12 runs ok, max log-likelihood error
  approximately `8.88e-16`, max filtered-mean RMSE versus reference
  approximately `4.50e-16`;
- cross-student comparable groups: 12;
- max filtered-mean RMSE between students approximately `3.94e-16`;
- max log-likelihood absolute difference between students approximately
  `8.88e-16`.

Interpretation:
- both student snapshots can now be called through quarantined adapters on small
  linear-Gaussian fixtures;
- both Kalman paths agree with the independent reference to numerical
  precision;
- this supports controlled baseline comparison, but still does not make student
  code production quality or correctness authority.

Next phase justified: G7 audit, tidy, and handoff.

### G7: audit, tidy, and handoff

Status: passed.

Checks:
- import-boundary search over `bayesfilter` and `tests` found no production or
  normal-test imports of `experiments/student_dpf_baselines`,
  `advanced_particle_filter`, or `2026MLCOE`;
- `py_compile` passed for all new experiment adapter, fixture, and runner
  modules;
- `git diff --check` passed for the student-baseline docs and experiment files;
- generated report outputs are small: the `reports/` subtree is approximately
  `268K`, with the largest panel JSON approximately `208K`;
- generated `__pycache__` directories are ignored and must not be staged.

Completion interpretation:
- all phases G0-G7 in the active gap-closure plan completed;
- no production `bayesfilter/` code was edited;
- no vendored student code was edited;
- unrelated DPF monograph-writing files remain dirty and must stay outside the
  student-baseline commit.

Next justified work:
- run a focused kernel PFF reproduction/debug pass for
  `advanced_particle_filter`, because G1 observed a `.F.` partial failure and
  non-completion in `test_kernel_pff.py`;
- after that, add nonlinear fixtures only if the kernel/flow behavior has a
  clear reproduction status;
- extend the MLCOE adapter beyond the Kalman path only after each target
  algorithm has its own reproduction gate.

## Stop rules

- Do not edit vendored student code unless the edit is isolated as an adapter or
  explicitly marked as a local patch.
- Do not claim either student implementation reproduces BayesFilter correctness.
- Do not write reader-facing monograph rewrite status into this memo.
- Do not write this stream's experimental status into the DPF monograph rebuild
  reset memo.
