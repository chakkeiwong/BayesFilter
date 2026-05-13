# Reset memo: student DPF experimental baseline

## Date

2026-05-09

## Active scope

This is the active reset memo for the quarantined student DPF
experimental-baseline stream.

Active master program:
`docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`.

Active next-phase plan:
none yet.  The next justified action is to write a student-lane clean-room
implementation plan for `experiments/controlled_dpf_baseline/`, based on
`docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`.

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

This memo records continuity and phase results for the student-baseline lane.
The master program above records durable scope, gates, current evidence, and
next-phase ordering.

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

Write a student-lane clean-room implementation plan for
`experiments/controlled_dpf_baseline/`, then audit it before implementation.
The implementation plan should use the clean-room controlled-baseline
specification:
`docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`.

Do not implement immediately from the specification alone.  The next plan must
remain outside production `bayesfilter/`, outside monograph rebuild/enrichment
files, and outside vendored student snapshots.

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

## Execution log: hypothesis-closure cycle started 2026-05-10

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-hypothesis-closure-plan-2026-05-10.md`.

Independent audit:
`docs/plans/bayesfilter-student-dpf-baseline-hypothesis-closure-plan-audit-2026-05-10.md`.

Initial decision:
- proceed in the student-baseline lane only;
- do not stage unrelated monograph reset or DPF monograph plan files;
- use bounded commands for kernel PFF reproduction.

### H0: preflight and lane guard

Status: passed.

Observed state:
- current branch: `dpf-monograph-rebuild`;
- unrelated monograph files are dirty, including
  `docs/chapters/ch19c_dpf_implementation_literature.tex`,
  `docs/plans/bayesfilter-dpf-monograph-rebuild-reset-memo-2026-05-09.md`,
  `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`, and an untracked
  DPF monograph R3 plan;
- the student reset memo and prior student result report are present;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`.

Interpretation:
- continuing is justified because this cycle can be staged path-by-path under
  the student-baseline scope;
- final commit must not include monograph dirty files.

Next phase justified: H1 larger and low-noise linear stress panel.

### H1: larger and low-noise linear stress panel

Status: passed.

Files added:
- `experiments/student_dpf_baselines/fixtures/stress_fixtures.py`;
- `experiments/student_dpf_baselines/runners/run_linear_stress_panel.py`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/linear_stress_panel_2026-05-10.json`;
- `experiments/student_dpf_baselines/reports/outputs/linear_stress_panel_summary_2026-05-10.json`;
- `experiments/student_dpf_baselines/reports/student-dpf-baseline-linear-stress-result-2026-05-10.md`.

Panel:
- fixtures: `lgssm_1d_long`, `lgssm_cv_2d_long`,
  `lgssm_cv_2d_low_noise`;
- seeds: `0`, `1`, `2`, `3`, `4`;
- particle counts: `64`, `128`, `512`.

Reference agreement:
- `2026MLCOE`: 45/45 runs ok, max Kalman log-likelihood error
  approximately `2.84e-14`;
- `advanced_particle_filter`: 45/45 runs ok, max Kalman log-likelihood error
  approximately `7.11e-15`.

Advanced bootstrap-PF diagnostics:
- `lgssm_1d_long`: median PF log-likelihood error ranged from about `0.206`
  to `0.519`;
- `lgssm_cv_2d_long`: median PF log-likelihood error ranged from about `1.52`
  to `12.23`;
- `lgssm_cv_2d_low_noise`: median PF log-likelihood error ranged from about
  `0.767` at 512 particles to `24.15` at 64 particles;
- low-noise ESS was materially lower, with min average ESS about `10.17` at
  64 particles.

Interpretation:
- H1 is supported for particle diagnostics: Kalman paths remain
  reference-consistent, while advanced bootstrap-PF diagnostics degrade under
  lower observation noise and smaller particle counts;
- MLCOE particle diagnostics remain unsupported in this adapter cycle because
  the current MLCOE adapter covers only the Kalman smoke path.

Next phase justified: H2 focused kernel PFF reproduction.

### H2: focused kernel PFF reproduction

Status: passed as classified failure/timeout.

Report:
`experiments/student_dpf_baselines/reports/advanced-particle-filter-kernel-pff-reproduction-2026-05-10.md`.

Machine-readable record:
`experiments/student_dpf_baselines/reports/outputs/advanced_particle_filter_kernel_pff_reproduction_2026-05-10.json`.

Results:
- `test_kernel_pff_lgssm`: timed out after 90 seconds with exit code `124`;
- `test_kernel_pff_convergence`: failed in 4.60 seconds because average
  iterations were `100.0`, hitting the maximum rather than satisfying the test
  assertion `< 100`;
- `test_scalar_vs_matrix_kernel`: passed but took 85.92 seconds and emitted a
  pytest warning about returning `bool`.

Interpretation:
- H2 is supported: the earlier `.F.` and non-completion were reproducible and
  localize to kernel PFF tests;
- classification is `algorithm_test_sensitivity_and_long_runtime`;
- kernel PFF should remain excluded from routine comparison panels until a
  separate bounded debug/reproduction plan is approved.

Next phase justified: H3 nonlinear smoke fixtures, but only as smoke/blocker
classification, not as flow/kernel comparison.

### H3: nonlinear smoke fixtures

Status: passed as smoke classification.

Files added:
- `experiments/student_dpf_baselines/runners/run_nonlinear_smoke.py`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/nonlinear_smoke_2026-05-10.json`;
- `experiments/student_dpf_baselines/reports/student-dpf-baseline-nonlinear-smoke-result-2026-05-10.md`.

Results:
- `advanced_particle_filter` range-bearing Student-t smoke: ok, runtime about
  `0.565` seconds, EKF/UKF/PF paths ran;
- `2026MLCOE` range-bearing smoke: ok, runtime about `2.895` seconds, EKF/UKF
  step paths ran.

Caveats:
- this is not a reference-consistency result;
- MLCOE EKF final estimate stayed at zero while UKF moved, which indicates
  method-specific nonlinear behavior needs explicit checks before comparison;
- advanced PF average ESS was about `9.44` for 128 particles on the nonlinear
  smoke, suggesting particle degeneracy risk even in a short run.

Interpretation:
- H3 is supported only as a feasibility smoke: both snapshots expose nonlinear
  paths that can run through quarantined wrappers;
- nonlinear comparison is feasible as a next experimental phase, but it needs
  independent references, method-specific sanity checks, and blocker labels.

Next phase justified: H4 synthesis, audit, and handoff.

### H4: synthesis, audit, and handoff

Status: passed.

Result report:
`experiments/student_dpf_baselines/reports/student-dpf-baseline-hypothesis-closure-result-2026-05-10.md`.

Synthesis:
- H1 is supported for available diagnostics: Kalman paths remained
  reference-consistent and advanced bootstrap-PF diagnostics degraded under
  low-noise/small-particle stress;
- H2 is supported as classified kernel PFF failure/timeout:
  `algorithm_test_sensitivity_and_long_runtime`;
- H3 is supported as nonlinear feasibility smoke only, not as correctness or
  reference-consistency evidence.

Remaining gaps:
- MLCOE particle/flow diagnostics are not yet exposed through adapters;
- kernel PFF is not routine-panel ready;
- nonlinear fixtures lack independent references and method-specific
  tolerances;
- HMC/DPF/neural OT code remains outside the validated baseline harness.

Next justified work:
- add an MLCOE BPF adapter gate for linear stress fixtures;
- build a nonlinear reference/proxy-metric spine;
- run a separate bounded kernel PFF debug gate if kernel/flow comparison remains
  important.

## Stop rules

- Do not edit vendored student code unless the edit is isolated as an adapter or
  explicitly marked as a local patch.
- Do not claim either student implementation reproduces BayesFilter correctness.
- Do not write reader-facing monograph rewrite status into this memo.
- Do not write this stream's experimental status into the DPF monograph rebuild
  reset memo.

## Execution log: MP1 MLCOE particle adapter gate started 2026-05-10

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-mp1-mlcoe-particle-adapter-plan-2026-05-10.md`.

Independent audit:
`docs/plans/bayesfilter-student-dpf-baseline-mp1-mlcoe-particle-adapter-plan-audit-2026-05-10.md`.

Initial audit decision:
- proceed in the student-baseline lane only;
- treat MLCOE BPF resampling count as threshold-inferred, not as a direct event
  log;
- keep MLCOE BPF likelihood fields null because the vendored BPF path does not
  expose a defensible likelihood estimate;
- use explicit TensorFlow and NumPy seeding for every run.

### MP1.0: preflight and lane guard

Status: passed.

Observed state:
- current branch: `dpf-monograph-rebuild`;
- unrelated monograph-lane files are dirty or untracked in the working tree;
- the active student master program and MP1 plan are present;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`.

Interpretation:
- continuing is justified because all MP1 edits can remain under
  `experiments/student_dpf_baselines/` and student-baseline plan/memo files;
- final staging must be path-scoped and must exclude monograph-lane files.

Next phase justified: MP1.1 adapter API probe.

### MP1.1: adapter API probe

Status: passed.

Implementation:
- added `run_bpf_fixture(...)` to
  `experiments/student_dpf_baselines/adapters/mlcoe_adapter.py`;
- added an adapter-local TensorFlow model bridge for existing
  linear-Gaussian fixtures;
- kept the existing MLCOE Kalman adapter path unchanged;
- did not edit vendored MLCOE code.

Smoke result:
- command: direct Python smoke calling `run_bpf_fixture` on
  `lgssm_1d_short` with seed `0` and `64` particles;
- status: `ok`;
- particle mean trajectory shape: `(9, 1)`;
- particle covariance trajectory shape: `(9, 1, 1)`;
- first ESS values: approximately `27.10`, `20.25`, `14.23`;
- threshold-inferred resampling count: `1`;
- runtime: approximately `2.61` seconds.

Interpretation:
- H1 is supported for the API probe: MLCOE BPF can run through a quarantined
  adapter without vendored-code edits;
- H2 is supported for the probe: particle means, weighted covariances, ESS, and
  threshold-inferred resampling diagnostics are available;
- MLCOE BPF likelihood remains unavailable and must stay null.

Next phase justified: MP1.2 bounded linear particle panel.

### MP1.2: bounded linear particle panel

Status: passed.

Files added:
- `experiments/student_dpf_baselines/runners/run_mlcoe_particle_gate.py`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/mlcoe_particle_gate_2026-05-10.json`;
- `experiments/student_dpf_baselines/reports/outputs/mlcoe_particle_gate_summary_2026-05-10.json`;
- `experiments/student_dpf_baselines/reports/student-dpf-baseline-mlcoe-particle-gate-result-2026-05-10.md`.

Panel:
- fixtures: `lgssm_1d_short`, `lgssm_cv_2d_short`, `lgssm_1d_long`,
  `lgssm_cv_2d_long`, `lgssm_cv_2d_low_noise`;
- seeds: `0`, `1`, `2`;
- particle counts: `64`, `128`, `512`;
- implementations: MLCOE BPF and advanced bootstrap-PF.

Results:
- total records: 90;
- ok records: 90;
- `2026MLCOE` BPF: 45/45 ok;
- `advanced_particle_filter` bootstrap-PF: 45/45 ok;
- no vendored student code was modified.

MLCOE BPF diagnostics:
- maximum particle-mean RMSE against Kalman: approximately `0.5867`;
- maximum particle-covariance RMSE against Kalman: approximately `0.1253`;
- minimum average ESS: approximately `5.82`;
- median runtime: approximately `0.146` seconds per run;
- likelihood available runs: `0`.

Advanced bootstrap-PF diagnostics in matched panel:
- maximum particle-mean RMSE against Kalman: approximately `0.2759`;
- maximum particle-covariance RMSE against Kalman: approximately `0.1161`;
- minimum average ESS: approximately `10.17`;
- median runtime: approximately `0.0110` seconds per run;
- likelihood available runs: `45`.

Interpretation:
- H1 remains supported: MLCOE BPF runs through the quarantined adapter on all
  planned linear fixtures;
- H2 remains supported: MLCOE BPF exposes enough state, weight, and ESS data
  for defensible particle diagnostics;
- MLCOE BPF likelihood remains unavailable and must not be compared to the
  advanced PF likelihood.

Next phase justified: MP1.3 cross-student particle comparison.

### MP1.3: cross-student particle comparison

Status: passed.

Result report:
`experiments/student_dpf_baselines/reports/student-dpf-baseline-mlcoe-particle-gate-result-2026-05-10.md`.

Cross-student matched groups:
- matching keys: fixture, seed, particle count;
- groups summarized: 45 underlying matched records, aggregated into 15
  fixture/particle-count summaries.

Key observations:
- MLCOE BPF and advanced bootstrap-PF both show lower ESS under lower-noise
  and smaller-particle settings;
- MLCOE BPF generally has lower average ESS than advanced bootstrap-PF in the
  matched summaries;
- MLCOE BPF is slower in this local environment, with median runtime ratios
  from about `6.94` to `56.29` relative to advanced bootstrap-PF depending on
  fixture and particle count;
- cross-student particle mean differences are nonzero and grow on harder
  constant-velocity fixtures, which is expected for stochastic PF paths.

Hypothesis outcomes:
- H1: supported;
- H2: supported;
- H3: partially supported, because MLCOE BPF produced interpretable low-noise
  stress diagnostics, but the degradation pattern was weaker than the
  pre-specified qualitative threshold;
- H4: supported as comparison-only evidence.

Interpretation:
- the MP1 phase closes the linear particle-diagnostic asymmetry enough to use
  both student snapshots in a balanced linear particle baseline;
- the evidence is still comparison-only and does not validate either
  implementation as production quality;
- the missing MLCOE likelihood remains a real limitation.

Next phase justified: MP1.4 audit, tidy, and reset memo update.

### MP1.4: audit, tidy, and handoff

Status: passed.

Checks:
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- `py_compile` passed for
  `experiments/student_dpf_baselines/adapters/mlcoe_adapter.py` and
  `experiments/student_dpf_baselines/runners/run_mlcoe_particle_gate.py`;
- `git diff --check` passed for edited student-baseline files;
- no files under `experiments/student_dpf_baselines/vendor/` were modified;
- generated MP1 artifacts are moderate in size:
  `mlcoe_particle_gate_2026-05-10.json` is approximately `2.8M`,
  the summary JSON is approximately `20K`, and the report is approximately
  `8K`.

Completion interpretation:
- MP1 completed without touching production `bayesfilter/` code, monograph
  files, or vendored student code;
- the student-baseline lane now has balanced linear particle diagnostics for
  MLCOE BPF and advanced bootstrap-PF;
- the next major gap is nonlinear reference/proxy metrics, not another
  Kalman-only or reproduction gate.

Next justified work:
- MP2 nonlinear reference and proxy-metric spine, unless project priority
  shifts specifically to kernel PFF debugging;
- explicit next hypotheses should test whether nonlinear range-bearing panels
  can be interpreted with latent-state RMSE, EKF/UKF agreement bands, PF
  repeated-seed dispersion, and ESS/runtime diagnostics without comparing
  incompatible likelihood families.

## Execution log: MP2 nonlinear reference spine started 2026-05-10

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-mp2-nonlinear-reference-spine-plan-2026-05-10.md`.

Independent audit:
`docs/plans/bayesfilter-student-dpf-baseline-mp2-nonlinear-reference-spine-plan-audit-2026-05-10.md`.

Initial audit decision:
- proceed in the student-baseline lane only;
- use a shared Gaussian range-bearing fixture for MP2 target semantics;
- treat MLCOE origin-initialized EKF as a diagnostic, not as the main
  comparison target;
- keep nonlinear metrics proxy/reference labeled and comparison-only.

### MP2.0: preflight and lane guard

Status: passed.

Observed state:
- current branch: `dpf-monograph-rebuild`;
- unrelated monograph-lane files are dirty or untracked in the working tree;
- the active student master program, reset memo, and MP2 plan are present;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`.

Interpretation:
- continuing is justified because MP2 edits can remain under
  `experiments/student_dpf_baselines/` and student-baseline plan/memo files;
- final staging must remain path-scoped and exclude monograph-lane files.

Next phase justified: MP2.1 fixture and method design.

### MP2.1: fixture and method design

Status: passed.

Files added:
- `experiments/student_dpf_baselines/fixtures/nonlinear_fixtures.py`.

Fixtures:
- `range_bearing_gaussian_moderate`;
- `range_bearing_gaussian_low_noise`.

Validation:
- both fixtures have state shape `(21, 4)` and observation shape `(20, 2)`;
- transition, process covariance, observation covariance, initial mean, and
  initial covariance shapes are consistent;
- all generated states and observations are finite;
- range-bearing observation helper returns shape `(2,)`;
- range-bearing Jacobian helper returns shape `(2, 4)`.

Interpretation:
- N1 is not yet decided, but the shared fixture side of N1 is satisfied;
- runner implementation is justified because the fixture arrays are stable and
  target semantics are explicitly Gaussian range-bearing.

Next phase justified: MP2.2 nonlinear panel runner.

### MP2.2: nonlinear panel runner

Status: passed.

Files added:
- `experiments/student_dpf_baselines/runners/run_nonlinear_reference_panel.py`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/nonlinear_reference_panel_2026-05-10.json`;
- `experiments/student_dpf_baselines/reports/outputs/nonlinear_reference_panel_summary_2026-05-10.json`;
- `experiments/student_dpf_baselines/reports/student-dpf-baseline-nonlinear-reference-panel-result-2026-05-10.md`.

Panel:
- fixtures: `range_bearing_gaussian_moderate`,
  `range_bearing_gaussian_low_noise`;
- target semantics: shared Gaussian range-bearing;
- advanced methods: EKF, UKF, bootstrap-PF with seeds `0` through `4`;
- MLCOE methods: EKF, UKF, BPF with seeds `0` through `4`, plus
  origin-initialized EKF diagnostic.

Results:
- total records: 30;
- ok records: 30;
- `advanced_particle_filter`: 14/14 ok;
- `2026MLCOE`: 16/16 ok;
- no vendored student code was modified.

Interpretation:
- N1 is supported: both snapshots run EKF/UKF paths on shared nonlinear
  fixtures without vendored-code edits;
- the runner produced proxy/reference metrics suitable for hypothesis
  classification.

Next phase justified: MP2.3 hypothesis classification.

### MP2.3: hypothesis classification

Status: passed.

Result report:
`experiments/student_dpf_baselines/reports/student-dpf-baseline-nonlinear-reference-panel-result-2026-05-10.md`.

Key metrics:
- `advanced_particle_filter` position RMSE range: approximately `0.0456` to
  `0.0692`;
- `2026MLCOE` non-origin EKF/UKF/BPF position RMSE range: approximately
  `0.0469` to `0.0680` for the main methods;
- `2026MLCOE` origin-initialized EKF diagnostic position RMSE:
  approximately `0.985` on low noise and `1.278` on moderate noise;
- advanced BPF average ESS medians: approximately `113.9` on moderate noise
  and `49.5` on low noise;
- MLCOE BPF average ESS medians: approximately `70.2` on moderate noise and
  `31.9` on low noise.

Hypothesis outcomes:
- N1 shared nonlinear fixture: supported;
- N2 MLCOE EKF zero behavior: supported.  MLCOE EKF is usable away from the
  origin, while the origin diagnostic is materially worse, consistent with a
  range-bearing Jacobian initialization artifact;
- N3 nonlinear PF degeneracy: supported for this proxy panel.  Both PF paths
  show lower ESS or larger RMSE pressure on the low-noise fixture;
- N4 comparison-only reporting: supported.  Records include target labels and
  avoid direct likelihood comparison.

Interpretation:
- MP2 closes the immediate nonlinear reference/proxy-metric gap;
- nonlinear comparison is now interpretable for shared Gaussian range-bearing
  EKF/UKF/BPF proxy metrics;
- these results still do not validate flow, DPF, HMC, kernel PFF, or neural OT
  behavior.

Next phase justified: MP2.4 audit, tidy, reset memo, and commit.

### MP2.4: audit, tidy, and handoff

Status: passed.

Checks:
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- `py_compile` passed for
  `experiments/student_dpf_baselines/fixtures/nonlinear_fixtures.py` and
  `experiments/student_dpf_baselines/runners/run_nonlinear_reference_panel.py`;
- `git diff --check` passed for edited student-baseline files;
- no files under `experiments/student_dpf_baselines/vendor/` were modified;
- generated MP2 artifacts are small:
  the panel JSON is approximately `36K`, the summary JSON is approximately
  `8K`, and the report is approximately `8K`.

Completion interpretation:
- MP2 completed without touching production `bayesfilter/` code, monograph
  files, or vendored student code;
- the student-baseline lane now has both balanced linear particle diagnostics
  and an interpretable nonlinear proxy-metric spine.

Next justified work:
- MP3 kernel PFF debug gate, because kernel PFF remains the largest classified
  failure/timeout gap after the linear and nonlinear proxy panels;
- alternative only if project priority changes: a flow/DPF readiness review
  that inventories runnable flow paths but does not run kernel PFF in routine
  panels.

## Execution log: MP3 kernel PFF debug gate started 2026-05-11

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-mp3-kernel-pff-debug-gate-plan-2026-05-11.md`.

Independent audit:
`docs/plans/bayesfilter-student-dpf-baseline-mp3-kernel-pff-debug-gate-plan-audit-2026-05-11.md`.

Initial audit decision:
- proceed in the student-baseline lane only;
- use reduced local diagnostics rather than rerunning the slow vendored tests
  as the primary evidence;
- distinguish completed filter runs from converged flow iterations;
- keep kernel PFF excluded from routine panels unless bounded convergence is
  consistently demonstrated.

### MP3.0: preflight and lane guard

Status: passed.

Observed state:
- current branch: `dpf-monograph-rebuild`;
- unrelated monograph-lane files are dirty or untracked in the working tree;
- the active student master program, reset memo, and MP3 plan are present;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`.

Interpretation:
- continuing is justified because MP3 edits can remain under
  `experiments/student_dpf_baselines/` and student-baseline plan/memo files;
- final staging must remain path-scoped and exclude monograph-lane files.

Next phase justified: MP3.1 reduced diagnostic runner.

### MP3.1: reduced diagnostic runner

Status: passed.

Files added:
- `experiments/student_dpf_baselines/runners/run_kernel_pff_debug_gate.py`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/kernel_pff_debug_gate_2026-05-11.json`;
- `experiments/student_dpf_baselines/reports/outputs/kernel_pff_debug_gate_summary_2026-05-11.json`;
- `experiments/student_dpf_baselines/reports/advanced-particle-filter-kernel-pff-debug-gate-result-2026-05-11.md`.

Panel:
- fixtures: `lgssm_1d_reduced`, `lgssm_cv_2d_reduced`;
- kernels: scalar and matrix;
- tolerance labels: loose `1e-3`, strict `1e-5`;
- particle counts: `64`, `128`;
- max iterations: `40`;
- total runs: 16.

Results:
- completed runs: 16/16;
- failed runs: 0/16;
- every run completed as a filter run but every time step hit the
  `max_iterations=40` cap;
- median average iterations: `40` for both scalar and matrix kernels;
- maximum hit-max fraction: `1.0` for both scalar and matrix kernels;
- median RMSE versus Kalman: approximately `0.122` for scalar and `0.130` for
  matrix;
- median runtime: approximately `0.148` seconds for scalar and `0.129` seconds
  for matrix.

Interpretation:
- K1 is supported: reduced scalar and matrix kernel PFF runs are runnable;
- the prior timeout was not a missing dependency or import failure;
- completed filter runs do not imply converged flow iterations.

Next phase justified: MP3.2 classification and report.

### MP3.2: classification and report

Status: passed.

Result report:
`experiments/student_dpf_baselines/reports/advanced-particle-filter-kernel-pff-debug-gate-result-2026-05-11.md`.

Hypothesis outcomes:
- K1 reduced fixtures runnable: supported;
- K2 tolerance sensitivity: not supported.  Loose tolerance did not reduce
  median iterations or runtime relative to strict tolerance in this bounded
  panel;
- K3 max-iteration failure mode: supported.  Non-converged behavior appears as
  finite completed runs with hit-max diagnostics, not missing dependencies;
- K4 routine-panel readiness: supported as exclusion.  Kernel PFF should remain
  excluded from routine panels pending further debug.

Readiness decision:
`excluded_pending_debug`.

Interpretation:
- MP3 narrows the previous classification from broad
  `algorithm_test_sensitivity_and_long_runtime` to a specific bounded finding:
  reduced kernel PFF runs complete quickly, but flow iterations consistently
  hit the maximum iteration cap even under loose tolerance;
- kernel PFF can be used only as debug evidence, not as routine comparison
  evidence.

Next phase justified: MP3.3 audit, tidy, reset memo, and commit.

### MP3.3: audit, tidy, and handoff

Status: passed.

Checks:
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- `py_compile` passed for
  `experiments/student_dpf_baselines/runners/run_kernel_pff_debug_gate.py`;
- `git diff --check` passed for edited student-baseline files;
- no files under `experiments/student_dpf_baselines/vendor/` were modified;
- generated MP3 artifacts are small:
  the panel JSON is approximately `20K`, the summary JSON is approximately
  `4K`, and the report is approximately `4K`.

Completion interpretation:
- MP3 completed without touching production `bayesfilter/` code, monograph
  files, or vendored student code;
- the kernel PFF failure mode is now narrowed: reduced runs are fast and
  runnable, but flow iterations consistently hit `max_iterations`;
- kernel PFF remains excluded from routine comparison panels.

Next justified work:
- MP4 flow and DPF readiness review.  The review should inventory advanced and
  MLCOE flow/DPF entry points, classify runnable paths, and select at most one
  bounded candidate for a later comparison phase;
- kernel PFF should stay out of routine panels unless a later algorithm-specific
  debug plan modifies only adapter-owned experiment logic or records an
  explicit local patch policy.

## Execution log: MP4 flow and DPF readiness review started 2026-05-11

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-mp4-flow-dpf-readiness-review-plan-2026-05-11.md`.

### MP4 goal

Status: planned.

Goal:
- determine whether any flow or DPF path in the two student snapshots is ready
  for a later bounded comparison panel;
- inventory advanced and MLCOE flow/DPF entry points;
- classify importability, call-surface clarity, target semantics, dependency
  blockers, artifact blockers, and runtime blockers;
- select at most one bounded EDH/LEDH-family comparison candidate, or record a
  blocker-only decision if no candidate satisfies the gates.

Lane guard:
- this MP4 cycle remains inside `experiments/student_dpf_baselines/` and
  student-baseline plan/report/reset-memo files under `docs/plans/`;
- monograph writing files, monograph reset memos, production `bayesfilter/`
  code, `docs/references.bib`, and vendored student code remain out of scope.

### MP4 remaining gaps

Status: identified.

Gaps:
- flow and DPF entry points are not cataloged across the two snapshots;
- matching labels such as EDH, LEDH, PFF, PFPF, and DPF do not guarantee
  matching target semantics;
- importability and constructor/call signatures are not yet recorded;
- the first bounded comparison candidate is not selected;
- DPF, neural OT, differentiable resampling, and HMC paths may depend on
  training, pretrained artifacts, notebooks, plotting, or parameter-inference
  objectives.

### MP4 hypotheses

Status: planned.

Hypotheses:
- F1: EDH/LEDH is the most plausible first flow candidate because both
  snapshots expose EDH/LEDH-family entry points that may map to a shared
  Gaussian target without vendored-code edits;
- F2: kernel PFF remains excluded from first routine flow comparison because
  MP3 found consistent `max_iterations` hits under reduced diagnostics;
- F3: DPF, neural OT, differentiable resampling, and HMC paths require
  separate reproduction gates before comparison;
- F4: import and signature probes are sufficient to classify readiness without
  running large experiments or editing vendored code;
- F5: a later bounded EDH/LEDH comparison, if selected, should reuse the
  existing linear-Gaussian or nonlinear Gaussian range-bearing fixtures with
  explicit proxy/reference metrics.

Next phase justified: MP4.0 preflight and lane guard.

### MP4.0: preflight and lane guard

Status: passed.

Plan audit:
`docs/plans/bayesfilter-student-dpf-baseline-mp4-flow-dpf-readiness-review-plan-audit-2026-05-11.md`.

Plan tightening:
- MP4 is explicitly limited to static inventory, import probes, and signature
  inspection;
- MP4 must not instantiate filters, call filter/update/step methods, run
  notebooks, execute experiment scripts, train models, run HMC, or generate
  large artifacts;
- kernel PFF remains excluded from first routine flow comparison because MP3
  classified it as `excluded_pending_debug`.

Observed state:
- current working tree contains unrelated monograph-lane dirty and untracked
  files;
- student-baseline edits can remain path-scoped;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  are dirty.

Interpretation:
- continuing is justified because MP4 can execute fully inside the
  student-baseline lane;
- final staging must remain path-scoped and must exclude monograph-lane files.

Next phase justified: MP4.1 static inventory and MP4.2 import/signature probe.

### MP4.1: static inventory

Status: passed.

Runner:
`experiments/student_dpf_baselines/runners/run_flow_dpf_readiness_review.py`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/flow_dpf_readiness_inventory_2026-05-11.json`;
- `experiments/student_dpf_baselines/reports/outputs/flow_dpf_readiness_summary_2026-05-11.json`;
- `experiments/student_dpf_baselines/reports/student-dpf-baseline-flow-dpf-readiness-review-result-2026-05-11.md`.

Inventory scope:
- 28 flow, DPF, resampling, neural OT, stochastic-flow, kernel PFF, and HMC
  candidate surfaces were cataloged across the two snapshots;
- the runner used static AST inventory plus import/signature inspection only;
- no filter classes were instantiated;
- no filter/update/step methods, notebooks, experiment scripts, training, or
  HMC entry points were executed.

Category counts:
- `edh_ledh_flow`: 5;
- `edh_ledh_pfpf`: 4;
- `kernel_pff`: 3;
- `stochastic_flow`: 3;
- `dpf`: 2;
- `dpfpf`: 1;
- `differentiable_resampling`: 4;
- `neural_ot`: 1;
- `neural_resampling`: 1;
- `hmc_parameter_inference`: 2;
- `flow_solver`: 2.

Interpretation:
- MP4.1 closed the catalog gap without broadening scope or editing vendored
  student snapshots;
- static inventory is sufficient to proceed to import/signature readiness
  classification.

Next phase justified: MP4.2 import and signature probe.

### MP4.2: import and signature probe

Status: passed.

Command:
`python -m experiments.student_dpf_baselines.runners.run_flow_dpf_readiness_review`.

Observed environment:
- Python `3.13.13`;
- NumPy `2.1.3`;
- TensorFlow `2.20.0`;
- TensorFlow Probability `0.25.0`;
- TensorFlow physical devices: CPU only.

Results:
- records: 28;
- probe status counts: `{'importable': 28}`;
- readiness counts:
  `{'candidate_for_bounded_comparison': 9, 'reproduction_gate_required': 14,
  'excluded_pending_debug': 3, 'adapter_internal_only': 2}`.

Interpretation:
- importability is broad but not equivalent to comparison readiness;
- deterministic EDH/LEDH and EDH/LEDH-PFPF paths have inspectable call
  surfaces;
- DPF, dPFPF, differentiable resampling, neural OT, neural resampling,
  stochastic flow, and HMC paths are importable surfaces that still require
  separate reproduction gates before comparison claims;
- kernel PFF remains excluded from routine comparison due to MP3
  max-iteration diagnostics.

Next phase justified: MP4.3 candidate selection.

### MP4.3: candidate selection

Status: passed.

Decision:
`selected_bounded_candidate`.

Selected later candidate:
- family: `importance_corrected_edh_pfpf`;
- advanced path:
  `advanced_particle_filter.filters.edh.EDHParticleFilter`;
- MLCOE path: `src.filters.flow_filters.PFPF_EDH`;
- fixture: existing nonlinear Gaussian range-bearing fixture;
- first runtime cap: short horizon, at most 64 particles, at most 10 flow
  steps for an adapter spike;
- adapter scope: adapter-owned bridges only, no vendored-code edits, no
  production `bayesfilter/` imports.

Metrics for the later adapter spike:
- latent-state RMSE;
- final-position RMSE;
- average ESS if exposed;
- resampling count if exposed;
- runtime seconds;
- finite-output checks;
- EKF/UKF proxy comparison.

Hypothesis outcomes:
- F1 EDH/LEDH first candidate: `supported_with_adapter_caveat`;
- F2 kernel PFF excluded: `supported`;
- F3 DPF/neural OT/HMC reproduction gates: `supported`;
- F4 import/signature probe sufficiency: `supported`;
- F5 reuse existing fixtures: `supported`.

Interpretation:
- the EDH/PFPF-EDH pair is the only immediate comparison candidate;
- it is not evidence of performance or correctness yet;
- if either adapter requires unrecorded model assumptions, the next phase must
  stop and record `blocked_missing_assumption` rather than widening scope.

Next phase justified: MP4.4 report, audit, tidy, and reset memo.

### MP4.4: report, audit, tidy, and completion

Status: passed.

Checks:
- `py_compile` passed for
  `experiments/student_dpf_baselines/runners/run_flow_dpf_readiness_review.py`;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- `git diff --check` passed;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  were modified;
- generated artifacts are small: inventory JSON approximately `44K`, summary
  JSON approximately `2.3K`, report approximately `8K`.

Completion interpretation:
- MP4 completed without touching production code, monograph files, references,
  or vendored student code;
- broad flow/DPF surfaces are importable in the current environment, but only
  EDH/PFPF-EDH is ready for a bounded adapter spike;
- all neural/DPF/OT/HMC surfaces remain outside comparison claims until
  separate reproduction gates are written and executed.

Next justified work:
- create a scoped EDH/PFPF adapter-spike plan comparing advanced
  `EDHParticleFilter` against MLCOE `PFPF_EDH` on the existing nonlinear
  Gaussian range-bearing fixture;
- primary next hypothesis: both paths can run through adapter-owned bridges
  with no vendored-code edits and produce finite latent-state RMSE and runtime
  diagnostics on a reduced fixture;
- veto diagnostics for the next phase: unrecorded target assumptions,
  incompatible model interfaces, large runtime, missing ESS/resampling metrics
  without structured nulls, vendored-code edits, or production-code imports.

## Execution log: EDH/PFPF adapter spike started 2026-05-11

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-edh-pfpf-adapter-spike-plan-2026-05-11.md`.

Independent audit:
`docs/plans/bayesfilter-student-dpf-baseline-edh-pfpf-adapter-spike-plan-audit-2026-05-11.md`.

### Adapter-spike goal

Status: planned.

Goal:
- run the bounded candidate selected by MP4:
  `advanced_particle_filter.filters.edh.EDHParticleFilter` versus MLCOE
  `src.filters.flow_filters.PFPF_EDH`;
- use adapter-owned model bridges on the existing nonlinear Gaussian
  range-bearing fixture;
- record finite-output, RMSE, ESS/resampling availability, and runtime
  diagnostics;
- decide whether a replicated EDH/PFPF panel is justified.

Hypotheses:
- E1: advanced `EDHParticleFilter` runs on the reduced fixture through an
  adapter-owned bridge;
- E2: MLCOE `PFPF_EDH` runs on the same reduced fixture through an
  adapter-owned TensorFlow bridge;
- E3: proxy comparison is interpretable without treating student agreement as
  correctness;
- E4: the spike produces a clear next decision:
  `edh_pfpf_panel_ready`, `adapter_spike_success_needs_replication`,
  `blocked_missing_assumption`, or `excluded_due_to_runtime_or_numerics`.

Next phase justified: S0 preflight and lane guard.

### S0: preflight and lane guard

Status: passed.

Observed state:
- current working tree contains unrelated monograph-lane dirty and untracked
  files;
- student-baseline edits can remain path-scoped;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  are dirty;
- the adapter-spike plan and independent audit are present.

Interpretation:
- continuing is justified because the spike can execute fully inside the
  student-baseline lane;
- final staging must remain path-scoped and must exclude monograph-lane files.

Next phase justified: S1 runner implementation.

### S1: runner implementation

Status: passed.

File added:
`experiments/student_dpf_baselines/runners/run_edh_pfpf_adapter_spike.py`.

Implementation:
- reuses the existing nonlinear Gaussian range-bearing fixture and MP2
  adapter-owned model bridges;
- reduces `range_bearing_gaussian_moderate` to an 8-observation fixture;
- runs only the MP4-selected pair:
  - advanced `EDHParticleFilter`;
  - MLCOE `PFPF_EDH`;
- uses 64 particles, 10 flow steps, seed 17, and a runtime-warning threshold
  of 30 seconds;
- records missing likelihood/covariance fields as null or structured
  diagnostics rather than fabricating them.

Interpretation:
- S1 closes the implementation gap without editing vendored student code or
  production `bayesfilter/` modules;
- bounded execution is justified.

Next phase justified: S2 execute, classify, and report.

### S2: execute, classify, and report

Status: passed.

Command:
`python -m experiments.student_dpf_baselines.runners.run_edh_pfpf_adapter_spike`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/edh_pfpf_adapter_spike_2026-05-11.json`;
- `experiments/student_dpf_baselines/reports/outputs/edh_pfpf_adapter_spike_summary_2026-05-11.json`;
- `experiments/student_dpf_baselines/reports/student-dpf-baseline-edh-pfpf-adapter-spike-result-2026-05-11.md`.

Observed environment:
- Python `3.13.13`;
- NumPy `2.1.3`;
- TensorFlow `2.20.0`;
- TensorFlow Probability `0.25.0`;
- TensorFlow physical devices: CPU only.

Panel:
- base fixture: `range_bearing_gaussian_moderate`;
- reduced fixture: `range_bearing_gaussian_moderate_edh_pfpf_h8`;
- horizon: 8 observations;
- particles: 64;
- flow steps: 10;
- seed: 17.

Results:
- advanced `EDHParticleFilter`: status `ok`, runtime about `0.559` seconds,
  state RMSE about `0.118`, position RMSE about `0.0778`, final-position error
  about `0.108`, average ESS about `8.45`, min ESS about `2.96`, resampling
  count `8`;
- MLCOE `PFPF_EDH`: status `ok`, runtime about `3.91` seconds, state RMSE
  about `0.0824`, position RMSE about `0.0710`, final-position error about
  `0.103`, average ESS about `24.14`, min ESS about `12.84`, inferred
  resampling count `6`;
- both outputs had finite means;
- no runtime warning was triggered.

Hypothesis outcomes:
- E1 advanced EDH PFPF runs: `supported`;
- E2 MLCOE PFPF_EDH runs: `supported`;
- E3 proxy comparison interpretable: `supported_proxy_only`;
- E4 next-phase decision: `adapter_spike_success_needs_replication`.

Interpretation:
- the selected EDH/PFPF pair is runnable under bounded adapter-owned bridges;
- the results are proxy diagnostics only, not correctness or performance
  claims;
- ESS and resampling-count semantics differ by implementation and must remain
  labeled in any replicated panel;
- a replicated EDH/PFPF panel is justified if it keeps the same lane and
  reporting constraints.

Next phase justified: S3 audit, tidy, reset memo, and commit.

### S3: audit, tidy, and completion

Status: passed.

Checks:
- `py_compile` passed for
  `experiments/student_dpf_baselines/runners/run_edh_pfpf_adapter_spike.py`;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- `git diff --check` passed;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  were modified;
- generated artifacts are small: panel JSON approximately `5.3K`, summary JSON
  approximately `1.9K`, report approximately `2.2K`.

Completion interpretation:
- the adapter spike completed without touching production code, monograph
  files, references, or vendored student code;
- both selected EDH/PFPF implementations run on the same reduced nonlinear
  Gaussian range-bearing fixture through adapter-owned bridges;
- the correct next step is not production work and not DPF/neural/OT/HMC work;
  it is a replicated EDH/PFPF panel that tests robustness of this spike across
  fixtures and seeds.

Next justified work:
- create and execute a replicated EDH/PFPF panel plan using both nonlinear
  Gaussian range-bearing fixtures and multiple seeds;
- primary hypotheses for that panel:
  - R1: both EDH/PFPF paths remain runnable across moderate and low-noise
    fixtures without vendored-code edits;
  - R2: low observation noise reduces ESS and increases resampling pressure in
    at least one implementation;
  - R3: EDH/PFPF proxy RMSE remains interpretable against EKF/UKF/BPF context
    without using student agreement as correctness evidence;
  - R4: runtime remains bounded enough for an experimental panel;
- veto diagnostics for the replicated panel: unbounded runtime, nonfinite
  outputs, new unrecorded target assumptions, vendored-code edits, production
  imports, or generated artifacts too large for normal repository history.

## Execution log: replicated EDH/PFPF panel planning started 2026-05-11

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-replicated-edh-pfpf-panel-plan-2026-05-11.md`.

### Replicated-panel goal

Status: planned.

Goal:
- test whether the EDH/PFPF adapter-spike success survives both existing
  nonlinear Gaussian range-bearing fixtures and multiple seeds;
- compare advanced `EDHParticleFilter` against MLCOE `PFPF_EDH` using
  adapter-owned bridges only;
- report latent-state RMSE, final-position error, observation proxy RMSE, ESS,
  resampling pressure, runtime, and finite-output checks;
- preserve comparison-only interpretation and implementation-specific
  diagnostics.

### Remaining gaps

Status: identified.

Gaps:
- the adapter spike is single-seed evidence only;
- low-noise EDH/PFPF behavior is untested;
- ESS and resampling-count semantics differ between implementations;
- replicated runtime and numerical stability are unknown;
- comparison must remain proxy-only and must not become a correctness claim.

### Replicated-panel hypotheses

Status: planned.

Hypotheses:
- R1: both EDH/PFPF paths remain runnable across moderate and low-noise fixtures
  without vendored-code edits;
- R2: low observation noise increases pressure in ESS, resampling count,
  runtime, or RMSE for at least one implementation;
- R3: proxy comparison remains interpretable with explicit target labels and
  missing-metric handling;
- R4: runtime remains bounded enough for experimental use;
- R5: the phase produces a clear decision: `replicated_panel_ready`,
  `replicated_panel_ready_with_caveats`, `needs_targeted_debug`, or
  `blocked_or_excluded`.

Next phase justified: RP0 preflight and lane guard.

### RP0: preflight and lane guard

Status: passed.

Observed state:
- current Git status contains dirty and untracked monograph-lane files from
  other work, plus student-baseline plan/reset/master edits;
- the replicated EDH/PFPF panel plan has been tightened for fixed seeds,
  runtime threshold, artifact names, and reset-memo ownership;
- independent plan audit exists at
  `docs/plans/bayesfilter-student-dpf-baseline-replicated-edh-pfpf-panel-plan-audit-2026-05-12.md`;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  are dirty.

Interpretation:
- the phase remains inside the student experimental-baseline lane;
- known monograph-lane dirty files must remain unstaged and untouched;
- continuing to runner implementation is justified.

Next phase justified: RP1 replicated panel runner.

### RP1: replicated panel runner

Status: passed.

File added:
`experiments/student_dpf_baselines/runners/run_replicated_edh_pfpf_panel.py`.

Implementation:
- parameterizes the adapter-spike EDH/PFPF bridges over both existing nonlinear
  Gaussian range-bearing fixtures;
- uses fixed seeds `17`, `23`, and `31`;
- reduces each fixture to 8 observations;
- uses 64 particles, 10 flow steps, and a 30 second per-run runtime-warning
  threshold;
- records one structured record per implementation, fixture, and seed for 12
  planned records total;
- preserves implementation-specific ESS and resampling-count semantics;
- writes the planned JSON, summary JSON, and Markdown report artifacts under
  `experiments/student_dpf_baselines/reports/`.

Check:
- `python -m py_compile experiments/student_dpf_baselines/runners/run_replicated_edh_pfpf_panel.py`
  passed.

Interpretation:
- the runner remains adapter-owned and does not edit vendored student code or
  production BayesFilter code;
- executing the panel is justified.

Next phase justified: RP2 execute, classify, and report.

### RP2: execute, classify, and report

Status: passed.

Command:
`python -m experiments.student_dpf_baselines.runners.run_replicated_edh_pfpf_panel`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/replicated_edh_pfpf_panel_2026-05-12.json`;
- `experiments/student_dpf_baselines/reports/outputs/replicated_edh_pfpf_panel_summary_2026-05-12.json`;
- `experiments/student_dpf_baselines/reports/student-dpf-baseline-replicated-edh-pfpf-panel-result-2026-05-12.md`.

Panel:
- fixtures: `range_bearing_gaussian_moderate`,
  `range_bearing_gaussian_low_noise`;
- reduced horizon: 8 observations;
- seeds: `17`, `23`, `31`;
- implementations: advanced `EDHParticleFilter`, MLCOE `PFPF_EDH`;
- particles: 64;
- flow steps: 10;
- planned records: 12.

Results:
- 12/12 records returned status `ok`;
- no runtime warnings were triggered;
- all successful records had finite means;
- generated artifacts are small: panel JSON about `26K`, summary JSON about
  `8.4K`, report about `3.7K`.

Implementation summary:
- `advanced_particle_filter`: 6/6 ok, median runtime about `0.0179` seconds,
  maximum runtime about `0.565` seconds, median position RMSE about `0.0649`,
  median average ESS about `13.2`;
- `2026MLCOE`: 6/6 ok, median runtime about `0.615` seconds, maximum runtime
  about `3.67` seconds, median position RMSE about `0.0648`, median average ESS
  about `18.3`.

Fixture summary:
- advanced moderate-noise median position RMSE about `0.0654`, median average
  ESS about `9.26`, median resampling count `8`;
- advanced low-noise median position RMSE about `0.0644`, median average ESS
  about `17.3`, median resampling count `6`;
- MLCOE moderate-noise median position RMSE about `0.0642`, median average ESS
  about `24.1`, median resampling count `6`;
- MLCOE low-noise median position RMSE about `0.0652`, median average ESS about
  `11.7`, median resampling count `8`.

Hypothesis results:
- R1 runnable across fixtures and seeds: `supported_all_planned_runs_ok`;
- R2 low-noise pressure: `supported_directional_pressure_signal_observed`;
- R3 proxy comparison interpretability: `supported_proxy_only`;
- R4 bounded runtime: `supported_no_runtime_warnings`;
- R5 decision: `replicated_panel_ready`.

Interpretation:
- the replicated EDH/PFPF panel is now a usable quarantined experimental
  baseline artifact;
- the result remains proxy-only and does not certify correctness, production
  quality, HMC suitability, or cross-student superiority;
- ESS and resampling-count semantics remain implementation-specific:
  advanced uses implementation resampled flags, while MLCOE uses an inferred
  `ESS < 0.5N` post-step threshold;
- low-noise pressure is clear for MLCOE across ESS, resampling count, runtime,
  and position RMSE; for advanced it appears mainly in lower minimum ESS and
  slightly higher runtime, while median average ESS and position RMSE did not
  degrade in this small panel.

Next phase justified:
- RP3 audit, tidy, reset-memo completion update, and scoped commit.

Proposed next experimental phase after RP3:
- create a controlled full-horizon EDH/PFPF sensitivity plan before any
  expansion to new algorithms;
- hypotheses to test next:
  - H1: the replicated-panel success survives full 20-observation horizons
    without runtime warnings or nonfinite outputs;
  - H2: increasing particles from 64 to 128 or 256 reduces low-noise ESS
    pressure more reliably than increasing flow steps alone;
  - H3: increasing flow steps from 10 to 20 improves observation-proxy RMSE or
    position RMSE only if it does not materially increase runtime;
  - H4: the advanced and MLCOE ESS/resampling diagnostics remain useful for
    within-implementation pressure tracking but should not be collapsed into a
    single cross-implementation correctness metric.

### RP3: audit, tidy, and completion

Status: passed.

Checks:
- `python -m py_compile experiments/student_dpf_baselines/runners/run_replicated_edh_pfpf_panel.py`
  passed;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- `git diff --check` passed;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  were modified;
- generated artifacts are small: panel JSON about `26K`, summary JSON about
  `8.4K`, report about `3.7K`.

Completion interpretation:
- the replicated EDH/PFPF phase completed fully inside the student
  experimental-baseline lane;
- the phase did not edit production `bayesfilter/`, monograph chapter files,
  references, monograph reset memos, or vendored student snapshots;
- known dirty and untracked monograph-lane files remain outside this lane and
  must remain unstaged by this work;
- the replicated panel is ready as a quarantined experimental baseline artifact
  with comparison-only interpretation.

Next justified work:
- write a controlled full-horizon EDH/PFPF sensitivity plan under the student
  baseline lane;
- do not expand to kernel PFF, stochastic flow, dPFPF, neural OT,
  differentiable resampling, HMC, or production code until a separate plan and
  gate justify those lanes;
- primary next hypotheses:
  - H1: full-horizon EDH/PFPF remains runnable for both implementations without
    runtime warnings or nonfinite outputs;
  - H2: particle-count increases reduce low-noise ESS pressure more reliably
    than flow-step increases alone;
  - H3: extra flow steps improve proxy RMSE only up to a bounded runtime/benefit
    threshold;
  - H4: ESS/resampling diagnostics should remain implementation-specific and
    should be used for within-implementation pressure trends, not cross-student
    correctness claims.

## Execution log: full-horizon EDH/PFPF sensitivity started 2026-05-12

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-full-horizon-edh-pfpf-sensitivity-plan-2026-05-12.md`.

### Full-horizon sensitivity goal

Status: planned.

Goal:
- test whether the replicated EDH/PFPF baseline remains usable on the full
  20-observation nonlinear range-bearing fixtures;
- compare 64 versus 128 particles and 10 versus 20 flow steps;
- preserve adapter-owned bridges, implementation-specific diagnostics, and
  comparison-only interpretation;
- avoid monograph, production, HMC, neural OT, differentiable resampling, and
  vendored-code work.

### Remaining gaps

Status: identified.

Gaps:
- full-horizon EDH/PFPF behavior is untested;
- particle-count sensitivity is unknown;
- flow-step sensitivity and runtime/RMSE tradeoff are unknown;
- ESS and resampling semantics remain implementation-specific;
- comparison must remain proxy-only.

### Hypotheses

Status: planned.

Hypotheses:
- FH1: full-horizon EDH/PFPF remains runnable for both implementations without
  vendored-code edits;
- FH2: 128 particles reduce low-noise pressure relative to 64 particles for at
  least one implementation;
- FH3: 20 flow steps produce a bounded RMSE/runtime tradeoff relative to 10
  steps;
- FH4: proxy comparison remains interpretable with explicit labels and missing
  metric handling;
- FH5: the phase produces a clear decision:
  `full_horizon_sensitivity_ready`,
  `full_horizon_sensitivity_ready_with_caveats`, `needs_targeted_debug`, or
  `blocked_or_excluded`.

Next phase justified: FH0 preflight and lane guard.

### FH0: preflight and lane guard

Status: passed.

Observed state:
- current Git status contains dirty and untracked monograph-lane files from
  other work, plus the new student full-horizon sensitivity plan and audit;
- independent plan audit exists at
  `docs/plans/bayesfilter-student-dpf-baseline-full-horizon-edh-pfpf-sensitivity-plan-audit-2026-05-12.md`;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  are dirty.

Interpretation:
- continuing is justified because the phase remains inside the student
  experimental-baseline lane;
- monograph-lane files must remain untouched and unstaged.

Next phase justified: FH1 sensitivity runner.

### FH1: full-horizon sensitivity runner

Status: passed.

File added:
`experiments/student_dpf_baselines/runners/run_full_horizon_edh_pfpf_sensitivity.py`.

Implementation:
- reuses the same adapter-owned EDH/PFPF bridges as the replicated panel;
- uses both full 20-observation nonlinear Gaussian range-bearing fixtures;
- runs seeds `17` and `23`;
- compares particles `64` and `128`;
- compares flow steps `10` and `20`;
- records 32 planned records across both implementations;
- uses a 45 second per-run runtime-warning threshold;
- preserves implementation-specific ESS and resampling-count semantics;
- writes the planned JSON, summary JSON, and Markdown result artifacts.

Check:
- `python -m py_compile experiments/student_dpf_baselines/runners/run_full_horizon_edh_pfpf_sensitivity.py`
  passed.

Interpretation:
- the runner remains inside the student experimental-baseline lane;
- executing the panel is justified.

Next phase justified: FH2 execute, classify, and report.

### FH2: execute, classify, and report

Status: passed.

Command:
`python -m experiments.student_dpf_baselines.runners.run_full_horizon_edh_pfpf_sensitivity`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/full_horizon_edh_pfpf_sensitivity_2026-05-12.json`;
- `experiments/student_dpf_baselines/reports/outputs/full_horizon_edh_pfpf_sensitivity_summary_2026-05-12.json`;
- `experiments/student_dpf_baselines/reports/student-dpf-baseline-full-horizon-edh-pfpf-sensitivity-result-2026-05-12.md`.

Panel:
- fixtures: `range_bearing_gaussian_moderate`,
  `range_bearing_gaussian_low_noise`;
- horizon: full fixture horizon, 20 observations;
- seeds: `17`, `23`;
- particles: `64`, `128`;
- flow steps: `10`, `20`;
- implementations: advanced `EDHParticleFilter`, MLCOE `PFPF_EDH`;
- planned records: 32.

Results:
- 32/32 records returned status `ok`;
- no runtime warnings were triggered;
- all successful records had finite means;
- generated artifacts are small: panel JSON about `77K`, summary JSON about
  `33K`, report about `6.1K`.

Implementation summary:
- `advanced_particle_filter`: 16/16 ok, median runtime about `0.0721` seconds,
  maximum runtime about `0.577` seconds, median position RMSE about `0.0519`,
  median average ESS about `19.3`;
- `2026MLCOE`: 16/16 ok, median runtime about `0.991` seconds, maximum runtime
  about `3.67` seconds, median position RMSE about `0.0537`, median average ESS
  about `30.4`.

Low-noise particle sensitivity:
- advanced, 10 steps: 128 particles increased median average ESS from about
  `18.4` to `30.4` and improved observation proxy RMSE from about `0.0164` to
  `0.0154`, while median position RMSE rose slightly from about `0.0464` to
  `0.0473`;
- advanced, 20 steps: 128 particles increased median average ESS from about
  `16.3` to `29.6`, while median position RMSE improved from about `0.0479` to
  `0.0466`;
- MLCOE, 10 steps: 128 particles increased median average ESS from about
  `14.2` to `28.6`, but median position RMSE rose from about `0.0470` to
  `0.0485`;
- MLCOE, 20 steps: 128 particles increased median average ESS from about
  `22.4` to `43.8` and improved median observation proxy RMSE from about
  `0.0172` to `0.0166`.

Flow-step sensitivity:
- 20 flow steps produced at least one bounded benefit signal in 7 of 8 grid
  summaries;
- the exception was advanced moderate-noise at 128 particles, where 20 steps
  increased runtime and did not improve median position RMSE or observation
  proxy RMSE;
- runtime ratios for 20 versus 10 steps were all below the `2.5` bounded-cost
  threshold used by the runner.

Hypothesis results:
- FH1 full horizon remains runnable: `supported_all_planned_runs_ok`;
- FH2 more particles reduce low-noise pressure:
  `supported_pressure_reduction_signal_observed`;
- FH3 more flow steps have bounded benefit:
  `supported_bounded_benefit_signal_observed`;
- FH4 proxy comparison remains interpretable: `supported_proxy_only`;
- FH5 decision: `full_horizon_sensitivity_ready`.

Interpretation:
- the full-horizon EDH/PFPF sensitivity panel is ready as a quarantined
  experimental baseline artifact;
- the evidence supports using higher particle counts for low-noise pressure
  reduction, mostly through ESS and sometimes through proxy RMSE;
- 20 flow steps are not uniformly better, but they remain bounded in this grid
  and often improve observation proxy RMSE;
- the result is not a production, correctness, HMC, or cross-student
  superiority claim;
- ESS and resampling-count semantics remain implementation-specific.

Next phase justified:
- FH3 audit, tidy, reset-memo completion update, and scoped commit.

Proposed next experimental phase after FH3:
- write a small confirmation plan that fixes a pragmatic setting, likely
  full horizon with 128 particles and 20 flow steps for the low-noise fixture
  plus 128 particles and 10 or 20 flow steps for the moderate fixture depending
  on implementation-specific cost/benefit;
- test additional seeds before any clean-room controlled baseline extraction;
- hypotheses to test next:
  - C1: the chosen full-horizon setting remains stable over additional seeds;
  - C2: the low-noise ESS gain from 128 particles persists across seeds;
  - C3: 20 flow steps improve observation proxy RMSE often enough to justify
    the runtime cost, or should be reserved for low-noise cases only;
  - C4: a clean-room controlled baseline can be specified from the fixture,
    metric, and reporting contract without copying student implementation code.

### FH3: audit, tidy, and completion

Status: passed.

Checks:
- `python -m py_compile experiments/student_dpf_baselines/runners/run_full_horizon_edh_pfpf_sensitivity.py`
  passed;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- `git diff --check` passed;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  were modified;
- generated artifacts are small: panel JSON about `77K`, summary JSON about
  `33K`, report about `6.1K`.

Completion interpretation:
- the full-horizon EDH/PFPF sensitivity phase completed fully inside the
  student experimental-baseline lane;
- the phase did not edit production `bayesfilter/`, monograph chapter files,
  references, monograph reset memos, or vendored student snapshots;
- known dirty and untracked monograph-lane files remain outside this lane and
  must remain unstaged by this work;
- the full-horizon panel is ready as a quarantined experimental baseline
  artifact with comparison-only interpretation.

Next justified work:
- write a student-lane full-horizon confirmation plan before clean-room
  extraction;
- fix a small number of pragmatic settings based on the sensitivity result
  rather than expanding the grid further;
- primary next hypotheses:
  - C1: the selected setting remains stable over additional seeds;
  - C2: 128 particles continue to reduce low-noise ESS pressure across seeds;
  - C3: 20 flow steps should be used selectively where observation proxy RMSE
    improves at bounded runtime cost;
  - C4: a clean-room controlled baseline can be specified from the fixture,
    metric, and reporting contract without copying student code.

## Execution log: full-horizon EDH/PFPF confirmation started 2026-05-12

Controlling plan:
`docs/plans/bayesfilter-student-dpf-baseline-full-horizon-edh-pfpf-confirmation-plan-2026-05-12.md`.

### Confirmation goal

Status: planned.

Goal:
- confirm the full-horizon EDH/PFPF experimental baseline over additional seeds
  using fixed pragmatic settings selected from the sensitivity panel;
- avoid broad grid expansion;
- decide whether the student-baseline evidence is stable enough to inform a
  later clean-room controlled baseline specification;
- preserve adapter-owned bridges, implementation-specific diagnostics, and
  comparison-only interpretation.

### Drift review

Status: passed.

Assessment:
- no active drift from the student experimental-baseline lane was found;
- the plan explicitly excludes monograph work, production code, HMC, neural OT,
  differentiable resampling, kernel PFF, and vendored-code edits;
- the plan was tightened before execution to define low-noise material
  degradation thresholds and moderate-noise flow-step policy rules.

Independent audit:
`docs/plans/bayesfilter-student-dpf-baseline-full-horizon-edh-pfpf-confirmation-plan-audit-2026-05-12.md`.

### Remaining gaps

Status: identified.

Gaps:
- selected full-horizon setting needs seed-stability confirmation;
- low-noise 128-particle pressure reduction needs additional-seed confirmation;
- moderate-noise 10-step versus 20-step policy remains unresolved;
- clean-room baseline inputs are not frozen;
- interpretation must remain proxy-only and comparison-only.

### Hypotheses

Status: planned.

Hypotheses:
- C1: selected full-horizon setting is seed-stable;
- C2: low-noise 128-particle pressure reduction persists;
- C3: moderate-noise flow-step policy can be resolved;
- C4: clean-room baseline specification is ready without copying student code;
- C5: next baseline decision is clear.

Next phase justified: C0 preflight and lane guard.

### C0: preflight and lane guard

Status: passed.

Observed state:
- current Git status contains dirty and untracked monograph-lane files from
  other work, plus the new student confirmation plan and audit;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  are dirty;
- independent plan audit exists.

Interpretation:
- continuing is justified because the phase can remain fully inside the student
  experimental-baseline lane;
- monograph-lane files must remain untouched and unstaged.

Next phase justified: C1 confirmation runner.

### C1: confirmation runner

Status: passed.

File added:
`experiments/student_dpf_baselines/runners/run_full_horizon_edh_pfpf_confirmation.py`.

Implementation:
- reuses the adapter-owned EDH/PFPF bridges from the prior full-horizon runner;
- uses full 20-observation nonlinear Gaussian range-bearing fixtures;
- runs additional seeds `31`, `43`, `59`, `71`, and `83`;
- fixes particles at `128`;
- uses 20 flow steps for the low-noise fixture;
- compares 10 and 20 flow steps for the moderate-noise fixture;
- records 30 planned records across both implementations;
- uses a 45 second per-run runtime-warning threshold;
- preserves implementation-specific ESS and resampling-count semantics;
- writes the planned JSON, summary JSON, and Markdown result artifacts.

Check:
- `python -m py_compile experiments/student_dpf_baselines/runners/run_full_horizon_edh_pfpf_confirmation.py`
  passed.

Interpretation:
- the runner remains inside the student experimental-baseline lane;
- executing the confirmation panel is justified.

Next phase justified: C2 execute, classify, and report.

### C2: execute, classify, and report

Status: passed.

Command:
`python -m experiments.student_dpf_baselines.runners.run_full_horizon_edh_pfpf_confirmation`.

Outputs:
- `experiments/student_dpf_baselines/reports/outputs/full_horizon_edh_pfpf_confirmation_2026-05-12.json`;
- `experiments/student_dpf_baselines/reports/outputs/full_horizon_edh_pfpf_confirmation_summary_2026-05-12.json`;
- `experiments/student_dpf_baselines/reports/student-dpf-baseline-full-horizon-edh-pfpf-confirmation-result-2026-05-12.md`.

Panel:
- fixtures: `range_bearing_gaussian_moderate`,
  `range_bearing_gaussian_low_noise`;
- horizon: full fixture horizon, 20 observations;
- seeds: `31`, `43`, `59`, `71`, `83`;
- particles: `128`;
- low-noise flow steps: `20`;
- moderate-noise flow steps: `10` and `20`;
- implementations: advanced `EDHParticleFilter`, MLCOE `PFPF_EDH`;
- planned records: 30.

Results:
- 30/30 records returned status `ok`;
- no runtime warnings were triggered;
- all successful records had finite means;
- generated artifacts are small: panel JSON about `58K`, summary JSON about
  `17K`, report about `4.7K`.

Implementation summary:
- `advanced_particle_filter`: 15/15 ok, median runtime about `0.0780` seconds,
  maximum runtime about `0.581` seconds, median position RMSE about `0.0653`,
  median average ESS about `22.6`;
- `2026MLCOE`: 15/15 ok, median runtime about `1.00` seconds, maximum runtime
  about `3.88` seconds, median position RMSE about `0.0640`, median average ESS
  about `59.3`.

Low-noise confirmation:
- advanced low-noise N128/steps20: median average ESS about `27.6`, median
  position RMSE about `0.0475`, median observation proxy RMSE about `0.0162`,
  median resampling count `18`; all material-degradation checks passed against
  the sensitivity-panel reference;
- MLCOE low-noise N128/steps20: median average ESS about `42.7`, median
  position RMSE about `0.0468`, median observation proxy RMSE about `0.0156`,
  median resampling count `17`; all material-degradation checks passed against
  the sensitivity-panel reference.

Moderate-noise flow-step policy:
- advanced moderate N128: 20 steps did not improve median position RMSE or
  observation proxy RMSE relative to 10 steps; runtime ratio about `1.55`;
- MLCOE moderate N128: 20 steps improved median observation proxy RMSE but not
  median position RMSE; runtime ratio about `1.64`;
- policy recommendation: `moderate_keep_both_as_diagnostic`;
- rationale: implementation-specific benefit differs.

Hypothesis results:
- C1 selected full-horizon setting seed-stable: `supported_seed_stable`;
- C2 low-noise 128-particle pressure reduction persists:
  `supported_low_noise_pressure_reduction_persists`;
- C3 moderate-noise flow-step policy resolved:
  `resolved_moderate_keep_both_as_diagnostic`;
- C4 clean-room baseline specification ready:
  `supported_clean_room_inputs_ready`;
- C5 next baseline decision: `confirmation_ready_with_caveats`.

Interpretation:
- the full-horizon confirmation panel is stable enough to inform a caveated
  clean-room controlled-baseline specification;
- low-noise N128/steps20 is confirmed for both implementations under proxy
  metrics and implementation-specific diagnostics;
- moderate-noise flow-step policy must remain caveated because advanced and
  MLCOE show different 20-step benefit patterns;
- no production, monograph, HMC, or student-code promotion claims are supported.

Next phase justified:
- C3 audit, tidy, reset-memo completion update, and scoped commit.

Proposed next phase after C3:
- write a student-lane clean-room controlled-baseline specification plan;
- do not copy student implementation code;
- carry forward these explicit hypotheses:
  - S1: a clean-room BayesFilter-owned experimental baseline can reproduce the
    fixture contract and metrics without importing student code;
  - S2: low-noise full-horizon baseline should use 128 particles and 20 flow
    steps as the first clean-room target setting;
  - S3: moderate-noise baseline should keep 10 and 20 flow steps as diagnostic
    variants until a BayesFilter-owned implementation establishes a clearer
    policy;
  - S4: ESS and resampling diagnostics remain within-implementation pressure
    metrics and must not be used as cross-implementation correctness evidence.

### C3: audit, tidy, and completion

Status: passed.

Checks:
- `python -m py_compile experiments/student_dpf_baselines/runners/run_full_horizon_edh_pfpf_confirmation.py`
  passed;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- `git diff --check` passed;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  were modified;
- generated artifacts are small: panel JSON about `58K`, summary JSON about
  `17K`, report about `4.7K`.

Completion interpretation:
- the confirmation phase completed fully inside the student
  experimental-baseline lane;
- the phase did not edit production `bayesfilter/`, monograph chapter files,
  references, monograph reset memos, or vendored student snapshots;
- known dirty and untracked monograph-lane files remain outside this lane and
  must remain unstaged by this work;
- the correct next step is a caveated clean-room controlled-baseline
  specification plan, not another student-code grid expansion.

Next justified work:
- write the clean-room controlled-baseline specification plan under the student
  baseline lane;
- include the confirmed fixture/metric/setting contract;
- carry forward the moderate-noise flow-step caveat;
- keep all production implementation work behind a separate future plan.

## Execution log: clean-room controlled-baseline specification started 2026-05-13

Controlling plans:
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-plan-2026-05-12.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-completion-plan-2026-05-13.md`.

### SP0: preflight and lane guard

Status: passed.

Observed state:
- current Git status contains one unrelated dirty monograph reset memo:
  `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`;
- current Git status also contains the student clean-room specification plan
  files for this phase;
- import-boundary search over `bayesfilter` and `tests` found no imports or
  references to `experiments/student_dpf_baselines`,
  `advanced_particle_filter`, or `2026MLCOE`;
- no files under `experiments/student_dpf_baselines/vendor/`,
  `experiments/student_dpf_baselines/fixtures/`,
  `experiments/student_dpf_baselines/runners/`, or
  `experiments/student_dpf_baselines/reports/` are dirty for this phase.

Plan tightening:
- the completion plan was updated from planned-only status to execution
  authorized by the current user instruction;
- the completion plan now requires one path-scoped student-lane commit and
  explicitly forbids push unless separately requested.

Drift assessment:
- no active drift from the student experimental-baseline lane was found;
- the unrelated monograph reset memo must remain unstaged and uncommitted by
  this work.

Interpretation:
- continuing is justified because all required edits can remain in student
  baseline plan/reset/master documents;
- no production code, monograph files, references, vendored student files, or
  experiment outputs are needed for this specification phase.

Next phase justified: SP1 clean-room specification.

### SP1: clean-room specification

Status: passed.

File added:
`docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`.

Specification content:
- clean-room scope and provenance;
- explicit no-copy/no-import boundary for student code;
- fixture contract for `range_bearing_gaussian_moderate` and
  `range_bearing_gaussian_low_noise`;
- state and observation conventions;
- transition, observation, covariance, initial distribution, horizon, and seed
  policy;
- first target settings: low-noise N128/steps20 and moderate N128/steps10 plus
  N128/steps20 as diagnostic variants;
- required metrics, optional diagnostics, null-handling rules, and result
  schema;
- acceptance, execution, and commit gates for a later implementation phase;
- caveats preserving proxy-only, comparison-only interpretation.

Interpretation:
- S1 through S4 remain testable by audit because the specification states the
  fixture and metric contract without importing or copying student vendor code;
- the moderate-noise flow-step caveat is preserved rather than flattened into a
  single winner;
- no production, monograph, references, vendored student, or experiment-output
  files were needed.

Next phase justified: SP2 independent audit.

### SP2: independent audit

Status: passed with nonblocking tightening.

Audit note:
`docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-audit-2026-05-13.md`.

Audit result:
- blocking findings: none;
- lane boundary: passed;
- monograph drift: passed;
- production drift: passed;
- vendor edit risk: passed;
- student import/copying risk: passed;
- fixture completeness: passed;
- metric completeness: passed;
- moderate-noise caveat preservation: passed;
- proxy-only interpretation: passed;
- future gate enforceability: passed.

Tightening applied:
- the 2026-05-12 source plan now points to the concrete 2026-05-13 execution
  artifacts;
- the 2026-05-13 completion plan now reflects execution authorization, scoped
  commit authorization, and no push;
- the specification now states that the first controlled target must be a
  BayesFilter-owned particle-flow or flow-assisted particle filtering baseline
  with an explicit flow-step or integration-step control; non-flow references
  must be labeled separately.

Interpretation:
- the specification can support a later clean-room implementation planning
  phase;
- implementation is not yet authorized by this specification phase alone;
- the next result should classify the phase using the H5 decision labels.

Next phase justified: SP3 result note and memo/master update.

### SP3: result note and continuity update

Status: passed.

Result note:
`docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-result-2026-05-13.md`.

Decision:
`clean_room_spec_ready_for_implementation_plan`.

Hypothesis results:
- H1 clean-room fixture contract without student code: supported;
- H2 first target settings fixed enough: supported with moderate caveat;
- H3 metric contract preserves proxy-only interpretation: supported;
- H4 implementation boundary can be audited: supported;
- H5 next decision is clear: supported.

Continuity updates:
- this reset memo now points to the clean-room implementation planning phase as
  the next justified action;
- the student master program now records the completed clean-room specification,
  audit, and result, and points to a separate implementation plan under
  `experiments/controlled_dpf_baseline/`.

Interpretation:
- the current phase closes the planning-level fixture, metric, target-setting,
  caveat, and clean-room-boundary gaps;
- it does not authorize immediate production work, monograph work, or broad
  experiment expansion;
- the next phase should test implementation hypotheses I1 through I4 from the
  result note.

Next phase justified: SP4 verification and scoped commit.

### SP4: verification and scoped commit

Status: passed before commit.

Checks:
- `git diff --check` passed;
- import-boundary search over `bayesfilter` and `tests` found no imports of
  `experiments/student_dpf_baselines`, `advanced_particle_filter`, or
  `2026MLCOE`;
- no vendored student files under `experiments/student_dpf_baselines/vendor/`
  are dirty;
- no experiment outputs were created by this specification phase;
- no production `bayesfilter/`, monograph chapter, reference, vendored student,
  or generated-output files are part of the scoped commit.

Known unrelated dirty file excluded from this lane:
- `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`.

Commit scope:
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-plan-2026-05-12.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-completion-plan-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-audit-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-result-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`.

Completion interpretation:
- the clean-room controlled-baseline specification phase completed inside the
  student experimental-baseline lane;
- the next justified phase is a separate student-lane implementation plan for
  `experiments/controlled_dpf_baseline/`, not immediate implementation,
  production promotion, or monograph work.
