# Reset memo: student DPF experimental baseline

## Date

2026-05-09

## Active scope

This is the active reset memo for the quarantined student DPF
experimental-baseline stream.

Active master program:
`docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`.

Active next-phase plan:
`docs/plans/bayesfilter-student-dpf-baseline-mp3-kernel-pff-debug-gate-plan-2026-05-11.md`.

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
