# Specification: student DPF clean-room controlled baseline

## Date

2026-05-13

## Status

Specification complete for student-lane audit.  This document is not an
implementation plan, does not execute experiments, and does not authorize
production `bayesfilter/` edits.

## Scope

This specification converts the confirmed student DPF experimental-baseline
evidence into a BayesFilter-owned controlled-baseline contract.  The intended
use is a later clean-room implementation under the student experimental lane,
most likely under `experiments/controlled_dpf_baseline/`.

The specification is derived from BayesFilter-owned fixture definitions and
student-baseline result reports.  Student implementations remain
comparison-only evidence.  They are not production code, API authority, or
correctness certificates.

## Provenance

BayesFilter-owned evidence:

- nonlinear fixture definitions:
  `experiments/student_dpf_baselines/fixtures/nonlinear_fixtures.py`;
- confirmation result:
  `experiments/student_dpf_baselines/reports/student-dpf-baseline-full-horizon-edh-pfpf-confirmation-result-2026-05-12.md`;
- controlling plan:
  `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-plan-2026-05-12.md`;
- completion plan:
  `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-completion-plan-2026-05-13.md`.

Student snapshot commits are cited only to identify the comparison evidence:

- `2026MLCOE`: `020cfd7f2f848afa68432e95e6c6e747d3d2402d`;
- `advanced_particle_filter`: `d2a797c330e11befacbb736b5c86b8d03eb4a389`.

## Clean-room boundary

A later controlled-baseline implementation may use this specification, the
BayesFilter-owned fixture definitions, the result schemas, and the reported
comparison numbers.  It must not use student implementation source as a design
source.

Prohibited actions:

- import from `experiments/student_dpf_baselines/vendor/`;
- import `advanced_particle_filter`, `2026MLCOE`, or student `src.*` modules;
- call existing student adapters as part of the clean-room baseline algorithm;
- copy student classes, functions, control flow, tuning tricks, or numerical
  shortcuts;
- edit vendored student snapshots in place;
- edit production `bayesfilter/` code without a separate production plan;
- edit DPF monograph rebuild/enrichment files, `docs/chapters/ch19*.tex`, or
  `docs/references.bib`.

Permitted future references:

- fixture equations and constants listed below;
- metric formulas listed below;
- result field names and status labels;
- student result aggregates as external comparison benchmarks only.

## Fixture contract

### Fixture names

The first clean-room baseline must implement exactly these two fixtures:

- `range_bearing_gaussian_moderate`;
- `range_bearing_gaussian_low_noise`.

### State and observation conventions

State vector:

```text
x_t = [px_t, py_t, vx_t, vy_t]^T
```

Observation vector:

```text
y_t = [range_t, bearing_t]^T
```

The state dimension is 4.  The observation dimension is 2.  The fixture horizon
is 20 observations and 21 latent states including the initial state.

### Transition model

The transition is linear Gaussian constant velocity:

```text
x_0 ~ Normal(m0, P0)
x_{t+1} ~ Normal(A x_t, Q),  t = 0, ..., 19
```

with:

```text
dt = 0.1

A = [[1.0, 0.0, dt,  0.0],
     [0.0, 1.0, 0.0, dt ],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]

Q = diag([0.0015, 0.0015, 0.0008, 0.0008])

m0 = [1.2, 0.7, 0.18, -0.06]

P0 = diag([0.04, 0.04, 0.01, 0.01])
```

### Observation model

For each post-transition state `x_{t+1}`, generate:

```text
h(x) = [sqrt(px^2 + py^2 + 1e-12), atan2(py, px)]^T
y_t ~ Normal(h(x_{t+1}), R)
```

The generated bearing component must be wrapped into `[-pi, pi)`.

When evaluating residuals, the range residual is ordinary subtraction and the
bearing residual must be wrapped into `[-pi, pi)`.

### Observation covariance regimes

Moderate fixture:

```text
fixture_name = range_bearing_gaussian_moderate
sigma_range = 0.12
sigma_bearing = 0.04
R = diag([0.12^2, 0.04^2])
fixture_generation_seed = 701
```

Low-noise fixture:

```text
fixture_name = range_bearing_gaussian_low_noise
sigma_range = 0.035
sigma_bearing = 0.012
R = diag([0.035^2, 0.012^2])
fixture_generation_seed = 702
```

### Reproducibility policy

The fixture generator should reproduce the BayesFilter-owned fixture arrays
exactly when implemented with NumPy `default_rng(seed)` and multivariate normal
draws in the order specified above.  If a later implementation uses a different
random engine, it must either store the generated fixture arrays or record the
numerical differences against the BayesFilter-owned fixture arrays before using
the results as comparison evidence.

Algorithm randomness is separate from fixture generation randomness.  The first
clean-room comparison run should use these algorithm seeds:

```text
31, 43, 59, 71, 83
```

Every future run must record the fixture-generation seed, algorithm seed list,
particle count, flow-step policy, horizon, and random library versions.

## Target settings

The first clean-room controlled-baseline implementation must avoid expanding
the grid.  Use the confirmed target settings only:

| Fixture | Particles | Flow steps | Status |
| --- | ---: | ---: | --- |
| `range_bearing_gaussian_low_noise` | 128 | 20 | first target setting |
| `range_bearing_gaussian_moderate` | 128 | 10 | diagnostic variant |
| `range_bearing_gaussian_moderate` | 128 | 20 | diagnostic variant |

The moderate-noise fixture must keep both 10 and 20 flow-step variants because
the confirmation panel did not show a single cross-implementation winner.

The default runtime-warning threshold for first comparison runs is 45 seconds
per implementation/fixture/seed/settings record.

## Algorithm-scope rule

The first controlled target is a BayesFilter-owned experimental particle-flow
or flow-assisted particle filtering baseline with an explicit flow-step or
integration-step control.  The later implementation plan must define that
algorithm from mathematical statements and BayesFilter-owned code only.

A plain bootstrap particle filter, EKF, UKF, or other non-flow sanity baseline
may be useful as a reference, but it must be labeled separately and must not be
reported as satisfying the N128/steps10 or N128/steps20 controlled target.

## Metric contract

Candidate filtered means should be shaped `(21, 4)`, including the initial mean
and one state estimate after each observation.  If an algorithm naturally emits
only 20 post-observation means, the result schema must state this explicitly and
the metric code must align the means to `x_1, ..., x_20` without silently
changing definitions.

Required metrics:

- `state_rmse`: square root of the mean squared error over all available state
  coordinates and aligned time indices;
- `position_rmse`: square root of the mean squared error over `px, py` and
  aligned time indices;
- `final_position_error`: Euclidean norm of the final `px, py` error;
- `observation_proxy_rmse`: unweighted RMSE of wrapped residuals between
  `h(mean_t)` and the observed range-bearing sequence for post-observation
  states;
- `runtime_seconds`;
- finite-output checks for means, covariances if emitted, particles if emitted,
  and scalar metrics.

Optional diagnostics:

- `average_ess`;
- `min_ess`;
- `resampling_count`;
- `log_likelihood`;
- covariance RMSE if an independent covariance reference exists.

Null-handling rules:

- unavailable optional diagnostics must be recorded as `null`;
- unavailable required metrics must produce a structured blocker status, not a
  silently missing field;
- failed runs must preserve fixture name, seed, target setting, failure reason,
  and elapsed runtime.

Interpretation rules:

- state and position metrics are simulated-fixture proxy metrics;
- observation proxy RMSE is not a likelihood and is not whitened by `R`;
- ESS and resampling counts are pressure diagnostics whose semantics must be
  labeled for each implementation;
- likelihood values are optional and cannot be primary cross-implementation
  evidence unless target semantics are independently verified.

## Result schema

Each record should include at least:

- `implementation_name`;
- `implementation_type`, for example `clean_room_controlled_baseline`;
- `fixture_name`;
- `target`;
- `status`;
- `failure_reason`;
- `seed`;
- `num_particles`;
- `flow_steps`;
- `horizon`;
- `runtime_seconds`;
- `metrics`;
- `diagnostics`;
- `provenance`.

The summary should group records by implementation, fixture, particle count,
and flow steps.  It should report planned records, successful records, failed
records, median required metrics, runtime warnings, finite-output counts, and
all structured blockers.

## Acceptance gates for later implementation

Preflight gates:

- implementation plan exists and has an independent audit note;
- path boundary is limited to `experiments/controlled_dpf_baseline/` and
  student-baseline plan/reset/master files unless separately approved;
- production import-boundary search still returns no student-baseline imports:

```bash
rg -n "experiments/student_dpf_baselines|advanced_particle_filter|2026MLCOE" bayesfilter tests
```

Implementation gates:

- no imports from `experiments/student_dpf_baselines/vendor/`;
- no calls to student adapters inside the clean-room algorithm;
- code review confirms no copied student implementation code;
- fixture generation is either implemented directly from this specification or
  cross-checked against the BayesFilter-owned fixture arrays;
- all generated artifacts are small enough for repository history or are
  explicitly excluded before commit.

Execution gates:

- all planned records complete or fail with structured blocker records;
- all successful records have finite required metrics;
- runtime-warning counts are recorded;
- low-noise N128/steps20 and moderate N128/steps10/20 are all present;
- no HMC, kernel PFF, stochastic flow, DPF, dPFPF, neural OT, differentiable
  resampling, notebook, plotting, production, or monograph claims are added.

Commit gates:

- `git diff --check` passes;
- only student-baseline or controlled-baseline files are staged;
- unrelated monograph-lane files remain unstaged;
- no vendored student files are modified.

## Confirmation evidence to preserve

The full-horizon confirmation panel produced 30/30 successful records over the
two student implementations, two fixtures, five algorithm seeds, 128 particles,
and the flow-step policy specified above.  No runtime warnings or nonfinite
outputs were reported.

Low-noise N128/steps20 comparison medians:

| Implementation | Position RMSE | Observation proxy RMSE | Average ESS |
| --- | ---: | ---: | ---: |
| `2026MLCOE` | 0.0468415 | 0.0156068 | 42.7296 |
| `advanced_particle_filter` | 0.0475423 | 0.0162379 | 27.6236 |

Moderate-noise caveat:

- `advanced_particle_filter` N128/steps20 did not improve median position RMSE
  or observation proxy RMSE relative to steps10;
- `2026MLCOE` N128/steps20 improved median observation proxy RMSE but not median
  position RMSE relative to steps10;
- therefore moderate N128/steps10 and N128/steps20 remain diagnostic variants.

## Decision labels for the next phase

A later result must choose one of:

- `clean_room_spec_ready_for_implementation_plan`;
- `clean_room_spec_ready_with_caveats`;
- `needs_spec_revision`;
- `blocked_or_excluded`.

This specification supports `clean_room_spec_ready_for_implementation_plan`
only if independent audit finds no blocking clean-room or lane-boundary defects.

## Later implementation outline

The next justified phase, if audit passes, should be an implementation plan
rather than immediate broad experiment execution:

1. Write and audit a clean-room implementation plan.
2. Implement a BayesFilter-owned fixture generator or controlled fixture module
   from this specification.
3. Implement a minimal controlled baseline algorithm or algorithm wrapper
   without student imports or copied student code.
4. Run only the first target settings listed above.
5. Compare against student result aggregates as external benchmark evidence.
6. Write a result note with structured blockers, caveats, and next hypotheses.
7. Update only the student reset memo and student master program.
8. Make a path-scoped commit with no monograph, production, reference, vendored
   student, or generated oversized artifacts.

## Caveats

This specification intentionally does not claim production readiness.  It does
not validate EDH/PFPF mathematical correctness, production BayesFilter APIs, or
the DPF monograph.  It only fixes the first clean-room controlled-baseline
contract that can be used to compare a BayesFilter-owned experimental baseline
against quarantined student evidence.
