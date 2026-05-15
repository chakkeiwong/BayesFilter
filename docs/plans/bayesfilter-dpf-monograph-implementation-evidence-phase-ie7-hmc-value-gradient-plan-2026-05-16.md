# Phase IE7 plan: same-scalar HMC value-gradient tests

## Date

2026-05-16

## Purpose

Verify the same-scalar contract for HMC-facing DPF targets: the scalar value
used by accept/reject logic must be the scalar whose gradient is supplied to
the integrator.  This phase checks value-gradient consistency and compiled
repeatability before any HMC tuning or posterior interpretation.  Passing IE7
does not validate HMC, DPF-HMC, or posterior correctness.

## Allowed Write Set

- `experiments/dpf_monograph_evidence/fixtures/`;
- `experiments/dpf_monograph_evidence/diagnostics/`;
- `experiments/dpf_monograph_evidence/runners/`;
- `experiments/dpf_monograph_evidence/reports/`;
- implementation-evidence result plan files and reset memo continuity.

## Prerequisites

- IE2 harness ready;
- IE3--IE5 completed or blocked with reasons that do not invalidate the fixed
  scalar target fixture.

## Tasks

1. Define a small deterministic scalar target with explicit random-seed policy.
2. Evaluate scalar value repeatedly in eager and compiled modes when available.
3. Compare autodiff gradients with finite-difference step ladders.
4. Record value identity, gradient path, finite status, repeatability, and
   failure layer.
5. Add a fixed-target leapfrog reversibility or energy-drift smoke diagnostic,
   or explicitly record that all HMC chain-based posterior sensitivity is
   prohibited in IE8.
6. Prohibit HMC tuning until value-gradient checks pass.

## IE7 Evidence Contract

IE7 owns exactly one canonical diagnostic id: `hmc_value_gradient`.  It must
write one schema-valid top-level JSON object, not an array:

- `experiments/dpf_monograph_evidence/reports/outputs/hmc_value_gradient.json`.

The fixed target must be a deterministic scalar fixture:

- fixture id: `fixed_scalar_hmc_target_fixture`;
- scalar function:
  `U(q) = 0.5 * q^T M q + beta * sum(q_i^4) + gamma * sin(c^T q)`;
- fixed dimension: `2`;
- fixed evaluation point: `[0.4, -0.25]`;
- fixed mass/inverse-mass for the leapfrog smoke;
- no stochastic terms in value, gradient, compiled path, or leapfrog smoke.

The same callable/wrapper must provide both:

- `accept_reject_scalar_value`;
- `differentiated_scalar_value`.

The row must record a same-scalar proof mechanism:

- `same_scalar_source="single_target_function_call"`;
- the same target wrapper id for accept/reject and differentiated paths;
- scalar equality residual;
- explicit verdict that the differentiated scalar and accept/reject scalar came
  from the same function call path, not two separately implemented expressions.

## Row Schema Checklist

The JSON `tolerance` object must include these keys, each with shape
`{threshold, observed, finite}`:

- `same_scalar_value_abs`;
- `analytic_gradient_abs_max`;
- `finite_difference_stable_window_abs_max`;
- `eager_repeat_value_abs_max`;
- `eager_repeat_gradient_abs_max`;
- `compiled_value_abs_max`;
- `compiled_gradient_abs_max`;
- `leapfrog_reversibility_position_abs_max`;
- `leapfrog_reversibility_momentum_abs_max`;
- `energy_drift_abs`;

The `finite_checks` object must record:

- fixture id, dimension, evaluation point, scalar formula, target wrapper id;
- accept/reject scalar value;
- differentiated scalar value;
- same-scalar source and verdict;
- analytic gradient;
- autodiff or analytic-gradient path used by the runner;
- finite-difference ladder table with step, central-difference gradient,
  absolute residual, relative residual, and classification
  (`truncation_region`, `stable_window`, or `cancellation_region`);
- selected stable-window step(s);
- eager repeat values and gradients;
- compiled path status:
  - if available, compiled values/gradients/repeatability and parity verdict;
  - if unavailable, explicit `not_applicable` reason;
- leapfrog reversibility position/momentum residuals;
- fixed-target energy before/after one forward-and-reverse smoke;
- assertion that no HMC chain, adaptation, mass tuning, or posterior summary
  was run.

The `shape_checks` object must record state dimension, gradient shape,
finite-difference ladder length, repeat count, leapfrog step count, and scalar
output shape.

## Finite-Difference And Smoke Contract

Finite-difference ladder:

- steps: `[1e-2, 1e-3, 1e-4, 1e-5, 1e-6]`;
- central difference only;
- stable window: `[1e-3, 1e-4, 1e-5]`;
- primary gradient threshold:
  `finite_difference_stable_window_abs_max <= 1e-5`;
- cap accounting:
  `max_finite_difference_evaluations = 2 * dimension * len(steps)`.

Failure interpretation:

- large residual at `1e-2` may be truncation-region explanatory evidence;
- large residual at `1e-6` may be cancellation-region explanatory evidence;
- stable-window failure is a repair trigger;
- any nonfinite value is a repair trigger.

Fixed-target smoke:

- run both a leapfrog reversibility check and an energy-drift smoke;
- fixed step size: `0.05`;
- fixed leapfrog step count: `3`;
- fixed initial momentum: `[0.3, -0.2]`;
- no chain, no adaptation, no tuning, no posterior summary;
- reversibility thresholds:
  - position max absolute residual `<= 1e-10`;
  - momentum max absolute residual `<= 1e-10`;
- forward energy-drift threshold `<= 1e-4`.

The energy-drift smoke is a bounded fixed-step integration sanity check, not a
claim of exact energy conservation.  Round-trip reversibility remains the
near-machine-precision check.

## Compiled/Eager Status

IE7 must explicitly report one of:

- compiled path available: record compiled values, compiled gradients,
  repeatability, and eager/compiled parity verdicts;
- compiled path unavailable: record `compiled_status=not_available` and a
  reason in the JSON row.

Compiled unavailability is not a blocker if eager same-scalar, gradient, and
fixed-target smoke checks pass and the row states that compiled parity was not
tested.

## Seed And Determinism Policy

IE7 must set:

- `seed_policy=deterministic_no_rng_fixed_scalar_target`;
- `replication_count=1`;
- `uncertainty_status=not_applicable`;
- no RNG use in value calls, gradient calls, compiled calls, or leapfrog smoke.

The row must assert that repeated evaluations do not mutate the target state.

## Coverage Semantics

Every IE7 row must include the full canonical coverage object and carry forward:

- `linear_gaussian_recovery=passed`;
- `synthetic_affine_flow=passed`;
- `pfpf_algebra_parity=passed`;
- `soft_resampling_bias=passed`;
- `sinkhorn_residual=passed`;
- `learned_map_residual=deferred`.

Set `hmc_value_gradient=passed` only if IE7 passes; otherwise set it to
`blocked`.  Set `posterior_sensitivity_summary=missing`.

## Required Run Manifest

The IE7 runner must record the full IE2 run-manifest keys.  It must set
`CUDA_VISIBLE_DEVICES=-1` before importing NumPy or any scientific dependency
and must record:

- `cpu_only=true`;
- `cuda_visible_devices="-1"`;
- `gpu_devices_visible=[]`;
- `gpu_hidden_before_import=true`;
- `pre_import_cuda_visible_devices="-1"`;
- `pre_import_gpu_hiding_assertion=true`;
- branch, commit, dirty-state summary, Python version, NumPy version, command,
  wall-clock cap, started/ended UTC timestamps, and artifact paths.

## Explicit Prohibitions

IE7 does not authorize:

- HMC chain execution;
- step-size tuning;
- mass-matrix tuning;
- posterior summaries;
- real DPF-HMC target instantiation;
- DSGE or MacroFinance target sampling;
- production `bayesfilter/` imports.

## Required Non-Implication Text

```text
IE7 fixed-scalar value-gradient diagnostics validate only same-scalar,
finite-difference, repeatability, and fixed-target leapfrog/energy-smoke checks
on a deterministic clean-room fixture. They do not validate HMC correctness,
DPF-HMC correctness, posterior/reference agreement, tuning readiness beyond
controlled-fixture eligibility, production bayesfilter code, banking use,
model-risk use, or production readiness.
```

## Per-Diagnostic Decision Rules

For `hmc_value_gradient`:

- `promotion_criterion_status=pass` only if same-scalar identity,
  stable-window finite-difference gradient, eager repeatability, compiled status
  reporting, leapfrog reversibility, and energy smoke satisfy the declared
  criteria.
- `promotion_veto_status=fail` if target identity, same-callable proof,
  finite-difference ladder, compiled/eager status, seed policy, CPU manifest,
  no-chain assertion, source support, tolerance object, or exact
  non-implication text is missing.
- `continuation_veto_status=fail` if same-scalar identity fails,
  nondeterministic target behavior is observed, or any HMC chain/tuning is run.
- `repair_trigger` must name same-scalar mismatch, stable-window gradient
  mismatch, compiled/eager discrepancy, repeatability failure, reversibility
  failure, energy-drift failure, nondeterminism, prohibited HMC execution, or
  schema failure.
- explanatory-only diagnostics include truncation-region and cancellation-region
  finite-difference residuals and any HMC/posterior commentary.

## Outcome / Exit Mapping

IE7 result notes must record both a master-program label and a local label.
Master-program labels:

- `ie_phase_passed`;
- `ie_phase_passed_with_caveats`;
- `ie_phase_blocked`;
- `ie_phase_deferred_with_recorded_reason`;
- `ie_phase_rejected_for_lane_drift`.

Local-to-master mapping:

- `ie7_hmc_value_gradient_passed` -> `ie_phase_passed`;
- `ie7_hmc_value_gradient_passed_with_caveats` ->
  `ie_phase_passed_with_caveats`;
- `ie7_hmc_value_gradient_blocks_hmc` -> `ie_phase_blocked`;
- `ie7_hmc_value_gradient_blocked` -> `ie_phase_blocked`.

## Primary Criterion

The same-scalar value-gradient contract passes on the fixed target or produces
a blocker that prevents HMC tuning and posterior claims.

## Veto Diagnostics

- accept/reject value and differentiated value are not proven identical;
- finite-difference tolerance is not justified;
- stochastic randomness changes across value and gradient calls;
- compiled/eager mismatch is ignored;
- HMC sampling is run before this phase passes.
- IE7 success is interpreted as posterior validity rather than eligibility for
  controlled-fixture sensitivity only.

## Outcome Classification

- Promotion/pass criterion: same scalar identity, finite-difference gradient,
  repeatability, and fixed-target reversibility/energy smoke pass on the
  controlled fixture.
- Promotion veto: accept/reject scalar identity, seed policy, compiled/eager
  status, or non-implication text is missing.
- Continuation veto: value-gradient mismatch or nondeterministic scalar value.
- Repair trigger: finite-difference ladder instability, compiled/eager
  discrepancy, energy-drift smoke failure, or nonfinite gradient.
- Explanatory-only diagnostics: any HMC chain or posterior commentary; IE7 does
  not authorize real DPF-HMC target sampling.

## Expected Artifacts

- `experiments/dpf_monograph_evidence/reports/hmc-value-gradient-result.md`;
- `experiments/dpf_monograph_evidence/reports/outputs/hmc_value_gradient.json`;
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie7-hmc-value-gradient-result-{YYYY-MM-DD}.md`.

## Exit Labels

- `ie7_hmc_value_gradient_passed`;
- `ie7_hmc_value_gradient_passed_with_caveats`;
- `ie7_hmc_value_gradient_blocks_hmc`;
- `ie7_hmc_value_gradient_blocked`.
