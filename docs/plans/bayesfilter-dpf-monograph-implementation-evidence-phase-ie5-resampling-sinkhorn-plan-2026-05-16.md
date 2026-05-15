# Phase IE5 plan: soft-resampling and Sinkhorn controlled tests

## Date

2026-05-16

## Purpose

Execute controlled diagnostics for relaxed resampling layers: two-particle
soft-resampling bias and small Sinkhorn/EOT residual behavior.  These tests
check the relaxed target and solver layer, not posterior validity.

## Allowed Write Set

- `experiments/dpf_monograph_evidence/fixtures/`;
- `experiments/dpf_monograph_evidence/diagnostics/`;
- `experiments/dpf_monograph_evidence/runners/`;
- `experiments/dpf_monograph_evidence/reports/`;
- implementation-evidence result plan files and reset memo continuity.

## Prerequisites

- IE2 harness ready;
- IE3 completed or blocked with a reason that does not affect resampling
  fixtures.

## Tasks

1. Build a two-particle soft-resampling fixture with nonlinear test functions.
2. Check exact relaxed-target expectation arithmetic for affine summaries and
   record deltas to the categorical reference for affine and nonlinear tests.
3. Build a small positive-cost Sinkhorn problem with known marginals.
4. Record row/column residuals, regularization epsilon, stabilization mode,
   iteration budget, and residual trend under budget changes.
5. Preserve the distinction between relaxed target evidence and categorical
   resampling evidence.
6. Define trusted comparator roles:
- exact relaxed-target arithmetic for soft-resampling expectations;
- manual arithmetic comparator for categorical-reference deltas;
   - constrained marginal residual checks for Sinkhorn;
   - explanatory-only stabilized objective trends unless paired with residual
     thresholds.
7. Define repair triggers for residual thresholds, epsilon sensitivity, and
   finite-solver failure.

## IE2 Schema Row Contract

IE5 must produce one schema-valid JSON result file per canonical diagnostic ID.
The IE2 validator accepts one top-level result object per JSON file, so IE5 must
not write a multi-row JSON array as the primary validation target.

| Diagnostic ID | Diagnostic role | Comparator ID | Required status on pass | Source support |
| --- | --- | --- | --- | --- |
| `soft_resampling_bias` | `promotion_criterion` | `closed_form_two_particle_soft_resampling_reference` | `pass` | `bibliography_spine_only` |
| `sinkhorn_residual` | `promotion_criterion` | `manual_balanced_sinkhorn_marginal_reference` | `pass` | `bibliography_spine_only` |

Both rows must set `source_support_class` and
`row_level_source_support_class` to `bibliography_spine_only`.  The local
fixtures are clean-room arithmetic, but the diagnostics remain tied to Chapter
26 relaxed-resampling and Sinkhorn/EOT discussion without an IE1 reviewed-source
upgrade.

## Coverage Semantics

Each IE5 JSON row file must include a full canonical `coverage` object.  On a
successful IE5 run, the row files must carry forward these prior states:

- `linear_gaussian_recovery=passed`;
- `synthetic_affine_flow=passed`;
- `pfpf_algebra_parity=passed`.

The two IE5-owned diagnostics must be marked `passed` only if their own row
passes.  Later-phase diagnostics must remain `missing`:

- `learned_map_residual=missing`;
- `hmc_value_gradient=missing`;
- `posterior_sensitivity_summary=missing`.

Any asymmetry between the two IE5 rows must be explained in the Markdown result.

## Soft-Resampling Row Schema

The `soft_resampling_bias` row must use a deterministic two-particle fixture.
For nontrivial fixed-support mixture relaxation toward uniform, affine
expectations generally do not equal the original categorical-reference
expectations.  Therefore IE5 must not use categorical-law affine preservation as
a promotion criterion.  It must instead test exact relaxed-target arithmetic and
record categorical-reference deltas as explicit caveated evidence.

The JSON `tolerance` object must include these keys, each with shape
`{threshold, observed, finite}`:

- `relaxed_probability_formula_abs`;
- `relaxed_constant_expectation_abs`;
- `relaxed_identity_expectation_abs`;
- `relaxed_linear_summary_abs`;
- `relaxed_nonlinear_expectation_abs`;
- `categorical_identity_delta_abs`;
- `categorical_linear_summary_delta_abs`;
- `categorical_nonlinear_delta_abs`;
- `categorical_nonlinear_delta_sign_expected`;
- `probability_sum_abs`;

The row must also record in `finite_checks`:

- normalized base weights;
- relaxed mixture parameter;
- relaxed probabilities;
- categorical-reference probabilities;
- affine test-function values, closed-form relaxed expectations, and
  categorical-reference expectations;
- nonlinear test-function values, relaxed expectation, categorical
  expectation, delta value, and delta sign;
- `finite_summary`.

The `shape_checks` object must record `particle_count=2`, one-dimensional
particle values, number of test functions, and deterministic fixture id.

Promotion passes only if relaxed-probability formula residuals, closed-form
relaxed expectation residuals, and probability-sum residuals are finite and
below tolerance, while categorical-reference deltas are finite and the nonlinear
delta has the predeclared expected sign.  Categorical-reference affine deltas
are recorded as caveated evidence of law change, not promotion failures.  The
nonlinear delta is evidence that relaxed resampling can change nonlinear
observables; it must not be treated as an unbiasedness pass.

## Sinkhorn Row Schema

The `sinkhorn_residual` row must use a deterministic small positive-cost
balanced transport fixture.  The JSON `tolerance` object must include these
keys, each with shape `{threshold, observed, finite}`:

- `row_marginal_abs_max`;
- `column_marginal_abs_max`;
- `total_mass_abs`;
- `nonnegative_plan_violation_abs`;
- `finite_plan_violation_abs`;
- `budget_residual_nonincrease_violations`;

The row must also record in `finite_checks`:

- epsilon;
- stabilization mode;
- budget ladder;
- row and column marginals;
- cost matrix;
- final transport plan;
- per-budget row residual max;
- per-budget column residual max;
- per-budget total residual max;
- whether residuals are nonincreasing over the declared budget ladder;
- `finite_summary`.

The `shape_checks` object must record source/target support sizes, cost-matrix
shape, marginal-vector shapes, final plan shape, and number of budget points.

Promotion passes only if the final row/column marginal residuals, mass residual,
nonnegativity violation, and finite-plan violation are finite and below
predeclared thresholds.  Residual decrease over the budget ladder is
explanatory-only unless the final marginal threshold passes.

## Deterministic Tolerance Contract

Soft-resampling thresholds:

| Check | Threshold |
| --- | ---: |
| `relaxed_probability_formula_abs` | `1e-12` |
| `relaxed_constant_expectation_abs` | `1e-12` |
| `relaxed_identity_expectation_abs` | `1e-12` |
| `relaxed_linear_summary_abs` | `1e-12` |
| `relaxed_nonlinear_expectation_abs` | `1e-12` |
| `categorical_identity_delta_abs` | finite descriptive value, no zero threshold |
| `categorical_linear_summary_delta_abs` | finite descriptive value, no zero threshold |
| `categorical_nonlinear_delta_abs` | finite descriptive value, nonzero expected |
| `categorical_nonlinear_delta_sign_expected` | expected sign `nonzero` |
| `probability_sum_abs` | `1e-12` |

Sinkhorn thresholds:

| Check | Threshold |
| --- | ---: |
| `row_marginal_abs_max` | `1e-9` |
| `column_marginal_abs_max` | `1e-9` |
| `total_mass_abs` | `1e-12` |
| `nonnegative_plan_violation_abs` | `0.0` |
| `finite_plan_violation_abs` | `0.0` |
| `budget_residual_nonincrease_violations` | `0` violations expected |

Sinkhorn epsilon and budget contract:

- epsilon: `0.3`;
- stabilization mode: `plain_log_domain_scaling`;
- iteration budget ladder: `[5, 20, 100]`;
- primary promotion threshold is evaluated at the final budget `100`;
- residual trend over the ladder is recorded and treated as explanatory unless
  a nonincrease violation occurs, in which case the row must set a repair
  trigger.

## Seed And Uncertainty Policy

Both IE5 rows are deterministic:

- `seed_policy=deterministic_no_rng_resampling_sinkhorn_fixture`;
- `replication_count=1`;
- `uncertainty_status=not_applicable`;
- `mcse_or_interval` must state that uncertainty intervals are not applicable
  because no stochastic sampling or Monte Carlo replication is performed.

If a future executor wants randomized resampling draws, IE5 must stop and write
a new phase plan; randomized draws are not authorized here.

## Required Run Manifest

The IE5 runner must record the full IE2 run-manifest keys.  It must set
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

## Required Non-Implication Text

For `soft_resampling_bias`:

```text
IE5 soft-resampling diagnostics validate only deterministic two-particle
relaxed-target expectation arithmetic for selected affine and nonlinear test
functions. They do not validate categorical resampling law preservation,
unbiasedness for nonlinear observables, posterior equivalence, production
bayesfilter code, banking use, model-risk use, or production readiness.
```

For `sinkhorn_residual`:

```text
IE5 Sinkhorn diagnostics validate only finite-budget marginal residuals for a
small deterministic regularized transport fixture. They do not validate exact
unregularized OT equivalence, exact EOT equivalence at finite epsilon/iteration
budget, posterior equivalence, production bayesfilter code, banking use,
model-risk use, or production readiness.
```

## Per-Diagnostic Decision Rules

For `soft_resampling_bias`:

- `promotion_criterion_status=pass` only if relaxed-probability formula,
  relaxed expectation, and probability residuals pass tolerance, and
  categorical-reference deltas are finite with the nonlinear delta nonzero in
  the expected direction.
- `promotion_veto_status=fail` if comparator id, relaxed-target caveat,
  source-support class, tolerance object, seed policy, or exact row-specific
  non-implication text is missing.
- `continuation_veto_status=fail` if the IE2 schema cannot represent affine
  relaxed-target pass criteria and categorical-reference-delta caveats in the
  same row.
- `repair_trigger` must name probability formula, probability normalization,
  relaxed expectation, categorical delta finiteness, nonlinear delta sign,
  finite-value failure, or schema failure.
- explanatory-only diagnostics include any statement about categorical
  resampling, posterior quality, or exact method equivalence.

For `sinkhorn_residual`:

- `promotion_criterion_status=pass` only if final marginal, mass,
  nonnegativity, finite-plan, and budget-ladder checks pass tolerance.
- `promotion_veto_status=fail` if epsilon, budget ladder, stabilization mode,
  comparator id, source-support class, tolerance object, or exact row-specific
  non-implication text is missing.
- `continuation_veto_status=fail` if no trusted marginal comparator exists for
  the deterministic fixture.
- `repair_trigger` must name row marginal, column marginal, mass,
  nonnegativity, finite-plan, epsilon sensitivity, budget trend, or schema
  failure.
- explanatory-only diagnostics include residual trend before final-threshold
  success and all posterior-equivalence commentary.

## Artifact Mapping

IE5 must produce one shared Markdown report plus two row-level JSON files:

- `experiments/dpf_monograph_evidence/reports/resampling-sinkhorn-result.md`;
- `experiments/dpf_monograph_evidence/reports/outputs/soft_resampling_bias.json`;
- `experiments/dpf_monograph_evidence/reports/outputs/sinkhorn_residual.json`;
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie5-resampling-sinkhorn-result-{YYYY-MM-DD}.md`.

Each JSON row's `artifact_paths` and `run_manifest.artifact_paths` must include
its own JSON file and the shared Markdown report.

## Outcome / Exit Mapping

IE5 result notes must record both a master-program label and a local label.
Normal master-program labels are:

- `ie_phase_passed`;
- `ie_phase_passed_with_caveats`;
- `ie_phase_blocked`.

IE5 also inherits the full master-program label set for exceptional outcomes:

- `ie_phase_deferred_with_recorded_reason`;
- `ie_phase_rejected_for_lane_drift`.

Allowed local labels:

- `ie5_resampling_sinkhorn_passed`;
- `ie5_resampling_sinkhorn_passed_with_caveats`;
- `ie5_resampling_sinkhorn_blocked`.

## Primary Criterion

Soft-resampling and Sinkhorn diagnostics produce finite, tolerance-classified
records with explicit relaxed-target caveats.

## Veto Diagnostics

- soft-resampling is described as unbiased for nonlinear tests;
- Sinkhorn residual success is described as posterior equivalence;
- epsilon, iteration budget, or residual tolerance is missing;
- finite-solver failure is treated as method-level impossibility.
- Sinkhorn residual decrease is used as promotion evidence without meeting the
  predeclared marginal residual threshold.

## Outcome Classification

- Promotion/pass criterion: soft-resampling expectations and Sinkhorn marginal
  residuals meet predeclared comparator/tolerance contracts.
- Promotion veto: epsilon, iteration budget, residual threshold, or relaxed-
  target caveat is missing.
- Continuation veto: no trusted marginal comparator exists for the Sinkhorn
  fixture.
- Repair trigger: nonfinite values, residual threshold miss, unstable epsilon
  behavior, or unexpected soft-bias sign.
- Explanatory-only diagnostics: residual trend without target-equivalence
  evidence and stabilized objective trends.

## Expected Artifacts

- `experiments/dpf_monograph_evidence/reports/resampling-sinkhorn-result.md`;
- `experiments/dpf_monograph_evidence/reports/outputs/soft_resampling_bias.json`;
- `experiments/dpf_monograph_evidence/reports/outputs/sinkhorn_residual.json`;
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie5-resampling-sinkhorn-result-{YYYY-MM-DD}.md`.

## Exit Labels

- `ie5_resampling_sinkhorn_passed`;
- `ie5_resampling_sinkhorn_passed_with_caveats`;
- `ie5_resampling_sinkhorn_blocked`.
