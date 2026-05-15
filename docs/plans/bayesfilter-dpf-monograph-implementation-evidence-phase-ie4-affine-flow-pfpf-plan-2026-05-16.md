# Phase IE4 plan: affine-flow PF-PF density and log-det tests

## Date

2026-05-16

## Purpose

Test PF-PF proposal correction algebra on synthetic affine flows where the
proposal density, inverse map, Jacobian determinant, and corrected importance
weights are known analytically.

## Allowed Write Set

- `experiments/dpf_monograph_evidence/fixtures/`;
- `experiments/dpf_monograph_evidence/diagnostics/`;
- `experiments/dpf_monograph_evidence/runners/`;
- `experiments/dpf_monograph_evidence/reports/`;
- implementation-evidence result plan files and reset memo continuity.

## Prerequisites

- IE2 harness ready;
- IE3 completed or blocked with a reason that does not invalidate affine-flow
  fixtures.

## Tasks

1. Define one-dimensional and low-dimensional invertible affine maps.
2. Compute analytic density ratios and log determinants.
3. Compare implemented forward/inverse density and log-weight calculations to
   analytic references.
4. Include sign-convention checks for log-det accumulation.
5. Record what each failure implicates: map inversion, determinant sign,
   proposal density, or normalization.
6. Record mandatory non-implication text: affine PF-PF checks do not validate
   nonlinear flow integration, solver stability, filtering correctness, or
   posterior quality.

## IE2 Schema Row Contract

IE4 must produce one schema-valid JSON result file per canonical diagnostic ID.
The current IE2 validator accepts one top-level result object per JSON file, so
IE4 must not write a multi-row JSON array as the primary validation target.

| Diagnostic ID | Diagnostic role | Comparator ID | Required status on pass | Source support |
| --- | --- | --- | --- | --- |
| `synthetic_affine_flow` | `promotion_criterion` | `analytic_affine_pushforward_density_reference` | `pass` | `bibliography_spine_only` |
| `pfpf_algebra_parity` | `promotion_criterion` | `closed_form_pfpf_log_weight_reference` | `pass` | `bibliography_spine_only` |

Both rows must set `source_support_class` and
`row_level_source_support_class` to `bibliography_spine_only`.  The local
fixture algebra is clean-room and deterministic, but the phase is still tied to
Chapter 26 PF-PF proposal-correction diagnostics and IE1 did not upgrade source
provenance beyond bibliography-spine support.  Therefore IE4 must use the
weakest applicable source-support class rather than treating local derivation as
a source-provenance upgrade.  The Markdown report must state this explicitly.

Each JSON file must contain exactly one top-level IE2 result object.  The
`coverage` object must include all canonical diagnostic IDs and must be
interpreted as the within-program known coverage state at the time IE4 emits
the file.  Therefore successful IE4 row files must mark IE3
`linear_gaussian_recovery` as `passed`, both IE4 IDs as `passed`, and all
later-phase diagnostic IDs as `missing`.  Failed or blocked IE4 row files must
mark IE3 as `passed`, the affected IE4 diagnostic as `blocked`, the unaffected
IE4 diagnostic as `passed` only if its own row passed, and all later-phase
diagnostic IDs as `missing`.  The Markdown result must explain any asymmetry
between the two IE4 row files.

The `synthetic_affine_flow` row must cover fixture integrity:

- affine map id, dimension, matrix/scale, offset, determinant sign, and inverse;
- forward reconstruction residual and inverse reconstruction residual;
- analytic base log-density, pushforward log-density, and log-determinant;
- absolute residual between implemented and analytic density/log-det values.

The `pfpf_algebra_parity` row must cover proposal-correction algebra:

- prior, proposal, and target-density log values for the same particles;
- explicit sign convention for `log q(x_t | x_{t-1}, y_t)` under the affine
  inverse map;
- corrected log-weight formula with both unnormalized and normalized parity;
- normalization discrepancy recorded before any normalization is used.

## Deterministic Tolerance Contract

All IE4 promotion checks are deterministic absolute-tolerance checks in
float64 NumPy arithmetic:

| Check | Threshold |
| --- | ---: |
| Forward reconstruction max absolute residual | `1e-12` |
| Inverse reconstruction max absolute residual | `1e-12` |
| Log-determinant residual | `1e-12` |
| Proposal or pushforward log-density residual | `1e-12` |
| Corrected unnormalized log-weight residual | `1e-12` |
| Normalized weight residual after log-sum-exp normalization | `1e-12` |
| Probability-sum residual before reporting normalized weights | `1e-12` |

If any residual is non-finite or exceeds tolerance, the affected row must exit
as `fail` or `blocked` with `repair_trigger` populated.  A failed algebra check
is a repair trigger unless it invalidates the analytic comparator or IE2 schema
itself; comparator/schema invalidation is a continuation veto.

The JSON `tolerance` object must be machine-readable.  Each residual-family key
below must map to an object with exactly this shape:

```json
{
  "threshold": 1e-12,
  "observed": 0.0,
  "finite": true
}
```

The required residual-family keys are:

- `forward_reconstruction_abs_max`;
- `inverse_reconstruction_abs_max`;
- `log_det_abs_max`;
- `pushforward_log_density_abs_max`;
- `proposal_log_density_abs_max`;
- `corrected_log_weight_abs_max`;
- `normalized_weight_abs_max`;
- `probability_sum_abs_max`.

The JSON `finite_checks` object must include a `residuals` object with the same
keys and the same `{threshold, observed, finite}` shape, plus a
`finite_summary` boolean.  The Markdown report must quote the maximum residuals
and thresholds, but Markdown text alone is not adequate evidence for IE4
pass/fail.

## Per-Diagnostic Decision Rules

For `synthetic_affine_flow`:

- `promotion_criterion_status=pass` only if the affine fixture is invertible,
  determinant sign is explicit, forward/inverse residuals meet tolerance, and
  analytic pushforward density/log-det parity meets tolerance.
- `promotion_veto_status=fail` if determinant sign, comparator id,
  source-support class, tolerance, finite checks, or non-implication text is
  missing, or if the row's `non_implication` field does not exactly match the
  required `synthetic_affine_flow` text in this plan.
- `continuation_veto_status=fail` if the IE2 schema cannot represent
  affine-only non-implication, analytic comparator identity, or row-level
  source support.
- `repair_trigger` must name map inversion, determinant sign, base density, or
  pushforward-density mismatch when any residual fails.
- explanatory-only diagnostics include all nonlinear-flow commentary.

For `pfpf_algebra_parity`:

- `promotion_criterion_status=pass` only if corrected log-weights and
  normalized weights match the closed-form reference within tolerance before
  any normalization hides an unnormalized discrepancy.
- `promotion_veto_status=fail` if the proposal-density sign convention,
  determinant contribution, comparator id, source-support class, tolerance, or
  non-implication text is missing, or if the row's `non_implication` field does
  not exactly match the required `pfpf_algebra_parity` text in this plan.
- `continuation_veto_status=fail` if the harness cannot record separate
  unnormalized and normalized parity residuals.
- `repair_trigger` must name proposal density, log-det sign, target density,
  log-sum-exp normalization, or finite-value failure when any residual fails.
- explanatory-only diagnostics include any filtering, posterior-quality, or
  nonlinear-flow interpretation.

## Required Run Manifest

The IE4 runner must record the full IE2 run-manifest keys.  It must set
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

The phase is deterministic, so `seed_policy` must be
`deterministic_no_rng_affine_fixture`, `replication_count` must be `1`, and
`uncertainty_status` must be `not_applicable`.

## Required Non-Implication Text

Each JSON row and Markdown result must include a row-specific version of this
statement:

For `synthetic_affine_flow`:

```text
IE4 synthetic affine-flow checks validate only closed-form affine
pushforward-density, inverse-map, and Jacobian-sign parity on deterministic
clean-room fixtures. They do not validate nonlinear flow integration, solver
stability, PF-PF filtering correctness, production bayesfilter code, real
DPF-HMC targets, posterior quality, banking use, model-risk use, or production
readiness.
```

For `pfpf_algebra_parity`:

```text
IE4 PF-PF algebra parity checks validate only closed-form proposal-density,
Jacobian-sign, unnormalized corrected-log-weight, and normalized-weight parity
on deterministic affine clean-room fixtures. They do not validate nonlinear
flow integration, solver stability, filtering correctness, production
bayesfilter code, real DPF-HMC targets, posterior quality, banking use,
model-risk use, or production readiness.
```

## Required Result-Note Sections

The IE4 result note must include:

- skeptical audit before execution;
- research-intent ledger;
- evidence contract with row-level comparator ids, tolerances, and
  source-support class;
- pre-mortem/failure-mode map;
- artifact list;
- per-diagnostic decision table;
- inference-status table;
- run manifest;
- post-run red-team note;
- next-phase justification or blocker.

## Primary Criterion

PF-PF density correction and log-det diagnostics pass on affine fixtures or
produce a structured algebraic blocker.

## Veto Diagnostics

- nonlinear flow evidence is claimed from affine-only tests;
- determinant sign conventions are not explicit;
- tolerance is not deterministic;
- failed normalization is hidden by renormalizing without recording it.

## Outcome Classification

- Promotion/pass criterion: affine density, inverse-map, determinant, and
  corrected-weight checks match analytic references within deterministic
  tolerance.
- Promotion veto: determinant sign, proposal-density, or source-support fields
  are missing.
- Continuation veto: the harness cannot record affine-only non-implication or
  analytic comparator identity.
- Repair trigger: any deterministic algebra mismatch, nonfinite value, or
  normalization discrepancy.
- Explanatory-only diagnostics: any nonlinear-flow commentary derived from this
  phase.

## Expected Artifacts

- `experiments/dpf_monograph_evidence/reports/affine-flow-pfpf-result.md`;
- `experiments/dpf_monograph_evidence/reports/outputs/affine_flow_synthetic_affine_flow.json`;
- `experiments/dpf_monograph_evidence/reports/outputs/affine_flow_pfpf_algebra_parity.json`;
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie4-affine-flow-pfpf-result-{YYYY-MM-DD}.md`.

## Exit Labels

IE4 normal execution outcomes must use these master-program phase labels:

- `ie_phase_passed`;
- `ie_phase_passed_with_caveats`;
- `ie_phase_blocked`.

IE4 also inherits the full master-program label set for exceptional outcomes:

- `ie_phase_deferred_with_recorded_reason`;
- `ie_phase_rejected_for_lane_drift`.

The phase result may additionally record the descriptive local label
`ie4_affine_flow_pfpf_passed`, `ie4_affine_flow_pfpf_passed_with_caveats`, or
`ie4_affine_flow_pfpf_blocked`, but the master-program exit label must be one
of the three labels above.
