# Phase IE6 plan: learned-OT teacher/student/OOD residual tests

## Date

2026-05-16

## Purpose

Evaluate learned or amortized OT only as a surrogate-map problem: a student map
must be compared to a declared teacher map, training envelope, and out-of-
distribution regime.  This phase does not validate posterior correctness.

## Allowed Write Set

- `experiments/dpf_monograph_evidence/fixtures/`;
- `experiments/dpf_monograph_evidence/diagnostics/`;
- `experiments/dpf_monograph_evidence/runners/`;
- `experiments/dpf_monograph_evidence/reports/`;
- implementation-evidence result plan files and reset memo continuity.

## Prerequisites

- IE2 harness ready;
- IE5 completed or blocked with a reason that leaves learned-map residual tests
  meaningful.
- pre-existing approved teacher/student artifact with provenance.  If none
  exists, IE6 must exit `ie6_learned_ot_residual_deferred_no_artifact`.

IE6 does not authorize substituting a newly invented analytic teacher/student
fixture for the required pre-existing approved artifact.  If no artifact is
present with provenance, IE6 must emit a structured deferred
`learned_map_residual` row and must not execute residual tests.

## Tasks

1. Define a small deterministic teacher map or teacher-output fixture.
2. Define student residual metrics: map residual, marginal residual,
   permutation residual, and scalar-summary residual.
3. Tag all records as in-distribution, edge-of-distribution, or OOD.
4. Run only bounded smoke-level inference or deterministic residual checks; do
   not train networks unless a later plan authorizes it.
5. Record whether residuals implicate the student approximation, teacher
   quality, or distribution shift.
6. Record teacher-origin class:
   - `analytic`;
   - `exact_discrete_solver`;
   - `finite_sinkhorn_teacher`;
   - `imported_prior_artifact`;
   - `unknown_or_missing`.
   Conclusions about teacher adequacy are forbidden unless the teacher-origin
   class and IE1 source status support them.

## Approved-Artifact Gate

Before residual execution, IE6 must identify:

- teacher artifact path or id;
- approval source and approval date;
- teacher variant: `unregularized_ot`, `entropic_ot`,
  `finite_sinkhorn_teacher`, `analytic_map`, or `unknown`;
- teacher output object: `coupling`, `barycentric_map`, `scalar_summary`, or
  `unknown`;
- student checkpoint id;
- training run id;
- training commit;
- training data envelope;
- artifact mode, which must be `preexisting_inference_only`;
- proof that IE6 performs no optimizer steps and does not mutate checkpoint
  files.

If any field is missing, the phase must defer.  Missing artifact provenance is
not a method failure and must not be reported as evidence against learned OT.

## IE2 Schema Row Contract

IE6 owns exactly one canonical diagnostic id: `learned_map_residual`.  IE6 must
write one schema-valid top-level JSON object, not an array:

- `experiments/dpf_monograph_evidence/reports/outputs/learned_ot_residual.json`.

If an approved artifact exists, the row must include a machine-readable
`tolerance` object with `{threshold, observed, finite}` entries for:

- `map_residual_abs_max`;
- `marginal_residual_abs_max`;
- `permutation_residual_abs_max`;
- `scalar_summary_residual_abs`;
- `ood_residual_abs_max`;
- `artifact_provenance_missing_flag`;
- `checkpoint_mutation_flag`;
- `optimizer_step_count`.

The `finite_checks` object must record teacher variant, teacher output object,
teacher origin class, artifact ids, training provenance ids, envelope tags,
residual table, artifact mode, and no-training proof.

The `shape_checks` object must record map input/output shape, number of test
points, support size, residual vector shape, and number of envelope slices.

If no approved artifact exists, the deferred row must use:

- `status=deferred`;
- `coverage.learned_map_residual=deferred`;
- `blocker_class=source`;
- `diagnostic_role=continuation_veto`;
- `source_support_class=row_level_source_support_class=bibliography_spine_only`;
- `comparator_id=no_approved_teacher_student_artifact`;
- `tolerance={"deferred_no_artifact": {"threshold": 0.0, "observed": 0.0, "finite": true}}`;
- non-empty `finite_checks`, `shape_checks`, and `mcse_or_interval` objects
  explaining that no residual was executed.

## OOD And Envelope Tags

Executed IE6 rows must classify every residual slice with:

- `envelope_tag`: `in_distribution`, `edge_of_distribution`, or `ood`;
- `shift_axis`: one of `particle_count_shift`, `dimension_shift`,
  `weight_degeneracy_shift`, `epsilon_shift`, `model_regime_shift`, or
  `none`;
- `training_envelope_id`;
- `evaluation_envelope_id`;
- reason for the tag.

For a deferred row, these fields must be present as deferred/not-applicable
metadata rather than omitted.

## Coverage Semantics

Every IE6 row must include the full canonical coverage object.  It must carry
forward:

- `linear_gaussian_recovery=passed`;
- `synthetic_affine_flow=passed`;
- `pfpf_algebra_parity=passed`;
- `soft_resampling_bias=passed`;
- `sinkhorn_residual=passed`.

If no approved artifact exists, set:

- `learned_map_residual=deferred`;
- `hmc_value_gradient=missing`;
- `posterior_sensitivity_summary=missing`.

If an approved artifact exists and the residual test executes, set
`learned_map_residual=passed` or `blocked` according to the row result, with
later diagnostics still `missing`.

## Required Run Manifest

The IE6 runner must record the full IE2 run-manifest keys.  It must set
`CUDA_VISIBLE_DEVICES=-1` before importing NumPy or any scientific dependency
and must record:

- `cpu_only=true`;
- `cuda_visible_devices="-1"`;
- `gpu_devices_visible=[]`;
- `gpu_hidden_before_import=true`;
- `pre_import_cuda_visible_devices="-1"`;
- `pre_import_gpu_hiding_assertion=true`;
- branch, commit, dirty-state summary, Python version, package versions,
  command, wall-clock cap, started/ended UTC timestamps, and artifact paths.

## Required Non-Implication Text

For a deferred row:

```text
IE6 learned-OT diagnostics were deferred because no approved pre-existing
teacher/student artifact with provenance was available. This deferral does not
validate or invalidate learned OT, neural OT, surrogate-map quality, posterior
quality, production bayesfilter code, banking use, model-risk use, or
production readiness.
```

For an executed row:

```text
IE6 learned-OT diagnostics validate only declared teacher/student surrogate-map
residuals on the recorded envelope slices and provenance-bearing artifacts.
They do not validate teacher adequacy, posterior equivalence, production
bayesfilter code, banking use, model-risk use, or production readiness.
```

## Per-Diagnostic Decision Rules

For `learned_map_residual`:

- `promotion_criterion_status=pass` only when an approved artifact exists and
  map, marginal, permutation, scalar-summary, provenance, checkpoint-mutation,
  and no-training checks pass.
- `promotion_veto_status=fail` if teacher variant, teacher output object,
  artifact provenance, checkpoint id, training envelope, OOD tags, source
  support, tolerance object, seed/no-training policy, or exact non-implication
  text is missing.
- `continuation_veto_status=fail` if no approved artifact exists and the phase
  attempts residual execution anyway.
- `repair_trigger` must name missing artifact provenance, map residual,
  marginal residual, permutation residual, scalar-summary residual, OOD
  residual, checkpoint mutation, optimizer-step execution, or schema failure.
- explanatory-only diagnostics include OOD residual behavior and all
  teacher-quality or posterior-quality commentary.

## Outcome / Exit Mapping

IE6 result notes must record both a master-program label and a local label.
Master-program labels:

- `ie_phase_passed`;
- `ie_phase_passed_with_caveats`;
- `ie_phase_deferred_with_recorded_reason`;
- `ie_phase_blocked`;
- `ie_phase_rejected_for_lane_drift`.

Allowed local labels:

- `ie6_learned_ot_residual_passed`;
- `ie6_learned_ot_residual_passed_with_caveats`;
- `ie6_learned_ot_residual_deferred_no_artifact`;
- `ie6_learned_ot_residual_blocked`.

Expected local label in the current repo state is
`ie6_learned_ot_residual_deferred_no_artifact` unless a provenance-bearing
artifact is found before execution.

## Primary Criterion

Learned-OT residual records are usable as surrogate-quality evidence without
being misread as posterior validation.

## Veto Diagnostics

- teacher and student maps are conflated;
- training or checkpoint provenance is absent;
- OOD tags are missing;
- neural success is treated as clean-room method validation;
- phase attempts new training or broad dependency work.
- missing artifact provenance is treated as a failed method rather than a
  deferred phase.

## Outcome Classification

- Promotion/pass criterion: approved artifact provenance exists and residuals
  meet declared in-envelope thresholds.
- Promotion veto: teacher-origin class, artifact provenance, OOD tag, or
  source-support class is missing.
- Continuation veto: no approved teacher/student artifact exists.
- Repair trigger: residual threshold miss, permutation failure, nonfinite
  output, or provenance inconsistency.
- Explanatory-only diagnostics: OOD residual behavior and teacher-quality
  commentary without reviewed teacher specification.

## Expected Artifacts

- `experiments/dpf_monograph_evidence/reports/learned-ot-residual-result.md`;
- `experiments/dpf_monograph_evidence/reports/outputs/learned_ot_residual.json`;
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie6-learned-ot-result-{YYYY-MM-DD}.md`.

## Exit Labels

- `ie6_learned_ot_residual_passed`;
- `ie6_learned_ot_residual_passed_with_caveats`;
- `ie6_learned_ot_residual_deferred_no_artifact`;
- `ie6_learned_ot_residual_blocked`.
