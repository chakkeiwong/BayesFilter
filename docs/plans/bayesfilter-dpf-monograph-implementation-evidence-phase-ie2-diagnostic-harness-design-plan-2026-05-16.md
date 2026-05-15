# Phase IE2 plan: Chapter 26 diagnostic harness design

## Date

2026-05-16

## Purpose

Design the non-production diagnostic harness that all executable evidence phases
must use.  IE2 converts Chapter 26 tables into concrete schemas, fixtures,
runner conventions, result records, and validation commands.

## Allowed Write Set

- `experiments/dpf_monograph_evidence/`;
- implementation-evidence plan/result files;
- reviewer-grade reset memo continuity section.

Production `bayesfilter/` remains read-only.

## Inputs

- Chapter 26 debugging crosswalk;
- reviewer-grade final readiness report;
- IE0 preflight result;
- IE1 source-status register or source-review deferral record.

## Tasks

1. Create the evidence root skeleton with README, fixture, diagnostic, runner,
   and report directories.
2. Define a result schema with fields for diagnostic id, chapter label, source
   support class, seed policy, status, tolerance, finite checks, shape checks,
   runtime, blocker class, and non-implication.
   The schema must also include:
   - comparator or baseline id;
   - diagnostic role classification;
   - promotion criterion status;
   - promotion veto status;
   - continuation veto status;
   - repair trigger;
   - explanatory-only diagnostics;
   - environment, Python/package versions, commit, branch, and dirty-state
     summary;
   - CPU/GPU mode, including whether GPU was intentionally hidden;
   - command, wall time, wall-clock cap, and artifact paths;
   - uncertainty status, replication count, and MCSE or interval fields when
     stochastic outputs are reported;
   - source-support class copied from IE1 or marked `not_source_dependent`;
   - post-run red-team note.
3. Define fixture contracts for:
   - linear-Gaussian recovery;
   - synthetic affine flow;
   - two-particle soft-resampling bias;
   - small Sinkhorn problem;
   - learned-map teacher/student residual;
   - fixed scalar HMC target.
4. Define bounded runner commands and `--validate-only` behavior.  Validation
   must check:
   - required schema fields;
   - allowed enum values;
   - artifact-path containment under `experiments/dpf_monograph_evidence/`;
   - non-implication text presence;
   - source-support propagation from IE1;
   - CPU-only/GPU-policy fields;
   - boundedness cap fields;
   - run-manifest presence;
   - import-boundary failures.
5. Define a canonical run manifest embedded in every JSON result and validated
   by the harness.  Required keys:
   - `command`;
   - `branch`;
   - `commit`;
   - `dirty_state_summary`;
   - `python_version`;
   - `package_versions`;
   - `cpu_only`;
   - `cuda_visible_devices`;
   - `gpu_devices_visible`;
   - `gpu_hidden_before_import`;
   - `pre_import_cuda_visible_devices`;
   - `pre_import_gpu_hiding_assertion`;
   - `seed_policy`;
   - `wall_clock_cap_seconds`;
   - `started_at_utc`;
   - `ended_at_utc`;
   - `artifact_paths`.
6. Use one unambiguous import policy:
   - no imports from production `bayesfilter/`;
   - no imports from student-baseline modules;
   - no mutation of production, student-baseline, or controlled student-baseline
     trees;
   - validator and import-boundary scans must fail on violation.
7. Define boundedness defaults and canonical cap fields:
   - CPU-only is mandatory for all runners unless a later accepted transition
     plan overrides it;
   - runners must set `CUDA_VISIBLE_DEVICES=-1` before scientific imports;
     `gpu_hidden_before_import` must be true in the manifest.  A phase that
     cannot prove this must stop before running numerical diagnostics.
   - canonical cap keys:
     `max_particles`, `max_time_steps`, `max_sinkhorn_iterations`,
     `max_finite_difference_evaluations`, `max_replications`,
     `max_wall_clock_seconds`;
   - cap-key schema rule:
     `max_wall_clock_seconds` is required globally.  The other cap keys are
     required when applicable and otherwise must be recorded as JSON null plus a
     `cap_non_applicability_reasons` object explaining why the cap is not
     applicable.  String encodings such as `N/A` are forbidden.
   - no hidden retries;
   - no cached result reuse unless the cache key, source command, and commit are
     recorded;
   - fixed seeds plus rerun count for stochastic diagnostics;
   - explicit compiled/eager parity fields when compiled execution is used.
8. Define a diagnostic-source mapping rule:
   - every diagnostic row must identify its source family from the IE1 register
     or `not_source_dependent`;
   - a row may carry multiple source families only if it also records the
     weakest source-support class as the row-level class;
   - source-support weakness order from weakest to strongest is:
     `source_gap` < `bibliography_spine_only` < `reviewed_local_summary` <
     `local_derivation_only` < `not_source_dependent`;
   - mixed rows involving source-family provenance plus local fixture or schema
     reasoning must use the weakest applicable row-level class.  For example,
     a locally derived fixture tied to a bibliography-only source family remains
     `bibliography_spine_only`, not `local_derivation_only`.
   - validators must reject upgrades from `bibliography_spine_only` or
     `source_gap` to `reviewed_local_summary` unless an IE1 successor artifact
     records reviewed local summaries.
9. Define canonical diagnostic ids tied to Chapter 26 and IE0:
   - `linear_gaussian_recovery`;
   - `synthetic_affine_flow`;
   - `pfpf_algebra_parity`;
   - `soft_resampling_bias`;
   - `sinkhorn_residual`;
   - `learned_map_residual`;
   - `hmc_value_gradient`;
   - `posterior_sensitivity_summary`.
   Validators must enforce these as an enum, reject unknown row-level
   diagnostic ids, and report which required ids are missing, blocked, deferred,
   or passed so IE8 aggregation is stable.
   The harness must also validate a machine-readable coverage object keyed by
   every canonical diagnostic id.  Allowed coverage status values are:
   `missing`, `blocked`, `deferred`, and `passed`.
10. Bind phase ownership of diagnostic ids:
   - IE3 may emit `linear_gaussian_recovery`;
   - IE4 may emit `synthetic_affine_flow` and `pfpf_algebra_parity`;
   - IE5 may emit `soft_resampling_bias` and `sinkhorn_residual`;
   - IE6 may emit `learned_map_residual`;
   - IE7 may emit `hmc_value_gradient`;
   - IE8 may emit `posterior_sensitivity_summary` and aggregate coverage only.
   IE8 must not backfill earlier phase ids ad hoc.

## Primary Criterion

IE3--IE7 can implement diagnostics against one shared schema without inventing
new result semantics.

## Veto Diagnostics

- schema lacks source-support, seed, tolerance, blocker, or non-implication
  fields;
- schema lacks comparator id, promotion/continuation/repair statuses,
  environment, command, artifact, CPU/GPU, or uncertainty fields;
- fixtures are too broad or tied to production internals;
- runner commands are unbounded;
- validation cannot detect missing required diagnostics.
- validation cannot detect production or student-lane imports.
- run manifest keys are not canonical or validator-enforced;
- CPU-only/GPU-hiding policy is omitted or optional;
- source-support mapping can silently upgrade IE1 source classes;
- diagnostic ids are not stable enough for IE8 aggregation.
- cap keys are not canonical or validator-enforced.
- diagnostic coverage status lacks the canonical machine-readable object;
- phase ownership of diagnostic ids is ambiguous.

## Expected Artifacts

- `experiments/dpf_monograph_evidence/README.md`;
- `experiments/dpf_monograph_evidence/results.py` or equivalent schema module;
- `experiments/dpf_monograph_evidence/runners/validate_results.py`;
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie2-diagnostic-harness-design-result-{YYYY-MM-DD}.md`.

## Exit Labels

- `ie2_harness_ready`;
- `ie2_harness_ready_with_caveats`;
- `ie2_harness_blocked`.
