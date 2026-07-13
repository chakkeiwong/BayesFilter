# Phase 2S Subplan: Geometry Centering And MAP-Local Reference Repair

Date: 2026-07-09
Status: `DRAFT_PENDING_CODEX_SUBSTITUTE_REVIEW`

## Phase Objective

Repair the Phase 2 local-reference mismatch by building a CPU-hidden
MAP-local quadratic geometry diagnostic for the same scalar filtering target.
The phase uses the existing quadratic MAP covariance initializer:

- `bayesfilter/inference/quadratic_map_covariance.py`
- `bayesfilter/inference/quadratic_geometry.py`

The optimizer is a finite-neighborhood locator only.  The covariance authority
is the constrained low-rank SPD quadratic regression and its regularized
precision-to-covariance conversion.

## Entry Conditions

- Phase 1R finite/acceptance screen passed.
- Phase 2 local quadratic reference agreement failed.
- Phase 2R selected exactly one localization outcome:
  `outside_geometry_trust_region`.
- Phase 2R transform bookkeeping checks passed.
- Native divergence remains unavailable and must not be treated as zero.
- Phase 3 GPU/XLA is blocked until a reviewed repair branch creates a valid
  handoff.

## Required Artifacts

- Phase 2S harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py`
- Phase 2S tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair.py`
- Phase 2S JSON/Markdown diagnostic artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2s_geometry_centering_repair.log`
- Phase 2S result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-result-2026-07-09.md`
- Refreshed next subplan.  If Phase 2S passes, draft a MAP-local
  reference-agreement or retuned fixed-kernel HMC screen subplan.  If Phase 2S
  fails, draft a narrower geometry initializer repair subplan or stop with a
  blocker.

## Required Checks, Tests, And Reviews

- Review this subplan before runtime.
- Run focused tests before runtime:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_quadratic_map_covariance.py tests/test_quadratic_geometry.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair.py
```

- Run telemetry regression tests:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_common_inference_runtime_contracts.py::test_hmc_screen_does_not_use_log_accept_threshold_as_native_divergence tests/test_common_inference_runtime_contracts.py::test_hmc_screen_keeps_unavailable_diagnostics_from_passing_as_zero
```

- Run `git diff --check`.
- Create the quiet log directory before redirected runtime:

```bash
mkdir -p docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09
```

- Planned runtime command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2s_geometry_centering_repair.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can a MAP-local SPD quadratic geometry, initialized from the Phase 2R reference mean/center neighborhood, produce a usable covariance/reference handoff for the scalar filtering target? |
| Baseline/comparator | Phase 2/2R truth-free local quadratic geometry and reference artifacts. |
| Primary pass criterion | The quadratic MAP covariance initializer returns `accepted=True`, `status=usable`, the optimizer locator accepts a finite position, finite locator/map-candidate values and scores, SPD regularized precision/covariance with condition number at most `1e5`, finite sample count at least five times the regression parameter count, nonzero holdout with holdout fit accepted, and no hard-veto diagnostics. |
| Veto diagnostics | Invalid input artifacts, nonfinite value/score at initial/locator/map-candidate positions, locator fallback or exception, geometry rejection, zero holdout count, holdout rejection, mass-matrix regularization failure, non-SPD precision/covariance, condition cap failure, insufficient finite samples, missing artifact fields, telemetry semantics mismatch, or unsupported claims. |
| Explanatory diagnostics | Locator movement, locator/log-prob/score changes, surrogate map candidate, center-refinement status, holdout RMSE, score RMSE, precision/covariance eigen summaries, distance from truth-free center, distance from Phase 1R pooled HMC mean, target replay at truth-free center/reference mean/locator/map candidate/pooled HMC mean. |
| Not concluded | No certified global MAP, posterior covariance correctness, HMC readiness/convergence, zero divergences, sampler superiority, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2S JSON, Markdown, quiet log, result file, and refreshed next subplan. |

## Implementation Design

- Load the scalar filtering geometry target from
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py`
  to preserve the exact data-generating path, free-parameter names, scale,
  prior, and value/score route.
- Use Phase 2R and Phase 2 artifacts to define diagnostic starting points:
  truth-free center, local reference mean transformed back to free parameters,
  and Phase 1R pooled HMC mean transformed back to free parameters.
- Run `estimate_quadratic_map_covariance` once, using the local reference mean
  free-parameter point as the initial position unless it is nonfinite.  If it
  is nonfinite, fail closed rather than silently choosing another starting
  point.
- Keep `QuadraticMapCovarianceLocatorConfig.enabled=True`; its result is a
  locator only and its inverse-Hessian output must not be used.
- Treat locator fallback, locator exception, or
  `accepted_optimizer_position=False` as a Phase 2S veto.  A fallback may still
  be useful for later debugging, but it cannot pass this MAP-local repair gate.
- Use `LowRankSPDQuadraticGeometryConfig` with:
  - `rank=4`;
  - `sample_count=90`;
  - `min_samples_per_parameter=5`;
  - `trust_radius=0.60`;
  - `pilot_radius=0.10`;
  - `pilot_direction_count=96`;
  - `eigenvalue_floor=1e-4`;
  - `max_condition_number=1e5`;
  - `holdout_rmse_abs_tolerance=0.10`;
  - `holdout_rmse_rel_tolerance=0.01`;
  - `seed=(20260709, 6201)`.
- Use `QuadraticMapCovarianceMassConfig` with:
  - `jitter=1e-9`;
  - `eigenvalue_floor=1e-4`;
  - `max_condition_number=1e5`;
  - `dense=True`.

These settings are an engineering diagnostic, not a default policy.  With
dimension four and effective low-rank rank three, the regression parameter
count is `9`, so `sample_count=90` is ten times the parameter count.  The
required finite sample floor is `45`; the planned sample count is two times
that floor and, because `holdout_fraction=0.25`, should leave a nonzero
holdout while preserving the five-samples-per-parameter training floor.  A
zero holdout count is a Phase 2S veto.

## Required Decision Fields

The JSON artifact must include:

- `decision.phase2s_geometry_centering_repair_passed`;
- `decision.vetoes`;
- `decision.viable_for_map_local_reference_subplan`;
- `decision.zero_divergence_claim_made`;
- `initializer.accepted`;
- `initializer.status`;
- `initializer.map_candidate_role`;
- `initializer.locator_diagnostics.uses_optimizer_inverse_hessian`;
- geometry diagnostics including finite sample count, required finite samples,
  holdout status, score RMSE, and eigen summaries;
- mass-matrix regularization report;
- target replay diagnostics;
- metric-role classification;
- inference-status table;
- run manifest.

## Forbidden Claims And Actions

- Do not claim a certified global MAP.
- Do not claim posterior covariance correctness.
- Do not use optimizer inverse-Hessian output as covariance evidence.
- Do not run GPU/XLA in this phase.
- Do not change HMC tuning, trajectory length, step size, or default policy.
- Do not claim HMC readiness, convergence, zero divergences, sampler
  superiority, production/default readiness, or Zhao-Cui source faithfulness.

## Exact Next-Phase Handoff Conditions

If Phase 2S passes, draft and review one of the following before any runtime:

- a MAP-local reference-agreement subplan that compares the old Phase 1R draws
  against the new local reference only as a diagnostic; or
- a same-target retuned fixed-kernel HMC screen subplan that explicitly uses
  the new covariance/mass handoff and retunes the trajectory under a separate
  evidence contract.

If Phase 2S fails, write a result that names the specific initializer blocker
and draft a narrower repair only if the blocker is fixable without changing
the scientific question.  Phase 3 GPU/XLA remains blocked in both cases until
the repair branch has a reviewed handoff.

## Stop Conditions

Stop for invalid artifacts, nonfinite target evaluations, geometry or mass
rejection, review nonconvergence, unsupported claims, or any need to change
runtime/model/default-policy boundaries.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is Phase 2/2R failed truth-free local reference, not HMC success. |
| Proxy metrics promoted | Holdout RMSE and score RMSE gate geometry usability only; they do not prove posterior correctness or HMC readiness. |
| Missing stop conditions | Stop conditions include invalid artifacts, nonfinite target values, geometry/mass rejection, and unsupported claims. |
| Unfair comparison | No method ranking occurs.  The old Phase 1R draws are not used to promote the new covariance. |
| Hidden assumptions | Numeric settings are scoped to a four-dimensional scalar diagnostic and are recorded as non-default. |
| Stale context | Phase 2S loads current Phase 2/2R artifacts and current target code before runtime. |
| Environment mismatch | CPU-hidden diagnostic cannot support GPU/XLA or default-readiness claims. |
| Artifact mismatch | JSON/Markdown/result/log paths are predeclared and must preserve metric roles and nonclaims. |

Audit status: `PASSED_FOR_REVIEW_ONLY`.  Runtime may begin only after review
converges.
