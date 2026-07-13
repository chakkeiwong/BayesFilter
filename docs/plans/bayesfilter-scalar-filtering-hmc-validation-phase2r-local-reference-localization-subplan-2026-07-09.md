# Phase 2R Subplan: Local Reference Mismatch Localization

Date: 2026-07-09
Status: `REVISED_AFTER_CODEX_SUBSTITUTE_REVIEW_ROUND_1`

## Phase Objective

Localize the Phase 2 local quadratic reference mismatch before any GPU/XLA
runtime.  This phase should decide whether the mismatch is primarily due to
local reference centering/geometry, short-chain transient behavior, transform
bookkeeping, or target nonquadraticity.

## Entry Conditions

- Phase 2 result exists and records
  `FAILED_LOCAL_REFERENCE_SCREEN_REPAIR_REQUIRED`.
- Phase 1R finite/acceptance screen passed.
- Geometry/mass whitening identity checks passed in Phase 2.
- Native divergence remains unavailable/not-zero-divergence evidence.

## Required Artifacts

- Phase 2R localization harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py`
- Phase 2R JSON/Markdown localization artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2r_localization.log`
- Phase 2R result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2r-local-reference-localization-result-2026-07-09.md`
- Refreshed next subplan that targets the selected localization outcome.

## Required Checks, Tests, And Reviews

- Review this subplan before runtime.
- Run focused tests:
  - `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization.py`
  - `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2_reference.py`
- Run `git diff --check`.
- Create the quiet log directory before redirected runtime:

```bash
mkdir -p docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09
```

- Planned runtime command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 240 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2r_localization.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | What is the smallest discriminating explanation for the Phase 2 local-reference mismatch? |
| Baseline/comparator | Phase 2 local quadratic reference result. |
| Primary criterion | Produce a valid localization artifact with one and only one primary outcome from the predeclared outcome set below, or `inconclusive_needs_longer_cpu_chain` if multiple explanations remain viable. |
| Veto diagnostics | Invalid input artifacts, nonfinite localization metrics, unsupported posterior/HMC/GPU/default claims, or telemetry semantics mismatch. |
| Explanatory diagnostics | Distance of HMC summaries from local reference, trust-radius exceedance, local quadratic predicted log-density drop, target replay values at cheap summary/replay points if cheap enough, and transform identity checks. |
| Not concluded | Posterior correctness, HMC readiness/convergence, GPU/XLA readiness, default readiness, zero divergences, or sampler superiority. |

## Predeclared Localization Outcomes

| Outcome | Required evidence | Next subplan |
| --- | --- | --- |
| `transform_bookkeeping_mismatch` | Any coordinate/matrix identity check fails: `K_u = F.T @ K_z @ F`, `C_u = inv(K_u)`, `z = F u`, or artifact coordinate contracts disagree | Code/test repair before any new sampling |
| `outside_geometry_trust_region` | Phase 1R pooled mean norm or any seed mean norm in `u` exceeds the Phase 1 geometry trust radius by more than `2x`, or local quadratic predicted drop exceeds a predeclared large-drop threshold | Geometry/centering repair or MAP-local reference subplan |
| `local_quadratic_reference_center_weak` | Local reference mean is near the origin but HMC summaries and target replay at HMC mean show higher log density than at the geometry center | MAP/centering repair subplan |
| `short_chain_transient_or_multimodality_possible` | Transform checks pass, HMC cloud is outside reference but cheap target replay does not support a simple center-quality diagnosis | Longer CPU-hidden chain/replication subplan with uncertainty diagnostics |
| `inconclusive_needs_longer_cpu_chain` | Required diagnostics are finite but no single outcome dominates under the rules above | Longer CPU-hidden chain/replication subplan |

Concrete diagnostic thresholds for Phase 2R:

- `transform_identity_max_abs_error <= 1e-8` must hold for transform
  bookkeeping to pass.
- `trust_radius = 0.30` in Phase 1 `z` coordinates; because Phase 2 `u`
  covariance is approximately identity, use `2 * trust_radius = 0.60` as the
  conservative outside-trust-region warning for reference-local summaries.
- `large_quadratic_drop_threshold = 10.0` in local log-density units.
- Cheap target replay, if implemented, may evaluate only the center,
  local-reference mean, Phase 1R pooled mean, and seed means.  It is diagnostic
  only unless the result subplan explicitly promotes it in a later reviewed
  repair.

## Forbidden Claims And Actions

- Do not run GPU/XLA as a repair for Phase 2 mismatch.
- Do not change the Phase 2 pass thresholds post hoc.
- Do not claim exact posterior mismatch from local quadratic mismatch alone.
- Do not claim zero divergences while native divergence is unavailable.

## Exact Next-Phase Handoff Conditions

Advance only to a reviewed repair that directly targets the selected
localization outcome.  The Phase 2R result must name exactly one primary
outcome from the predeclared table or mark the result inconclusive.  If
localization points to weak geometry/centering, draft a geometry/MAP-local
reference repair.  If it points to short-chain transients, draft a longer
CPU-hidden chain plan with uncertainty diagnostics.  If it points to transform
bookkeeping, repair code/tests before any more runtime.

## Stop Conditions

Stop for invalid artifacts, nonfinite localization metrics, missing diagnostic
role discipline, unsupported claims, or review nonconvergence.
