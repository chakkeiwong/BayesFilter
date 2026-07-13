# Phase 2S Result: Geometry Centering And MAP-Local Reference Repair

Date: 2026-07-09
Status: `PASSED_MAP_LOCAL_GEOMETRY_HANDOFF_DIAGNOSTIC`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2S produced a usable MAP-local SPD quadratic geometry and covariance handoff candidate | Passed: initializer accepted, locator accepted a finite position, holdout was nonzero and passed, precision/covariance were SPD and well-conditioned | No Phase 2S vetoes | This is a local diagnostic initializer, not a posterior covariance proof and not HMC evidence | Draft and review a MAP-local reference/handoff diagnostic subplan before any retuned HMC run | No certified global MAP, posterior covariance correctness, HMC readiness/convergence, zero-divergence claim, sampler superiority, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered: a MAP-local SPD quadratic geometry can be built for the scalar filtering target from the Phase 2R reference-mean neighborhood. |
| Baseline/comparator | Phase 2/2R truth-free local quadratic geometry and reference artifacts. |
| Primary criterion | Passed. |
| Veto diagnostics | None. |
| Explanatory diagnostics | Locator movement, target replay, fit residuals, eigen summaries, and distances to old reference/HMC summaries. |
| Not concluded | No certified global MAP, posterior covariance correctness, HMC readiness/convergence, zero divergences, sampler superiority, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2s_geometry_centering_repair.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py`
- Tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair.py`

## Key Diagnostics

| Diagnostic | Value | Role |
| --- | --- | --- |
| Locator status | `tfp_lbfgs_locator_accepted` | Hard gate; fallback would have vetoed |
| Optimizer inverse Hessian used | `False` | Hard boundary |
| Locator iterations | `9` | Explanatory |
| Initial log probability | `-37.786099036348105` | Explanatory |
| Locator log probability | `-37.77528495512359` | Explanatory |
| Locator score norm | `1.017238315038726e-10` | Explanatory/local geometry quality signal |
| Geometry finite samples | `90` | Hard sample-count gate |
| Required finite samples | `45` | Hard sample-count gate |
| Regression parameter count | `9` | Sample-ratio contract |
| Holdout count | `22` | Hard nonzero-holdout gate |
| Holdout RMSE | `0.058211774395612294` | Geometry fit gate only |
| Holdout threshold | `0.3777528495512359` | Geometry fit gate only |
| Score RMSE | `0.30615107387554297` | Explanatory geometry diagnostic |
| Regularized precision condition number | `45.0073152832043` | Hard SPD/condition gate |
| Covariance condition number | `45.007315283204306` | Hard SPD/condition gate |
| Diagonal fallback used | `False` | Hard mass-regularization gate |

The initializer's `map_candidate_role` is
`locator_position_geometry_covariance_only`.  That means the accepted locator
position is the handoff center, but the covariance authority remains the
low-rank SPD quadratic geometry plus `covariance_from_precision`, not the
optimizer inverse Hessian.

## Target Replay

| Point | Value | Score norm | Role |
| --- | --- | --- | --- |
| Old truth-free center | `-37.84742912954012` | `2.860577873798304` | Explanatory |
| Phase 2 reference-mean initial position | `-37.786099036348105` | `0.9312012159249712` | Gate-required initial finite target |
| Locator/map candidate | `-37.77528495512359` | `1.017238315038726e-10` | Gate-required locator/map finite target |
| Phase 1R pooled HMC mean | `-38.12895023622733` | `4.05561296780585` | Explanatory only |

The target replay suggests a better local center than the old truth-free
geometry, but it does not validate the posterior or HMC sampler.

## Handoff Candidate

The JSON artifact records a MAP-local handoff candidate with:

- center free parameters:
  `[0.5704394246369003, -0.1242247342531544, 0.6609123192759063, 0.1354211218811133]`;
- `precision_theta` and `covariance_theta` from the regularized quadratic
  precision in free-parameter coordinates;
- `precision_z`, `covariance_z`, and `factor_z` for a new local whitened
  coordinate;
- coordinate formula:
  `free = center_free_parameter_values + scale * (factor_z @ u_new)`.

This is a candidate for the next reviewed subplan, not an automatic HMC
runtime authorization.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `f297b103303c64019302ed5d9b9aaf2c8f919b64` |
| Git dirty status | Dirty; artifact records planned HMC validation edits and unrelated user work |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seeds | Phase 1R seeds `(20260709, 6101)`, `(20260709, 6102)`, `(20260709, 6103)`; quadratic geometry seed `(20260709, 6201)` |
| Wall time | `58.14801573997829` seconds |
| Plan/result paths | Master, Phase 2S subplan, JSON, Markdown, quiet log, and this result file |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Passed for Phase 2S initializer usability. |
| Statistically supported ranking | None; single diagnostic initializer. |
| Descriptive-only differences | Locator movement, target replay values, fit residuals, eigen summaries, and distances to old reference/HMC summaries. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked until local repair handoff and later HMC gates pass. |
| Default readiness | Not assessed. |
| Next evidence needed | Reviewed MAP-local reference/handoff diagnostic or retuned fixed-kernel HMC screen subplan. |

## Checks

| Check | Status |
| --- | --- |
| Phase 2S subplan Codex substitute review round 1 | `VERDICT: REVISE` |
| Phase 2S subplan repair | Fixed sample-count wording, locator-fallback gate, and nonzero-holdout gate |
| Phase 2S subplan Codex substitute review round 2 | `VERDICT: AGREE` |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_quadratic_map_covariance.py tests/test_quadratic_geometry.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair.py` | Passed before runtime: `20 passed, 4 warnings` |
| `git diff --check` | Passed before runtime |
| Phase 2S runtime command | Exited `0`; artifact decision passed |

## Final Nonclaims

- No certified global MAP.
- No posterior covariance correctness.
- No HMC readiness.
- No HMC convergence.
- No posterior correctness.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
