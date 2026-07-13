# Phase 2R Result: Local Reference Mismatch Localization

Date: 2026-07-09
Status: `PASSED_LOCALIZATION_OUTCOME_OUTSIDE_GEOMETRY_TRUST_REGION`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2r-local-reference-localization-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2R localized the Phase 2 mismatch to `outside_geometry_trust_region` | Passed: exactly one predeclared localization outcome was selected and transform checks passed | No Phase 2R vetoes. Native divergence remains unavailable and is not zero-divergence evidence | This localizes the failed local quadratic reference screen; it does not say whether a MAP-local geometry will repair HMC behavior | Draft, review, and then execute a geometry/centering or MAP-local reference repair subplan | No posterior correctness, HMC readiness/convergence, zero-divergence claim, sampler superiority, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered: the Phase 1R HMC summaries lie outside the original local geometry trust region, while transform bookkeeping checks pass. |
| Baseline/comparator | Phase 2 local quadratic reference result. |
| Primary criterion | Passed as a localization diagnostic. |
| Veto diagnostics | None. |
| Explanatory diagnostics | Pooled HMC mean norm in `u` was `3.2079965478482895`; seed mean norms were `1.9314939055758316`, `3.024038476914353`, and `5.718576264956027`; trust warning threshold was `0.6`. |
| Not concluded | No exact posterior reference, posterior correctness, HMC readiness/convergence, GPU/XLA readiness, default readiness, or zero-divergence claim. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2r_localization.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py`

## Localization Diagnostics

| Diagnostic | Value | Role |
| --- | --- | --- |
| Transform identity max absolute error | `3.3306690738754696e-15` | Hard transform-veto evidence; passed threshold `1e-8` |
| Pooled HMC mean norm in `u` | `3.2079965478482895` | Outcome evidence for outside-trust-region localization |
| Seed mean norms in `u` | `[1.9314939055758316, 3.024038476914353, 5.718576264956027]` | Outcome evidence |
| Outside trust threshold | `0.6` | Predeclared warning threshold, `2 * trust_radius` |
| Large quadratic drop points | `seed_2_mean: 15.838221711125513` | Outcome evidence; threshold `10.0` |
| Target replay pooled minus center | `-0.2815211066872152` | Explanatory only; does not support a pooled-mean-higher-than-center claim |

Selected outcome:

`outside_geometry_trust_region`

Next justified action:

Draft a geometry/centering repair or MAP-local reference subplan.  Phase 3
GPU/XLA remains blocked until this repair branch is reviewed and produces a
valid handoff.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `f297b103303c64019302ed5d9b9aaf2c8f919b64` |
| Git dirty status | Dirty; artifact records planned HMC validation edits and unrelated user work |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 240 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; CPU-hidden artifact analysis |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seeds | Inherited Phase 1R seeds `(20260709, 6101)`, `(20260709, 6102)`, `(20260709, 6103)` |
| Wall time | `48.841003492998425` seconds |
| Plan/result paths | Master, Phase 2R subplan, JSON, Markdown, quiet log, and this result file |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Passed for Phase 2R localization. |
| Statistically supported ranking | None; no method comparison and no uncertainty interval. |
| Descriptive-only differences | HMC/reference distances, target replay values, and local quadratic drops. |
| Default readiness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked until a targeted repair branch passes. |
| Next evidence needed | Reviewed geometry/centering or MAP-local reference repair. |

## Checks

| Check | Status |
| --- | --- |
| Phase 2R subplan Codex substitute review | `VERDICT: AGREE` |
| Phase 2R runtime command | Exited `0`; artifact selected `outside_geometry_trust_region` |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2_reference.py` | Passed before this result write: `6 passed` |
| `git diff --check` | Passed before this result write |

## Final Nonclaims

- No HMC readiness.
- No HMC convergence.
- No posterior correctness.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
