# Phase 2 Result: Local Quadratic Reference Agreement

Date: 2026-07-09
Status: `FAILED_LOCAL_REFERENCE_SCREEN_REPAIR_REQUIRED`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2 local quadratic reference agreement does not pass | Failed: max marginal mean error was `2.728680904681481` against threshold `0.5`; standard-deviation ratios were outside `[0.5, 2.0]` | Vetoes: `mean_abs_error_above_0p5`, `std_ratio_outside_0p5_2p0` | The comparator is a local quadratic Gaussian, not an exact posterior; mismatch could reflect local geometry error, short-chain behavior, transform issues, or target nonquadraticity | Draft/review a localization repair before any GPU/XLA phase | No posterior correctness, HMC convergence/readiness, zero-divergence claim, sampler superiority, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered: Phase 1R HMC marginal summaries do not agree with the local quadratic Gaussian reference under the predeclared screen. |
| Baseline/comparator | Local quadratic reference from 2026-07-08 geometry/mass artifacts. |
| Primary criterion | Failed. |
| Veto diagnostics | `mean_abs_error_above_0p5`, `std_ratio_outside_0p5_2p0`. |
| Explanatory diagnostics | Reference mean `[-0.044565226160942085, 0.14339340258862682, 0.12440293581509161, 0.3807781120936724]`; HMC pooled mean `[2.684115678520539, 0.6267063734817137, 1.5582468030301495, 0.5156267037406872]`; HMC/reference std ratios `[3.5135873864369875, 2.78503939933846, 2.0517916307643334, 2.266402929702481]`. |
| Not concluded | No exact posterior failure, no HMC readiness/convergence, no GPU/XLA readiness, no default readiness, and no source-faithfulness claim. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2_local_quadratic_reference.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py`

## Reference Construction

The local reference used:

- local density: `c + l_z^T z - 0.5 z^T K_z z`
- coordinate map: `z = F u`, with `F = chol(M_z)`
- `K_u = F.T @ K_z @ F`
- `C_u = inv(K_u)`
- `m_u = C_u @ F.T @ l_z`

The artifact reports `precision_u_identity_max_abs_error =
3.219646771412954e-15`, so the geometry/mass whitening handoff is internally
consistent.  The failure is between the short HMC summaries and this local
quadratic reference, not between `K_z` and `M_z`.

## Native-Divergence Telemetry

Native divergence remained `not_exposed_by_kernel` for all Phase 1R seeds.
Unavailable native divergence was not treated as zero divergences.  The artifact
records no zero-divergence claim.

## Checks

| Check | Status |
| --- | --- |
| Phase 2 subplan Codex substitute review round 2 | `VERDICT: AGREE` |
| `git diff --check` before runtime | Passed |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2_reference.py` | Passed: `3 passed` |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase1.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase1r.py` | Passed: `8 passed` |
| Phase 2 runtime command | Exited `0`; artifact decision failed by predeclared local-reference screen |

## Repair Trigger

This is a localization trigger:

- Geometry/mass whitening internally checks out.
- Phase 1R finite/acceptance telemetry passed.
- Local quadratic reference agreement failed by large HMC marginal mean and
  dispersion differences.

The next repair should localize whether this is caused by the local quadratic
reference being centered at a weak truth-free point, insufficient chain length,
transient short-chain drift, or target nonquadraticity.  GPU/XLA should not be
used as a substitute for this localization gate.

## Final Nonclaims

- No exact posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
