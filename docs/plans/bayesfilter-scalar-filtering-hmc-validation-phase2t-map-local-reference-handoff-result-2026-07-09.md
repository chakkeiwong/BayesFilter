# Phase 2T Result: MAP-Local Reference Handoff

Date: 2026-07-09
Status: `PASSED_MAP_LOCAL_REFERENCE_HANDOFF_DIAGNOSTIC`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2T validated the Phase 2S MAP-local handoff and exact Phase 2U screen contract | Passed: Phase 2S artifact valid, MAP-local matrices finite/SPD/consistent, target replay finite at the map candidate, old Phase 1R summaries excluded from pass/fail, and Phase 2U candidate/selection/veto policy predeclared | No final Phase 2T vetoes | This only validates a local affine handoff and next HMC-screen contract; it is not a sampler run | Draft, review, and only then run Phase 2U retuned MAP-local fixed-kernel CPU-hidden HMC screen | No posterior correctness, HMC readiness/convergence, zero-divergence claim, sampler superiority, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered for handoff only: the Phase 2S MAP-local geometry is internally consistent enough to justify a reviewed retuned fixed-kernel HMC screen. |
| Baseline/comparator | Phase 2S MAP-local geometry artifact; old Phase 1R summaries were diagnostic only and excluded from pass/fail. |
| Primary criterion | Passed. |
| Veto diagnostics | None after implementation repair. |
| Explanatory diagnostics | Matrix residuals, target replay, old-geometry projection metadata, and Phase 2U candidate contract. |
| Not concluded | No posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2t_map_local_reference_handoff.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py`
- Tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff.py`

## Key Diagnostics

| Diagnostic | Value | Role |
| --- | --- | --- |
| Phase 2T decision | `phase2t_map_local_reference_handoff_passed=True` | Primary Phase 2T pass/fail |
| Final vetoes | `[]` | Hard veto evidence |
| `precision_z @ covariance_z` identity max error | `1.1254733403169899e-15` | Matrix-consistency gate |
| `factor_z @ factor_z.T` reconstruction max error | `8.881784197001252e-16` | Matrix-consistency gate |
| Precision theta scale-transform max error | `1.0000285044498014e-09` | Theta/z transform gate |
| Covariance theta scale-transform max error | `4.887283910903761e-10` | Theta/z transform gate |
| Candidate grid | `(L, step) = (2, 0.785), (4, 0.3925), (8, 0.19625), (16, 0.098125)` | Phase 2U handoff contract |
| Trajectory length | `1.57` for every candidate | Phase 2U handoff contract |
| Phase 2U selection policy | First candidate in listed order passing hard vetoes and acceptance envelope | Phase 2U handoff contract |
| Native divergence policy | Positive native divergence is a hard veto when available; unavailable is not zero-divergence evidence | Telemetry boundary |

## Implementation Repair Note

The first Phase 2T runtime attempt failed because the harness symmetrized
`factor_z` before checking `factor_z @ factor_z.T`, which is wrong for a
Cholesky factor.  This was a harness implementation bug, not evidence against
the MAP-local geometry or target.  The harness was repaired to load `factor_z`
without symmetrization, and the focused tests now include a non-diagonal factor
fixture so the bug is covered.

## Phase 2U Handoff

Phase 2U may run only after its subplan review converges.  The predeclared
candidate list is:

| Candidate | Leapfrog steps | Step size | `L * step_size` |
| --- | --- | --- | --- |
| 0 | `2` | `0.785` | `1.57` |
| 1 | `4` | `0.3925` | `1.57` |
| 2 | `8` | `0.19625` | `1.57` |
| 3 | `16` | `0.098125` | `1.57` |

The next screen is a finite/acceptance screen in the new MAP-local `u_new`
coordinate.  It is not a posterior agreement phase and cannot advance to
Phase 3 GPU/XLA unless the reviewed MAP-local HMC repair branch passes its own
gates.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `f297b103303c64019302ed5d9b9aaf2c8f919b64` |
| Git dirty status | Dirty; artifact records planned scalar HMC validation edits and unrelated user work |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 180 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden artifact analysis |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seeds | Inherited from Phase 2S manifest; no HMC seeds consumed in Phase 2T |
| Wall time | `0.07020520098740235` seconds |
| Plan/result paths | Master, Phase 2T subplan, JSON, Markdown, quiet log, and this result file |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Passed for Phase 2T handoff consistency. |
| Statistically supported ranking | None; no sampler run and no method comparison. |
| Descriptive-only differences | Matrix residuals and old-geometry projection diagnostics. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed Phase 2U retuned MAP-local fixed-kernel HMC screen. |

## Checks

| Check | Status |
| --- | --- |
| Phase 2T subplan Codex substitute review round 1 | `VERDICT: REVISE` |
| Phase 2T subplan repair | Patched theta/z transform checks, old Phase 1R exclusion from pass/fail, and exact Phase 2U candidate/selection policy |
| Phase 2T subplan Codex substitute review round 2 | `VERDICT: AGREE` |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair.py` | Passed before runtime: `10 passed` |
| Initial Phase 2T runtime | Failed due harness-side `factor_z` symmetrization bug |
| Harness repair check | `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff.py` passed: `5 passed` |
| Final Phase 2T runtime command | Exited `0`; artifact decision passed |

## Final Nonclaims

- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
