# Phase 2V Result: Longer Selected MAP-Local Screen

Date: 2026-07-09
Status: `PASSED_REFERENCE_SUBPLAN_HANDOFF_PENDING_REVIEW`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2V longer selected MAP-local finite/acceptance screen passed | Passed: selected kernel started at `u_new=0`, retained samples were finite, target trace and log-accept ratios were finite, and acceptance was inside `(0.05, 0.99)` | No final Phase 2V vetoes | This is still a CPU-hidden finite/acceptance screen with unavailable native-divergence telemetry and no posterior reference comparison | Draft and review a scalar reference/posterior-agreement diagnostic subplan | No posterior correctness, HMC readiness/convergence, zero-divergence claim, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered for the longer selected-kernel screen: the Phase 2U selected MAP-local kernel remained finite and inside the acceptance envelope with 128 retained draws. |
| Baseline/comparator | Phase 2U selected candidate 0. |
| Primary criterion | Passed. |
| Veto diagnostics | None in final runtime. |
| Explanatory diagnostics | Acceptance, log-accept values, target-log-prob values, sample summaries, runtime, and native-divergence availability. |
| Not concluded | No posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2v_longer_selected_map_local_screen.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py`
- Tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py`

## Selected-Kernel Diagnostics

| Diagnostic | Value | Role |
| --- | --- | --- |
| Selected kernel | `L=2`, `step_size=0.785`, trajectory length `1.57` | Fixed Phase 2U handoff |
| Initial state | `u_new=[0.0, 0.0, 0.0, 0.0]` | Hard veto contract |
| Initial target value | `-37.77528495512359` | Hard finite target check |
| Initial score norm | `2.4416858704074592e-11` | Explanatory MAP-local centering diagnostic |
| Retained samples | `128` finite, `0` nonfinite | Hard veto evidence |
| Acceptance | `0.40625` | Primary acceptance-envelope gate |
| Log-accept finite/nonfinite | `128` finite, `0` nonfinite | Hard veto evidence |
| Log-accept max abs finite | `3290.2187092038007` | Explanatory tail diagnostic only |
| Target log-prob range | `[-41.29766007404644, -37.85051477370975]` | Hard finite target trace and explanatory range |
| Mean `u_new` | `[2.339329346843717, 0.6170706799855532, -2.158614641877545, 1.5823448512102032]` | Explanatory; not posterior agreement |
| Std `u_new` | `[1.4446082251234498, 1.4433962914543734, 1.5630226197334784, 1.2645246074262342]` | Explanatory; not posterior agreement |
| Max abs `u_new` | `5.960663881275316` | Explanatory |
| Native divergence | unavailable: `not_exposed_by_kernel` | Telemetry boundary; no zero-divergence claim |

## Telemetry Boundary

Native divergence telemetry was unavailable for Phase 2V.  Unavailable
telemetry is not zero-divergence evidence.  Log-accept tails are explanatory
and were not used as native-divergence telemetry.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `f297b103303c64019302ed5d9b9aaf2c8f919b64` |
| Git dirty status | Dirty; artifact records planned scalar HMC validation edits and unrelated user work |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seed | `(20260709, 6401)` |
| Wall time | `178.8504172930261` seconds |
| Plan/result paths | Master, Phase 2V subplan, JSON, Markdown, quiet log, and this result file |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Passed for the longer selected-kernel finite/acceptance screen. |
| Statistically supported ranking | None; single selected-kernel screen. |
| Descriptive-only differences | Acceptance, log-accept values, target ranges, sample summaries, and runtime. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed scalar reference/posterior-agreement diagnostic subplan. |

## Checks

| Check | Status |
| --- | --- |
| Phase 2V subplan Codex substitute review round 1 | `VERDICT: REVISE` |
| Phase 2V subplan repair | Added exact initial-state veto and blocked GPU/XLA handoff from finite/acceptance evidence alone |
| Phase 2V subplan Codex substitute review round 2 | `VERDICT: AGREE` |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.py` | Passed before runtime: `13 passed` |
| `git diff --check` | Passed before runtime |
| Phase 2V runtime command | Exited `0`; artifact decision passed |

## Post-Run Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | A passing longer screen may still only reflect local finite/acceptance behavior near the MAP-local center, not posterior validity. |
| What would overturn | Reference disagreement, nonfinite telemetry, positive native divergence when available, or GPU/XLA mismatch under a reviewed later phase. |
| Weakest evidence | Single CPU-hidden selected-kernel screen with unavailable native-divergence telemetry and no posterior reference comparison. |

## Final Nonclaims

- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
