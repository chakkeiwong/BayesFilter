# Phase 2U Result: Retuned MAP-Local HMC Screen

Date: 2026-07-09
Status: `PASSED_SELECTED_KERNEL_HANDOFF_PENDING_PHASE_2V_REVIEW`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2U selected the first passing MAP-local fixed-kernel candidate for a later longer screen | Passed: all four candidates passed hard vetoes and acceptance envelope; selected candidate is candidate 0 by the predeclared first-passing rule | No final Phase 2U vetoes | This is a short CPU-hidden finite/acceptance screen with no uncertainty analysis and unavailable native divergence telemetry | Draft and review Phase 2V longer selected-kernel MAP-local screen before any further runtime | No posterior correctness, HMC readiness/convergence, zero-divergence claim, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered for the short screen: at least one MAP-local equal-trajectory-length candidate passed hard vetoes and the acceptance envelope. |
| Baseline/comparator | Phase 2S/2T MAP-local affine handoff; old Phase 1R summaries were excluded from pass/fail. |
| Primary criterion | Passed. |
| Veto diagnostics | None in the final rerun. |
| Explanatory diagnostics | Candidate acceptance, log-accept ranges, target-log-prob ranges, sample ranges, runtime, and native-divergence availability. |
| Not concluded | No posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2u_retuned_map_local_hmc_screen.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py`
- Tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.py`

## Candidate Screen

| Candidate | Leapfrog steps | Step size | Trajectory | Seed | Acceptance | Hard vetoes | Runtime seconds | Max abs `u_new` | Log-accept max abs | Target log-prob range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `2` | `0.785` | `1.57` | `(20260709, 6301)` | `0.34375` | none | `97.46058729302604` | `5.7574913760263` | `3649.3851947870867` | `[-41.46616440775271, -38.101236770598966]` |
| 1 | `4` | `0.3925` | `1.57` | `(20260709, 6302)` | `0.546875` | none | `68.35223318601493` | `5.365143733766334` | `430.6216801541776` | `[-40.49214646573278, -37.86911620056172]` |
| 2 | `8` | `0.19625` | `1.57` | `(20260709, 6303)` | `0.96875` | none | `132.7876608769875` | `10.043862894094998` | `36.33968750224398` | `[-45.164080114603266, -38.0975892755761]` |
| 3 | `16` | `0.098125` | `1.57` | `(20260709, 6304)` | `0.984375` | none | `264.7582094660029` | `11.175519807356721` | `0.13317427725214337` | `[-43.396064352877815, -37.827440221983]` |

Selected candidate:

- candidate index: `0`
- leapfrog steps: `2`
- step size: `0.785`
- trajectory length: `1.57`
- acceptance: `0.34375`
- selection policy: `first_passing_candidate_in_predeclared_order`

The selected candidate is not asserted to be statistically superior.  It is
only the predeclared first passing handoff candidate.

## Telemetry Boundary

Native divergence telemetry was unavailable for all four candidates:
`not_exposed_by_kernel`.  Phase 2U therefore makes no zero-divergence claim.
Log-accept tails are recorded as explanatory diagnostics only and are not used
as native-divergence telemetry.

## Implementation Repair Note

The first Phase 2U runtime attempt completed enough HMC work to build a payload
but failed during artifact writing with `TypeError: Object of type EagerTensor
is not JSON serializable`.  This was a harness serialization bug, not
candidate-gate evidence and not scientific evidence against the target.  The
harness `json_ready(...)` function was repaired to convert TensorFlow tensors
through `.numpy()`, a focused serialization test was added, and the same
reviewed runtime command was rerun unchanged.  The final rerun passed.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `f297b103303c64019302ed5d9b9aaf2c8f919b64` |
| Git dirty status | Dirty; artifact records planned scalar HMC validation edits and unrelated user work |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 720 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seeds | `(20260709, 6301)`, `(20260709, 6302)`, `(20260709, 6303)`, `(20260709, 6304)` |
| Wall time | `610.7140235070256` seconds |
| Plan/result paths | Master, Phase 2U subplan, JSON, Markdown, quiet log, and this result file |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Passed for the short CPU-hidden MAP-local finite/acceptance screen. |
| Statistically supported ranking | None; fixed short grid and no uncertainty analysis. |
| Descriptive-only differences | Candidate acceptance, log-accept tails, target ranges, sample ranges, and runtime. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed longer selected-kernel MAP-local screen. |

## Checks

| Check | Status |
| --- | --- |
| Phase 2U subplan Codex substitute review | `VERDICT: AGREE`; weaker than full Claude material review |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff.py` | Passed before runtime: `11 passed` |
| First Phase 2U runtime attempt | Failed during artifact serialization; no JSON/Markdown artifact |
| Serialization repair focused tests | Passed: `12 passed` |
| `git diff --check` | Passed before final rerun |
| Final Phase 2U runtime command | Exited `0`; artifact decision passed |

## Post-Run Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | A passing candidate may reflect a short-chain local finite/acceptance screen near the MAP-local center, not posterior validity. |
| What would overturn | Longer selected-kernel screen failures, reference disagreement, nonfinite telemetry, positive native divergence when available, or GPU/XLA mismatch under a reviewed later phase. |
| Weakest evidence | One short CPU-hidden grid with no uncertainty analysis and no native divergence availability guarantee. |

## Final Nonclaims

- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
