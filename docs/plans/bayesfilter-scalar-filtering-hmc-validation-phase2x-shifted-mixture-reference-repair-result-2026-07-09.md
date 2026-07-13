# Phase 2X Result: Shifted-Mixture Reference Repair

Date: 2026-07-09
Status: `FAILED_REFERENCE_VALIDITY_BLOCKER`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2X shifted-mixture reference repair failed the reference-validity gate | Failed: reference ESS `33.4215730897076` and ESS ratio `0.01631912748520879` were below the predeclared `256` and `0.125` thresholds | Final vetoes: `reference_ess_below_threshold`, `reference_ess_ratio_below_threshold` | Both Phase 2W and Phase 2X proposals had finite evaluations but severe weight concentration, so the next step is to localize the target/proposal mismatch rather than make another blind proposal tweak | Draft and review a target-geometry localization diagnostic subplan; stop before additional runtime unless that blocker boundary is explicitly cleared | No posterior correctness, HMC readiness/convergence, zero-divergence claim, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered negatively for the shifted-mixture repair: the proposal did not produce a usable self-normalized importance reference. |
| Baseline/comparator | Phase 2W failed standard-normal proposal and Phase 2V selected-kernel HMC summaries. |
| Primary criterion | Failed at reference validity before HMC agreement. |
| Veto diagnostics | Reference ESS and ESS ratio vetoes fired. |
| Explanatory diagnostics | Mixture component counts, proposal log density range, target log probabilities, log weights, normalized weight summary, weighted moments, and runtime were recorded. |
| Not concluded | No posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2x_shifted_mixture_reference_repair.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py`
- Tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.py`

## Reference Diagnostics

| Diagnostic | Value | Role |
| --- | --- | --- |
| Proposal | `0.25 * N(0, I_4) + 0.75 * N(phase2w_pilot_center, diag(shifted_scale^2))` | Fixed repair design |
| Proposal draws | `2048` antithetic draws, seed `(20260709, 6601)` | Fixed repair design |
| Component counts | `512` standard, `1536` shifted | Fixed repair design |
| Shifted center | `[0.16900152112527375, 0.34590014590251295, 0.47216707577215133, -0.3362900480743778]` | Phase 2W pilot diagnostic only |
| Shifted scale | `[1.4111540908177695, 1.7433972704992957, 2.2347453201729985, 2.2206014796667195]` | Predeclared Phase 2W-pilot scale transform |
| Proposal log-density finite/nonfinite | `2048` finite, `0` nonfinite | Hard validity evidence |
| Target log-prob finite/nonfinite | `2048` finite, `0` nonfinite | Hard validity evidence |
| Log-weight finite/nonfinite | `2048` finite, `0` nonfinite | Hard validity evidence |
| Log-weight range | `[-416.04885854676746, -26.175504948369124]` | Explanatory |
| Max normalized weight | `0.11446321229118656` | Explanatory degeneracy diagnostic |
| Reference ESS | `33.4215730897076` | Reference validity veto |
| Reference ESS ratio | `0.01631912748520879` | Reference validity veto |
| Reference mean `u_new` | `[-0.043945514545527586, -0.03297944040803433, -0.2764198156051569, 0.091681845356501]` | Pilot diagnostic only; not a valid posterior reference |
| Reference std `u_new` | `[2.1533163913207503, 2.762122672572576, 3.1974869281363474, 2.308167189648201]` | Pilot diagnostic only; not a valid posterior reference |

## HMC Agreement Boundary

HMC-vs-reference agreement was not evaluated or interpreted.  The artifact
records `hmc_reference_agreement.evaluated=False` and
`phase2x_agreement_not_evaluated` because reference validity failed first.
This is a reference-proposal failure, not evidence against the Phase 2V HMC
chain, the target, or the broader research direction.

## Checks

| Check | Status |
| --- | --- |
| Phase 2X subplan Codex substitute review | `VERDICT: AGREE`; weaker than full Claude material review |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py` | Passed before runtime: `21 passed` |
| `git diff --check` | Passed before runtime |
| Phase 2X runtime command | Exited `0`; artifact decision failed at reference validity |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `f297b103303c64019302ed5d9b9aaf2c8f919b64` |
| Git dirty status | Dirty; artifact records planned scalar HMC validation edits and unrelated user work |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seed | `(20260709, 6601)` |
| Wall time | `113.19665156002156` seconds |
| Plan/result paths | Master, Phase 2X subplan, JSON, Markdown, quiet log, and this result file |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Failed at reference validity. |
| Reference validity | Failed: ESS and ESS ratio below threshold. |
| HMC-reference agreement | Not interpreted. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Reference ESS, log-weight summary, weighted moments, top-weight locations, and runtime. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed target-geometry/proposal-mismatch localization before any new reference agreement attempt. |

## Post-Run Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | The target has important high-density/proposal-mismatch regions in tails or along curved directions that diagonal Gaussian mixtures centered from a low-ESS pilot do not cover. |
| What would overturn | A geometry-localization diagnostic showing the failures were caused by a harness/log-density bug rather than genuine proposal mismatch, or a reviewed non-diagonal/multimodal/transport proposal that passes fresh ESS gates. |
| Weakest evidence | Two failed importance proposals diagnose proposal inadequacy but do not identify the target geometry by themselves. |

## Final Nonclaims

- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
