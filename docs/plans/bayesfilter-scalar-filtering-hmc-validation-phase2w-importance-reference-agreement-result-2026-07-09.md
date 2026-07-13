# Phase 2W Result: MAP-Local Importance Reference Agreement

Date: 2026-07-09
Status: `FAILED_REFERENCE_VALIDITY_REPAIR_TRIGGERED`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2W fixed standard-normal importance reference failed the reference-validity gate | Failed: reference ESS `22.894679726459746` and ESS ratio `0.022358085670370845` were below the predeclared `128` and `0.125` thresholds | Final vetoes: `reference_ess_below_threshold`, `reference_ess_ratio_below_threshold` | The fixed standard-normal proposal was too poorly matched to be a valid independent reference; HMC-vs-reference agreement was not interpreted | Draft and review a defensive shifted-mixture reference repair subplan using Phase 2W pilot diagnostics only | No posterior correctness, HMC readiness/convergence, zero-divergence claim, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered negatively for the fixed standard-normal proposal: it did not produce a usable self-normalized importance reference. |
| Baseline/comparator | Phase 2V selected-kernel HMC summaries were loaded, but comparison was not interpreted because the reference validity gate failed. |
| Primary criterion | Failed at reference validity before HMC agreement. |
| Veto diagnostics | Reference ESS and ESS ratio vetoes fired. |
| Explanatory diagnostics | Proposal values, target log probabilities, log weights, normalized weight summary, weighted moments, and runtime were recorded. |
| Not concluded | No posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2w_importance_reference_agreement.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py`
- Tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py`

## Reference Diagnostics

| Diagnostic | Value | Role |
| --- | --- | --- |
| Proposal | Standard normal `N(0, I_4)` in MAP-local `u_new` | Fixed reference design |
| Proposal draws | `1024` antithetic draws, seed `(20260709, 6501)` | Fixed reference design |
| Target log-prob finite/nonfinite | `1024` finite, `0` nonfinite | Hard validity evidence |
| Log-weight finite/nonfinite | `1024` finite, `0` nonfinite | Hard validity evidence |
| Log-weight range | `[-104.0199706824082, -27.425684702606866]` | Explanatory |
| Max normalized weight | `0.17637097762827186` | Explanatory degeneracy diagnostic |
| Nonzero normalized weights | `1024` | Explanatory |
| Reference ESS | `22.894679726459746` | Reference validity veto |
| Reference ESS ratio | `0.022358085670370845` | Reference validity veto |
| Reference mean `u_new` | `[0.16900152112527375, 0.34590014590251295, 0.47216707577215133, -0.3362900480743778]` | Pilot diagnostic only; not a valid posterior reference |
| Reference std `u_new` | `[1.1289232726542155, 1.3947178163994365, 1.7877962561383989, 1.7764811837333756]` | Pilot diagnostic only; not a valid posterior reference |
| Reference mean MCSE proxy | `[0.23593759044489337, 0.2914869140558837, 0.37363774057733806, 0.3712729643488579]` | Explanatory only because reference validity failed |

## HMC Agreement Boundary

HMC-vs-reference agreement was not evaluated or interpreted.  The artifact
records `hmc_reference_agreement.evaluated=False` and
`phase2w_agreement_not_evaluated` because reference validity failed first.
This is a reference-proposal failure, not evidence against the Phase 2V HMC
chain, the target, or the broader research direction.

## Review Record

| Check | Status |
| --- | --- |
| Phase 2W subplan Codex substitute review round 1 | `VERDICT: REVISE` |
| Phase 2W subplan repair | Fixed the MCSE definition to use a square root and added explicit Phase 2U artifact/selected-kernel handoff validity vetoes |
| Phase 2W subplan Codex substitute review round 2 | `VERDICT: AGREE` |
| Review strength | Codex substitute review, weaker than full Claude material review |
| Claude status | Unavailable for repo-context material review per prior handoff; not retried |

## Checks

| Check | Status |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py` | Passed before runtime: `15 passed` |
| `git diff --check` | Passed before runtime |
| Phase 2W runtime command | Exited `0`; artifact decision failed at reference validity |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `f297b103303c64019302ed5d9b9aaf2c8f919b64` |
| Git dirty status | Dirty; artifact records planned scalar HMC validation edits and unrelated user work |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seed | `(20260709, 6501)` |
| Wall time | `81.64405142096803` seconds |
| Plan/result paths | Master, Phase 2W subplan, JSON, Markdown, quiet log, and this result file |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Failed at reference validity. |
| Reference validity | Failed: ESS and ESS ratio below threshold. |
| HMC-reference agreement | Not interpreted. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Reference ESS, log-weight summary, weighted moments, and runtime. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed reference-proposal repair. |

## Post-Run Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | The standard-normal proposal is centered correctly in the MAP-local coordinate but is too diffuse or mis-scaled relative to the local posterior mass, causing weight degeneracy. |
| What would overturn | A repaired proposal with predeclared finite-weight and ESS gates could produce a valid reference and allow HMC moment agreement to be interpreted. |
| Weakest evidence | Phase 2W has one failed fixed proposal; it does not diagnose HMC convergence, posterior correctness, or GPU behavior. |

## Final Nonclaims

- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
