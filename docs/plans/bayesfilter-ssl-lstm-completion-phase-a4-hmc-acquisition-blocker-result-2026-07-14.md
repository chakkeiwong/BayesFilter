# Phase A4 HMC Acquisition Blocker Result

Date: 2026-07-14 (Asia/Shanghai)

Status: `BLOCKED_INVALID_CALIBRATION_INPUT_REPAIR_REQUIRED`

## Outcome

No existing SSL-LSTM HMC artifact qualified for A4 calibration, so the
authorized calibration-only four-chain GPU/XLA acquisition route was executed.
The target/transform engineering checks, repaired mechanics canary, and the
first 64-draw balanced-kernel tuning screen passed their bounded contracts.
The smallest serious retained rung then fired the prospective continuation
veto: chain 0 accepted `0/250` retained transitions and never moved, while the
other chains moved and accepted `49.2%`, `65.2%`, and `60.8%`.

The `[250,4,4]` archive is finite, hash-verified, and GPU/XLA-produced, but it is
not an admissible four-chain calibration input. No R-hat, ESS, or MCSE
calculation can rescue a chain with zero movement. The sequential acquisition
therefore stopped before any extension or forecast calibration.

This is evidence against the selected `(step_size=0.3925,
num_leapfrog_steps=4)` acquisition setting from these dispersed starts. It is
not evidence that the A1 target, affine adapter, retained archive runtime, HMC
as a research direction, or moment-based predictive validation is invalid.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject the balanced-kernel retained archive as A4 calibration input and stop this acquisition cycle | `FAIL`: chain 0 movement `false`; chain 0 acceptance `0.0 < 0.20` | Continuation veto `unmoved_chain`; promotion veto `per_chain_acceptance_outside_threshold`; no nonfinite-value veto | Whether the predeclared half-step/eight-leapfrog kernel prevents state-dependent rejection at the difficult dispersed start | Separately authorize and run the prospective repair plan with fresh tuning/acquisition artifacts; do not reuse `segment_0` for calibration | No posterior incorrectness, target invalidity, HMC-direction rejection, sampler ranking, predictive equivalence, NeuTra readiness, model adequacy, or default readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Failed for retained sampler admission because one chain was completely unmoved; target values, scores, samples, log-accept ratios, archive hashes, and GPU placement remained valid |
| Statistically supported ranking | None; one selected kernel failed the retained admission gate, and no stochastic superiority comparison was run |
| Descriptive-only differences | Aggregate/per-chain acceptance, maximum finite log-accept ratio, target-log-probability range, runtime, and movement counts for the three moving chains |
| Default-readiness | Not assessed and not supported |
| Next evidence needed | Fresh four-chain tuning and smallest retained acquisition with the prospective `(0.19625,8)` repair kernel, preserving the same target, starts, thresholds, and full budget lineage |

## Separate Evidence Ledgers

| Ledger | Status | Evidence |
| --- | --- | --- |
| Engineering correctness | `PASSED` | Focused suite `11/11`; locked A1 target/adapter identity; scalar/batch value and score parity; affine orientation and finite-difference check; private shard hash/shape readback; strict source and budget lineage |
| Numerical validity | `PASSED_FOR_EMITTED_ARTIFACTS` | All acquired samples, target values, final states, and log-accept ratios finite; GPU/XLA placement recorded; A0 factor reconstruction residual `8.881784197001252e-16 <= 1.1465297583454372e-13` |
| Sampler admission | `FAILED` | Retained chain movement `[false,true,true,true]`; per-chain acceptance `[0.0,0.492,0.652,0.608]`; `unmoved_chain` continuation veto |
| Posterior correctness | `NOT_ASSESSED` | A finite/moving HMC archive is a prerequisite, not posterior-reference evidence |
| Forecast calibration | `NOT_RUN` | Invalid HMC input blocked split-half forecast calibration before any A4 nomination/validation data were opened |
| Scientific interpretation | `BOUNDED_NEGATIVE_RESULT` | Rejects the current kernel acquisition attempt only; does not reject the target, HMC, NeuTra, or moment-law validation directions |

## Evidence Sequence

| Artifact | Result | Key evidence | SHA-256 |
| --- | --- | --- | --- |
| `existing-artifact-audit.json` | No existing artifact qualified | No locked-A1 `[draw,4,4]` archive with sampler-validity evidence | `7ce9dc96dc9e86a4ad8cba466712c1b9405ca57bcf297e854e613538d4ca00c8` |
| `cpu-transform-check.json` | `PASSED` | Value/score transform residuals `0.0`; finite-difference residual `3.951489690682646e-09` on the final CPU receipt | `aa29f1fd11c1d80273881a5a54142b966deb8c5d0dfe7a90c44b047f13906e13` |
| `gpu-canary.json` | `FAILED` under original canary gate | Valid GPU/XLA archive; movement `[false,true,true,true]`; preserved visible repair trigger | `d5aa099cc4835d427b570a7a22430a7b79498760dc99d8ed280c9bf39692c048` |
| `gpu-canary-repair-01.json` | `PASSED` | Movement `[true,true,true,true]`; acceptance `[0.875,0.875,0.5,0.625]`; finite GPU/XLA telemetry | `b30098f573fb2a7a22f8a1a71b910d2b931fac7c169f049ac9e9efe6af87ab2d` |
| `tune-0.json` | `SELECTED` | Movement all true; acceptance `[0.8125,0.25,0.65625,0.828125]`; first passing predeclared candidate | `9e70e8dbd04de09c0bc3946d100d24d67ce520c18f63e58c8b5d3502762fa76f` |
| `segment-0.json` | `HARD_VETO` | Movement `[false,true,true,true]`; acceptance `[0.0,0.492,0.652,0.608]`; no nonfinite telemetry | `d12e7aeb1c9760b9d4bba9f9827c027e371d227a3cf5b84d7775f3a922021892` |

All paths above are relative to
`docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/`.
The private retained sample shard is
`private/segment_0_retained_samples.tftensor`, SHA-256
`d39c1d198171cb0d0b9ec3d234f3193f9addd5654fc8298941ebcedc72ba5667`,
with shape `[250,4,4]`. It remains a private failed-attempt artifact and must
not be used for calibration, confirmation, or posterior claims.

## Run Manifest Summary

| Field | Value |
| --- | --- |
| Git commit | `3d353253dc93a102722e00cbca8803a1b3fce7fa` |
| Worktree | Dirty; unrelated Kalman/QR/Sylvester lane changes preserved and untouched |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; TensorFlow Probability `0.25.0` |
| Device route | Two NVIDIA GeForce RTX 4080 SUPER devices visible; evidence outputs on `GPU:0`; XLA JIT and TF32 enabled; `float64` target tensors |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Target | `ssl_lstm_completion:a1:masked_svd_ukf_four_parameter`; semantic SHA-256 `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e` |
| Selected kernel | `step_size=0.3925`; `num_leapfrog_steps=4`; trajectory length `1.57` |
| Seeds | Root `[20260714,1404]`; failed canary `1411`; repaired canary `1412`; tuning `1420`; retained rung `1430` |
| Trusted-GPU wall time | `1333.7487312000012` seconds = `0.370485758666667` hours |
| Authorized cap | `28800` seconds = `8` hours |
| Unspent budget after stop | `27466.2512688` seconds = `7.629514241333333` hours; unspent budget is not authority to cross the fired continuation veto |
| Plan | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-subplan-2026-07-11.md` |
| Result | This file |

The first canary compile/execute consumed `424.92757005395833` seconds, its
visible repair retry consumed `413.86419730097987`, tuning consumed
`212.44679685204756`, and the retained rung consumed `282.51016699301545`.
Every attempt is charged; no failed run was removed from the budget.

## Failure Classification

| Question | Answer |
| --- | --- |
| Did the harness fail? | No. It emitted hash-verified finite archives, detected the zero-movement chain, and stopped as designed. |
| Did the implementation or target fail numerically? | No. Values, scores, target log probabilities, samples, and log-accept telemetry were finite and GPU/XLA-produced. |
| Did tuning fail? | Yes. The 64-draw screen selected a kernel that failed the later retained admission gate; current evidence does not identify whether screen length, warmup path, start-specific geometry, or another tuning factor caused the mismatch. |
| Did the current candidate fail? | Yes. The balanced kernel cannot supply the required four usable chains from the fixed dispersed starts in this attempt. |
| Did HMC or moment-based validation fail as research directions? | No. The next discriminating repair is a prospectively smaller step at the same trajectory length. |

## Native Divergence Limitation

TensorFlow Probability `0.25.0` plain `HamiltonianMonteCarlo` did not expose a
native divergence boolean. Every artifact records
`native_divergence_status=not_exposed_by_kernel`. No zero-divergence claim is
made. The very large but finite maximum log-accept ratio magnitude
`1589512.814800619` is explanatory and consistent with an aggressive proposal
for the stuck chain, but it is not a native divergence substitute.

## Post-Run Red Team

The leading repair hypothesis is state-dependent tuning failure: the balanced
kernel looked viable in a 64-draw screen, but chain 0 later occupied a state at
which every retained proposal was rejected. Another possibility is that the
fixed dispersed start for chain 0 lies in a geometry region not represented by
the historical local tuning screen. The current evidence does not distinguish
these explanations, and neither supports deleting that chain or narrowing
initialization after seeing the result.

A fresh smaller-step kernel that moves all four chains and passes the same
R-hat/ESS/MCSE/acceptance gates would overturn the candidate-specific blocker.
A repeat zero-movement chain under the smaller-step repair would strengthen the
case for a more fundamental warmup/mass-geometry repair. The weakest evidence
is that native divergence telemetry is unavailable and no posterior reference
comparison has occurred.

## Stop And Handoff

- Do not extend `segment_0`; it contains an invalid unmoved chain.
- Do not use any canary, tuning, or `segment_0` sample as forecast-calibration
  input.
- Do not run forecast calibration, A5 confirmation, NeuTra training, or
  NeuTra-HMC from this result.
- The next eligible action is the separately prospective repair plan at
  `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-repair-plan-2026-07-14.md`.
