# Phase A4 HMC Acquisition Repair Result

Date: 2026-07-14 (Asia/Shanghai)

Status: `BLOCKED_REPAIR_TUNING_NOT_SELECTED`

## Outcome

The one authorized fresh smaller-step repair was executed with the locked A1
target, original four dispersed starts, `step_size=0.19625`, `8` leapfrog
steps, trajectory length `1.57`, and fresh `repair-01` artifacts. The trusted
GPU/XLA 64-retained/32-burn-in tuning screen produced four finite moving
chains, but it did not pass the prospective all-chain acceptance gate.

Per-chain acceptance was
`[0.984375, 0.890625, 0.9375, 0.953125]`. Chains 0 and 3 exceeded the frozen
upper bound `0.95`. The aggregate acceptance `0.94140625` was within the
aggregate bound, but the contract requires every chain to pass. The tuning
receipt is therefore `NOT_SELECTED`.

The sequential contract stopped before retained acquisition. No repair
`segment-*.json` exists, no tuning sample or final state is admissible for
calibration, and no forecast calibration was run.

This result shows that the smaller-step candidate repaired the previous
zero-movement symptom on this tuning path but over-corrected under the frozen
all-chain acceptance screen. It is a tuning-candidate failure, not an
implementation, target, posterior, HMC-direction, or predictive-validation
failure.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject the smaller-step kernel for retained A4 acquisition and stop the authorized repair | `FAIL`: two per-chain acceptance rates exceed `0.95`; tuning status `NOT_SELECTED` | No hard veto: all four chains moved; samples and telemetry were finite; native divergence was not exposed | Whether an intermediate step size, adaptive warmup, or mass-geometry repair could satisfy both movement and acceptance at the fixed starts | Write a new prospective tuning-repair plan only if the owner authorizes another attempt; do not interpolate post hoc under this result | No posterior incorrectness, HMC-direction rejection, sampler ranking, predictive equivalence, NeuTra readiness, model adequacy, or default readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the emitted tuning artifact: all chains moved; retained samples, target values, final state, and log-accept telemetry were finite; no positive native divergence was available or observed |
| Tuning promotion screen | Failed because chain acceptance `0.984375` and `0.953125` exceeded the prospective `0.95` ceiling |
| Statistically supported ranking | None; the balanced and smaller-step candidates failed different prospective gates on different realized paths, and no uncertainty-supported sampler comparison was run |
| Descriptive-only differences | The smaller-step tuning path had all-chain movement and higher acceptance than the failed balanced retained path; these observations do not rank kernels |
| Default-readiness | Not assessed and not supported |
| Next evidence needed | A separately authorized prospective warmup/kernel-geometry design that can distinguish an intermediate-step repair from adaptive or mass-matrix repair without post-hoc seed hunting |

## Separate Evidence Ledgers

| Ledger | Status | Evidence |
| --- | --- | --- |
| Engineering correctness | `PASSED` | Original-plus-repair focused suite `17/17`; post-review repair suite `7/7`; compile and whitespace checks passed; exact target, geometry, source, receipt, namespace, and no-overwrite bindings |
| Numerical validity | `PASSED_FOR_EMITTED_TUNING_ARTIFACT` | All samples and telemetry finite; four chains moved; trusted output device `GPU:0`; XLA JIT used; private tensor and manifest hashes verified |
| Sampler admission | `FAILED_BEFORE_RETAINED_ACQUISITION` | Tuning candidate was not selected because two chains exceeded the acceptance ceiling; no retained admission diagnostics were eligible |
| Posterior correctness | `NOT_ASSESSED` | A finite moving tuning screen is not posterior-reference evidence |
| Forecast calibration | `NOT_RUN` | Conditional authorization required an admitted retained archive; none exists |
| Scientific interpretation | `BOUNDED_NEGATIVE_RESULT` | Rejects this frozen smaller-step tuning attempt only; it does not reject HMC, NeuTra, the SSL-LSTM target, or moment-law validation |

## Tuning Evidence

| Field | Value |
| --- | --- |
| Public receipt | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/repair-01/tune.json` |
| Public receipt SHA-256 | `c374b2e8197ee39272c020d8bcac6e29d598e28ff8142a708126fe89ada52dde` |
| Private manifest | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/repair-01/private/repair_01_tune_smaller_step_private_manifest.json` |
| Private manifest SHA-256 | `c265c16bb15a95bddd675728efad2898b55c6ccb4185a42bc3c88697095e3fc9` |
| Private retained shard SHA-256 | `919cdeff4e7510507f8c39bfb79fb69940c2cb42089fb4c401e159fa4a54f3d2` |
| Shape | `[64,4,4]` |
| Kernel | `step_size=0.19625`; `num_leapfrog_steps=8`; trajectory length `1.57` |
| Seed | `[20260714,1521]` |
| Starts | Original fixed four dispersed latent states; no failed-attempt state was read |
| Chain movement | `[true,true,true,true]` |
| Per-chain acceptance | `[0.984375,0.890625,0.9375,0.953125]` |
| Aggregate acceptance | `0.94140625` |
| Acceptance criterion | Every chain and aggregate in `[0.20,0.95]` |
| Native divergence | `not_exposed_by_kernel`; this is unavailability, not a zero-divergence claim |
| Log-accept telemetry | `256/256` finite; maximum absolute finite value `1010.3160791363323` |
| Target-log-probability telemetry | `256/256` finite; range `[-44.95143515923172,-37.821134920565555]` |

The private tuning shard and final state are diagnostic artifacts only. They
must not be used as retained HMC draws, current state for another attempt,
forecast-calibration input, confirmation input, or posterior evidence.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `3d353253dc93a102722e00cbca8803a1b3fce7fa` |
| Worktree | Dirty; unrelated Kalman/QR/Sylvester lane changes preserved and untouched |
| Command | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_a4_hmc_repair_2026_07_14.py tune` |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; TensorFlow Probability `0.25.0` |
| Device/JIT | Two RTX 4080 SUPER devices visible; evidence output on `GPU:0`; XLA JIT and TF32 enabled; target tensors `float64` |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Target | `ssl_lstm_completion:a1:masked_svd_ukf_four_parameter`; semantic SHA-256 `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e` |
| This run wall time | `222.9857433269499s` |
| This HMC call | `219.99778789503034s` |
| Prior charged GPU time | `1333.7487312000012s` |
| Total charged GPU time | `1556.7344745269511s` = `0.4324262429241531h` |
| Shared cap | `28800s` = `8h` |
| Unspent budget after stop | `27243.26552547305s` = `7.567573757075847h`; unspent budget is not authority for another attempt |
| Plan | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-repair-plan-2026-07-14.md` |
| Review | `docs/reviews/bayesfilter-ssl-lstm-a4-hmc-repair-native-review-2026-07-14.md` |
| Result | This file |

The first non-trusted invocation failed before HMC with
`CUDA_ERROR_NO_DEVICE`, produced no repair artifact, and is sandbox evidence
only. The identical command was rerun with trusted GPU access as required by
policy. Only the trusted run is scientific GPU evidence and charged HMC work.

## Failure Classification

| Question | Answer |
| --- | --- |
| Did the harness fail? | No. It enforced prior hashes and budget, wrote fresh hash-verified artifacts, classified the candidate `NOT_SELECTED`, and prevented the retained rung. |
| Did the target or numerical path fail? | No. Values, scores, samples, final state, target log probabilities, and log-accept ratios were finite; device and XLA evidence passed. |
| Did the candidate move all chains? | Yes, on this 64-draw tuning path. That repairs the prior tuning symptom but does not establish retained movement or convergence. |
| Did the tuning candidate pass? | No. Two chains exceeded the prospective acceptance ceiling. |
| Did HMC or moment-based validation fail as research directions? | No. The result narrows the kernel/warmup problem but does not test the predictive-moment comparison. |

## Post-Run Red Team

The strongest alternative explanation is that the fixed 64-draw/32-burn-in
screen is noisy near the `0.95` ceiling: chain 3 exceeded it by only
`0.003125`, while chain 0 exceeded it by `0.034375`. That does not permit
waiving the prospective gate, extending the screen, or trying another seed
after observing the result. Acceptance close to one may also indicate a
conservative step rather than invalid sampling, but retained mixing and
stationarity were never evaluated because the tuning gate failed.

An intermediate fixed step could plausibly balance the balanced kernel's
state-dependent rejection against this smaller kernel's over-acceptance.
Adaptive warmup or a better mass matrix could be more principled. Current
evidence does not rank those repairs. A new plan should prospectively select
which mechanism is being tested and should not use the remaining budget as a
reason to search kernels or seeds post hoc.

The weakest evidence remains native-divergence unavailability and the short,
single-seed tuning screen. The result would be overturned only by a fresh,
prospectively authorized design that passes its own four-chain tuning and
retained admission gates; it cannot be overturned by reinterpreting this
failed receipt.

## Stop And Handoff

- Do not run repair segment 0 or any retained extension from this tuning
  receipt.
- Do not use the tuning shard or final state for calibration or another HMC
  attempt.
- Do not run forecast calibration, A5 confirmation, NeuTra training, or
  NeuTra-HMC from this result.
- Another HMC repair requires a new prospective mechanism and explicit owner
  authorization. The scientifically relevant choice is now between an
  intermediate fixed step and an adaptive/mass-geometry repair, not additional
  seed search.
