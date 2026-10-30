# KDM trained-checkpoint update localization result — 8 October 2026

The tested TensorFlow and author updates agree in float64 at both completed
training origins. Float32 MDPF gradient and optimizer-moment discrepancies remain.
The experiment completed successfully; the strict float32 numerical screen did
not pass. Both replications remain incomplete.

## Target, actual computation and evidence

The target was the same author full-sequence training loss, IWSG derivatives and
one Adam update at the actual learned MDPF and A-MDPF parameters. The independent
CPU author worker captured its native float32 particles and proposal/resampling
draws, then exactly replayed them before casting those identical inputs to
float64. All 32 exact replay controls pass. TensorFlow used its shared GPU/XLA
training kernel. All tested loss, forward, parameter-update and second-moment
checks pass under the unchanged bounds; the float64 gradient match supports the
mathematical translation on these cases. This does not prove the whole training
trajectory or stochastic sampling law equivalent.

Evidence: [run manifest](artifacts/controlled-smc-kdm-replication-20260928/kdm-trained-update-20261008-01/manifest.json),
[comparison](artifacts/controlled-smc-kdm-replication-20260928/kdm-trained-update-20261008-01/tf/comparison.json),
[precision analysis](artifacts/controlled-smc-kdm-replication-20260928/kdm-trained-update-20261008-01/precision-analysis.json)
and [verified summary](artifacts/controlled-smc-kdm-replication-20260928/kdm-trained-update-reviewed-20261008.json).
The [pre-run plan](controlled-smc-kdm-trained-update-localization-20261008.md)
records source anchors, defaults, fixed data rows 384:448, seed 202610086,
B=64, T=17 and N=25. The run manifest preserves code/input hashes and commands.

## Results

| Checkpoint / variant / precision | Passed checks | Trace count |
|---|---:|---:|
| source_trained/mdpf/float32 | 207/211 | 1 |
| source_trained/mdpf/float64 | 211/211 | 1 |
| tf_trained/mdpf/float32 | 185/211 | 1 |
| tf_trained/mdpf/float64 | 211/211 | 1 |
| source_trained/am_dpf/float32 | 253/253 | 1 |
| source_trained/am_dpf/float64 | 253/253 | 1 |
| tf_trained/am_dpf/float32 | 253/253 | 1 |
| tf_trained/am_dpf/float64 | 253/253 | 1 |

There are 30 failed float32 checks: 14 raw-gradient tensors, the corresponding
14 clipped-gradient tensors, one gradient norm and one Adam first moment.
The largest raw-gradient absolute difference is 0.0003533363; the largest
component exceeds the existing scaled screen by a factor of 6.2874. Bounds are
unchanged. Both A-MDPF origins pass in both precisions.

The following maximum gradient errors use author float64 on the exact same
float32 inputs as a numerical reference. They are descriptive observations.

| Checkpoint / variant | Author float32 error | TF float32 error | TF float64 error |
|---|---:|---:|---:|
| source_trained/mdpf | 9.16603e-05 | 6.90105e-05 | 2.27374e-13 |
| source_trained/am_dpf | 1.76213e-06 | 1.82233e-06 | 4.19109e-15 |
| tf_trained/mdpf | 0.000774788 | 0.000421452 | 5.01821e-14 |
| tf_trained/am_dpf | 4.61914e-06 | 8.29036e-06 | 1.66533e-14 |

The largest checked float64 endpoint difference is 2.27374e-13. For the two
selected MDPF checkpoints, TF float32 has a smaller maximum gradient error than
author float32 relative to that reference. This does not establish a general
accuracy ranking or show that rounding caused the downstream quality gap.

The author worker intentionally hid GPU devices. TF used the RTX 5080,
TensorFlow 2.20.0-dev0+selfbuilt, XLA enabled, TF32 disabled and verified memory
growth. Peak TF allocator usage was 278500864 bytes. Each case traced once.
The generic warning about several newly created functions is not evidence of
repeated tracing within one case.

## Decision and inference

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve float32 failure; continue replication diagnosis | 928/928 float64 and 898/928 float32 checks pass | Float32 MDPF promotion veto remains; no continuation veto | Accumulated rounding versus ordinary training variability | Fixed source-only independent-training calibration | Replication complete or quality gap explained |

| Inference class | Status |
|---|---|
| Hard veto screen | Strict float32 MDPF numerical screen failed; exact source controls and finite-value checks pass |
| Statistically supported ranking | None; four selected checkpoints and one batch are not independent training-population evidence |
| Descriptive-only differences | Precision errors above and earlier one-pair downstream quality gap |
| Default-readiness | Not established; no algorithm, default or tolerance changed |
| Next evidence needed | Source-only training variance calibration, independently justified numerical accuracy criteria, then untouched trained-unit qualification |

Post-run skeptical review: the strongest alternative explanation is that a
single batch misses a trajectory-dependent implementation problem. Fresh zero
Adam moments do not reproduce final optimizer history. Exact draw replay also
does not test native random-number generation. Agreement here narrows the
algebraic hypothesis but cannot erase those limitations. A mismatch on additional
matched valid inputs, optimizer history or sampling laws would reopen that
hypothesis. No author-code repair or superiority claim follows.

## Accounting and next action

CPU author worker: 9.3875287422 seconds; GPU worker: 82.9294193760 seconds.
Charge 129.3875287422 CPU seconds including the plan's 120 routine seconds,
and 82.9294193760 GPU seconds once in reference-diagnostics.json. One of two
permitted attempts used; fixed diagnostic stops here. Remaining allocations are
reported by the live checkpoint, not added by this phase.

Next: source-only independent-training variability calibration, with frozen
source hyperparameters and data, followed by the same author evaluator.
A small calibration cannot certify the accepted 0.01-nat/1% margins.
