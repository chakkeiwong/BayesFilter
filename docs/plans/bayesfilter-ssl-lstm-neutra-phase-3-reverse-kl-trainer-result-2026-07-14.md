# SSL-LSTM NeuTra Phase 3 Reverse-KL Trainer Result

Date: 2026-07-14

Status: `PHASE_3_REVERSE_KL_TRAINER_CORRECTNESS_PASSED`

## Outcome

BayesFilter now has a TensorFlow reverse-KL trainer for diagonal-affine and
one-stage dense-IAF transports. It uses stateless initialization/base noise,
manual Adam state, global-norm clipping, stable restart keys/hashes, and frozen
transport export. The target value/score is evaluated outside the optimizer
tape and stopped; the tape recomputes only the transport/logdet surrogate, so
it cannot differentiate through the SSL-LSTM filter.

## Decision Table

| Decision | Primary status | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept Phase 3 | `32/32` final CPU-hidden controls and final trusted actual-target GPU/XLA canary passed | No sign, reduction, batch, gradient, clipping, state, snapshot, target-boundary, placement, finiteness, or XLA veto | Reverse KL can still mode-seek; two updates do not assess transport quality | Obtain prospective Phase 4 material-training budget and run the bounded candidate ladder | Learned quality, sampler validity, posterior correctness, superiority, predictive equivalence, or readiness |

## Correctness Evidence

| Contract | Evidence |
| --- | --- |
| Objective sign | Diagonal-Gaussian analytic gradient matches; an intentionally reversed score produces the opposite gradient and fails the expected contract |
| Target-score boundary | Curved-ridge manual-score transport gradient matches debug full autodiff; production tape watches transport variables only |
| Mean reduction | Row duplication and batch permutation leave loss gradients invariant for affine and dense-IAF families |
| Optimizer | Finite Adam update, global-norm clipping, step counter, and nonzero parameter changes pass |
| Restart | Restored state reproduces the next update exactly; tampered hash rejects |
| Frozen handoff | Exported payload reloads through Phase 2 and exactly replays forward/logdet |
| Seeds | Training `2101` replays; validation `2201` is distinct |
| Fail closed | Nonfinite target and wrong target shapes reject |
| Default route | Trainer configuration defaults to `jit_compile=True`; no serious CPU fallback exists |

Gaussian and curved-ridge targets are implementation controls on separate known
problems, not SSL-LSTM posterior references.

## Final Trusted Canary

Final receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-3/trainer-gpu-xla-canary-r3.json`,
SHA-256 `550fcf533bbbb4a91c5469abe49db99bf17bf14a0ea8560e135c3230bb42f3a9`.

The canary ran exactly two updates with batch size `4` and role seed
`[20260714,2101]` against locked target semantic SHA-256
`549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e`.
It used conda `tfgpu`, Python `3.13.13`, TensorFlow `2.20.0`, `float64`, TF32,
two visible RTX 4080 SUPER devices, output on `GPU:0`, soft placement disabled,
`jit_compile=True`, and trust basis
`owner_designated_managed_session_visible_gpu_trusted`.

| Metric | Value | Role |
| --- | --- | --- |
| Compile plus first step | `100.51297287503257s` | Engineering/throughput diagnostic |
| Steady second step | `0.04816120304167271s` | Engineering/throughput diagnostic |
| First/second loss | `75.8418348218555` / `75.68918977626205` | Descriptive only |
| First/second raw gradient norm | `72.18139991747977` / `71.97144013117304` | Descriptive only |
| First/second clipped norm | approximately `10.0` / `10.0` | Clipping contract passed |
| Frozen forward/logdet replay | exact zero residual | Engineering gate |

All returned and host-synchronized diagnostics were finite and GPU-resident.
TensorFlow logged actual XLA cluster compilation. XLA also warned that internal
assert ops were ignored; therefore future serious runners must synchronize and
perform host-side finite checks at every recorded step/checkpoint rather than
relying only on in-graph assertions.

Final source bindings:

| Path | SHA-256 |
| --- | --- |
| `bayesfilter/inference/neutra_training.py` | `ed45945097d4cae35012b4d5eb16d19127f838f4d629eff24cc81bd9d15891e0` |
| `tests/test_neutra_reverse_kl_training.py` | `f9b300a382cfe766a3bb4ba8111620cf090b387115e594b81133f6db2cb392f7` |
| Actual-target canary runner | `4b5772b0e5b732155b2f32365c3ba569be7e23ec3e0baff2c72ffc000cd421d0` |
| Locked target source | `6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667` |

## Review And Red Team

Review caught and repaired three issues before close: process-local TensorFlow
variable names were replaced by stable logical restart keys; nonfinite target
checks were moved before total-loss checks for correct failure attribution; and
the negative sign control plus host-synchronized XLA finite checks were added.
All focused checks and the source-bound trusted receipt were rerun.

Strongest alternative explanation: the trainer may be mathematically correct
yet learn a narrow reverse-KL mode that fails dispersed-start or tail coverage.
Phase 4 treats that as a candidate-selection/repair risk and uses independent
seeds and frozen sensitivity probes. Loss reduction alone cannot promote a
candidate.

## Handoff

Phase 4 may execute only under its prospective resource and selection plan. It
must preserve independent A/B training (`2101`, `2102`), independent validation
(`2201`, `2202`), host finite checks, immutable failed artifacts, and the rule
that the enhanced topology is attempted only after the plain arm is evaluated.

