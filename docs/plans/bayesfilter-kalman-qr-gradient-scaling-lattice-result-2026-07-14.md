# Kalman QR Gradient Scaling Lattice Result

Date: 2026-07-14

Status: `BLOCKED_GPU_RESOURCE_NOT_EXCLUSIVE`

Plan:
`docs/plans/bayesfilter-kalman-qr-gradient-scaling-lattice-live-plan-2026-07-14.md`

## Result

The `r6` continuation stopped at its predeclared resource veto. Physical GPU 1
did not become exclusive during the 7,200-second prelaunch window, so no `r6`
GPU benchmark child was launched and no GPU timing was admitted.

The terminal artifact preserves and revalidates the nine accepted `r4` CPU
schedules: 108 method records covering all `D/P` cells, both methods, all three
batch sizes, and TensorFlow thread limits 1, 4, and 16. Every inherited record
has return code zero, finite output, direct-output parity, XLA enabled, five
synchronized warm calls, and CPU placement; every schedule aggregate check is
true. This is valid CPU/XLA correctness and descriptive timing evidence, but it
does not satisfy the full 180-record lattice criterion.

The six planned GPU schedules remain unmeasured:
`gpu-b1-float32`, `gpu-b4-float32`, `gpu-b16-float32`,
`gpu-b1-float64`, `gpu-b4-float64`, and `gpu-b16-float64`. Only the first is
materialized as a zero-attempt pending schedule in `r6/status.json`; the other
five were not reached because the supervisor stopped at the first continuation
veto.

The final resource snapshot at `2026-07-13T21:25:29.348867+00:00` reported
physical GPU 1 UUID `GPU-a1ea1946-07c0-8ed5-2ba1-d96f82c89cd3` at 30,748 MiB
used with foreign PID 3705283 holding 30,724 MiB. One-minute host load was
5.74, below the load-64 veto. This is a resource-exclusivity blocker, not an
XLA compilation, numerical, placement, cleanup, target, or method failure.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep the repaired supervisor and CPU evidence; do not close or interpret the full lattice | Incomplete: 108/180 method records and 54/90 method-pair cells are admitted; all admitted CPU records pass | Continuation veto fired before GPU launch: GPU 1 remained nonexclusive for 7,200 seconds. No method or parity veto fired | When an exclusive GPU window will be available; no `r6` GPU measurement exists | Continue under the same frozen contract in a fresh result root after trusted census shows GPU 1 exclusive; inherit only the same pinned 108 CPU rows and rerun all six GPU schedules | No full-lattice pass, GPU/XLA result from `r6`, speed ranking, default/production readiness, HMC/posterior correctness, or scientific conclusion |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | The resource-exclusivity continuation veto fired. Within the admitted CPU subset, there was no crash, timeout, non-finite output, dtype/shape failure, direct-output failure, or analytical/autodiff parity failure. |
| Statistically supported ranking | None. The run was not designed to support a performance ranking, and five within-process warm calls are not independent replications. |
| Descriptive-only differences | CPU first-call, warm-call, GraphDef, thread-limit, dimension, parameter-count, and batch-size values in `summary.json` are descriptive only. No GPU values were admitted. |
| Default-readiness | Not established. XLA remains the repository default by owner directive, not by this incomplete benchmark. |
| Next evidence needed | Six clean GPU schedules under exclusive physical GPU 1, producing 72 valid records and 36 valid analytical/autodiff pairs, followed by full 180-record validation. Independent-process paired replications with prospective uncertainty analysis would be required for any ranking. |

## Evidence Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | The supervisor terminated fail-closed with `blocked_resource_not_exclusive`, exit code 2, and a structured final snapshot. Nine inherited schedule statuses and the measurement-source hashes revalidated. |
| Numerical validity | The 108 CPU records are finite and pass direct-output and pair-comparator checks on the deterministic fixture. GPU numerical validity was not tested by `r6`. |
| Performance | CPU timings and graph sizes are preserved as descriptive observations. The requested CPU/GPU lattice is incomplete and no method ranking is supported. |
| Scientific interpretation | The resource result says nothing against the Kalman target or either gradient mechanism. The next phase remains viable because the failure occurred before any GPU method launch. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit recorded at launch | `3d353253dc93a102722e00cbca8803a1b3fce7fa` |
| Command | Exact command in the live plan and below |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; Python 3.13.13; Linux 6.8.0-124-generic x86_64 |
| CPU/GPU status | CPU records deliberately hide GPU with `CUDA_VISIBLE_DEVICES=-1`; `r6` GPU launch blocked because physical GPU 1 was occupied |
| JIT/TF32/XLA | All admitted CPU records: `jit_compile=true`, TF32 disabled, input `XLA_FLAGS=UNSET`; no `r6` GPU setting was executed |
| Data/fixture | Deterministic generated LGSSM fixture; seed `null`; no external input data |
| Matrix | `T=120`; `D={10,20,30}`; `P={50,150}`; `B={1,4,16}`; CPU thread limits `{1,4,16}`; GPU dtypes `{float32,float64}` |
| Methods | `batch_native_analytical_qr_score`; `batch_native_autodiff_qr_score` |
| Warm calls | Exactly five synchronized warm executions for every admitted record |
| `r6` wall time | `2026-07-13T19:25:02.025465+00:00` to `2026-07-13T21:25:29.590599+00:00` (about 7,227.6 seconds) |
| Inherited CPU measurement wall time | Accepted schedule elapsed-time sum 4,038.54 seconds; originally measured from `2026-07-13T16:51:56.019662+00:00` through `2026-07-13T18:21:31.615766+00:00` |
| Trust basis | CPU debug/reference exception for inherited CPU rows; no managed-session GPU measurement admitted in `r6` |
| Plan | `docs/plans/bayesfilter-kalman-qr-gradient-scaling-lattice-live-plan-2026-07-14.md` |
| Status artifact | `docs/benchmarks/kalman_qr_gradient_scaling_lattice_r6_2026-07-14/status.json` |
| Summary artifact | `docs/benchmarks/kalman_qr_gradient_scaling_lattice_r6_2026-07-14/summary.json` |
| Result | This file |

Exact command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_gradient_scaling_lattice_2026_07_14.py \
  --output-root \
    docs/benchmarks/kalman_qr_gradient_scaling_lattice_r6_2026-07-14 \
  --inherit-passed-cpu-from \
    docs/benchmarks/kalman_qr_gradient_scaling_lattice_r4_2026-07-14/status.json \
  --method-timeout-seconds 600 \
  --resource-wait-seconds 7200 \
  --resource-poll-seconds 30
```

The command exited 2 after writing terminal status
`blocked_resource_not_exclusive` and a 108-row summary.

## Artifact Integrity

| Artifact | SHA-256 |
| --- | --- |
| `r6/status.json` | `9ab6a71f6380c0e039790c743908ff356e0f15993fcb10e7ae95bb20e097eee3` |
| `r6/summary.json` | `c265be4c57cd9c816f862cb1ef7beadfb5b1b09ce11cfbe938488a2fe8fac968` |
| Inherited `r4/status.json` | `cc02d95a91a08d5997c1bb4011cc4b114cf203c4d817361b2fd58ada5f596d04` |
| Lattice supervisor | `5bdbf54134367dcefd2d82b32f63444a9db1faae1a8f7bd0f01cc8b1087e5fe9` |
| Frozen live plan at launch | `386da598a609fb2481b9bba7446407fad3c03b0ec4f778d3e6029645c6b0dc2e` |

The six measurement-affecting hashes match the source manifest recorded in
`r6/status.json`. No Kalman mathematics changed during `r6`.

## Negative-Result Classification

- Implementation failure: none observed in `r6`; no GPU child launched.
- Numerical failure: none in the 108 admitted CPU records.
- Tuning failure: not applicable.
- Diagnostic failure: none in the terminal resource classification; trusted
  GPU census and the supervisor agreed on the foreign context.
- Resource failure: yes. Physical GPU 1 did not become exclusive within the
  prospective two-hour wait budget.
- Evidence against the scientific or engineering direction: none. The full
  benchmark question remains unanswered because the missing device arm was not
  executed.

## Post-Run Red Team

The strongest alternative explanation is simply scheduling: another lane's
valid long-running workload occupied the designated GPU for the entire window.
The low host load does not rescue the run because GPU exclusivity, not CPU
load, was the binding comparator condition. Conversely, the two-hour timeout
does not show that the GPU/XLA path is slow or broken because compilation never
started.

The result would be overturned by a clean continuation that obtains the 72
GPU records and passes all schedule, placement, cleanup, finite-output, and
pair-parity checks. The weakest part of the overall lattice evidence is the
missing GPU arm. The strongest part is the structurally revalidated 108-record
CPU subset and the fail-closed proof that no contaminated GPU timing entered
the summary.

## Handoff

Do not edit the frozen execution-affecting files before continuation. First
confirm in a trusted census that physical GPU 1 has no compute PID, at most
512 MiB used, and at most 5% utilization. Then launch a fresh root under the
same evidence and resource contract, inheriting only the pinned `r4` CPU
statuses and rerunning all six GPU schedules. If another two-hour exclusivity
window expires, retain that as a resource blocker and coordinate a GPU window;
do not weaken ownership, cleanup, placement, parity, or XLA gates.
