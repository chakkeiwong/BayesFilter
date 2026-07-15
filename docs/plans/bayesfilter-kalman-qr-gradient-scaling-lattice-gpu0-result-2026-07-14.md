# Kalman QR Gradient Scaling Lattice GPU 0 Result

Date: 2026-07-14

Status: `PASS_COMPLETE_GPU0_MEMORY_GROWTH`

Plan: `docs/plans/bayesfilter-kalman-qr-gradient-scaling-lattice-gpu0-plan-2026-07-14.md`

## Result

The repaired lattice completed on physical GPU 0 under the approved
memory-growth and shared-device admission policy. All 15 schedules passed:
the nine validated CPU schedules were inherited from `r4`, and all six GPU
schedules ran fresh in `gpu0_r2`. The structured summary contains 180 method
records: 108 CPU records plus 72 GPU records. Every method pair passed the
analytical/autodiff comparator and all schedule aggregate checks are true.

The prior `gpu0_r1` attempt remains a separate terminal artifact. Its
`gpu-b4-float64` retry was rejected because a foreign process entered GPU 0;
that was a shared-device overlap veto, not algorithm, XLA, parity, OOM, or
memory-growth evidence. No `r1` record is included in this result.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit the complete CPU/GPU lattice as valid descriptive benchmark evidence | Passed: 15/15 schedules, 180/180 records, 90/90 method pairs, finite expected outputs, direct-output parity, and five warm calls per record | No timeout, crash, non-finite output, dtype/shape failure, parity failure, source drift, GPU overlap, OOM, placement, growth, allocator, or cleanup veto | GPU 0 display contexts make timing and utilization non-exclusive; one run is not uncertainty evidence | Use the artifacts for descriptive scaling inspection; run independent paired replications only if a speed ranking is needed | No statistically supported speed ranking, physical-core pinning claim, universal hardware/framework claim, HMC/posterior/default/production readiness, or scientific validity claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed. All 15 schedule statuses are `passed`; every child aggregate check is true. |
| Statistically supported ranking | None. Five synchronized warm calls per method are within-process repetitions, not independent replications. |
| Descriptive-only differences | Warm timings, first executable call, GraphDef size, batch/dimension/parameter scaling, thread-limit effects, dtype effects, and allocator peaks. GPU 0 also had the two authorized display contexts. |
| Default-readiness | Not established by this benchmark. XLA and memory growth remain owner policies; this artifact validates the requested repaired lattice only. |
| Next evidence needed for ranking | Independent process/seed replications with a predeclared paired uncertainty analysis and a resource policy suitable for timing comparison. |

## Gate And Telemetry Checks

- `status.json`: `complete`.
- Schedules: `15/15` passed; attempts: `15`, with no retry required in `r2`.
- Summary: `180` rows (`108` inherited CPU, `72` fresh GPU).
- Warm executions: exactly five per row (`180/180`).
- GPU logical placement: `/GPU:0` for all `72` GPU records.
- GPU trust basis: `owner_designated_managed_session_visible_gpu_trusted` for all `72` GPU records.
- GPU memory growth: enabled for all `72` GPU records; allocator current and peak telemetry present for all `72`.
- GPU JIT: `jit_compile=true` for all `72`; effective flag `--xla_gpu_enable_triton_gemm=false`.
- GPU output dtypes: `float32` for the three float32 schedules and `float64` for the three float64 schedules.
- All method outputs are finite and have expected dtype/shape metadata.
- Per-schedule aggregate checks: `comparator_parity`, `expected_dtype_shape`, `finite_output_metadata`, `identity_integrity`, `primary_pair_complete`, and `record_integrity` are true for all 15; `gpu_memory_growth` is true for all six GPU schedules.
- Post-run census: GPU 0 returned to the authorized display-only contexts, with about `1188 MiB` used at the final check; no BayesFilter compute context remained.

## Allocator Results

The largest TensorFlow allocator peak was `317,148,928` bytes (about 302.5
MiB), for `batch_native_autodiff_qr_score` at `float64`, `B=16`, `D=30`,
`P=150`. The corresponding allocator current value was `7,422,208` bytes.
The per-schedule peak maxima were:

| GPU dtype | Batch | Analytical peak (bytes) | Autodiff peak (bytes) |
| --- | ---: | ---: | ---: |
| float32 | 1 | 19,993,088 | 19,993,088 |
| float32 | 4 | 36,775,680 | 70,330,112 |
| float32 | 16 | 137,465,344 | 271,683,072 |
| float64 | 1 | 24,155,648 | 40,932,864 |
| float64 | 4 | 74,498,048 | 141,606,912 |
| float64 | 16 | 275,877,376 | 317,148,928 |

These values directly support the memory-growth repair diagnosis: the prior
approximately 30 GiB process reservation was TensorFlow initialization policy,
not the live Kalman tensor footprint measured by the repaired children.

## Descriptive Timing Snapshot

The summary preserves exact per-row timings. Representative medians of the
five warm calls, aggregated over the six `D/P` cells, are shown only as
descriptive observations:

| Device arm | Setting | Batch | Analytical median (s) | Autodiff median (s) |
| --- | --- | ---: | ---: | ---: |
| CPU | 1 thread | 1 | 0.336883 | 0.037177 |
| CPU | 1 thread | 4 | 1.549919 | 0.113172 |
| CPU | 1 thread | 16 | 8.815233 | 0.439186 |
| CPU | 4 threads | 1 | 0.892837 | 0.077601 |
| CPU | 4 threads | 4 | 5.402805 | 0.195096 |
| CPU | 4 threads | 16 | 16.030133 | 0.428928 |
| CPU | 16 threads | 1 | 0.862217 | 0.079528 |
| CPU | 16 threads | 4 | 4.035982 | 0.217441 |
| CPU | 16 threads | 16 | 14.889855 | 0.459197 |
| GPU | float32 | 1 | 0.126616 | 0.159329 |
| GPU | float32 | 4 | 0.259629 | 0.169466 |
| GPU | float32 | 16 | 0.867350 | 0.198613 |
| GPU | float64 | 1 | 0.341743 | 0.308408 |
| GPU | float64 | 4 | 0.678612 | 0.390550 |
| GPU | float64 | 16 | 1.944504 | 0.462856 |

The table is not a ranking or superiority claim. The GPU was shared with
authorized display contexts, and the run has no independent replication or
uncertainty interval.

The CPU thread rows must also not be interpreted as physical-core scaling.
They set both TensorFlow intra-op and inter-op pool limits, without CPU affinity
or NUMA binding, on a shared host. A focused follow-up reproduced the analytical
slowdown with affinity fixed and attributed it primarily to CPU/XLA overhead or
poor scheduling/locality for many small `[B,P,...]` linear-algebra kernels. See
`docs/plans/bayesfilter-kalman-qr-cpu-analytical-scaling-diagnostic-result-2026-07-14.md`.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit recorded at launch | `3d353253dc93a102722e00cbca8803a1b3fce7fa` |
| Command | `PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_kalman_qr_gradient_scaling_lattice_2026_07_14.py --output-root docs/benchmarks/kalman_qr_gradient_scaling_lattice_gpu0_r2_2026-07-14 --inherit-passed-cpu-from docs/benchmarks/kalman_qr_gradient_scaling_lattice_r4_2026-07-14/status.json --method-timeout-seconds 600 --resource-wait-seconds 7200 --resource-poll-seconds 30` |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; Linux managed GPU session |
| Matrix | `T=120`; `D={10,20,30}`; `P={50,150}`; `B={1,4,16}`; CPU threads `{1,4,16}`; GPU dtypes `{float32,float64}` |
| Methods | `batch_native_analytical_qr_score`; `batch_native_autodiff_qr_score` |
| JIT/TF32/XLA | XLA enabled; float32/float64 GPU schedules use the recorded no-Triton flag; TF32 follows the schedule contract |
| GPU | Physical index `0`, logical `/GPU:0`, RTX 4080 SUPER; prelaunch utilization below the authorized 50% gate |
| Warm calls | Five synchronized warm executions per method record |
| Wall time | `2026-07-14T05:49:39.253751+00:00` to `2026-07-14T06:06:52.413338+00:00` (about `1033.16 s`) |
| GPU schedule elapsed sum | `1027.40 s` |
| CPU source | Nine schedules inherited and revalidated from `docs/benchmarks/kalman_qr_gradient_scaling_lattice_r4_2026-07-14/status.json` |
| Output root | `docs/benchmarks/kalman_qr_gradient_scaling_lattice_gpu0_r2_2026-07-14/` |
| Status artifact | `docs/benchmarks/kalman_qr_gradient_scaling_lattice_gpu0_r2_2026-07-14/status.json` |
| Summary artifact | `docs/benchmarks/kalman_qr_gradient_scaling_lattice_gpu0_r2_2026-07-14/summary.json` |
| Plan | `docs/plans/bayesfilter-kalman-qr-gradient-scaling-lattice-gpu0-plan-2026-07-14.md` |

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| `gpu0_r2/status.json` | `ac6a36463f4143e2b0c47b1875a247fc1013a04f59389b39e3e53361c5a0a088` |
| `gpu0_r2/summary.json` | `87111e17dba3c4076085fedb0ad692cf8155b958709bd0ecb9a94e6d8ebd5af0` |
| Inherited `r4/status.json` | `cc02d95a91a08d5997c1bb4011cc4b114cf203c4d817361b2fd58ada5f596d04` |
| Lattice supervisor | `46b399f0d50c1d82f1b03af4050dca557fd43c306a1cafc43aa738df0a4b9b04` |
| GPU0 plan | `78bd3467b5cfa56d5267772637092c31bf266db44b2dd2357bef9e951c9e7919` |

## Metadata Follow-Up

The generic child runner's serialized `execution_contract.trust_basis` retains
the historical label `gpu_hidden_cpu_debug_reference`. This field is stale
metadata, not the admission basis used for this run. The actual GPU method
records independently contain `/GPU:0`, growth telemetry, allocator telemetry,
XLA settings, and the owner-designated managed-session trust basis; the lattice
supervisor validated those fields for all 72 GPU records. A future focused
cleanup may align the generic child contract label, but changing it after the
measurement would not improve this result and was intentionally not performed
post-run.

## Negative-Result Classification

- Implementation failure: none in the completed lattice.
- Numerical failure: none; all records are finite and pass the method-pair comparator.
- Tuning failure: not applicable.
- Diagnostic failure: none in the supervisor or structured child checks.
- Resource failure: the earlier `gpu0_r1` attempt had a shared-device overlap veto; `gpu0_r2` passed its resource contract.
- Evidence against the scientific or engineering direction: none. The result establishes benchmark execution and repaired-path viability only.

## Post-Run Red Team

The strongest alternative explanation for the timing variation is shared display
activity and normal XLA compilation/cache behavior, not a stable method effect.
The result would be overturned as a complete lattice only by discovering a
source/record drift, invalid inherited CPU artifact, missing GPU telemetry,
incorrect device placement, or a failed parity/finite check. None occurred.
The weakest part of the evidence is performance inference from one shared-GPU
run; the strongest part is the complete structured correctness and allocator
contract across all 180 records.

## Handoff

The requested GPU/CPU benchmark is complete for descriptive correctness and
scaling evidence. Preserve `gpu0_r1` as contamination history and `gpu0_r2` as
the admitted result. Do not describe the timing table as a statistically
supported ranking. If ranking is needed later, create a separate experiment
plan for independent paired replications and uncertainty analysis.
