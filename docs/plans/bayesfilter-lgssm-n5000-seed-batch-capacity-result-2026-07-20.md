# LGSSM N=5000 Seed-Batch Capacity Result

Date: 2026-07-20
Status: `MEMORY_FEASIBLE_BATCH8_SEMANTIC_PARITY_FAIL`
Plan:
`docs/plans/bayesfilter-lgssm-n5000-seed-batch-capacity-plan-2026-07-20.md`

## Outcome

Eight concurrent seeds are memory-feasible for the exact
`T=50,N=5000,K=2500,2 x 2` canonical Contract E--Chol value-and-total-score
program. TensorFlow allocator peak was `3,161,939,712` bytes (`3015.46 MiB`),
only `36.81%` of the fixed 8192 MiB logical-device limit. The graph compiled
with XLA and passed finite, bitwise replay, chart, reset, marginal, work, GPU,
TF32, chunk, and scope gates.

Batch eight is nevertheless **not claim-admissible**. Against the same eight
seeds executed as source-matched singleton microbatches, the largest absolute
value difference was `0.2393494` and the largest absolute total-score
difference was `0.3355436`, far above the predeclared `1e-4` parity tolerance.
The value shift was systematic (`0.2368..0.2393`) across all eight seeds.

A CPU-only preparation diagnostic confirmed that each batch row's initial
noise, transition noise, residual design, reset mask, and prepared ridge is
bitwise identical to its corresponding singleton tensor. Therefore the
observed mismatch is inside the batch-size-dependent canonical numerical
computation, not seed generation or prepared inputs.

Follow-up localization resolved the cause category. At `T=2,N=32`, the maximum
same-row batch-versus-singleton discrepancies were:

| Backend | Value | Total score | Final particles |
| --- | ---: | ---: | ---: |
| CPU eager float32 | `0` | `4.77e-7` | `3.58e-7` |
| CPU/XLA float32 | `9.54e-7` | `2.38e-6` | `3.58e-7` |
| GPU/XLA float32, TF32 disabled | `1.91e-6` | `2.86e-6` | `5.36e-7` |
| GPU/XLA float32, TF32 enabled | `1.75e-4` | `2.82e-2` | `8.05e-4` |

The first localized TF32 discrepancy occurs before transport in the diagonal
LGSSM transition matrix application: batch size one versus two differed by up
to `1.21e-3`, while TF32-disabled execution was equal there. Replacing that one
operator with elementwise diagonal multiplication reduced that local source but
did not restore end-to-end parity; other TF32-eligible batched kernels remain
shape-sensitive. The experimental replacement was therefore reverted.

Verdict: this is a **numerical-contract bug**. There is no evidence of seed-row
mixing, and small ordinary floating-point reordering is expected, but the
existing campaign explicitly claimed that microbatching changes only execution
schedule, not the per-seed finite scalar. TF32 batch-shape effects amplified by
the 50-step reset recursion violate that claim by amounts large enough to change
the value/score bias screen.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Record batch eight as memory-feasible | PASS: `3015.46 MiB < 8192 MiB` | No OOM/resource veto | Physical device was shared with another TensorFlow process | Reuse this as capacity evidence for the exact scope | No clean speed claim |
| Do not nominate batch eight for claim-bearing runs | FAIL: same-seed value/score parity | Semantic parity veto fired | Multiple TF32-eligible batched kernels are shape-sensitive | Define and implement a reviewed batch-stable TF32 route, or bind batch geometry as a different finite-program scope and retune it; then rerun parity/claim evidence | No claim-equivalence or bias evidence |
| Keep singleton execution as correctness fallback | Existing singleton claim remains the comparison authority | Size one was not required by memory | Its low parallelism is inefficient | Use only until a larger microbatch passes same-seed parity | No assertion that size one is optimal |

## Engineering Evidence

| Field | Result |
| --- | --- |
| Candidate | Eight concurrent seeds `82220..82227`, controls `(20,5)` |
| Canonical node | `PASS`; `hard_valid=true` |
| Replay | Bitwise exact within the batch-eight graph |
| Maximum `TV_col` | `2.1602459e-06` |
| Maximum `E_row` | `0.007997632` |
| Peak TensorFlow allocator | `3,161,939,712` bytes (`3015.46 MiB`) |
| Memory-limit fraction | `0.3681` of 8192 MiB |
| Maximum value disagreement | `0.2393493652` |
| Maximum total-score disagreement | `0.3355436325` |
| Preparation row parity | Bitwise exact for all five checked prepared tensor families and all eight seeds |
| Timing context | `externally_contended`; another TensorFlow process shared the GPU |
| Batch-eight cold / warm | `973.37 s` / `927.10 s` |
| Matched singleton cold / warm | `452.43 s` / `441.92 s` |
| Speed interpretation | Batch eight was about `0.47x` the singleton throughput in this contended observation; this is descriptive and not a clean speed comparison |
| Wall time | `1919.62 s` |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Resource and canonical node gates pass; same-seed semantic parity fails. |
| Statistically supported ranking | None; this is a deterministic paired engineering test under external contention. |
| Descriptive-only differences | Observed timing and throughput under contention. |
| Default readiness | Batch eight is not ready for claim-bearing or bias-comparison use. |
| Next evidence needed | A reviewed batch-stability repair and a fresh same-seed parity run at increasing `T,N`; alternatively, an explicit new tuning scope that admits batch shape as part of the finite numerical program. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | Recorded in the run manifest and terminal JSON |
| Environment | `tf-gpu`, TensorFlow `2.19.1` |
| Hardware | NVIDIA GeForce RTX 4080 SUPER; externally contended |
| Backend | GPU, float32, TF32 enabled, XLA JIT |
| Scope | `T=50,N=5000,K=2500`, `2 x 2`, scope SHA-256 `77b2dcc58c3f716cadee575d394e3c94e3a94be2afda22e4cc2fabe65e2521f7` |
| Seeds | `82220..82227` |
| Result | `docs/benchmarks/artifacts/lgssm_n5000_seed_batch_capacity_20260720/attempt01/result.json` |
| Manifest | `docs/benchmarks/artifacts/lgssm_n5000_seed_batch_capacity_20260720/attempt01/run_manifest.json` |
| Result SHA-256 | `b56c368c9bafeb707bef564efe8c008fbfa685c6e379a767a9c45ea747001416` |
| Manifest SHA-256 | `92434047f1d5789dcd04fe1ba7dde95be081e5bdd433920baa33c654b1c0e3d2` |

## Post-Run Red Team

The diagnostics support float32/TF32 GPU kernel selection, not an indexing or
RNG bug. That does not rescue claim equivalence: the computed batch-eight finite
program is measurably different from the singleton program used for the
existing bias evidence. A reviewed repair that restores same-seed parity would
overturn the current non-nomination. The weakest evidence is speed because the
GPU was shared; allocator capacity and deterministic parity are the stronger
outputs.
