# GPU Memory Growth Kalman Canary Result

Date: 2026-07-14

Status: `PASS_MEMORY_GROWTH_REPAIR`

Plan:
`docs/plans/bayesfilter-gpu-memory-growth-kalman-canary-plan-2026-07-14.md`

## Result

The full-device allocation was a TensorFlow initialization bug, not live
Kalman tensor memory. The repaired smallest-cell GPU/XLA canary completed both
methods with memory growth enabled, finite correctly shaped outputs, five warm
calls, and analytical/autodiff parity.

| Method | TensorFlow allocator current | TensorFlow allocator peak | Result |
| --- | ---: | ---: | --- |
| `batch_native_analytical_qr_score` | 157,696 bytes | 564,992 bytes | Passed |
| `batch_native_autodiff_qr_score` | 157,696 bytes | 2,064,128 bytes | Passed |

Before repair, an identical-cell child initialized TensorFlow without memory
growth and NVIDIA reported about 30,724 MiB assigned to the process. During the
repaired real canary, an independent live sample showed total GPU 0 memory at
about 1,516 MiB versus a prelaunch display baseline of about 1,240 MiB, an
increase of roughly 276 MiB. After completion it returned to about 1,241 MiB
and only the pre-existing display contexts remained.

TensorFlow still logged `Created device ... with 29194/29196 MB memory`. Under
memory growth this is the allocator's device limit, not current allocation.
The in-process allocator telemetry and independent NVIDIA delta demonstrate
that the benchmark no longer reserves the full device.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Adopt required GPU memory growth and keep the repaired Kalman path | Passed for the exact smallest cell and both methods | No OOM, timeout, CPU placement, numerical, parity, missing-telemetry, or cleanup veto | Peak memory of larger `D/P/B/float64` cells remains unmeasured | Run an upward capacity ladder with the same allocator telemetry before completing the GPU lattice | No largest-cell capacity, timing ranking, production readiness, HMC/posterior correctness, or scientific claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for `D=10,T=120,P=50,B=1,float32`; both methods and comparator passed. |
| Statistically supported ranking | None. Timings are descriptive on a shared display GPU. |
| Descriptive-only differences | First/warm calls, allocator magnitudes, GraphDef sizes, and NVIDIA utilization. |
| Default-readiness | Memory growth is an owner policy and resource-safety repair; this canary does not validate every GPU path in the repository. |
| Next evidence needed | Prospective upward `B/D/P/dtype` capacity ladder, stopping on allocator/OOM/cleanup veto, then the six-schedule GPU lattice. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit recorded by runner | `3d353253dc93a102722e00cbca8803a1b3fce7fa` |
| Device | Physical GPU 0 exposed as logical `/GPU:0`; RTX 4080 SUPER |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Prelaunch gate | 40% utilization, below owner-authorized 50%; 1,240 MiB used; only `gnome-remote-desktop-daemon` and `nxnode.bin` contexts |
| Environment | `CUDA_VISIBLE_DEVICES=0`; `TF_FORCE_GPU_ALLOW_GROWTH=true`; one TensorFlow/OpenMP thread |
| Compiler | `jit_compile=true`; TF32 enabled; effective `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` |
| Problem | Deterministic fixture; `D=10,T=120,P=50,B=1,float32`; no random seed |
| Methods | True-batched analytical and reverse-mode autodiff QR score |
| Warm calls | Five per method |
| Method elapsed | Analytical 9.535 s; autodiff 9.376 s, descriptive only |
| Artifact | `docs/benchmarks/kalman_qr_gpu_memory_growth_canary_2026-07-14/status.json` |
| Artifact SHA-256 | `24c5928bc960ad261903cb3d4a5636dd6cf60956770b5afabbd0693f1fd1f134` |

The executed measurement sources are pinned in `schedule.json`. In particular,
the executed Kalman child script hash was
`18ba3042c3b9c7261fb6f0fb290eae324e4169e1593b0ab001f01f8c8d03b4e3`
and the executed method-isolated runner hash was
`71bac42f5e026ed381d10acbce964621f41a3911e3d7d82ccae006bba93a585a`.
After the run, only the runner's result-admission logic was hardened to require
the already-recorded growth/allocator fields; the TensorFlow execution path was
not changed or rerun. The current validator accepts the pinned canary records
with `gpu_memory_growth=true`.

Exact command:

```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true \
OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --dimensions 10 --parameter-counts 50 --timesteps 120 --batch-size 1 \
  --dtype float32 --device gpu --cpu-threads 1 --repeats 5 \
  --timeout-seconds 600 \
  --methods batch_native_analytical_qr_score batch_native_autodiff_qr_score \
  --output-dir docs/benchmarks/kalman_qr_gpu_memory_growth_canary_2026-07-14 \
  --plan-path docs/plans/bayesfilter-gpu-memory-growth-kalman-canary-plan-2026-07-14.md \
  --result-path docs/plans/bayesfilter-gpu-memory-growth-kalman-canary-result-2026-07-14.md \
  --no-resume --jit-compile --tf32-enabled
```

## Checks

- Focused CPU-hidden suite: 71 tests passed before the canary and 71 passed
  after the admission-gate hardening.
- Structured status: `complete`, two records, all aggregate checks true,
  comparator complete and parity true.
- Both records: logical `/GPU:0`, growth enabled, allocator telemetry present,
  XLA compiled, no-Triton policy applied, finite output, five warm calls.
- Trusted post-run census: no Kalman context; GPU 0 returned to display-only
  baseline.

## Post-Run Red Team

The strongest alternative explanation is that the smallest cell understates
larger-cell XLA temporaries. That does not weaken the narrow repair result:
full-device reservation occurred before work under the old initialization,
whereas the repaired identical cell used bounded growth and released cleanly.

The result would be overturned if a fresh child with the repaired policy again
reserved nearly all VRAM before executing, if telemetry were absent or
inconsistent, or if cleanup left a context. None occurred. The weakest evidence
is capacity beyond the smallest cell; the strongest is agreement between
per-process TensorFlow allocator telemetry and independent live/post-run
NVIDIA census.
