# Kalman QR Core/Batch/Dtype Grid Result

Date: 2026-07-09

## Decision

`PARTIAL_CPU_ARTIFACT_COMPLETE_GPU_XLA_AUTODIFF_BLOCKED`

The requested grid was started under a governed benchmark subplan.  The harness
was extended to record `--cpu-threads`, and the CPU thread-control preflight
passed.  One full CPU artifact completed for `cpu_threads=1`, `batch_size=1`,
`dtype=float32`, `T=120`, dimensions `10,20,30`, and parameter counts `50,150`.

The full requested CPU grid was not completed in this visible turn because the
first 6-row artifact took `971.9157942309976` seconds.  Running the remaining
CPU thread/batch artifacts sequentially is an overnight-scale job.  The GPU
grid is blocked for the current three-arm harness because trusted GPU/XLA
preflight fails in `autodiff_row_loop_qr_score`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | How do descriptive JIT warm-call runtimes vary across dimensions, parameter count, batch size, CPU thread count, and GPU dtype? |
| Baseline/comparator | `batch_native_analytical_qr_score`, `scalar_analytical_row_loop`, and `autodiff_row_loop_qr_score`. |
| Primary criterion | Every launched row passes dtype, shape, finite-output, and parity checks and records thread/device/dtype/JIT provenance. |
| Veto diagnostics | Row parity failure, nonfinite timed output, dtype mismatch, missing provenance, TensorFlow GPU invisibility, GPU/XLA compile failure, or timeout without artifact. |
| Explanatory diagnostics | Warm medians, compile+first time, warm-start time, row subprocess metadata, TF32 flag, CPU thread manifest, and device placement. |
| Not concluded | Statistical ranking, universal superiority, production/default readiness, HMC readiness, posterior correctness, or scientific validity. |
| Artifact | JSON/Markdown/log artifacts plus this result note. |

## Completed Artifacts

- Subplan: `docs/plans/bayesfilter-kalman-qr-core-batch-dtype-grid-subplan-2026-07-09.md`
- CPU preflight JSON/Markdown:
  - `docs/benchmarks/kalman_qr_core_batch_grid_preflight_cpu_threads1_float32_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_preflight_cpu_threads1_float32_2026-07-09.md`
- GPU preflight logs/artifacts:
  - `docs/benchmarks/logs/kalman_qr_core_batch_grid_preflight_gpu_float32_2026-07-09.log`
  - `docs/benchmarks/kalman_qr_core_batch_grid_preflight_gpu_float32_xla_autotune0_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_preflight_gpu_float32_xla_autotune0_2026-07-09.md`
  - `docs/benchmarks/logs/kalman_qr_core_batch_grid_preflight_gpu_float32_xla_autotune0_2026-07-09.log`
- Completed CPU artifact:
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads1_batch1_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads1_batch1_xla_2026-07-09.md`
  - `docs/benchmarks/logs/kalman_qr_core_batch_grid_cpu_threads1_batch1_xla_2026-07-09.log`

## Completed CPU Rows

All rows below used `CUDA_VISIBLE_DEVICES=-1`,
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, `jit_compile=True`,
`dtype=float32`, `T=120`, `batch_size=1`, and `cpu_threads=1`.  The CPU thread
manifest records TensorFlow intra-op/inter-op thread counts as `1` and
`OMP_NUM_THREADS=1`.

| dims `(n,m)` | params | parity | batch-native warm median s | scalar analytical row-loop warm median s | autodiff row-loop warm median s |
| --- | ---: | --- | ---: | ---: | ---: |
| `(10,10)` | 50 | `True` | `0.1764923520386219` | `0.20492676395224407` | `0.019021357991732657` |
| `(10,10)` | 150 | `True` | `0.5623855529702269` | `0.7172949240193702` | `0.019179252034518868` |
| `(20,20)` | 50 | `True` | `0.3474214029847644` | `0.3938408530084416` | `0.04053309897426516` |
| `(20,20)` | 150 | `True` | `1.1760862480150536` | `1.4606184359872714` | `0.04051107901614159` |
| `(30,30)` | 50 | `True` | `0.6281680039828643` | `0.7019986739614978` | `0.06928448600228876` |
| `(30,30)` | 150 | `True` | `2.1426030439906754` | `2.456200279004406` | `0.0709458669880405` |

These timings are descriptive only.  They do not support a speed ranking.

## GPU Status

Trusted GPU visibility probe with
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python` showed two logical GPUs.  The
non-escalated GPU path could not see GPUs and was treated as sandbox evidence
only.

Trusted GPU/XLA preflight then failed:

- Default XLA GPU preflight aborted in GEMM fusion autotuning with
  `FAILED_PRECONDITION: Can not combine dim orders and requirements`.
- Retrying with `XLA_FLAGS=--xla_gpu_autotune_level=0` avoided the process
  abort but the structured row still failed with
  `FailedPreconditionError` in `__inference_autodiff_row_loop_score`.

Therefore the requested GPU `float32` versus `float64` grid is blocked for the
current three-arm harness.  A different GPU-only analytical benchmark contract
would be a new benchmark, not the same requested analytical/autodiff comparison.

## Checks Run

| Check | Result |
| --- | --- |
| `python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py` | passed |
| `git diff --check -- scripts docs/benchmarks docs/plans bayesfilter/linear tests` | passed |
| CPU thread preflight | passed; requested and actual TensorFlow/env thread settings recorded as `1` |
| Trusted GPU visibility probe | passed; two TensorFlow logical GPUs visible |
| GPU three-arm XLA preflight | blocked in `autodiff_row_loop_qr_score` |
| CPU `threads=1,batch=1` artifact | passed; 6/6 applicable rows parity passed |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| `PARTIAL_CPU_ARTIFACT_COMPLETE_GPU_XLA_AUTODIFF_BLOCKED` | Passed for launched CPU rows; not complete for full requested grid | GPU/XLA autodiff preflight veto fired | Whether GPU XLA failure is specific to the autodiff row-loop comparator or repairable with kernel/flag changes | Resume remaining CPU artifacts as overnight sequential jobs; separately repair or redesign GPU comparator contract | No full CPU table, no GPU dtype comparison, no speed ranking, no production/default readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | CPU launched rows passed; GPU three-arm preflight blocked. |
| Statistically supported ranking | Not assessed. |
| Descriptive-only differences | Completed CPU `threads=1,batch=1` warm medians only. |
| Default-readiness | Not assessed. |
| Next evidence needed | Remaining CPU thread/batch artifacts and repaired GPU three-arm XLA preflight. |

## Resume Commands

Run remaining CPU artifacts sequentially to avoid confounding the CPU thread
comparison.  Replace `{threads}` with `1`, `4`, `16` and `{batch}` with `1`,
`4`, `16`, skipping the completed `threads=1,batch=1` artifact:

```bash
CUDA_VISIBLE_DEVICES=-1 timeout 43200 /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 20 30 --parameter-counts 50 150 --timesteps 120 --repeats 1 --batch-size {batch} --device cpu --jit-compile --dtype float32 --cpu-threads {threads} --isolate-each-row --row-subprocess-timeout-seconds 3600 --output-json docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads{threads}_batch{batch}_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads{threads}_batch{batch}_xla_2026-07-09.md > docs/benchmarks/logs/kalman_qr_core_batch_grid_cpu_threads{threads}_batch{batch}_xla_2026-07-09.log 2>&1
```

Do not launch the GPU dtype grid until the three-arm GPU/XLA preflight passes
for `autodiff_row_loop_qr_score` or the user approves a new GPU benchmark
contract that excludes or changes that comparator.

## Overnight Launch

After the partial result above, the user explicitly approved an overnight run.
The detached runner was launched as:

```bash
systemd-run --user --unit=kalman-qr-core-batch-grid-20260709 --working-directory=/home/ubuntu/python/BayesFilter /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_kalman_qr_core_batch_grid_overnight_2026_07_09.py
```

Runner script:

- `docs/benchmarks/run_kalman_qr_core_batch_grid_overnight_2026_07_09.py`

Status artifact:

- `docs/benchmarks/kalman_qr_core_batch_grid_overnight_status_2026-07-09.json`

Initial service status:

- `kalman-qr-core-batch-grid-20260709.service` was active and running.
- Initial status JSON showed the GPU preflight running.
