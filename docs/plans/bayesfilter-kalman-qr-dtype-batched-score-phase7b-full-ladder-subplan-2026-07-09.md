# Phase 7B Subplan: Full CPU/GPU Ladder

Date: 2026-07-09

## Phase Objective

Run the full descriptive parameter-count ladder for the refreshed batched-score
benchmark harness across CPU/GPU, FP32/FP64, dimensions `(10, 10)`, `(20, 20)`,
`(30, 30)`, parameter counts `50, 100, 150, 200, 300, 400`, fixed
`T=120`, fixed `batch_size=2`, and JIT compilation enabled.

## Entry Conditions Inherited From Previous Phase

- Phase 7 smoke result exists and records
  `PHASE_7_SMOKE_GATE_PASSED_FULL_LADDER_DEFERRED`.
- `scripts/benchmark_kalman_qr_parameter_count_scaling.py` exposes:
  - `batch_native_analytical_qr_score`;
  - `scalar_analytical_row_loop`;
  - `autodiff_row_loop_qr_score`;
  - non-timed `batched_static_autodiff_probe`.
- CPU-hidden FP32 XLA smoke passed dtype, shape, finite-output, and parity
  gates for one row.
- GPU runtime may be used only if visible device provenance is recorded and
  the run satisfies the BayesFilter managed-session GPU trust policy.

## Required Artifacts

- CPU FP32 ladder:
  - `docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float32_cpu_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float32_cpu_xla_2026-07-09.md`
  - `docs/benchmarks/logs/kalman_qr_batched_score_parameter_count_scaling_float32_cpu_xla_2026-07-09.log`
- CPU FP64 ladder:
  - `docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float64_cpu_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float64_cpu_xla_2026-07-09.md`
  - `docs/benchmarks/logs/kalman_qr_batched_score_parameter_count_scaling_float64_cpu_xla_2026-07-09.log`
- GPU FP32 ladder if GPU provenance gate passes:
  - `docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float32_gpu_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float32_gpu_xla_2026-07-09.md`
  - `docs/benchmarks/logs/kalman_qr_batched_score_parameter_count_scaling_float32_gpu_xla_2026-07-09.log`
- GPU FP64 ladder if GPU provenance gate passes:
  - `docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float64_gpu_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float64_gpu_xla_2026-07-09.md`
  - `docs/benchmarks/logs/kalman_qr_batched_score_parameter_count_scaling_float64_gpu_xla_2026-07-09.log`
- Phase 7B result or blocker result.
- Refreshed Phase 8 closeout subplan.

## Required Checks, Tests, And Reviews

Preflight checks:

```bash
python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py
git diff --check -- scripts docs/benchmarks docs/plans bayesfilter/linear tests
nvidia-smi
python -c "import json, os, tensorflow as tf; print(json.dumps({'tensorflow_version': tf.__version__, 'physical_gpus': [device.name for device in tf.config.list_physical_devices('GPU')], 'logical_gpus': [device.name for device in tf.config.list_logical_devices('GPU')], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSET')}))"
```

CPU FP32 command:

```bash
CUDA_VISIBLE_DEVICES=-1 timeout 43200 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 20 30 --parameter-counts 50 100 150 200 300 400 --timesteps 120 --repeats 1 --batch-size 2 --device cpu --jit-compile --dtype float32 --isolate-each-row --row-subprocess-timeout-seconds 3600 --output-json docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float32_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float32_cpu_xla_2026-07-09.md > docs/benchmarks/logs/kalman_qr_batched_score_parameter_count_scaling_float32_cpu_xla_2026-07-09.log 2>&1
```

CPU FP64 command:

```bash
CUDA_VISIBLE_DEVICES=-1 timeout 43200 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 20 30 --parameter-counts 50 100 150 200 300 400 --timesteps 120 --repeats 1 --batch-size 2 --device cpu --jit-compile --dtype float64 --isolate-each-row --row-subprocess-timeout-seconds 3600 --output-json docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float64_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float64_cpu_xla_2026-07-09.md > docs/benchmarks/logs/kalman_qr_batched_score_parameter_count_scaling_float64_cpu_xla_2026-07-09.log 2>&1
```

GPU provenance gate, required before either GPU ladder:

```bash
python -c "import json, os, tensorflow as tf; print(json.dumps({'tensorflow_version': tf.__version__, 'physical_gpus': [device.name for device in tf.config.list_physical_devices('GPU')], 'logical_gpus': [device.name for device in tf.config.list_logical_devices('GPU')], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSET'), 'required_trust_basis': 'owner_designated_managed_session_visible_gpu_trusted'}))" > docs/benchmarks/logs/kalman_qr_batched_score_gpu_tensorflow_provenance_2026-07-09.log 2>&1
```

This gate passes only if TensorFlow reports at least one logical GPU.  The
benchmark JSON must also record selected device `/GPU:0`, physical/logical GPU
lists, TF32 flag, requested dtype, JIT status, and trust basis
`owner_designated_managed_session_visible_gpu_trusted`.

GPU FP32 command, only after GPU provenance gate passes:

```bash
timeout 43200 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 20 30 --parameter-counts 50 100 150 200 300 400 --timesteps 120 --repeats 1 --batch-size 2 --device gpu --jit-compile --dtype float32 --isolate-each-row --row-subprocess-timeout-seconds 3600 --output-json docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float32_gpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float32_gpu_xla_2026-07-09.md > docs/benchmarks/logs/kalman_qr_batched_score_parameter_count_scaling_float32_gpu_xla_2026-07-09.log 2>&1
```

GPU FP64 command, only after GPU provenance gate passes:

```bash
timeout 43200 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 20 30 --parameter-counts 50 100 150 200 300 400 --timesteps 120 --repeats 1 --batch-size 2 --device gpu --jit-compile --dtype float64 --isolate-each-row --row-subprocess-timeout-seconds 3600 --output-json docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float64_gpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_batched_score_parameter_count_scaling_float64_gpu_xla_2026-07-09.md > docs/benchmarks/logs/kalman_qr_batched_score_parameter_count_scaling_float64_gpu_xla_2026-07-09.log 2>&1
```

Post-run checks:

```bash
python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py
git diff --check -- scripts docs/benchmarks docs/plans bayesfilter/linear tests
```

Review:

- Claude review remains unavailable in this run unless the user separately
  approves the external-disclosure boundary.
- Use a fresh bounded Codex substitute review for this subplan and for the
  Phase 7B result.  Record that this is weaker than Claude review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | How do descriptive warm-call runtimes vary across device, dtype, dimension, and parameter count for the refreshed batched-score harness? |
| Baseline/comparator | `batch_native_analytical_qr_score`, `scalar_analytical_row_loop`, and `autodiff_row_loop_qr_score`, where the autodiff arm differentiates the scalar QR value per row inside one compiled row-loop function. |
| Primary criterion | All applicable rows in completed artifacts pass dtype, shape, finite-output, and parity gates; artifacts record compile+first, warm-start, warm summaries, device provenance, JIT, dtype, TF32 flag, batch size, and nonclaims. |
| Veto diagnostics | Timed-output nonfinite value/score, row parity failure, dtype mismatch, missing device provenance, GPU run without visible trusted provenance, child-row timeout without artifact record, or unsupported ranking claim. |
| Explanatory diagnostics | Compile+first time, warm-start time, warm median, first-minus-warm, TF32 flag, device placement, row subprocess metadata, and non-timed batch-static autodiff diagnostic. |
| Not concluded | Statistical speed ranking, universal superiority, production/default readiness, HMC readiness, posterior correctness, or scientific validity. |
| Artifact | Four JSON/Markdown/log ladder artifacts when CPU and GPU both run, otherwise completed CPU artifacts plus a blocker/result explaining skipped GPU. |

## Forbidden Claims And Actions

- Do not claim a statistically supported speed ranking from `repeats=1`.
- Do not compare GPU and CPU rows unless both rows passed parity and device
  provenance gates.
- Do not treat the non-timed batch-static autodiff probe as a benchmark arm.
- Do not treat CPU-hidden artifacts as GPU/default-readiness evidence.
- Do not run GPU commands if TensorFlow cannot see a logical GPU or if the
  artifact cannot record managed-session GPU trust basis.
- Do not use Claude review without separate approval for external disclosure.

## Exact Next-Phase Handoff Conditions

Advance to Phase 8 closeout if:

- all launched CPU/GPU artifacts pass row gates; or a material failure has a
  structured blocker result and Phase 8 is explicitly a blocker closeout rather
  than a normal pass closeout;
- artifact interpretation remains descriptive-only;
- GPU skipped/running/unavailable status is explicitly recorded if applicable;
- Phase 7B result includes a decision table and inference-status table.

## Stop Conditions

Stop and write a blocker result if:

- any timed row fails parity or emits nonfinite timed outputs;
- the benchmark cannot preserve dtype/device/JIT provenance;
- GPU is unavailable or untrusted and the next required action is GPU-only;
- row subprocess timeouts prevent a meaningful partial artifact;
- full-ladder runtime exceeds the visible execution budget and cannot be
  launched under an approved detached/overnight plan;
- review does not converge after five rounds for the same material blocker.

## Numeric Provenance

| Number | Provenance | Role |
| --- | --- | --- |
| Dimensions `10,20,30` | User-requested benchmark grid | Grid axis |
| Parameter counts `50,100,150,200,300,400` | User-requested benchmark grid | Grid axis |
| `T=120` | User-requested benchmark setting | Grid setting |
| `batch_size=2` | Phase 7 smoke setting and minimum independent proposal batch beyond scalar | Fixed batch setting |
| `repeats=1` | Existing benchmark request and runtime-control convenience | Descriptive timing only |
| Parent timeout `43200s` | Runtime guard for visible long ladder, not a scientific threshold | Execution guard |
| Row timeout `3600s` | Runtime guard for isolated row subprocesses, not a scientific threshold | Execution guard |
