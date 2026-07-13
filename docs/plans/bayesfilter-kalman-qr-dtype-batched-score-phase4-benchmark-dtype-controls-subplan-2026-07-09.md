# Phase 4 Subplan: Benchmark Dtype Controls

Date: 2026-07-09

## Phase Objective

Add benchmark controls that explicitly request, verify, and report dtype for
analytical and autodiff QR score benchmarks.

## Entry Conditions Inherited From Previous Phase

- Phase 3 analytical score dtype cleanup passes.
- FP32 and FP64 CPU/XLA smoke tests exist.
- Benchmark harness write set is limited to dtype controls and artifact fields.

## Required Artifacts

- Updated benchmark harness with `--dtype float32|float64`.
- JSON/Markdown fields for requested dtype, observed value dtype, observed score
  dtype, TF32 mode, device, and JIT flag.
- Focused CPU-hidden smoke artifacts for FP32 and FP64.
- Phase 4 result and refreshed Phase 5 subplan.

## Required Checks, Tests, And Reviews

Run:

```bash
CUDA_VISIBLE_DEVICES=-1 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 --parameter-counts 50 --timesteps 8 --repeats 1 --device cpu --jit-compile --dtype float32 --output-json docs/benchmarks/kalman_qr_dtype_smoke_float32_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_dtype_smoke_float32_cpu_xla_2026-07-09.md
CUDA_VISIBLE_DEVICES=-1 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 --parameter-counts 50 --timesteps 8 --repeats 1 --device cpu --jit-compile --dtype float64 --output-json docs/benchmarks/kalman_qr_dtype_smoke_float64_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_dtype_smoke_float64_cpu_xla_2026-07-09.md
git diff --check -- scripts docs/benchmarks docs/plans
```

GPU smoke is deferred unless explicitly approved under Phase 4 or Phase 7.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can benchmark artifacts fail closed if requested dtype differs from observed output dtype? |
| Baseline/comparator | Existing FP64-only benchmark harness and artifacts. |
| Primary criterion | FP32 and FP64 CPU-hidden smoke artifacts record matching requested/observed dtype for both analytical and autodiff arms. |
| Veto diagnostics | Missing dtype field, mismatch not failing, TF32 conflated with dtype, or non-JIT run mislabeled as XLA evidence. |
| Explanatory diagnostics | Smoke timing and device placement. |
| Not concluded | Full performance ladder, GPU performance, or statistical speed ranking. |
| Artifact | Phase 4 result and refreshed Phase 5 subplan. |

## Forbidden Claims And Actions

- Do not run the full benchmark ladder in this phase unless explicitly approved
  by a refreshed subplan.
- Do not infer FP32 correctness from requested dtype alone.
- Do not treat TF32-enabled as FP32 evidence for FP64 tensors.

## Exact Next-Phase Handoff Conditions

Advance to Phase 5 only if benchmark dtype smoke artifacts prove fail-closed
requested/observed dtype behavior.

## Stop Conditions

Stop if harness cannot observe output dtype reliably or if smoke artifacts show
hidden FP64 execution for FP32 requests.
