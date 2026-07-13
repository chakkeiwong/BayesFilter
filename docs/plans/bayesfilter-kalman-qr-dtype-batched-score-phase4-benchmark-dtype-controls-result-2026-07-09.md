# Phase 4 Result: Benchmark Dtype Controls

Date: 2026-07-09

## Status

`PASSED_CPU_HIDDEN_XLA_SMOKE`

## Phase Objective

Add benchmark controls that explicitly request, verify, and report dtype for
analytical and autodiff QR score benchmark artifacts.

## Skeptical Execution Audit

| Risk | Finding |
| --- | --- |
| Wrong baseline | Baseline remained the existing parameter-count benchmark harness with implicit FP64 behavior. |
| Proxy metric misuse | The pass criterion required observed value/score tensor dtypes, not only CLI requested dtype. |
| Missing stop condition | Mismatch between requested and observed dtype now fails row parity. |
| Unfair comparison | Only one-row CPU-hidden smoke artifacts were run; no full performance ladder or ranking claim is made. |
| Environment mismatch | CPU artifacts intentionally hide GPU and record the CPU debug/reference trust basis. |
| Artifact mismatch | JSON and Markdown artifacts record requested dtype, observed dtypes, JIT, TF32 mode, device provenance, and nonclaims. |

Audit status: `PASSED_FOR_PHASE_4_CPU_HIDDEN_DTYPE_SMOKE`.

## Source Changes

- Added `--dtype float32|float64` to
  `scripts/benchmark_kalman_qr_parameter_count_scaling.py`, preserving
  historical `float64` as the explicit CLI default.
- Threaded requested dtype through deterministic fixture construction,
  lower-triangular matrices, covariance factors, derivative bases,
  observations, compiled analytical score, and compiled autodiff score.
- Added requested/observed dtype fields to JSON rows and Markdown output:
  `requested_dtype`, `observed_value_dtypes`, `observed_score_dtypes`, and
  `observed_dtype_check`.
- Made observed dtype mismatch fail the row parity screen.
- Recorded TF32 execution status separately from requested tensor dtype.
- Forwarded dtype and plan path through isolated child-row execution.

## Checks Run

```bash
python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py
```

Result: passed.

```bash
CUDA_VISIBLE_DEVICES=-1 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 --parameter-counts 50 --timesteps 8 --repeats 1 --device cpu --jit-compile --dtype float32 --output-json docs/benchmarks/kalman_qr_dtype_smoke_float32_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_dtype_smoke_float32_cpu_xla_2026-07-09.md
```

Result: passed with `all_applicable_rows_parity_passed=true`.

```bash
CUDA_VISIBLE_DEVICES=-1 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 --parameter-counts 50 --timesteps 8 --repeats 1 --device cpu --jit-compile --dtype float64 --output-json docs/benchmarks/kalman_qr_dtype_smoke_float64_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_dtype_smoke_float64_cpu_xla_2026-07-09.md
```

Result: passed with `all_applicable_rows_parity_passed=true`.

```bash
git diff --check -- scripts docs/benchmarks docs/plans
```

Result: passed.

## Smoke Artifact Summary

| Artifact | Requested dtype | Observed analytical value/score | Observed autodiff value/score | JIT | TF32 reported | Device | Parity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `docs/benchmarks/kalman_qr_dtype_smoke_float32_cpu_xla_2026-07-09.json` | `float32` | `float32` / `float32` | `float32` / `float32` | `true` | `true` | `/CPU:0` | `true` |
| `docs/benchmarks/kalman_qr_dtype_smoke_float64_cpu_xla_2026-07-09.json` | `float64` | `float64` / `float64` | `float64` / `float64` | `true` | `true` | `/CPU:0` | `true` |

## Repair Log

| Issue | Status |
| --- | --- |
| Initial FP32 smoke failed because the main loop did not pass parsed dtype into `benchmark_case`. | Repaired by resolving dtype in `main()` and passing it into the row benchmark. |
| Isolated child-row execution could have fallen back to default dtype. | Repaired by forwarding `--dtype` and `--plan-path` to child commands. |
| TF32 could be confused with requested dtype. | Repaired in artifacts by reporting TF32 separately from requested/observed tensor dtype. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Benchmark artifacts can fail closed if requested dtype differs from observed output dtype. |
| Baseline/comparator | Existing implicit-FP64 benchmark harness was updated with explicit dtype controls. |
| Primary criterion | Passed: FP32 and FP64 CPU-hidden smoke artifacts record matching requested/observed dtype for analytical and autodiff arms. |
| Veto diagnostics | No active Phase 4 veto: dtype fields exist, observed dtype checks passed, JIT was enabled, and TF32 was reported separately. |
| Explanatory diagnostics | Smoke timings are descriptive only; TensorFlow emitted the expected CPU/XLA and CUDA-hidden diagnostics. |
| Not concluded | Full performance ladder, GPU performance, statistical speed ranking, batch-native score, HMC readiness, posterior correctness, default-readiness, or scientific validity. |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Advance to Phase 5 | Passed scoped CPU-hidden FP32/FP64 dtype smoke artifacts | No Phase 4 veto active | Batch-native score contract still needs exact shape/API/review gate | Execute Phase 5 contract subplan before implementation | GPU evidence, full ladder, batch-native implementation, and speed claims |

## Phase 5 Handoff

Phase 5 may start after the Phase 5 subplan review converges:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase5-batched-score-contract-subplan-2026-07-09.md`

Known handoff items:

- Phase 5 must keep `B` batch size and `P` parameter count distinct.
- `tf.vectorized_map` over scalar score may be used only as a comparator or
  fallback reference, not as the final batch-native kernel.
- The active review path is bounded Codex substitute review unless the user
  explicitly approves the Claude external-disclosure boundary.
