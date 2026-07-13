# Kalman QR Parameter Count Scaling Subplan

Date: 2026-07-09

Owning root: `/home/ubuntu/python/BayesFilter`

## Status

`PLAN_READY_FOR_BOUNDED_BENCHMARK`

## Skeptical Audit

Status: passed for a bounded CPU/GPU timing diagnostic.

- Wrong baseline: the comparator remains TensorFlow `GradientTape` over the
  same QR square-root Kalman log likelihood, not a different Kalman likelihood
  or covariance-form filter.
- Proxy metrics: compile+first-call time, warm-start time, warm-call medians,
  and ratios are engineering diagnostics only. They do not certify posterior
  correctness, HMC readiness, production readiness, or statistical superiority.
- Missing stop conditions: mark a row invalid if either method emits nonfinite
  value/score output, if analytical and autodiff values disagree beyond
  tolerance, if score residual exceeds tolerance, if the selected device is not
  recorded, or if the artifact is not written. Mark a row N/A when the requested
  number of independent lower-triangular slots exceeds model capacity.
- Unfair comparisons: keep state/measurement dimensions and `T=120` fixed
  within each row, use the same deterministic fixture for analytical and
  autodiff, and use the dynamic QR `while_loop` value backend for the autodiff
  score so the comparison is not confounded by Python time unroll.
- Hidden assumptions: the benchmark uses dense, time-invariant, square-root QR
  LGSSMs with equal state and measurement dimensions `(10,10)`, `(20,20)`,
  `(30,30)`, fixed observations, and independent parameter slots assigned to
  lower-triangular transition/observation matrices and lower-triangular
  covariance factors, with vector slots only after matrix/factor slots are
  exhausted.
- Applicability boundary: under the independent-slot interpretation,
  `(10,10)` has `5 * 55 + 3 * 10 = 305` available slots, so `p=400` is N/A for
  `(10,10)`. `(20,20)` and `(30,30)` can support all requested counts.
- Compilation timing boundary: TensorFlow/XLA compilation is triggered by the
  first materialized call, so the artifact records compile+first-call time,
  first post-compile warm-start time, warm-call medians, and an explanatory
  first-minus-warm estimate. It does not claim a pure compiler-only time.
- Execution repair: if a monolithic grid accumulates too many XLA-compiled
  shapes and fails in LLVM/codegen memory, rerun with isolated per-row child
  processes. This preserves the per-row compile+first-call and warm-call
  contract while avoiding cumulative compiled-code memory from previous grid
  rows.
- Environment mismatch: GPU/XLA is the BayesFilter default target. CPU-only
  runs are labeled debug/reference exceptions and hide GPU devices before
  TensorFlow import.
- Artifact adequacy: JSON and Markdown artifacts under `docs/benchmarks/` must
  preserve command, environment, device visibility, JIT setting, parameter
  counts, applicability, timings, finite checks, and value/score agreement
  diagnostics.

Reason to proceed: the requested grid is a bounded engineering benchmark with
explicit comparators, applicability limits, veto checks, and nonclaims.

## Research Intent Ledger

| Field | Entry |
| --- | --- |
| Main question | How do QR analytical score and autodiff score timings vary as parameter count changes for fixed `(n,m)` and `T=120`, on CPU and GPU? |
| Candidate or mechanism under test | Public analytical QR square-root score propagation in `tf_qr_sqrt_kalman_score`, with parameter slots carried in first-derivative tensors. |
| Comparator | TensorFlow `GradientTape` score of the dynamic QR square-root log likelihood on the same lower-triangular-parameterized fixture. |
| Expected failure mode | XLA compilation failure, excessive compile time, nonfinite numerical output, score disagreement, or requested parameter count exceeding available independent slots. |
| Promotion criterion | No promotion. The run answers a descriptive timing question only. |
| Promotion veto | Any row with nonfinite outputs or failed value/score parity cannot be used for a timing ratio interpretation. |
| Continuation veto | TensorFlow runtime unavailable, selected GPU unavailable for the GPU artifact, all rows fail to compile/run, or artifacts cannot be written. |
| Repair trigger | XLA failure triggers a smaller diagnostic row or explicit result note; capacity overflow triggers an N/A row, not duplicate/conflicting parameters. |
| Explanatory diagnostics | Compile+first-call time, warm-start time, warm-call median/mean/min/max, first-minus-warm estimate, output devices, value residual, score max-abs residual, and score relative residual. |
| What must not be concluded | No HMC readiness, posterior correctness, production default change, statistically supported ranking, or universal speed claim is concluded. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific or engineering question | Measure descriptive compile+first-call and warm-call speed scaling for analytical QR score and autodiff score across devices, dimensions, and parameter counts. |
| Exact baseline/comparator | Autodiff is `GradientTape` over `tf_qr_sqrt_kalman_log_likelihood_while_loop` on the same observations, parameters, and model tensors. |
| Primary pass/fail criterion | Each reported timing row must have finite value/score outputs and pass analytical-vs-autodiff value/score agreement tolerances. |
| Veto diagnostics | Nonfinite outputs, value residual above `1e-8`, score max residual above `1e-5`, missing timing samples, missing device manifest, or missing artifact. |
| Explanatory only | Runtime medians, runtime ratios, compile+first-call time, warm-start time, first-minus-warm estimate, device placement, and TensorFlow warning text. |
| Not concluded if passed | Passing rows do not prove analytical QR is universally faster, statistically superior, HMC-ready, or production-ready for all LGSSMs. |
| Artifact | `docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.{json,md}` and `docs/benchmarks/kalman_qr_parameter_count_scaling_gpu_xla_2026-07-09.{json,md}`. |

## Planned Commands

CPU debug/reference:

```bash
python scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  --dimensions 10 20 30 \
  --parameter-counts 50 100 150 200 300 400 \
  --timesteps 120 \
  --repeats 1 \
  --jit-compile \
  --device cpu \
  --isolate-each-row \
  --flush-after-row \
  --output-json docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.json \
  --output-md docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.md
```

Trusted GPU/XLA:

```bash
python scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  --dimensions 10 20 30 \
  --parameter-counts 50 100 150 200 300 400 \
  --timesteps 120 \
  --repeats 1 \
  --jit-compile \
  --device gpu \
  --isolate-each-row \
  --flush-after-row \
  --output-json docs/benchmarks/kalman_qr_parameter_count_scaling_gpu_xla_2026-07-09.json \
  --output-md docs/benchmarks/kalman_qr_parameter_count_scaling_gpu_xla_2026-07-09.md
```
