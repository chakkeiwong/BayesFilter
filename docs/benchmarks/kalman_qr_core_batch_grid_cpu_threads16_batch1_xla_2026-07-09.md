# Kalman QR Parameter Count Scaling Result

- JSON artifact: `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads16_batch1_xla_2026-07-09.json`
- Plan: `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7-correctness-benchmark-subplan-2026-07-09.md`
- Command: `/home/ubuntu/python/BayesFilter/scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 20 30 --parameter-counts 50 150 --timesteps 120 --repeats 1 --batch-size 1 --device cpu --jit-compile --dtype float32 --isolate-each-row --row-subprocess-timeout-seconds 3600 --output-json docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads16_batch1_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads16_batch1_xla_2026-07-09.md --cpu-threads 16`
- Device: `/CPU:0`
- JIT compile: `True`
- Requested dtype: `float32`
- Batch size: `1`
- CPU threads: `16`
- TF32 execution enabled: `True`
- Benchmark methods: `['batch_native_analytical_qr_score', 'scalar_analytical_row_loop', 'autodiff_row_loop_qr_score']`
- Autodiff value backend: `scalar_while_loop_row_loop`
- Autodiff execution: `compiled_static_batch_row_loop`
- Trust basis: `cpu_debug_or_reference_exception`

## Execution Note

All measured arms are compiled TensorFlow functions when `jit_compile=True`. The batch-native analytical arm calls `tf_qr_sqrt_kalman_score_batched_static`; the scalar comparator loops over batch rows inside one compiled function and calls the scalar analytical score; the autodiff comparator loops over batch rows inside one compiled function and differentiates the scalar QR value for each row. The first timed call is compile+first-call, the second timed call is the first warm-start call, and repeated calls provide the warm-call summary. The first-minus-warm value is explanatory only and is not a pure compiler-only measurement.

Requested dtype is checked against observed analytical/autodiff value and score tensor dtypes. A mismatch fails the row parity screen. TF32 mode is reported separately and is not treated as the requested tensor dtype.

Transition and observation matrices are lower triangular. Covariance inputs are SPD matrices formed from lower-triangular factors because the public Kalman API consumes covariance matrices, not covariance factors.

## Decision Table

| Field | Status | Notes |
| --- | --- | --- |
| Decision | `DESCRIPTIVE_TIMING_RECORDED` | No default, HMC, or scientific promotion claim. |
| Primary criterion | `True` | Finite outputs, requested/observed dtype match, and analytical/autodiff value-score parity for applicable rows. |
| Veto diagnostics | `see rows` | Nonfinite outputs or parity failure invalidates a row timing ratio. |
| Applicability | `capacity_checked` | Rows above independent lower-triangular slot capacity are marked N/A. |
| Main uncertainty | `single-run wall timing` | Repeats are descriptive and not a statistical ranking. |
| Not concluded | `no promotion` | No HMC readiness, posterior correctness, or universal speed superiority. |

## Timing Table

| dims `(n,m)` | params | batch | batch-native compile+first s | batch-native warm-start s | batch-native warm median s | scalar row-loop warm median s | autodiff row-loop warm median s | autodiff row-loop / batch-native warm | scalar row-loop / batch-native warm | observed dtypes | batch/autodiff score max abs residual | parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| (10,10) | 50 | 1 | 21.1334 | 0.427165 | 0.408621 | 0.38974 | 0.039919 | 0.0976921 | 0.953792 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 4.470e-08 | `True` |
| (10,10) | 150 | 1 | 94.6359 | 1.18882 | 1.16218 | 1.10067 | 0.0408235 | 0.0351266 | 0.947068 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 1.043e-07 | `True` |
| (20,20) | 50 | 1 | 25.8223 | 0.734276 | 0.788152 | 0.829744 | 0.0924065 | 0.117245 | 1.05277 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 5.960e-08 | `True` |
| (20,20) | 150 | 1 | 124.359 | 2.68421 | 2.59798 | 2.54363 | 0.0930728 | 0.0358251 | 0.979079 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 8.941e-08 | `True` |
| (30,30) | 50 | 1 | 25.3635 | 1.53989 | 1.53137 | 1.65286 | 0.139781 | 0.0912786 | 1.07934 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 8.941e-08 | `True` |
| (30,30) | 150 | 1 | 133.127 | 5.01561 | 4.70671 | 4.62045 | 0.141287 | 0.0300181 | 0.981672 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 5.960e-08 | `True` |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | `True` |
| Statistically supported ranking | `not assessed` |
| Descriptive-only differences | `compile+first, warm-start, warm medians, and ratios only` |
| Default-readiness | `not assessed` |
| Next evidence needed | `replicate runs and broaden model families if making a speed claim` |

## Run Manifest

- Git commit: `52ee244498988e046a6356f926003b581103083b`
- TensorFlow: `2.20.0`
- Requested dtype: `float32`
- Batch size: `1`
- CPU thread manifest: `{'requested_cpu_threads': 16, 'tf_intra_op_parallelism_threads': 16, 'tf_inter_op_parallelism_threads': 16, 'intra_op_set_status': 'set', 'inter_op_set_status': 'set', 'omp_num_threads': '16', 'tf_num_intraop_threads_env': '16', 'tf_num_interop_threads_env': '16'}`
- TF32 execution enabled: `True`
- Physical GPUs: `[]`
- Logical GPUs: `[]`
- CUDA_VISIBLE_DEVICES: `-1`
- Data version: `deterministic synthetic lower-triangular LGSSM fixture generated by this script`
- Random seeds: `N/A deterministic fixture`

## Post-Run Red Team

The strongest alternative explanation is device/runtime noise or XLA compile/runtime behavior specific to this synthetic lower-triangular parameterization. A result that would overturn a speed interpretation is a replicated run on the target deployment device where warm median ratios change materially or parity fails.
