# Kalman QR Parameter Count Scaling Result

- JSON artifact: `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads4_batch4_xla_2026-07-09.json`
- Plan: `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7-correctness-benchmark-subplan-2026-07-09.md`
- Command: `/home/ubuntu/python/BayesFilter/scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 20 30 --parameter-counts 50 150 --timesteps 120 --repeats 1 --batch-size 4 --device cpu --jit-compile --dtype float32 --isolate-each-row --row-subprocess-timeout-seconds 3600 --output-json docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads4_batch4_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads4_batch4_xla_2026-07-09.md --cpu-threads 4`
- Device: `/CPU:0`
- JIT compile: `True`
- Requested dtype: `float32`
- Batch size: `4`
- CPU threads: `4`
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
| (10,10) | 50 | 4 | 24.0256 | 1.16002 | 1.01002 | 1.64716 | 0.164651 | 0.163017 | 1.63082 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 1.341e-07 | `True` |
| (10,10) | 150 | 4 | 119.873 | 3.45599 | 3.44284 | 5.04617 | 0.139252 | 0.0404469 | 1.4657 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 1.341e-07 | `True` |
| (20,20) | 50 | 4 | 27.5116 | 2.56255 | 2.49628 | 3.29694 | 0.263007 | 0.10536 | 1.32074 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 4.172e-07 | `True` |
| (20,20) | 150 | 4 | 135.54 | 7.66123 | 7.48693 | 11.1975 | 0.297015 | 0.0396711 | 1.49561 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 4.023e-07 | `True` |
| (30,30) | 50 | 4 | 30.6068 | 4.34803 | 4.87456 | 6.15586 | 0.483362 | 0.0991601 | 1.26285 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 3.055e-07 | `True` |
| (30,30) | 150 | 4 | 139.974 | 12.5514 | 12.45 | 18.5071 | 0.470395 | 0.0377828 | 1.48652 | `value={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}; score={'autodiff_row_loop_qr_score': 'float32', 'batch_native_analytical_qr_score': 'float32', 'scalar_analytical_row_loop': 'float32'}` | 2.682e-07 | `True` |

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
- Batch size: `4`
- CPU thread manifest: `{'requested_cpu_threads': 4, 'tf_intra_op_parallelism_threads': 4, 'tf_inter_op_parallelism_threads': 4, 'intra_op_set_status': 'set', 'inter_op_set_status': 'set', 'omp_num_threads': '4', 'tf_num_intraop_threads_env': '4', 'tf_num_interop_threads_env': '4'}`
- TF32 execution enabled: `True`
- Physical GPUs: `[]`
- Logical GPUs: `[]`
- CUDA_VISIBLE_DEVICES: `-1`
- Data version: `deterministic synthetic lower-triangular LGSSM fixture generated by this script`
- Random seeds: `N/A deterministic fixture`

## Post-Run Red Team

The strongest alternative explanation is device/runtime noise or XLA compile/runtime behavior specific to this synthetic lower-triangular parameterization. A result that would overturn a speed interpretation is a replicated run on the target deployment device where warm median ratios change materially or parity fails.
