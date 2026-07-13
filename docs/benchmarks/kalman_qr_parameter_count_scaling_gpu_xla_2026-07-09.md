# Kalman QR Parameter Count Scaling Result

- JSON artifact: `docs/benchmarks/kalman_qr_parameter_count_scaling_gpu_xla_2026-07-09.json`
- Plan: `docs/plans/bayesfilter-kalman-qr-parameter-count-scaling-subplan-2026-07-09.md`
- Command: `scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 20 30 --parameter-counts 50 100 150 200 300 400 --timesteps 120 --repeats 1 --jit-compile --device gpu --isolate-each-row --row-subprocess-timeout-seconds 1800 --flush-after-row --output-json docs/benchmarks/kalman_qr_parameter_count_scaling_gpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_parameter_count_scaling_gpu_xla_2026-07-09.md`
- Device: `/GPU:0`
- JIT compile: `True`
- Autodiff value backend: `while_loop`
- Autodiff execution: `compiled_full`
- Trust basis: `owner_designated_managed_session_visible_gpu_trusted`

## Execution Note

Both measured arms are compiled TensorFlow functions when `jit_compile=True`. The analytical arm wraps `tf_qr_sqrt_kalman_score`; the autodiff arm computes the dynamic QR `while_loop` value and `GradientTape` score inside one compiled function. The first timed call is compile+first-call, the second timed call is the first warm-start call, and repeated calls provide the warm-call summary. The first-minus-warm value is explanatory only and is not a pure compiler-only measurement.

Transition and observation matrices are lower triangular. Covariance inputs are SPD matrices formed from lower-triangular factors because the public Kalman API consumes covariance matrices, not covariance factors.

## Decision Table

| Field | Status | Notes |
| --- | --- | --- |
| Decision | `DESCRIPTIVE_TIMING_RECORDED` | No default, HMC, or scientific promotion claim. |
| Primary criterion | `True` | Finite outputs and analytical/autodiff value-score parity for applicable rows. |
| Veto diagnostics | `see rows` | Nonfinite outputs or parity failure invalidates a row timing ratio. |
| Applicability | `capacity_checked` | Rows above independent lower-triangular slot capacity are marked N/A. |
| Main uncertainty | `single-run wall timing` | Repeats are descriptive and not a statistical ranking. |
| Not concluded | `no promotion` | No HMC readiness, posterior correctness, or universal speed superiority. |

## Timing Table

| dims `(n,m)` | params | analytical compile+first s | analytical warm-start s | analytical warm median s | autodiff compile+first s | autodiff warm-start s | autodiff warm median s | autodiff / analytical warm | score max abs residual | parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| (10,10) | 50 | 28.0134 | 0.4803 | 0.493052 | 3.28267 | 0.17737 | 0.177696 | 0.360399 | 1.388e-16 | `True` |
| (10,10) | 100 | 53.7835 | 1.00003 | 1.05117 | 3.31969 | 0.209415 | 0.20946 | 0.199264 | 1.388e-16 | `True` |
| (10,10) | 150 | 79.9176 | 1.48542 | 1.45105 | 3.69856 | 0.202517 | 0.198722 | 0.13695 | 1.804e-16 | `True` |
| (10,10) | 200 | 118.525 | 1.61492 | 1.66235 | 3.47261 | 0.172846 | 0.179981 | 0.10827 | 1.527e-16 | `True` |
| (10,10) | 300 | 178.778 | 6.23985 | 5.76623 | 3.833 | 0.368084 | 0.383895 | 0.0665764 | 1.388e-16 | `True` |
| (10,10) | 400 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `N/A capacity` |
| (20,20) | 50 | 36.7752 | 2.1676 | 1.83831 | 4.03717 | 0.677148 | 0.370094 | 0.201323 | 1.388e-16 | `True` |
| (20,20) | 100 | 67.3656 | 1.86224 | 1.88936 | 3.73546 | 0.35007 | 0.352471 | 0.186556 | 1.388e-16 | `True` |
| (20,20) | 150 | 107.513 | 2.58789 | 2.65757 | 4.24214 | 0.340634 | 0.3474 | 0.130721 | 1.249e-16 | `True` |
| (20,20) | 200 | 153.12 | 3.56645 | 3.60868 | 4.18458 | 0.354347 | 0.360656 | 0.0999412 | 1.943e-16 | `True` |
| (20,20) | 300 | 247.002 | 5.30281 | 5.4899 | 4.17287 | 0.352976 | 0.367716 | 0.0669804 | 1.110e-16 | `True` |
| (20,20) | 400 | 344.233 | 7.45178 | 7.48562 | 4.44949 | 0.370255 | 0.366729 | 0.0489912 | 2.082e-16 | `True` |
| (30,30) | 50 | 37.6419 | 1.45877 | 1.53926 | 4.14597 | 0.491531 | 0.497057 | 0.322919 | 1.110e-16 | `True` |
| (30,30) | 100 | 77.5309 | 2.75773 | 2.76458 | 4.16549 | 0.497217 | 0.496398 | 0.179556 | 8.327e-17 | `True` |
| (30,30) | 150 | 122.325 | 3.95363 | 3.88505 | 4.67521 | 0.496428 | 0.487871 | 0.125576 | 1.388e-16 | `True` |
| (30,30) | 200 | 161.468 | 5.13611 | 5.15251 | 4.4607 | 0.496218 | 0.493088 | 0.0956985 | 1.110e-16 | `True` |
| (30,30) | 300 | 276.183 | 8.69356 | 8.46208 | 4.66935 | 0.560844 | 0.554951 | 0.0655809 | 1.665e-16 | `True` |
| (30,30) | 400 | 400.405 | 12.1251 | 12.0522 | 4.875 | 0.554168 | 0.587438 | 0.0487413 | 1.943e-16 | `True` |

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
- Physical GPUs: `['/physical_device:GPU:0', '/physical_device:GPU:1']`
- Logical GPUs: `['/device:GPU:0', '/device:GPU:1']`
- CUDA_VISIBLE_DEVICES: `UNSET`
- Data version: `deterministic synthetic lower-triangular LGSSM fixture generated by this script`
- Random seeds: `N/A deterministic fixture`

## Post-Run Red Team

The strongest alternative explanation is device/runtime noise or XLA compile/runtime behavior specific to this synthetic lower-triangular parameterization. A result that would overturn a speed interpretation is a replicated run on the target deployment device where warm median ratios change materially or parity fails.
