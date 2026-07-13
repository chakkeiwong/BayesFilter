# Kalman QR Parameter Count Scaling Result

- JSON artifact: `docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.json`
- Plan: `docs/plans/bayesfilter-kalman-qr-parameter-count-scaling-subplan-2026-07-09.md`
- Command: `scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 20 30 --parameter-counts 50 100 150 200 300 400 --timesteps 120 --repeats 1 --jit-compile --device cpu --isolate-each-row --row-subprocess-timeout-seconds 1800 --flush-after-row --output-json docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.md`
- Device: `/CPU:0`
- JIT compile: `True`
- Autodiff value backend: `while_loop`
- Autodiff execution: `compiled_full`
- Trust basis: `cpu_debug_or_reference_exception`

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
| (10,10) | 50 | 28.5463 | 0.546856 | 0.540804 | 1.91322 | 0.0538734 | 0.0515387 | 0.0953002 | 1.527e-16 | `True` |
| (10,10) | 100 | 75.115 | 0.890944 | 0.899949 | 2.05133 | 0.0440715 | 0.046987 | 0.0522108 | 1.665e-16 | `True` |
| (10,10) | 150 | 144.519 | 1.44968 | 1.47839 | 2.51792 | 0.0475344 | 0.0464087 | 0.0313914 | 1.110e-16 | `True` |
| (10,10) | 200 | 228.341 | 1.68574 | 2.02659 | 2.27105 | 0.04569 | 0.0493125 | 0.0243327 | 1.388e-16 | `True` |
| (10,10) | 300 | 458.868 | 4.02503 | 3.77564 | 2.54425 | 0.0518629 | 0.0498868 | 0.0132128 | 1.665e-16 | `True` |
| (10,10) | 400 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `N/A capacity` |
| (20,20) | 50 | 30.2391 | 1.21872 | 1.23966 | 2.1159 | 0.109838 | 0.107795 | 0.0869555 | 1.249e-16 | `True` |
| (20,20) | 100 | 83.8741 | 2.55237 | 2.5552 | 2.30114 | 0.111858 | 0.111481 | 0.0436289 | 1.249e-16 | `True` |
| (20,20) | 150 | 150.096 | 3.96948 | 3.92029 | 2.75489 | 0.111845 | 0.110068 | 0.0280765 | 1.388e-16 | `True` |
| (20,20) | 200 | 249.236 | 5.51922 | 5.44749 | 2.53816 | 0.10612 | 0.116017 | 0.0212973 | 1.804e-16 | `True` |
| (20,20) | 300 | 463.559 | 9.03958 | 8.74239 | 2.86278 | 0.115509 | 0.112006 | 0.0128118 | 1.527e-16 | `True` |
| (20,20) | 400 | 796.69 | 12.7737 | 12.6482 | 3.10071 | 0.0989602 | 0.10983 | 0.00868347 | 1.527e-16 | `True` |
| (30,30) | 50 | 32.5951 | 3.12333 | 3.20053 | 2.15251 | 0.222123 | 0.175801 | 0.0549288 | 1.110e-16 | `True` |
| (30,30) | 100 | 90.0086 | 5.97208 | 6.0443 | 2.43413 | 0.209726 | 0.183506 | 0.0303602 | 1.943e-16 | `True` |
| (30,30) | 150 | 149.675 | 9.05076 | 8.85353 | 2.97674 | 0.200137 | 0.203607 | 0.0229972 | 1.665e-16 | `True` |
| (30,30) | 200 | 251.364 | 12.6502 | 12.4727 | 3.09712 | 0.208697 | 0.21825 | 0.0174982 | 1.110e-16 | `True` |
| (30,30) | 300 | 514.727 | 21.1813 | 21.5872 | 3.16996 | 0.225061 | 0.221083 | 0.0102414 | 1.527e-16 | `True` |
| (30,30) | 400 | 818.525 | 26.534 | 25.7324 | 3.57077 | 0.218836 | 0.204631 | 0.00795228 | 1.943e-16 | `True` |

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
- Physical GPUs: `[]`
- Logical GPUs: `[]`
- CUDA_VISIBLE_DEVICES: `-1`
- Data version: `deterministic synthetic lower-triangular LGSSM fixture generated by this script`
- Random seeds: `N/A deterministic fixture`

## Post-Run Red Team

The strongest alternative explanation is device/runtime noise or XLA compile/runtime behavior specific to this synthetic lower-triangular parameterization. A result that would overturn a speed interpretation is a replicated run on the target deployment device where warm median ratios change materially or parity fails.
