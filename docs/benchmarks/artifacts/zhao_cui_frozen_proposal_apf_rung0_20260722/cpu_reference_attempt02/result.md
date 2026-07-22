# Zhao-Cui Frozen-Proposal APF Rung-0 Result

Status: `PASS_ENGINEERING_RUNG0`

This is a 24D fully adapted diagonal-Gaussian mechanics witness. It is not a TT fit, Austria SIR result, HMC result, or NAWM result.

## Gates

| Gate | Status |
| --- | --- |
| finite | `True` |
| same_scalar_fd_max_abs_error_le_2e-2 | `True` |
| minimum_ess_fraction_ge_0p999 | `True` |
| final_log_weight_spread_le_2e-3 | `True` |
| xla_enabled | `True` |
| expected_device | `True` |
| memory_growth_verified | `True` |

## Diagnostics

| Field | Value |
| --- | --- |
| dimension | `24` |
| time_steps | `10` |
| particle_count | `1024` |
| seed | `220722` |
| log_likelihood | `-287.92333984375` |
| exact_kalman_log_likelihood | `-287.9295959472656` |
| descriptive_value_error | `0.006256103515625` |
| score | `[-17.47868537902832, -6.02589750289917]` |
| same_scalar_fd_score | `[-17.4713134765625, -6.027221202850342]` |
| same_scalar_fd_max_abs_error | `0.0073719024658203125` |
| exact_kalman_score | `[-17.25143051147461, -5.865792751312256]` |
| descriptive_score_error | `[-0.22725486755371094, -0.16010475158691406]` |
| minimum_ess | `1023.997802734375` |
| minimum_ess_fraction | `0.9999978542327881` |
| maximum_log_weight_spread | `5.340576171875e-05` |
| final_log_weight_spread | `5.340576171875e-05` |
| compile_inclusive_seconds | `1.1147397150052711` |
| warmed_seconds | `0.00320273099350743` |
| output_device | `/job:localhost/replica:0/task:0/device:CPU:0` |
| gpu_allocator_current_bytes | `0` |
| gpu_allocator_peak_bytes | `0` |

## Nonclaims

No source-faithful Zhao-Cui, TT fit quality, posterior correctness, HMC convergence, Austria SIR, NAWM, default-readiness, or superiority claim is made.
