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
| log_likelihood | `-287.922607421875` |
| exact_kalman_log_likelihood | `-287.9295959472656` |
| descriptive_value_error | `0.006988525390625` |
| score | `[-17.245010375976562, -5.880650997161865]` |
| same_scalar_fd_score | `[-17.242431640625, -5.889892101287842]` |
| same_scalar_fd_max_abs_error | `0.009241104125976562` |
| exact_kalman_score | `[-17.25143051147461, -5.865792751312256]` |
| descriptive_score_error | `[0.006420135498046875, -0.014858245849609375]` |
| minimum_ess | `1023.99853515625` |
| minimum_ess_fraction | `0.9999985694885254` |
| maximum_log_weight_spread | `4.57763671875e-05` |
| final_log_weight_spread | `4.57763671875e-05` |
| compile_inclusive_seconds | `3.224812582018785` |
| warmed_seconds | `0.0008308029791805893` |
| output_device | `/job:localhost/replica:0/task:0/device:GPU:0` |
| gpu_allocator_current_bytes | `1154048` |
| gpu_allocator_peak_bytes | `9904896` |

## Nonclaims

No source-faithful Zhao-Cui, TT fit quality, posterior correctness, HMC convergence, Austria SIR, NAWM, default-readiness, or superiority claim is made.
