# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/fixed-sir-t20-n10000-seed81120-fd.json`
- Status: `failed_fd`
- Row: `zhao_cui_spatial_sir_austria_j9_T20`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'fail', 'step': 0.001, 'atol': 0.01, 'rtol': 0.05, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 7.853515625, 'max_relative_error': 0.05667001008987427, 'parameters': [{'parameter': 'log_kappa_scale', 'score': -2990.702392578125, 'finite_difference': -2982.848876953125, 'abs_error': 7.853515625, 'relative_error': 0.002625976921990514}, {'parameter': 'log_nu_scale', 'score': 27.304161071777344, 'finite_difference': 25.756834030151367, 'abs_error': 1.5473270416259766, 'relative_error': 0.05667001008987427}, {'parameter': 'log_obs_noise_scale', 'score': 422.55389404296875, 'finite_difference': 416.9921569824219, 'abs_error': 5.561737060546875, 'relative_error': 0.013162195682525635}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
