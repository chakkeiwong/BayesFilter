# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/fixed-sir-t1-n10000-seed81120-fd.json`
- Status: `completed`
- Row: `zhao_cui_spatial_sir_austria_j9_T20`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'pass', 'step': 0.001, 'atol': 0.01, 'rtol': 0.05, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 0.07912921905517578, 'max_relative_error': 0.017157360911369324, 'parameters': [{'parameter': 'log_kappa_scale', 'score': -8.849845886230469, 'finite_difference': -8.852005004882812, 'abs_error': 0.00215911865234375, 'relative_error': 0.00024391294573433697}, {'parameter': 'log_nu_scale', 'score': 3.3377413749694824, 'finite_difference': 3.337859869003296, 'abs_error': 0.00011849403381347656, 'relative_error': 3.550000110408291e-05}, {'parameter': 'log_obs_noise_scale', 'score': 4.532839775085449, 'finite_difference': 4.611968994140625, 'abs_error': 0.07912921905517578, 'relative_error': 0.017157360911369324}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
