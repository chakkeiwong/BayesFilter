# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/fixed-sir-t5-n10000-seed81120-fd.json`
- Status: `completed`
- Row: `zhao_cui_spatial_sir_austria_j9_T20`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'pass', 'step': 0.001, 'atol': 0.01, 'rtol': 0.05, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 2.568359375, 'max_relative_error': 0.0020681782625615597, 'parameters': [{'parameter': 'log_kappa_scale', 'score': -1358.4566650390625, 'finite_difference': -1355.8883056640625, 'abs_error': 2.568359375, 'relative_error': 0.001890645013190806}, {'parameter': 'log_nu_scale', 'score': 638.9251708984375, 'finite_difference': 637.603759765625, 'abs_error': 1.3214111328125, 'relative_error': 0.0020681782625615597}, {'parameter': 'log_obs_noise_scale', 'score': 155.28147888183594, 'finite_difference': 155.00640869140625, 'abs_error': 0.2750701904296875, 'relative_error': 0.001771429437212646}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
