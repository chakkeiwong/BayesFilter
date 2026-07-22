# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/fixed-sir-t1-n4-seed81120-fd.json`
- Status: `completed`
- Row: `zhao_cui_spatial_sir_austria_j9_T20`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'pass', 'step': 0.001, 'atol': 0.01, 'rtol': 0.05, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 0.15147781372070312, 'max_relative_error': 0.017033180221915245, 'parameters': [{'parameter': 'log_kappa_scale', 'score': -9.453929901123047, 'finite_difference': -9.60540771484375, 'abs_error': 0.15147781372070312, 'relative_error': 0.01577005535364151}, {'parameter': 'log_nu_scale', 'score': 3.5755198001861572, 'finite_difference': 3.601073980331421, 'abs_error': 0.025554180145263672, 'relative_error': 0.007096266373991966}, {'parameter': 'log_obs_noise_scale', 'score': 5.438969612121582, 'finite_difference': 5.533217906951904, 'abs_error': 0.09424829483032227, 'relative_error': 0.017033180221915245}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
