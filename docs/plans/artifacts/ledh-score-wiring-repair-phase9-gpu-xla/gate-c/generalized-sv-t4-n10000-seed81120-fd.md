# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/generalized-sv-t4-n10000-seed81120-fd.json`
- Status: `failed_fd`
- Row: `zhao_cui_generalized_sv_synthetic_from_estimated_values`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'fail', 'step': 0.0001, 'atol': 0.005, 'rtol': 0.005, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 0.015154637396335602, 'max_relative_error': 0.44275397062301636, 'parameters': [{'parameter': 'gamma_unconstrained', 'score': -0.01528964750468731, 'finite_difference': -0.019073486328125, 'abs_error': 0.0037838388234376907, 'relative_error': 0.19838212430477142}, {'parameter': 'log_tau', 'score': -0.0342281237244606, 'finite_difference': -0.019073486328125, 'abs_error': 0.015154637396335602, 'relative_error': 0.44275397062301636}, {'parameter': 'mu', 'score': -0.029310978949069977, 'finite_difference': -0.0286102294921875, 'abs_error': 0.0007007494568824768, 'relative_error': 0.023907406255602837}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
