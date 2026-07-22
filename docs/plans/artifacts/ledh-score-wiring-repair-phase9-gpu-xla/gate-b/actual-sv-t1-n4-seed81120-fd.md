# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/actual-sv-t1-n4-seed81120-fd.json`
- Status: `completed`
- Row: `zhao_cui_sv_actual_nongaussian_T1000`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'pass', 'step': 0.0001, 'atol': 0.005, 'rtol': 0.005, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 0.0011738166213035583, 'max_relative_error': 0.019119782373309135, 'parameters': [{'parameter': 'gamma_unconstrained', 'score': -0.060218967497348785, 'finite_difference': -0.061392784118652344, 'abs_error': 0.0011738166213035583, 'relative_error': 0.019119782373309135}, {'parameter': 'log_beta', 'score': -0.40035516023635864, 'finite_difference': -0.4011392593383789, 'abs_error': 0.0007840991020202637, 'relative_error': 0.0019546805415302515}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
