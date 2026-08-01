# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/generalized-sv-t1-n4-seed81120-fd.json`
- Status: `completed`
- Row: `zhao_cui_generalized_sv_synthetic_from_estimated_values`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'pass', 'step': 0.0001, 'atol': 0.005, 'rtol': 0.005, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 0.0009666457772254944, 'max_relative_error': 0.04161176085472107, 'parameters': [{'parameter': 'gamma_unconstrained', 'score': 0.021767405793070793, 'finite_difference': 0.02086162567138672, 'abs_error': 0.0009057801216840744, 'relative_error': 0.04161176085472107}, {'parameter': 'log_tau', 'score': 0.03053959645330906, 'finite_difference': 0.03039836883544922, 'abs_error': 0.0001412276178598404, 'relative_error': 0.004624410066753626}, {'parameter': 'mu', 'score': -0.0365842804312706, 'finite_difference': -0.037550926208496094, 'abs_error': 0.0009666457772254944, 'relative_error': 0.025742262601852417}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
