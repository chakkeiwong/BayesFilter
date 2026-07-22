# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/actual-sv-t4-n10000-seed81120-fd.json`
- Status: `failed_fd`
- Row: `zhao_cui_sv_actual_nongaussian_T1000`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'fail', 'step': 0.0001, 'atol': 0.005, 'rtol': 0.005, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 0.00948423147201538, 'max_relative_error': 0.06029246747493744, 'parameters': [{'parameter': 'gamma_unconstrained', 'score': -0.2760983407497406, 'finite_difference': -0.2765655517578125, 'abs_error': 0.0004672110080718994, 'relative_error': 0.0016893318388611078}, {'parameter': 'log_beta', 'score': 0.15730375051498413, 'finite_difference': 0.14781951904296875, 'abs_error': 0.00948423147201538, 'relative_error': 0.06029246747493744}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
