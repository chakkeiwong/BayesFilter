# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/ksc-sv-t1-n4-seed81120-fd.json`
- Status: `completed`
- Row: `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'pass', 'step': 0.0001, 'atol': 0.005, 'rtol': 0.005, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 0.0008873343467712402, 'max_relative_error': 0.0031043512281030416, 'parameters': [{'parameter': 'gamma_unconstrained', 'score': -0.04694149270653725, 'finite_difference': -0.047087669372558594, 'abs_error': 0.00014617666602134705, 'relative_error': 0.0031043512281030416}, {'parameter': 'log_beta', 'score': -0.4008479714393616, 'finite_difference': -0.4017353057861328, 'abs_error': 0.0008873343467712402, 'relative_error': 0.0022087537217885256}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
