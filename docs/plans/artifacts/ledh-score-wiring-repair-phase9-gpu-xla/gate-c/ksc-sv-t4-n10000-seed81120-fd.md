# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/ksc-sv-t4-n10000-seed81120-fd.json`
- Status: `failed_fd`
- Row: `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'fail', 'step': 0.0001, 'atol': 0.005, 'rtol': 0.005, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 0.010241001844406128, 'max_relative_error': 0.03693515062332153, 'parameters': [{'parameter': 'gamma_unconstrained', 'score': -0.27726981043815613, 'finite_difference': -0.26702880859375, 'abs_error': 0.010241001844406128, 'relative_error': 0.03693515062332153}, {'parameter': 'log_beta', 'score': 0.1562928855419159, 'finite_difference': 0.1621246337890625, 'abs_error': 0.0058317482471466064, 'relative_error': 0.0359707735478878}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
