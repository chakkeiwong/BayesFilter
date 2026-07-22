# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/predator-prey-t1-n2-seed81120-fd.json`
- Status: `failed_fd`
- Row: `zhao_cui_predator_prey_T20`
- Stage: `fd-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'fail', 'step': 0.0001, 'atol': 0.005, 'rtol': 0.005, 'pass_rule': 'max_abs_error <= atol OR max_relative_error <= rtol', 'max_abs_error': 0.3162194490432739, 'max_relative_error': 1.0, 'parameters': [{'parameter': 'r', 'score': -183.59210205078125, 'finite_difference': -183.4869384765625, 'abs_error': 0.10516357421875, 'relative_error': 0.0005728109972551465}, {'parameter': 'K', 'score': -1.308040738105774, 'finite_difference': -0.9918212890625, 'abs_error': 0.3162194490432739, 'relative_error': 0.24175046384334564}, {'parameter': 'a', 'score': -0.1401509791612625, 'finite_difference': 0.0, 'abs_error': 0.1401509791612625, 'relative_error': 1.0}, {'parameter': 's', 'score': 26.561914443969727, 'finite_difference': 26.58843994140625, 'abs_error': 0.026525497436523438, 'relative_error': 0.0009976327419281006}, {'parameter': 'u', 'score': 8.439687728881836, 'finite_difference': 8.27789306640625, 'abs_error': 0.16179466247558594, 'relative_error': 0.019170692190527916}, {'parameter': 'v', 'score': -11.972293853759766, 'finite_difference': -11.8255615234375, 'abs_error': 0.14673233032226562, 'relative_error': 0.01225599180907011}], 'uses_value_only_scalar_route': True}`
- Memory: `None`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
