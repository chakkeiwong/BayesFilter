# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [gpu-mechanics-gaussian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-terminal-r1/gpu-mechanics-gaussian/attempt-001-result.json) | gaussian / frozen | mechanics | complete / matches_current_source | True | mechanics_passed | baseline_observed | 5.43 |
| [gpu-nonlinear-transport](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-terminal-r1/gpu-nonlinear-transport/attempt-001-result.json) | gaussian / fixed_transport | accuracy | complete / matches_current_source | True | pipeline_assessed | baseline_observed | 183.15 |
| [gpu-invariance-wrong_energy](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-terminal-r1/gpu-invariance-wrong_energy/attempt-001-result.json) | gaussian / frozen | invariance | complete / matches_current_source | True | discrepancy_detected | intended_discrepancy_detected | 7.63 |
| gpu-automatic-gaussian | gaussian / ordinary | accuracy | timed_out / matches_current_source | False | not assessed | unassessed | 600.78 |
| gpu-full-sbc | normal_conjugate / ordinary | sbc | timed_out / matches_current_source | False | not assessed | unassessed | 800.98 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=gpu, engine=mechanics, route=frozen, target=gaussian | executed | gpu-mechanics-gaussian |
| control=baseline, device=gpu, engine=accuracy, route=fixed_transport, target=gaussian, transport=bayesfilter.neutra.dense_iaf_frozen_transport.v1 | executed | gpu-nonlinear-transport |
| control=wrong_energy, device=gpu, engine=invariance, route=frozen, target=gaussian | executed | gpu-invariance-wrong_energy |
| control=baseline, device=gpu, engine=accuracy, route=ordinary, target=gaussian | uncovered | gpu-automatic-gaussian |
| control=baseline, device=gpu, engine=sbc, route=ordinary, target=normal_conjugate | uncovered | gpu-full-sbc |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
