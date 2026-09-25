# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [gpu-mechanics-gaussian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-r1/gpu-mechanics-gaussian/attempt-001-result.json) | gaussian / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 5.63 |
| [gpu-mechanics-gamma](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-r1/gpu-mechanics-gamma/attempt-001-result.json) | gamma / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 4.48 |
| [gpu-mechanics-funnel](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-r1/gpu-mechanics-funnel/attempt-001-result.json) | funnel / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 5.03 |
| [gpu-invariance-baseline](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-r1/gpu-invariance-baseline/attempt-001-result.json) | gaussian / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 6.83 |
| [gpu-invariance-wrong_energy](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-r1/gpu-invariance-wrong_energy/attempt-001-result.json) | gaussian / frozen | invariance | complete / stale | True | discrepancy_detected | intended_discrepancy_detected | 6.89 |
| [gpu-invariance-identity](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-r1/gpu-invariance-identity/attempt-001-result.json) | gaussian / frozen | invariance | complete / stale | True | no_discrepancy_detected | invariance_preserving_control_observed | 4.68 |
| [gpu-automatic-gaussian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-r1/gpu-automatic-gaussian/attempt-001-result.json) | gaussian / ordinary | accuracy | complete / stale | True | pipeline_assessed | baseline_observed | 500.96 |
| [gpu-fixed-transport](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-r1/gpu-fixed-transport/attempt-001-result.json) | gaussian / fixed_transport | accuracy | complete / stale | False | pipeline_assessed | baseline_observed | 144.52 |
| [gpu-full-sbc](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-r1/gpu-full-sbc/attempt-001-result.json) | normal_conjugate / ordinary | sbc | complete / stale | False | calibration_incomplete | baseline_observed | 681.75 |
| [gpu-hmc-defect-power](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/gpu-r1/gpu-hmc-defect-power/attempt-001-result.json) | gaussian / frozen | power | complete / stale | True | power_estimated | baseline_observed | 41.17 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=gpu, engine=mechanics, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | gpu-mechanics-gaussian |
| control=baseline, device=gpu, engine=mechanics, family=positive, phase=development, route=frozen, start=dispersed, target=gamma | stale | gpu-mechanics-gamma |
| control=baseline, device=gpu, engine=mechanics, family=hierarchical, phase=development, route=frozen, start=dispersed, target=funnel | stale | gpu-mechanics-funnel |
| control=baseline, device=gpu, engine=invariance, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | gpu-invariance-baseline |
| control=wrong_energy, device=gpu, engine=invariance, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | gpu-invariance-wrong_energy |
| control=identity, device=gpu, engine=invariance, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | gpu-invariance-identity |
| control=baseline, device=gpu, engine=accuracy, family=quadratic, phase=development, route=ordinary, start=dispersed, target=gaussian | stale | gpu-automatic-gaussian |
| control=baseline, device=gpu, engine=accuracy, family=quadratic, phase=development, route=fixed_transport, start=dispersed, target=gaussian | stale | gpu-fixed-transport |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate | stale | gpu-full-sbc |
| control=baseline, device=gpu, engine=power, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | gpu-hmc-defect-power |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
