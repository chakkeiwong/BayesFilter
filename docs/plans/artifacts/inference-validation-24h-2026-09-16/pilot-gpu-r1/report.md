# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [gpu-sbc-pilot](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/pilot-gpu-r1/gpu-sbc-pilot/attempt-001-result.json) | normal_conjugate / ordinary | sbc | complete / matches_current_source | False | calibration_incomplete | unassessed | 824.21 |
| gpu-simplex-pilot | dirichlet / prepared | accuracy | timed_out / matches_current_source | False | not assessed | unassessed | 600.62 |
| [gpu-stopping-pilot](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/pilot-gpu-r1/gpu-stopping-pilot/attempt-001-result.json) | gaussian / prepared | stopping | complete / matches_current_source | True | pipeline_assessed | baseline_observed | 676.02 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | executed | gpu-sbc-pilot |
| control=baseline, device=gpu, engine=accuracy, family=simplex, phase=development, route=prepared, start=dispersed, target=dirichlet, transport=none | uncovered | gpu-simplex-pilot |
| control=baseline, device=gpu, engine=stopping, family=quadratic, phase=development, route=prepared, start=dispersed, target=gaussian, transport=none | executed | gpu-stopping-pilot |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
