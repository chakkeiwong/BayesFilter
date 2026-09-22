# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [cpu-sbc-pilot](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/pilot-cpu-r1/cpu-sbc-pilot/attempt-001-result.json) | normal_conjugate / ordinary | sbc | complete / matches_current_source | True | no_discrepancy_detected | baseline_observed | 406.36 |
| [cpu-simplex-pilot](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/pilot-cpu-r1/cpu-simplex-pilot/attempt-001-result.json) | dirichlet / prepared | accuracy | complete / matches_current_source | True | pipeline_assessed | baseline_observed | 540.90 |
| [cpu-stopping-pilot](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/pilot-cpu-r1/cpu-stopping-pilot/attempt-001-result.json) | gaussian / prepared | stopping | complete / matches_current_source | True | pipeline_assessed | baseline_observed | 317.43 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=cpu_reference, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | executed | cpu-sbc-pilot |
| control=baseline, device=cpu_reference, engine=accuracy, family=simplex, phase=development, route=prepared, start=dispersed, target=dirichlet, transport=none | executed | cpu-simplex-pilot |
| control=baseline, device=cpu_reference, engine=stopping, family=quadratic, phase=development, route=prepared, start=dispersed, target=gaussian, transport=none | executed | cpu-stopping-pilot |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
