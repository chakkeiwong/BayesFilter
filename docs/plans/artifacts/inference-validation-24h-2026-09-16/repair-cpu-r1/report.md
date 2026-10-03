# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [cpu-native-search-0](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/repair-cpu-r1/cpu-native-search-0/attempt-001-result.json) | normal_conjugate / ordinary | sbc | complete / matches_current_source | False | calibration_incomplete | unassessed | 378.34 |
| [cpu-native-search-1](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/repair-cpu-r1/cpu-native-search-1/attempt-001-result.json) | normal_conjugate / ordinary | sbc | complete / matches_current_source | False | calibration_incomplete | unassessed | 255.42 |
| [cpu-native-search-2](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/repair-cpu-r1/cpu-native-search-2/attempt-001-result.json) | normal_conjugate / ordinary | sbc | complete / matches_current_source | True | no_discrepancy_detected | baseline_observed | 710.50 |
| [cpu-native-search-3](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/repair-cpu-r1/cpu-native-search-3/attempt-001-result.json) | normal_conjugate / ordinary | sbc | complete / matches_current_source | False | calibration_incomplete | unassessed | 479.60 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=cpu_reference, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | executed | cpu-native-search-0, cpu-native-search-1, cpu-native-search-2, cpu-native-search-3 |
| control=baseline, device=cpu_reference, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | executed | cpu-native-search-0, cpu-native-search-1, cpu-native-search-2, cpu-native-search-3 |
| control=baseline, device=cpu_reference, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | executed | cpu-native-search-0, cpu-native-search-1, cpu-native-search-2, cpu-native-search-3 |
| control=baseline, device=cpu_reference, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | executed | cpu-native-search-0, cpu-native-search-1, cpu-native-search-2, cpu-native-search-3 |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
