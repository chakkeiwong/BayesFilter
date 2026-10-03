# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| automatic-gaussian | gaussian / ordinary | accuracy | timed_out / stale | False | not assessed | unassessed | 180.22 |
| prepared-gaussian | gaussian / prepared | accuracy | timed_out / stale | False | not assessed | unassessed | 90.17 |
| [frozen-transport-gaussian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-pipeline-r1/frozen-transport-gaussian/attempt-001-result.json) | gaussian / fixed_transport | accuracy | complete / stale | False | pipeline_assessed | baseline_observed | 50.18 |
| stopped-gaussian | gaussian / prepared | stopping | timed_out / stale | False | not assessed | unassessed | 120.17 |
| [full-procedure-sbc](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-pipeline-r1/full-procedure-sbc/attempt-001-result.json) | normal_conjugate / ordinary | sbc | complete / stale | False | calibration_incomplete | baseline_observed | 103.31 |
| missed-mode-stopping | mixture / prepared | stopping | timed_out / stale | False | not assessed | unassessed | 100.17 |
| [constrained-beta](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-pipeline-r1/constrained-beta/attempt-001-result.json) | beta / prepared | accuracy | complete / stale | True | pipeline_assessed | baseline_observed | 52.73 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=cpu_reference, engine=accuracy, family=quadratic, phase=development, route=ordinary, start=dispersed, target=gaussian | stale | automatic-gaussian |
| control=baseline, device=cpu_reference, engine=accuracy, family=quadratic, phase=development, route=prepared, start=dispersed, target=gaussian | stale | prepared-gaussian |
| control=baseline, device=cpu_reference, engine=accuracy, family=quadratic, phase=development, route=fixed_transport, start=dispersed, target=gaussian | stale | frozen-transport-gaussian |
| control=baseline, device=cpu_reference, engine=stopping, family=quadratic, phase=development, route=prepared, start=dispersed, target=gaussian | stale | stopped-gaussian |
| control=baseline, device=cpu_reference, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate | stale | full-procedure-sbc |
| control=baseline, device=cpu_reference, engine=stopping, family=multimodal, phase=development, route=prepared, start=single_mode, target=mixture | stale | missed-mode-stopping |
| control=baseline, device=cpu_reference, engine=accuracy, family=bounded, phase=development, route=prepared, start=dispersed, target=beta | stale | constrained-beta |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
