# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [prepared-gaussian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-pipeline-r2/prepared-gaussian/attempt-001-result.json) | gaussian / prepared | accuracy | complete / stale | False | pipeline_assessed | baseline_observed | 93.53 |
| [stopped-gaussian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-pipeline-r2/stopped-gaussian/attempt-001-result.json) | gaussian / prepared | stopping | complete / stale | False | pipeline_assessed | baseline_observed | 186.48 |
| [missed-mode-stopping](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-pipeline-r2/missed-mode-stopping/attempt-001-result.json) | mixture / prepared | stopping | complete / stale | False | pipeline_discrepancy | discrepancy_under_baseline | 100.27 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=cpu_reference, engine=accuracy, family=quadratic, phase=development, route=prepared, start=dispersed, target=gaussian | stale | prepared-gaussian |
| control=baseline, device=cpu_reference, engine=stopping, family=quadratic, phase=development, route=prepared, start=dispersed, target=gaussian | stale | stopped-gaussian |
| control=baseline, device=cpu_reference, engine=stopping, family=multimodal, phase=development, route=prepared, start=single_mode, target=mixture | stale | missed-mode-stopping |
| control=baseline, device=cpu_reference, engine=accuracy, family=bounded, phase=development, route=prepared, start=dispersed, target=beta | uncovered | none |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
