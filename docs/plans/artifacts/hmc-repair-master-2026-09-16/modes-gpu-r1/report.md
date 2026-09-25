# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [master-mode-single-mode](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/hmc-repair-master-2026-09-16/modes-gpu-r1/master-mode-single-mode/attempt-001-result.json) | mixture / prepared | stopping | complete / matches_current_source | False | posterior_incomplete | unassessed | 644.75 |
| [master-mode-mode-dispersed](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/hmc-repair-master-2026-09-16/modes-gpu-r1/master-mode-mode-dispersed/attempt-001-result.json) | mixture / prepared | stopping | complete / matches_current_source | False | posterior_incomplete | unassessed | 607.02 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=gpu, engine=stopping, family=multimodal, phase=development, route=prepared, start=single_mode, target=mixture, transport=none | executed | master-mode-single-mode |
| control=baseline, device=gpu, engine=stopping, family=multimodal, phase=development, route=prepared, start=mode_dispersed, target=mixture, transport=none | executed | master-mode-mode-dispersed |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
