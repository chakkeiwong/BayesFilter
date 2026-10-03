# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [mechanics-gaussian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-gaussian/attempt-001-result.json) | gaussian / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 3.88 |
| [mechanics-rotated_gaussian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-rotated_gaussian/attempt-001-result.json) | rotated_gaussian / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 3.98 |
| [mechanics-banana](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-banana/attempt-001-result.json) | banana / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 3.79 |
| [mechanics-funnel](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-funnel/attempt-001-result.json) | funnel / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 4.39 |
| [mechanics-student_t](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-student_t/attempt-001-result.json) | student_t / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 3.83 |
| [mechanics-cauchy](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-cauchy/attempt-001-result.json) | cauchy / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 3.94 |
| [mechanics-mixture](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-mixture/attempt-001-result.json) | mixture / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 4.13 |
| [mechanics-gamma](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-gamma/attempt-001-result.json) | gamma / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 4.23 |
| [mechanics-beta](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-beta/attempt-001-result.json) | beta / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 4.33 |
| [mechanics-dirichlet](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-dirichlet/attempt-001-result.json) | dirichlet / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 3.98 |
| [mechanics-normal_conjugate](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-normal_conjugate/attempt-001-result.json) | normal_conjugate / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 4.33 |
| [mechanics-beta_binomial](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-beta_binomial/attempt-001-result.json) | beta_binomial / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 4.03 |
| [mechanics-lgssm_location](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/mechanics-lgssm_location/attempt-001-result.json) | lgssm_location / frozen | mechanics | complete / stale | True | mechanics_passed | baseline_observed | 4.13 |
| [defect-wrong_score](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/defect-wrong_score/attempt-001-result.json) | gaussian / frozen | mechanics | complete / stale | True | mechanics_discrepancy | intended_discrepancy_detected | 4.09 |
| [defect-wrong_metric](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/defect-wrong_metric/attempt-001-result.json) | gaussian / frozen | mechanics | complete / stale | True | mechanics_discrepancy | intended_discrepancy_detected | 4.59 |
| [defect-omit_jacobian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/defect-omit_jacobian/attempt-001-result.json) | gamma / frozen | mechanics | complete / stale | True | mechanics_discrepancy | intended_discrepancy_detected | 4.18 |
| [controller-baseline](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/controller-baseline/attempt-001-result.json) | gaussian / controller | search | complete / stale | True | inventory_passed | baseline_observed | 3.99 |
| [controller-drop_candidate](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/controller-drop_candidate/attempt-001-result.json) | gaussian / controller | search | complete / stale | True | inventory_discrepancy | intended_discrepancy_detected | 3.84 |
| [controller-cross_l_epsilon](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/controller-cross_l_epsilon/attempt-001-result.json) | gaussian / controller | search | complete / stale | True | inventory_discrepancy | intended_discrepancy_detected | 3.88 |
| [controller-lost_chunk](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-fast-r2/controller-lost_chunk/attempt-001-result.json) | gaussian / controller | search | complete / stale | True | inventory_discrepancy | intended_discrepancy_detected | 4.28 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=cpu_reference, engine=mechanics, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | mechanics-gaussian |
| control=baseline, device=cpu_reference, engine=mechanics, family=quadratic, phase=development, route=frozen, start=dispersed, target=rotated_gaussian | stale | mechanics-rotated_gaussian |
| control=baseline, device=cpu_reference, engine=mechanics, family=nonlinear_transform, phase=development, route=frozen, start=dispersed, target=banana | stale | mechanics-banana |
| control=baseline, device=cpu_reference, engine=mechanics, family=hierarchical, phase=development, route=frozen, start=dispersed, target=funnel | stale | mechanics-funnel |
| control=baseline, device=cpu_reference, engine=mechanics, family=heavy_tail, phase=development, route=frozen, start=dispersed, target=student_t | stale | mechanics-student_t |
| control=baseline, device=cpu_reference, engine=mechanics, family=heavy_tail, phase=development, route=frozen, start=dispersed, target=cauchy | stale | mechanics-cauchy |
| control=baseline, device=cpu_reference, engine=mechanics, family=multimodal, phase=development, route=frozen, start=dispersed, target=mixture | stale | mechanics-mixture |
| control=baseline, device=cpu_reference, engine=mechanics, family=positive, phase=development, route=frozen, start=dispersed, target=gamma | stale | mechanics-gamma |
| control=baseline, device=cpu_reference, engine=mechanics, family=bounded, phase=development, route=frozen, start=dispersed, target=beta | stale | mechanics-beta |
| control=baseline, device=cpu_reference, engine=mechanics, family=simplex, phase=development, route=frozen, start=dispersed, target=dirichlet | stale | mechanics-dirichlet |
| control=baseline, device=cpu_reference, engine=mechanics, family=conjugate, phase=development, route=frozen, start=dispersed, target=normal_conjugate | stale | mechanics-normal_conjugate |
| control=baseline, device=cpu_reference, engine=mechanics, family=conjugate, phase=development, route=frozen, start=dispersed, target=beta_binomial | stale | mechanics-beta_binomial |
| control=baseline, device=cpu_reference, engine=mechanics, family=state_space, phase=development, route=frozen, start=dispersed, target=lgssm_location | stale | mechanics-lgssm_location |
| control=wrong_score, device=cpu_reference, engine=mechanics, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | defect-wrong_score |
| control=wrong_metric, device=cpu_reference, engine=mechanics, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | defect-wrong_metric |
| control=omit_jacobian, device=cpu_reference, engine=mechanics, family=positive, phase=development, route=frozen, start=dispersed, target=gamma | stale | defect-omit_jacobian |
| control=baseline, device=cpu_reference, engine=search, family=quadratic, phase=development, route=controller, start=dispersed, target=gaussian | stale | controller-baseline |
| control=drop_candidate, device=cpu_reference, engine=search, family=quadratic, phase=development, route=controller, start=dispersed, target=gaussian | stale | controller-drop_candidate |
| control=cross_l_epsilon, device=cpu_reference, engine=search, family=quadratic, phase=development, route=controller, start=dispersed, target=gaussian | stale | controller-cross_l_epsilon |
| control=lost_chunk, device=cpu_reference, engine=search, family=quadratic, phase=development, route=controller, start=dispersed, target=gaussian | stale | controller-lost_chunk |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
