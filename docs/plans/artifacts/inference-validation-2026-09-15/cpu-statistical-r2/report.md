# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [invariance-gaussian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-gaussian/attempt-001-result.json) | gaussian / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.93 |
| [invariance-rotated_gaussian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-rotated_gaussian/attempt-001-result.json) | rotated_gaussian / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.58 |
| [invariance-banana](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-banana/attempt-001-result.json) | banana / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.53 |
| [invariance-funnel](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-funnel/attempt-001-result.json) | funnel / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.88 |
| [invariance-student_t](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-student_t/attempt-001-result.json) | student_t / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.53 |
| [invariance-cauchy](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-cauchy/attempt-001-result.json) | cauchy / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.64 |
| [invariance-mixture](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-mixture/attempt-001-result.json) | mixture / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.24 |
| [invariance-gamma](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-gamma/attempt-001-result.json) | gamma / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.14 |
| [invariance-beta](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-beta/attempt-001-result.json) | beta / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.64 |
| [invariance-dirichlet](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-dirichlet/attempt-001-result.json) | dirichlet / frozen | invariance | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.59 |
| [invariance-identity](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-identity/attempt-001-result.json) | gaussian / frozen | invariance | complete / stale | True | no_discrepancy_detected | invariance_preserving_control_observed | 4.14 |
| [invariance-two_cycle](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-two_cycle/attempt-001-result.json) | gaussian / frozen | invariance | complete / stale | True | no_discrepancy_detected | invariance_preserving_control_observed | 4.19 |
| [invariance-wrong_energy](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-wrong_energy/attempt-001-result.json) | gaussian / frozen | invariance | complete / stale | True | discrepancy_detected | intended_discrepancy_detected | 4.69 |
| [invariance-omit_jacobian](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/invariance-omit_jacobian/attempt-001-result.json) | gamma / frozen | invariance | complete / stale | True | discrepancy_detected | intended_discrepancy_detected | 4.54 |
| [sbc-reference-normal_conjugate](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/sbc-reference-normal_conjugate/attempt-001-result.json) | normal_conjugate / reference | sbc | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.79 |
| [sbc-reference-beta_binomial](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/sbc-reference-beta_binomial/attempt-001-result.json) | beta_binomial / reference | sbc | complete / stale | True | no_discrepancy_detected | baseline_observed | 4.39 |
| [sbc-reference-lgssm_location](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/sbc-reference-lgssm_location/attempt-001-result.json) | lgssm_location / reference | sbc | complete / stale | True | no_discrepancy_detected | baseline_observed | 7.19 |
| [sbc-ignored-data](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/sbc-ignored-data/attempt-001-result.json) | normal_conjugate / reference | sbc | complete / stale | True | discrepancy_detected | intended_discrepancy_detected | 4.89 |
| [primitive-power](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/primitive-power/attempt-001-result.json) | normal_conjugate / reference | power | complete / stale | True | power_estimated | baseline_observed | 9.45 |
| [diagnostics-stationary](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/diagnostics-stationary/attempt-001-result.json) | gaussian / reference | stopping | complete / stale | True | diagnostics_assessed | baseline_observed | 5.35 |
| [diagnostics-transient](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/diagnostics-transient/attempt-001-result.json) | gaussian / reference | stopping | complete / stale | True | diagnostics_assessed | baseline_observed | 5.09 |
| [diagnostics-constant](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/diagnostics-constant/attempt-001-result.json) | gaussian / reference | stopping | complete / stale | True | diagnostics_assessed | baseline_observed | 4.68 |
| [diagnostics-missed_mode](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-2026-09-15/cpu-statistical-r2/diagnostics-missed_mode/attempt-001-result.json) | mixture / reference | stopping | complete / stale | True | diagnostics_assessed | baseline_observed | 5.29 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=cpu_reference, engine=invariance, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | invariance-gaussian |
| control=baseline, device=cpu_reference, engine=invariance, family=quadratic, phase=development, route=frozen, start=dispersed, target=rotated_gaussian | stale | invariance-rotated_gaussian |
| control=baseline, device=cpu_reference, engine=invariance, family=nonlinear_transform, phase=development, route=frozen, start=dispersed, target=banana | stale | invariance-banana |
| control=baseline, device=cpu_reference, engine=invariance, family=hierarchical, phase=development, route=frozen, start=dispersed, target=funnel | stale | invariance-funnel |
| control=baseline, device=cpu_reference, engine=invariance, family=heavy_tail, phase=development, route=frozen, start=dispersed, target=student_t | stale | invariance-student_t |
| control=baseline, device=cpu_reference, engine=invariance, family=heavy_tail, phase=development, route=frozen, start=dispersed, target=cauchy | stale | invariance-cauchy |
| control=baseline, device=cpu_reference, engine=invariance, family=multimodal, phase=development, route=frozen, start=dispersed, target=mixture | stale | invariance-mixture |
| control=baseline, device=cpu_reference, engine=invariance, family=positive, phase=development, route=frozen, start=dispersed, target=gamma | stale | invariance-gamma |
| control=baseline, device=cpu_reference, engine=invariance, family=bounded, phase=development, route=frozen, start=dispersed, target=beta | stale | invariance-beta |
| control=baseline, device=cpu_reference, engine=invariance, family=simplex, phase=development, route=frozen, start=dispersed, target=dirichlet | stale | invariance-dirichlet |
| control=identity, device=cpu_reference, engine=invariance, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | invariance-identity |
| control=two_cycle, device=cpu_reference, engine=invariance, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | invariance-two_cycle |
| control=wrong_energy, device=cpu_reference, engine=invariance, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian | stale | invariance-wrong_energy |
| control=omit_jacobian, device=cpu_reference, engine=invariance, family=positive, phase=development, route=frozen, start=dispersed, target=gamma | stale | invariance-omit_jacobian |
| control=baseline, device=cpu_reference, engine=sbc, family=conjugate, phase=development, route=reference, start=dispersed, target=normal_conjugate | stale | sbc-reference-normal_conjugate |
| control=baseline, device=cpu_reference, engine=sbc, family=conjugate, phase=development, route=reference, start=dispersed, target=beta_binomial | stale | sbc-reference-beta_binomial |
| control=baseline, device=cpu_reference, engine=sbc, family=state_space, phase=development, route=reference, start=dispersed, target=lgssm_location | stale | sbc-reference-lgssm_location |
| control=ignore_data, device=cpu_reference, engine=sbc, family=conjugate, phase=development, route=reference, start=dispersed, target=normal_conjugate | stale | sbc-ignored-data |
| control=baseline, device=cpu_reference, engine=power, family=conjugate, phase=development, route=reference, start=dispersed, target=normal_conjugate | stale | primitive-power |
| control=baseline, device=cpu_reference, engine=stopping, family=quadratic, phase=development, route=reference, start=dispersed, target=gaussian | stale | diagnostics-stationary, diagnostics-transient, diagnostics-constant |
| control=baseline, device=cpu_reference, engine=stopping, family=quadratic, phase=development, route=reference, start=dispersed, target=gaussian | stale | diagnostics-stationary, diagnostics-transient, diagnostics-constant |
| control=baseline, device=cpu_reference, engine=stopping, family=quadratic, phase=development, route=reference, start=dispersed, target=gaussian | stale | diagnostics-stationary, diagnostics-transient, diagnostics-constant |
| control=baseline, device=cpu_reference, engine=stopping, family=multimodal, phase=development, route=reference, start=dispersed, target=mixture | stale | diagnostics-missed_mode |
| device=gpu, engine=sbc, phase=confirmation, route=ordinary, target=normal_conjugate | uncovered | none |
| engine=accuracy, route=external, target=macrofinance | uncovered | none |
| engine=accuracy, route=external, target=eight_schools | uncovered | none |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
