# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| gpu-normal-sbc-00 | normal_conjugate / ordinary | sbc | cancelled / stale | False | fixed_epsilon_baseline_mismatch_preserve_partial_evidence | unassessed | 5650.66 |
| [gpu-normal-sbc-01](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/main-gpu-r1/gpu-normal-sbc-01/attempt-001-result.json) | normal_conjugate / ordinary | sbc | complete / stale | False | calibration_incomplete | unassessed | 5317.02 |
| gpu-normal-sbc-02 | normal_conjugate / ordinary | sbc | cancelled / stale | False | fixed_epsilon_baseline_mismatch_preserve_partial_evidence | unassessed | 5650.64 |
| gpu-normal-sbc-03 | normal_conjugate / ordinary | sbc | cancelled / stale | False | fixed_epsilon_baseline_mismatch_preserve_partial_evidence | unassessed | 5650.63 |
| gpu-normal-sbc-04 | normal_conjugate / ordinary | sbc | cancelled / stale | False | fixed_epsilon_baseline_mismatch_preserve_partial_evidence | unassessed | 333.61 |
| gpu-normal-sbc-05 | normal_conjugate / ordinary | sbc | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-normal-sbc-06 | normal_conjugate / ordinary | sbc | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-normal-sbc-07 | normal_conjugate / ordinary | sbc | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-gaussian-stopping-0 | gaussian / prepared | stopping | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-gaussian-stopping-1 | gaussian / prepared | stopping | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mixture-stopping-0 | mixture / prepared | stopping | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mixture-stopping-1 | mixture / prepared | stopping | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-beta_binomial-sbc | beta_binomial / ordinary | sbc | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-lgssm_location-sbc | lgssm_location / ordinary | sbc | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-kernel-power-eps0p3 | gaussian / frozen | power | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-kernel-power-eps0p6 | gaussian / frozen | power | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-kernel-power-eps1p0 | gaussian / frozen | power | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-affine-rotated | rotated_gaussian / fixed_transport | accuracy | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-gaussian | gaussian / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-rotated_gaussian | rotated_gaussian / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-banana | banana / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-funnel | funnel / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-student_t | student_t / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-cauchy | cauchy / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-mixture | mixture / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-gamma | gamma / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-beta | beta / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-dirichlet | dirichlet / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-normal_conjugate | normal_conjugate / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-beta_binomial | beta_binomial / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-mechanics-lgssm_location | lgssm_location / frozen | mechanics | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-simplex-selected | dirichlet / prepared | accuracy | not_run / stale | False | not assessed | unassessed | 0.00 |
| gpu-funnel-ordinary | funnel / ordinary | accuracy | not_run / stale | False | not assessed | unassessed | 0.00 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | stale | gpu-normal-sbc-00, gpu-normal-sbc-01, gpu-normal-sbc-02, gpu-normal-sbc-03, gpu-normal-sbc-04, gpu-normal-sbc-05, gpu-normal-sbc-06, gpu-normal-sbc-07 |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | stale | gpu-normal-sbc-00, gpu-normal-sbc-01, gpu-normal-sbc-02, gpu-normal-sbc-03, gpu-normal-sbc-04, gpu-normal-sbc-05, gpu-normal-sbc-06, gpu-normal-sbc-07 |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | stale | gpu-normal-sbc-00, gpu-normal-sbc-01, gpu-normal-sbc-02, gpu-normal-sbc-03, gpu-normal-sbc-04, gpu-normal-sbc-05, gpu-normal-sbc-06, gpu-normal-sbc-07 |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | stale | gpu-normal-sbc-00, gpu-normal-sbc-01, gpu-normal-sbc-02, gpu-normal-sbc-03, gpu-normal-sbc-04, gpu-normal-sbc-05, gpu-normal-sbc-06, gpu-normal-sbc-07 |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | stale | gpu-normal-sbc-00, gpu-normal-sbc-01, gpu-normal-sbc-02, gpu-normal-sbc-03, gpu-normal-sbc-04, gpu-normal-sbc-05, gpu-normal-sbc-06, gpu-normal-sbc-07 |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | stale | gpu-normal-sbc-00, gpu-normal-sbc-01, gpu-normal-sbc-02, gpu-normal-sbc-03, gpu-normal-sbc-04, gpu-normal-sbc-05, gpu-normal-sbc-06, gpu-normal-sbc-07 |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | stale | gpu-normal-sbc-00, gpu-normal-sbc-01, gpu-normal-sbc-02, gpu-normal-sbc-03, gpu-normal-sbc-04, gpu-normal-sbc-05, gpu-normal-sbc-06, gpu-normal-sbc-07 |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | stale | gpu-normal-sbc-00, gpu-normal-sbc-01, gpu-normal-sbc-02, gpu-normal-sbc-03, gpu-normal-sbc-04, gpu-normal-sbc-05, gpu-normal-sbc-06, gpu-normal-sbc-07 |
| control=baseline, device=gpu, engine=stopping, family=quadratic, phase=development, route=prepared, start=dispersed, target=gaussian, transport=none | stale | gpu-gaussian-stopping-0, gpu-gaussian-stopping-1 |
| control=baseline, device=gpu, engine=stopping, family=quadratic, phase=development, route=prepared, start=dispersed, target=gaussian, transport=none | stale | gpu-gaussian-stopping-0, gpu-gaussian-stopping-1 |
| control=baseline, device=gpu, engine=stopping, family=multimodal, phase=development, route=prepared, start=single_mode, target=mixture, transport=none | stale | gpu-mixture-stopping-0, gpu-mixture-stopping-1 |
| control=baseline, device=gpu, engine=stopping, family=multimodal, phase=development, route=prepared, start=single_mode, target=mixture, transport=none | stale | gpu-mixture-stopping-0, gpu-mixture-stopping-1 |
| control=baseline, device=gpu, engine=sbc, family=conjugate, phase=development, route=ordinary, start=dispersed, target=beta_binomial, transport=none | stale | gpu-beta_binomial-sbc |
| control=baseline, device=gpu, engine=sbc, family=state_space, phase=development, route=ordinary, start=dispersed, target=lgssm_location, transport=none | stale | gpu-lgssm_location-sbc |
| control=baseline, device=gpu, engine=power, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian, transport=none | stale | gpu-kernel-power-eps0p3, gpu-kernel-power-eps0p6, gpu-kernel-power-eps1p0 |
| control=baseline, device=gpu, engine=power, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian, transport=none | stale | gpu-kernel-power-eps0p3, gpu-kernel-power-eps0p6, gpu-kernel-power-eps1p0 |
| control=baseline, device=gpu, engine=power, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian, transport=none | stale | gpu-kernel-power-eps0p3, gpu-kernel-power-eps0p6, gpu-kernel-power-eps1p0 |
| control=baseline, device=gpu, engine=accuracy, family=quadratic, phase=development, route=fixed_transport, start=dispersed, target=rotated_gaussian, transport=bayesfilter.neutra.frozen_affine_diag.v1 | stale | gpu-affine-rotated |
| control=baseline, device=gpu, engine=mechanics, family=quadratic, phase=development, route=frozen, start=dispersed, target=gaussian, transport=none | stale | gpu-mechanics-gaussian |
| control=baseline, device=gpu, engine=mechanics, family=quadratic, phase=development, route=frozen, start=dispersed, target=rotated_gaussian, transport=none | stale | gpu-mechanics-rotated_gaussian |
| control=baseline, device=gpu, engine=mechanics, family=nonlinear_transform, phase=development, route=frozen, start=dispersed, target=banana, transport=none | stale | gpu-mechanics-banana |
| control=baseline, device=gpu, engine=mechanics, family=hierarchical, phase=development, route=frozen, start=dispersed, target=funnel, transport=none | stale | gpu-mechanics-funnel |
| control=baseline, device=gpu, engine=mechanics, family=heavy_tail, phase=development, route=frozen, start=dispersed, target=student_t, transport=none | stale | gpu-mechanics-student_t |
| control=baseline, device=gpu, engine=mechanics, family=heavy_tail, phase=development, route=frozen, start=dispersed, target=cauchy, transport=none | stale | gpu-mechanics-cauchy |
| control=baseline, device=gpu, engine=mechanics, family=multimodal, phase=development, route=frozen, start=dispersed, target=mixture, transport=none | stale | gpu-mechanics-mixture |
| control=baseline, device=gpu, engine=mechanics, family=positive, phase=development, route=frozen, start=dispersed, target=gamma, transport=none | stale | gpu-mechanics-gamma |
| control=baseline, device=gpu, engine=mechanics, family=bounded, phase=development, route=frozen, start=dispersed, target=beta, transport=none | stale | gpu-mechanics-beta |
| control=baseline, device=gpu, engine=mechanics, family=simplex, phase=development, route=frozen, start=dispersed, target=dirichlet, transport=none | stale | gpu-mechanics-dirichlet |
| control=baseline, device=gpu, engine=mechanics, family=conjugate, phase=development, route=frozen, start=dispersed, target=normal_conjugate, transport=none | stale | gpu-mechanics-normal_conjugate |
| control=baseline, device=gpu, engine=mechanics, family=conjugate, phase=development, route=frozen, start=dispersed, target=beta_binomial, transport=none | stale | gpu-mechanics-beta_binomial |
| control=baseline, device=gpu, engine=mechanics, family=state_space, phase=development, route=frozen, start=dispersed, target=lgssm_location, transport=none | stale | gpu-mechanics-lgssm_location |
| control=baseline, device=gpu, engine=accuracy, family=simplex, phase=development, route=prepared, start=dispersed, target=dirichlet, transport=none | stale | gpu-simplex-selected |
| control=baseline, device=gpu, engine=accuracy, family=hierarchical, phase=development, route=ordinary, start=dispersed, target=funnel, transport=none | stale | gpu-funnel-ordinary |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
