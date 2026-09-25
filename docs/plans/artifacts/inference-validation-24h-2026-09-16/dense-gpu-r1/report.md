# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [gpu-dense-iaf-full-posterior](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/dense-gpu-r1/gpu-dense-iaf-full-posterior/attempt-001-result.json) | gaussian / fixed_transport | accuracy | complete / matches_current_source | True | pipeline_assessed | baseline_observed | 352.25 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=gpu, engine=accuracy, family=quadratic, phase=development, route=fixed_transport, start=dispersed, target=gaussian, transport=bayesfilter.neutra.dense_iaf_frozen_transport.v1 | executed | gpu-dense-iaf-full-posterior |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
