# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| [rank-power-n64-bias0p25](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/rank-power-r1/rank-power-n64-bias0p25/attempt-001-result.json) | normal_conjugate / reference | power | complete / matches_current_source | True | power_estimated | baseline_observed | 7.54 |
| [rank-power-n64-bias0p5](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/rank-power-r1/rank-power-n64-bias0p5/attempt-001-result.json) | normal_conjugate / reference | power | complete / matches_current_source | True | power_estimated | baseline_observed | 7.59 |
| [rank-power-n64-bias1p0](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/rank-power-r1/rank-power-n64-bias1p0/attempt-001-result.json) | normal_conjugate / reference | power | complete / matches_current_source | True | power_estimated | baseline_observed | 7.24 |
| [rank-power-n128-bias0p25](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/rank-power-r1/rank-power-n128-bias0p25/attempt-001-result.json) | normal_conjugate / reference | power | complete / matches_current_source | True | power_estimated | baseline_observed | 8.74 |
| [rank-power-n128-bias0p5](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/rank-power-r1/rank-power-n128-bias0p5/attempt-001-result.json) | normal_conjugate / reference | power | complete / matches_current_source | True | power_estimated | baseline_observed | 8.64 |
| [rank-power-n128-bias1p0](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/rank-power-r1/rank-power-n128-bias1p0/attempt-001-result.json) | normal_conjugate / reference | power | complete / matches_current_source | True | power_estimated | baseline_observed | 8.64 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=cpu_reference, engine=power, family=conjugate, phase=development, route=reference, start=dispersed, target=normal_conjugate, transport=none | executed | rank-power-n64-bias0p25, rank-power-n64-bias0p5, rank-power-n64-bias1p0, rank-power-n128-bias0p25, rank-power-n128-bias0p5, rank-power-n128-bias1p0 |
| control=baseline, device=cpu_reference, engine=power, family=conjugate, phase=development, route=reference, start=dispersed, target=normal_conjugate, transport=none | executed | rank-power-n64-bias0p25, rank-power-n64-bias0p5, rank-power-n64-bias1p0, rank-power-n128-bias0p25, rank-power-n128-bias0p5, rank-power-n128-bias1p0 |
| control=baseline, device=cpu_reference, engine=power, family=conjugate, phase=development, route=reference, start=dispersed, target=normal_conjugate, transport=none | executed | rank-power-n64-bias0p25, rank-power-n64-bias0p5, rank-power-n64-bias1p0, rank-power-n128-bias0p25, rank-power-n128-bias0p5, rank-power-n128-bias1p0 |
| control=baseline, device=cpu_reference, engine=power, family=conjugate, phase=development, route=reference, start=dispersed, target=normal_conjugate, transport=none | executed | rank-power-n64-bias0p25, rank-power-n64-bias0p5, rank-power-n64-bias1p0, rank-power-n128-bias0p25, rank-power-n128-bias0p5, rank-power-n128-bias1p0 |
| control=baseline, device=cpu_reference, engine=power, family=conjugate, phase=development, route=reference, start=dispersed, target=normal_conjugate, transport=none | executed | rank-power-n64-bias0p25, rank-power-n64-bias0p5, rank-power-n64-bias1p0, rank-power-n128-bias0p25, rank-power-n128-bias0p5, rank-power-n128-bias1p0 |
| control=baseline, device=cpu_reference, engine=power, family=conjugate, phase=development, route=reference, start=dispersed, target=normal_conjugate, transport=none | executed | rank-power-n64-bias0p25, rank-power-n64-bias0p5, rank-power-n64-bias1p0, rank-power-n128-bias0p25, rank-power-n128-bias0p5, rank-power-n128-bias1p0 |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
