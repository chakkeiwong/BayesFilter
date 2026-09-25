# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| c2-confirm-baseline | normal_conjugate / ordinary | reference_mean | failed / matches_current_source | False | not assessed | unassessed | 2443.98 |
| c2-confirm-quarter | normal_conjugate / ordinary | reference_mean | failed / matches_current_source | False | not assessed | unassessed | 1742.12 |
| c2-confirm-half | normal_conjugate / ordinary | reference_mean | failed / matches_current_source | False | not assessed | unassessed | 891.27 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, design_id=c2-confirm-baseline, device=gpu, engine=reference_mean, family=conjugate, phase=confirmation, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | uncovered | c2-confirm-baseline |
| control=location_shift, design_id=c2-confirm-quarter, device=gpu, engine=reference_mean, family=conjugate, phase=confirmation, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | uncovered | c2-confirm-quarter |
| control=location_shift, design_id=c2-confirm-half, device=gpu, engine=reference_mean, family=conjugate, phase=confirmation, route=ordinary, start=dispersed, target=normal_conjugate, transport=none | uncovered | c2-confirm-half |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
