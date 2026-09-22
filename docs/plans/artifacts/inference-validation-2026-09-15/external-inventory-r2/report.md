# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| external-regression | regression / external | accuracy | unavailable / matches_current_source | False | external observation and reference bundles required | unassessed | 0.00 |
| external-eight_schools | eight_schools / external | accuracy | unavailable / matches_current_source | False | external observation and reference bundles required | unassessed | 0.00 |
| external-macrofinance | macrofinance / external | accuracy | unavailable / matches_current_source | False | external observation and reference bundles required | unassessed | 0.00 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=cpu_reference, engine=accuracy, family=regression, phase=development, route=external, start=dispersed, target=regression | uncovered | external-regression |
| control=baseline, device=cpu_reference, engine=accuracy, family=hierarchical, phase=development, route=external, start=dispersed, target=eight_schools | uncovered | external-eight_schools |
| control=baseline, device=cpu_reference, engine=accuracy, family=consumer, phase=development, route=external, start=dispersed, target=macrofinance | uncovered | external-macrofinance |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
