# Inference validation results

Execution, statistical finding and expected defect detection are separate; no universal validation claim.

| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| external-regression | regression / external | accuracy | unavailable / stale | False | external observation and reference bundles required | unassessed | 0.00 |
| external-eight_schools | eight_schools / external | accuracy | unavailable / stale | False | external observation and reference bundles required | unassessed | 0.00 |
| external-macrofinance | macrofinance / external | accuracy | unavailable / stale | False | external observation and reference bundles required | unassessed | 0.00 |

## Required coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| control=baseline, device=cpu_reference, engine=accuracy, family=regression, phase=development, route=external, start=dispersed, target=regression | stale | external-regression |
| control=baseline, device=cpu_reference, engine=accuracy, family=hierarchical, phase=development, route=external, start=dispersed, target=eight_schools | stale | external-eight_schools |
| control=baseline, device=cpu_reference, engine=accuracy, family=consumer, phase=development, route=external, start=dispersed, target=macrofinance | stale | external-macrofinance |

Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.
