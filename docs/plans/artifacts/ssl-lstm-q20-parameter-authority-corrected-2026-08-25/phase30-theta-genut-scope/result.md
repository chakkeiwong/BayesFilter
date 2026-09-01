# Corrected q=20 Parameter-Space GenUT

Status: `PARAMETER_GENUT_GLOBAL_INFEASIBLE_SCOPE`

GenUT was evaluated in theta in R^4. Any global infeasibility is a scope result, not a theorem about all local uses.

| Scope | Status | Feasible | Target valid |
|---|---|---:|---:|
| global_theta_R4 | `PARAMETER_GENUT_INFEASIBLE_SCOPE` | `False` | `False` |
| negative_axis2_theta_R4 | `PARAMETER_GENUT_INFEASIBLE_SCOPE` | `False` | `False` |
| positive_axis2_theta_R4 | `PARAMETER_GENUT_INFEASIBLE_SCOPE` | `False` | `False` |

## Nonclaims

- sigma points are not IID posterior samples and receive no density claim
- moment residuals do not establish mode discovery or posterior correctness
- no clipping or negative-weight repair was applied
