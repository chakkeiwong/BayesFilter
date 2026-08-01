# GenUT Antithetic LGSSM And SV Result

Date: 2026-07-22

Status: `ANTITHETIC_PARTIAL_COORDINATE_NOMINATION_FEASIBILITY_ONLY`

The primary comparator is an equal-cost average of two independent complete
GenUT runs. The single-cloud arm is descriptive only.

## LGSSM

Selected controls: `{'epsilon': 2.0, 'sinkhorn_steps': 8, 'balance_steps': 8, 'ridge': 1e-05}`.

| Coordinate | Geometric variance ratio | Familywise 95% log-ratio CI | Datasets lower | Nominated |
|---|---:|---:|---:|---|
| value | 0.7298 | [-0.6909, 0.0609] | 7/8 | False |
| phi1 | 0.8034 | [-0.7947, 0.3569] | 4/8 | False |
| phi2 | 0.9237 | [-0.8105, 0.6519] | 4/8 | False |
| phi3 | 0.9345 | [-0.7344, 0.5988] | 5/8 | False |
| q_scale | 1.2850 | [-0.0615, 0.5631] | 1/8 | False |
| r_scale | 0.9641 | [-0.6655, 0.5924] | 4/8 | False |

## SV

Selected controls: `{'epsilon': 4.0, 'sinkhorn_steps': 4, 'balance_steps': 4, 'ridge': 1e-06}`.

| Coordinate | Geometric variance ratio | Familywise 95% log-ratio CI | Datasets lower | Nominated |
|---|---:|---:|---:|---|
| value | 0.4916 | [-1.1729, -0.2472] | 7/8 | True |
| theta_gamma | 0.6769 | [-0.8681, 0.0876] | 6/8 | False |
| theta_log_beta | 0.7091 | [-0.7321, 0.0444] | 7/8 | False |

## Decision

Antithetic averaging reduced conditional variance only for a subset of coordinates under the equal-cost screen. It remains an optional experimental coupling.

This feasibility campaign does not change the default. Dataset-level mean
error and MSE diagnostics are retained in `result.json`.
