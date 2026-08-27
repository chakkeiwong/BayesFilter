# Phase 20 Result: Source-Faithful GenUT Fixture

Status: `PASS_SOURCE_FAITHFUL_GENUT_FIXTURE`

The TensorFlow CPU-hidden 25-point/two-dimensional weighted grid passed the
Ebeigbe et al. asymmetric `2d+1` construction and selected-moment gates.

| Gate | Receipt |
|---|---:|
| feasible discriminants/offsets/weights | true |
| standardized skewness | `0.0503103` on both axes |
| standardized kurtosis | `2.56191` on both axes |
| central weight | `0.218559` |
| mean residual | `5.55e-17` |
| covariance residual | `1.00e-10` (ridge-limited) |
| diagonal third-moment residual | `8.19e-16` |
| diagonal fourth-moment residual | `4.44e-16` |
| sigma-weight sum residual | `0` |

The asymmetric offsets differ (`u=1.57485`, `v=1.62516`), so this is not the
symmetric rule identified as wrong in Phase 17. It remains a sigma-point
quadrature result, not a density or IID sample bank.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain GenUT as a source-faithful local quadrature candidate | feasibility and all selected moments pass | no fixture veto | actual q20 cloud feasibility unknown | run an actual-bank feasibility/status probe | no global density, mode discovery, posterior authority, HMC, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed |
| Statistically supported ranking | Not applicable |
| Descriptive-only differences | asymmetric sigma offsets and ridge-limited covariance residual |
| Default-readiness | Not ready; fixture only |
| Next evidence needed | q20 feasibility and target/status probe |

No HMC was launched.
