# Phase 18 Result: Source-Faithful Second-Order LETF/ETPF Fixture

Status: `PASS_SOURCE_FAITHFUL_ETPF_FIXTURE`

The TensorFlow CPU-hidden eight-particle/two-dimensional fixture passed the
source-operation and numerical gates after the missing-subplan artifact repair.
The implementation follows the checked Sinkhorn-plus-Riccati route; it is not
yet a q=20 authority arm.

| Gate | Receipt |
|---|---:|
| finite tensors | true |
| Riccati convergence | true, 38 Euler iterations |
| corrected column-sum residual | `4.44e-16` |
| corrected row-sum residual | `8.88e-16` |
| equal-weight mean residual | `1.94e-16` |
| covariance residual | `5.97e-4` |
| source Riccati increment at stop | `9.49e-4` (source threshold `1e-3`) |
| base transport nonnegative fraction | `1.0` |
| corrected transform negative-entry fraction | `0.46875` |

The negative corrected entries are retained as an explicit diagnostic, not
clipped away: clipping would break the source moment construction. They also
confirm why transformed rows cannot be presented as IID posterior samples or
as a bounded-support density representation.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain the ETPF implementation as a fixture candidate | source operations and finite-cloud constraints pass | no fixture veto | only small synthetic cloud; Sinkhorn regularization is a hypothesis | run a bounded q=20 small-N integration probe with target/status checks | no q20 posterior authority, density identity, IID law, mode discovery, HMC, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed |
| Statistically supported ranking | Not applicable; no comparison |
| Descriptive-only differences | corrected map may leave convex hull; covariance residual follows source stop rule |
| Default-readiness | Not ready; fixture only |
| Next evidence needed | q20 target/status integration and independent support diagnostics |

No HMC was launched.
