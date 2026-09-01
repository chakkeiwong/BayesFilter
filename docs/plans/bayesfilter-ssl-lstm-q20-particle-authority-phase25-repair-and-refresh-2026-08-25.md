# Phase 25 Repair and Refresh Note

Status: `DIRECT_FULL_STATE_LEDH_BLOCKED_REDUCED_REPAIR_REFRESHED_PHASE26`

## Repair matrix

| Finding | Classification | Required action |
|---|---|---|
| source/hash or target mismatch | harness/target | fail closed, preserve artifact, repair identity before rerun |
| nonfinite covariance or eigenvalues | numerical | isolate covariance construction and rerun the same fixed points |
| full-state induced covariance rank deficient | measure mismatch | do not invent a Lebesgue density; refresh a reduced-coordinate Phase 26 |
| reduced innovation density finite but parameter target unbound | scientific/interface | retain diagnostic only and investigate explicit parameter-space route separately |
| complete common-measure density/Jacobian identity | candidate-ready | refresh a source-bound adapter fixture with explicit nonclaims |

## Refresh rule

The next subplan must include the measured state/innovation/parameter
dimensions, eigenvalue spectrum and rank sensitivity, target signature, source
hashes, and the exact route classification. A candidate failure is not a
research-direction veto. Direct LEDH remains inadmissible until a common
measure and determinant identity are demonstrated.

## Executed result

The prescribed probe completed in `7.5 s` at
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase25-attempt1/`.
All finite/status hard checks passed. The induced covariance `G Q G^T` had
rank `20` at every tested point for state dimension `60`; the remaining `40`
coordinates are deterministic residual constraints. The reduced innovation
Gaussian density was finite, but the direct full-state LEDH route is blocked
relative to a Lebesgue density. The four-dimensional aggregate parameter
target remains unbound to this state-space density. Phase 26 is refreshed to a
reduced-coordinate fixture and explicit target-measure boundary check.
