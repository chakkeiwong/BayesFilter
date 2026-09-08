# Phase 23 Repair and Refresh Note

Status: `PASS_SOURCE_FAITHFUL_LEDHPFPF_FIXTURE_REFRESHED_PHASE24`

The fixture preserves the source map and density role boundary.

| Failure | Classification | Repair |
|---|---|---|
| missing/hash-mismatched source or plan | harness | fail closed and create a fresh receipt |
| nonfinite/noninvertible step | numerical/implementation | isolate pseudo-time step and covariance input |
| inverse or determinant mismatch | density-contract implementation | repair map ordering or Jacobian product before q20 use |
| finite target but omitted proposal term | scientific contract | block admission and preserve failure |
| all gates pass | role-limited candidate | refresh a q20 affine-flow identity probe, not HMC |

After execution, record step determinants, inverse residual, density residual,
target-weight finiteness, source hashes, and the Phase 24 entry decision.

The fixture passed all gates. Refresh Phase 24 to audit whether the q20
SSL-LSTM target exposes the required state-space callback lifecycle. A
parameter-space affine map without those terms must remain classified as an
extension, not promoted as LEDH-PFPF.
