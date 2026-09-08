# Phase 19 Repair and Refresh Note

Status: `PASS_ETPF_Q20_PROBE_ROLE_LIMITED_REFRESHED_PHASE20`

This phase is a bounded q=20 integration probe. It preserves the authority
bank, target, mode axis, and source fixture controls.

| Failure | Classification | Repair |
|---|---|---|
| protocol/hash/index mismatch | input harness | fail closed and refresh metadata-bound receipt |
| Sinkhorn/Riccati constraint failure | implementation/scale | isolate the equation or fixture control; retain raw rows |
| target/status failure | candidate integration | record transformed-row evidence; do not clip or alter target |
| resource/shape failure | infrastructure | reduce only the declared probe size in a new reviewed subplan |
| poor mode/whitening diagnostic | explanatory | retain role-limited result; no authority or HMC conclusion |

After completion, write a result and refresh Phase 20 with exact transformed
target/status counts, support excursions, hashes, and the next source-faithful
arm decision.

Attempt 1 passed all target/status and transport-marginal checks but failed the
`1e-3` covariance gate at N=32 (`0.01552`). This is a numerical integration
failure at the source stopping tolerance, not a target or support failure. The
declared repair tightens the Riccati increment tolerance to `1e-5` and raises
the maximum iteration cap to `5000`; both are recorded as repair hypotheses,
not new defaults. Attempt 2 used a fresh output root and passed all gates. All
32 target/status rows are valid. The transformed subset leaves the retained
source range in 17 coordinate entries and has a corrected negative fraction of
`0.50586`; these remain support diagnostics. Refresh Phase 20 to a
source-faithful GenUT fixture rather than promoting or scaling ETPF directly.
